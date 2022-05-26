# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List

import pytorch_lightning as pl
import torch
# from apex.optimizers import FusedAdam, FusedSGD # TODO: Look into Nvidia apex
from monai.networks.nets import DynUNet
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from torch.optim import SGD, Adam
from torchmetrics import MaxMetric, F1Score

from nnunet_pathology.utils.losses import LossFactory
from nnunet_pathology.utils.metrics import Dice
from nnunet_pathology.utils.utils import print0


class NNUnetModule(pl.LightningModule):
    def __init__(self, patch_size, spacings, exec_mode: str, deep_supervision: bool, deep_supr_num: int,
                 momentum: float, weight_decay: float, use_res_block: bool, use_tta: bool,
                 learning_rate: float, optimizer: str, use_focal_loss: bool, depth: int, num_classes: int,
                 steps: int, ignore_first_channel: bool, use_cosine_scheduler: bool = False, filters: List[int] = None,
                 in_channels: int = 3, min_fmap: int = 4):
        """

        Args:
            patch_size:
            spacings:
            use_tta: Enable test time augmentation
            in_channels: e.g. for RGB this will be 3
            min_fmap: Minimal dimension of feature map in the bottleneck
        """
        super(NNUnetModule, self).__init__()
        # Patch size is fixed in the initial version of this pathology framework.
        # Only depth is self-adapting. Both patch_size and pixel spacing can be used.
        self.save_hyperparameters(logger=False)
        self.model = None
        self.build_nnunet()
        self.loss = LossFactory(ignore_first_channel=self.hparams.ignore_first_channel,
                                focal=self.hparams.use_focal_loss)
        # TODO: Make what loss configurable, using factory method (not only CE or focal).
        self.tta_flips = [[2], [3], [2, 3]]
        self.train_dice = Dice(self.hparams.num_classes, ignore_first_channel=self.hparams.ignore_first_channel)
        # self.train_dice = F1Score(self.hparams.num_classes, average="macro", mdmc_average='samplewise',
        #                           ignore_index=0 if self.hparams.ignore_first_channel else None)
        # TODO: make this configurable
        self.val_dice = Dice(self.hparams.num_classes, ignore_first_channel=self.hparams.ignore_first_channel)
        # self.val_dice = F1Score(self.hparams.num_classes, average="macro", mdmc_average='samplewise',
        #                         ignore_index=0 if self.hparams.ignore_first_channel else None)
        # TODO: make this configurable
        self.val_best_dice = MaxMetric()
        # TODO: make this configurable

    def on_train_start(self) -> None:
        self.val_best_dice.reset()

    def on_validation_start(self) -> None:
        self.model.eval()
        self.model.train(False)

    def _forward(self, img):
        return self.tta_inference(img) if self.hparams.use_tta else self.do_inference(img)

    def compute_loss(self, preds, label):
        if self.model.training and self.hparams.deep_supervision:
            loss, weights = 0.0, 0.0
            for i in range(preds.shape[1]):
                loss += self.loss(preds[:, i], label) * 0.5 ** i
                weights += 0.5 ** i
            return loss / weights
        return self.loss(preds, label)

    def training_step(self, batch, batch_idx):
        patches, targets = batch[0].permute(0, 3, 1, 2), batch[1].permute(0, 3, 1, 2)
        predictions = self.model(patches)
        loss = self.compute_loss(predictions, targets)
        predictions = predictions[:, 0]  # Full-scale output, no deep supervision head.

        train_dice = self.train_dice(predictions, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/dice", train_dice, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        patches, targets = batch[0].permute(0, 3, 1, 2), batch[1].permute(0, 3, 1, 2)
        predictions = self._forward(patches)
        loss = self.loss(predictions, targets)
        val_dice = self.val_dice(predictions, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dice", val_dice, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        val_dice = self.val_dice.compute()  # get val dice from current epoch
        self.val_best_dice.update(torch.mean(val_dice))

        if self.hparams.num_classes > 1:
            for i in range(self.hparams.num_classes - (1 if self.hparams.ignore_first_channel else 0)):
                self.log(f"val/dice_class_{i}", val_dice[i], on_epoch=True, prog_bar=False)
        self.log("val/dice_best", self.val_best_dice.compute(), on_epoch=True, prog_bar=True)
        self.log("val/dice", torch.mean(val_dice), on_epoch=True, prog_bar=True)
        self.val_dice.reset()

    def on_train_epoch_end(self):
        self.train_dice.reset()

    def get_unet_params(self):
        strides, kernels, sizes, spacings = [], [], self.hparams.patch_size[:], self.hparams.spacings
        while True:
            spacing_ratio = [spacing / min(self.hparams.spacings) for spacing in self.hparams.spacings]
            stride = [
                2 if ratio <= 2 and size >= 2 * self.hparams.min_fmap else 1 for (ratio, size) in
                zip(spacing_ratio, sizes)
            ]
            kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
            if all(s == 1 for s in stride):
                break
            sizes = [i / j for i, j in zip(sizes, stride)]
            spacings = [i * j for i, j in zip(self.hparams.spacings, stride)]
            kernels.append(kernel)
            strides.append(stride)
            if len(strides) == self.hparams.depth:
                break
        strides.insert(0, len(spacings) * [1])
        kernels.append(len(spacings) * [3])
        return kernels, strides

    def build_nnunet(self):
        kernels, strides = self.get_unet_params()
        self.model = DynUNet(
            2,  # spatial_dims will always be 2 in the case of pathology
            self.hparams.in_channels,
            self.hparams.num_classes,
            kernels,
            strides,
            strides[1:],
            filters=self.hparams.filters,
            norm_name=("INSTANCE", {"affine": True}),
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            deep_supervision=self.hparams.deep_supervision,
            deep_supr_num=self.hparams.deep_supr_num,
            res_block=self.hparams.use_res_block,
            trans_bias=True,
        )
        print0(f"Filters: {self.model.filters},\nKernels: {kernels}\nStrides: {strides}")

    def do_inference(self, image):
        if self.hparams.exec_mode == "predict":
            return self.inference2d_test(image)
        return self.inference2d(image)

    def tta_inference(self, img):
        pred = self.do_inference(img)
        for flip_idx in self.tta_flips:
            pred += torch.flip(self.do_inference(torch.flip(img, flip_idx)), flip_idx)
        pred /= len(self.tta_flips) + 1
        return pred

    def inference2d(self, image):
        predictions = self.model(image)
        return predictions

    def inference2d_test(self, image):
        """
        TODO: There should be a WSI inference part. This sliding_window approach won't be necessary and also takes
              too much time.
        Args:
            image:

        Returns:

        """
        predictions_shape = (image.shape[0], self.hparams.num_classes, *image.shape[2:])
        predictions = torch.zeros(predictions_shape, dtype=image.dtype, device=image.device)
        for depth in range(image.shape[2]):
            predictions[:, :, depth] = self.sliding_window_inference(image[:, :, depth])
        return predictions

    def configure_optimizers(self):
        optimizer = {
            "sgd": SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum),
            "adam": Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay),
        }[self.hparams.optimizer.lower()]

        if self.hparams.use_cosine_scheduler:
            scheduler = {
                "scheduler": WarmupCosineSchedule(
                    optimizer=optimizer,
                    warmup_steps=250,
                    t_total=self.trainer.max_epochs * self.hparams.steps,
                ),
                "interval": "step",
                "frequency": 1,
            }
            return {"optimizer": optimizer, "monitor": "val_loss", "lr_scheduler": scheduler}
        return {"optimizer": optimizer, "monitor": "val_loss"}
