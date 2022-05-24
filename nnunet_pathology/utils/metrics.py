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

import torch
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot
from torchmetrics import Metric


class Dice(Metric):
    def __init__(self, num_classes, ignore_first_channel=False):
        super().__init__(dist_sync_on_step=False)
        self.num_classes = num_classes
        self.dice_metric = DiceMetric(include_background=not ignore_first_channel)
        self.add_state("steps", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("dice", default=torch.zeros((num_classes - 1 if ignore_first_channel else num_classes,)),
                       dist_reduce_fx="sum")

    def update(self, predictions, targets):
        predictions, targets = self.ohe(torch.argmax(predictions.detach(), dim=1)),\
                               self.ohe(torch.argmax(targets.detach(), dim=1))
        self.steps += 1
        self.dice += torch.mean(torch.nan_to_num(self.dice_metric(predictions, targets), nan=0),
                                dim=0)  # Take mean over batch

    def compute(self, mean=False):
        return torch.mean(self.dice / self.steps) if mean else self.dice / self.steps

    def ohe(self, x):
        return one_hot(x.unsqueeze(1), num_classes=self.num_classes, dim=1)
