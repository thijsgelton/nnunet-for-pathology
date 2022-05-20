from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.datamodules.datasets.wsi_dataset import WholeSlideDataset


class WholeSlideDataModule(LightningDataModule):
    """
    Encapsulating module that uses the WsiDataset. This in turn uses WholeSlideData as a generator for patches and
    this uses augmentations (if specified) on every batch.
    """

    def __init__(
            self,
            user_train_config,
            user_val_config,
            user_test_config,
            num_classes,
            steps_per_epoch: int = 1000,
            val_steps_per_epoch: int = 200,
            num_workers: int = 0,
            pin_memory: bool = False,
            context: str = "fork",
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return self.hparams.num_classes

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.

        # TODO: Only require the paths of all WSIs and use configuration to (stratified) split the data.
                Stratification could be done using a pattern selector.
        # TODO: Create a separate module specifically for WSI inference.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset_kwargs = dict(
                user_config=self.hparams.user_train_config,
                num_workers=self.hparams.num_workers,
                steps=self.hparams.steps_per_epoch,
                exec_mode="training",
                context=self.hparams.context
            )
            self.data_train = WholeSlideDataset(**dataset_kwargs)

            dataset_kwargs['exec_mode'] = "validation"
            dataset_kwargs['steps'] = self.hparams.val_steps_per_epoch
            dataset_kwargs['user_config'] = self.hparams.user_val_config
            self.data_val = WholeSlideDataset(**dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            collate_fn=lambda x: x[0],  # WSD provides batches and DataLoader wraps it. Therefore, unpacking is needed.
            pin_memory=self.hparams.pin_memory,
            num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            collate_fn=lambda x: x[0],
            pin_memory=self.hparams.pin_memory,
            num_workers=0
        )
