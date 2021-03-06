import numpy as np
import torch
from torch.utils.data import Dataset
from wholeslidedata.iterators import create_batch_iterator


class WholeSlideDataset(Dataset):

    def __init__(self, user_config, exec_mode, num_workers, steps, norm_mean=None, norm_std=None,
                 return_info=False, context="fork"):
        """
        # TODO: user_config should be replaced by a dictionary with the corresponding values. This way all config
                can be done using Hydra.

        Args:
            user_config:
            exec_mode:
            num_workers:
            steps:
            return_info:
            context:
        """
        self.norm_std = norm_std
        self.norm_mean = norm_mean
        self.steps = steps
        self.return_info = return_info
        self.iterator = create_batch_iterator(user_config=user_config,
                                              mode=exec_mode,
                                              cpus=num_workers,
                                              return_info=return_info,
                                              context=context)

    def __len__(self):
        return self.steps

    def __getitem__(self, index):
        if self.return_info:
            x_batch, y_batch, info = next(self.iterator)
        else:
            x_batch, y_batch = next(self.iterator)

        if isinstance(self.norm_mean, np.ndarray) and self.norm_mean.any():
            x_batch = (x_batch - self.norm_mean) / self.norm_std

        x_batch = x_batch.astype('float32')
        y_batch = y_batch.astype('int8')

        if self.return_info:
            return torch.from_numpy(x_batch), torch.from_numpy(y_batch), info
        return torch.from_numpy(x_batch), torch.from_numpy(y_batch)
