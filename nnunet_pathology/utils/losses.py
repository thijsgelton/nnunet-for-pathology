from monai.losses import DiceFocalLoss, DiceCELoss
from torch import nn


class LossFactory(nn.Module):
    def __init__(self, ignore_first_channel=False, focal=False):
        super(LossFactory, self).__init__()
        if focal:
            self.loss_fn = DiceFocalLoss(
                include_background=not ignore_first_channel, softmax=True, to_onehot_y=False, batch=True, gamma=2.0
            )
        else:
            self.loss_fn = DiceCELoss(include_background=not ignore_first_channel, softmax=True, to_onehot_y=False,
                                      batch=True)

    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)
