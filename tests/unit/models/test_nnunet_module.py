import unittest

from models.nnunet_module import NNUnetModule


class MyTestCase(unittest.TestCase):

    def test_get_unet_params(self):
        self.nnunet_module = NNUnetModule(patch_size=[1024, 1024], spacings=[0.5, 0.5], exec_mode="training",
                                          deep_supervision=True, deep_supr_num=1, use_res_block=False, use_tta=True,
                                          steps=1000)
        assert self.nnunet_module.get_unet_params()


if __name__ == '__main__':
    unittest.main()
