import os
from pathlib import Path

import gdown
import pytest
import torch
import yaml

from datamodules.wsi_datamodule import WholeSlideDataModule
from tests.helpers.module_available import _IS_WINDOWS

_WSI_DOWNLOAD_LINK = 'https://drive.google.com/uc?id=1noRtbC5fxBlnO7YnvktjIDhFI61PdOSB'
_WSI_NAME = Path('TCGA-21-5784-01Z-00-DX1.tif')

_WSA_DOWNLOAD_LINK = 'https://drive.google.com/uc?id=1jkTp0IJHHpmLd1yDO1L3KRFJgm0STh0d'
_WSA_NAME = Path('TCGA-21-5784-01Z-00-DX1.xml')


def _download(output_folder, download_link, name):
    output_path = Path(output_folder) / name
    if not output_path.exists():
        gdown.download(download_link, str(output_path))
    return str(output_path)


def download_wsi(output_folder):
    return _download(output_folder, _WSI_DOWNLOAD_LINK, _WSI_NAME)


def download_wsa(output_folder):
    return _download(output_folder, _WSA_DOWNLOAD_LINK, _WSA_NAME)


@pytest.fixture
def download():
    """
    Creates a data.yml file with the corresponding local path locations.
    Not using the teardown section (after yield) to delete the files, since the tif file is large and re-downloading
    everytime would slow down testing too much.
    """
    output_folder = "test_files"
    wsi = download_wsi(output_folder)
    wsa = download_wsa(output_folder)
    with open(os.path.join(output_folder, "data.yml"), "w") as yml:
        ws_locations = [{"wsi": {"path": wsi}, "wsa": {"path": wsa}}]
        yaml.dump({"training": ws_locations, "validation": ws_locations, "test": ws_locations}, yml)


def test_wsi_datamodule(download):
    with open("test_files/user_config.yml", "r") as fp_config:
        wsd_config = yaml.load(fp_config, yaml.FullLoader)

    datamodule = WholeSlideDataModule(
        user_train_config="test_files/user_config.yml",
        user_val_config="test_files/user_config.yml",
        user_test_config="test_files/user_config.yml",
        context="spawn" if _IS_WINDOWS else "fork",
        num_workers=6,
        num_classes=3
    )

    assert not datamodule.data_train and not datamodule.data_val and not datamodule.data_test

    datamodule.setup()

    assert datamodule.data_train and datamodule.data_val
    assert (
            len(datamodule.data_train) + len(datamodule.data_val) == 1200
    )

    assert datamodule.train_dataloader()
    assert datamodule.val_dataloader()

    batch = next(iter(datamodule.train_dataloader()))
    x, y = batch

    assert len(x) == wsd_config['wholeslidedata']['default']['batch_shape']['batch_size']
    assert len(y) == wsd_config['wholeslidedata']['default']['batch_shape']['batch_size']
    assert x.dtype == torch.float32
    assert y.dtype == torch.int8
