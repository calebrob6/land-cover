# pylint: disable=wrong-import-position
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import keras.backend as K
from keras.models import Model

from landcover.architectures import FC_DenseNet, UNet


def test_fc_dense_net():
    K.clear_session()
    i, o = FC_DenseNet((240, 240, 4), dims=[32, 16, 16, 16, 16], out_ch=5)
    model = Model(inputs=i, outputs=o)
    assert model.count_params() == 249177


def test_unet_large():
    K.clear_session()
    i, o = UNet((240, 240, 4), dims=[32, 64, 128, 256, 128], out_ch=5)
    model = Model(inputs=i, outputs=o)
    assert model.count_params() == 5998277


def test_unet_small():
    K.clear_session()
    i, o = UNet((240, 240, 4), dims=[64, 32, 32, 32, 32], out_ch=5)
    model = Model(inputs=i, outputs=o)
    assert model.count_params() == 480133
