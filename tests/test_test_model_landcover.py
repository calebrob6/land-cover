import sys
import os
import subprocess
from pathlib import Path
import tempfile
import rasterio as rio
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

FILE_DIR = Path(os.path.dirname(os.path.realpath(__file__)))


def test_test_model_landcover_hr():
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        mock_data = FILE_DIR / "mock_data"
        model_weights = mock_data / "hr_final_model.h5"
        input_csv = mock_data / "test_extended-test_tiles.csv"
        subprocess.run(
            "python3 landcover/testing_model_landcover.py --input %s --output %s --model %s"
            % (str(input_csv), str(temp), str(model_weights)),
            shell=True,
            check=True,
        )
        out_file = temp / "m_3907833_nw_17_1_naip-new_class.tif"
        assert out_file.is_file()
        with rio.open(out_file) as src:
            array = src.read()
            assert np.nonzero(array[0])
            assert array.shape == (1, 260, 260)


def test_test_model_landcover_sr():
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        mock_data = FILE_DIR / "mock_data"
        model_weights = mock_data / "sr_final_model.h5"
        input_csv = mock_data / "test_extended-test_tiles.csv"
        subprocess.run(
            "python3 landcover/testing_model_landcover.py --input %s --output %s --model %s --superres"
            % (str(input_csv), str(temp), str(model_weights)),
            shell=True,
            check=True,
        )
        out_file = temp / "m_3907833_nw_17_1_naip-new_class.tif"
        assert out_file.is_file()
        with rio.open(out_file) as src:
            array = src.read()
            assert np.nonzero(array[0])
            assert array.shape == (1, 260, 260)
