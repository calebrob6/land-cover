import sys
import os
import subprocess
from pathlib import Path
import tempfile
import rasterio as rio
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../landcover/"))
)
# pylint: disable=wrong-import-position
from landcover.train_model_landcover import Train

FILE_DIR = Path(os.path.dirname(os.path.realpath(__file__)))


def test_train_model_landcover_hr():
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        mock_data = FILE_DIR / "mock_data"
        subprocess.run(
            "python3 landcover/train_model_landcover.py --output %s --name test "
            "--data_dir %s --training_states test --validation_states test "
            "--model_type unet --epochs 1 --loss jaccard --batch_size 1"
            % (str(temp), str(mock_data)),
            shell=True,
            check=True,
        )
        print(os.listdir(str(temp)))
        output_dir = temp / "test"
        assert output_dir.is_dir()
        assert len(list(output_dir.rglob("*"))) == 6
        model_weights = output_dir / "final_model.h5"
        assert model_weights.is_file()


def test_train_model_landcover_transfer_hr():
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        mock_data = FILE_DIR / "mock_data"
        preload_weights = mock_data / "unet_final_model_transfer.h5"
        subprocess.run(
            "python3 landcover/train_model_landcover.py --output %s --name test "
            "--data_dir %s --training_states test --validation_states test "
            "--model_type unet --epochs 1 --loss jaccard --batch_size 1 --preload_weights %s"
            % (str(temp), str(mock_data), str(preload_weights)),
            shell=True,
            check=True,
        )
        print(os.listdir(str(temp)))
        output_dir = temp / "test"
        assert output_dir.is_dir()
        assert len(list(output_dir.rglob("*"))) == 6
        model_weights = output_dir / "final_model.h5"
        assert model_weights.is_file()


def test_train_model_landcover_sr():
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        mock_data = FILE_DIR / "mock_data"
        subprocess.run(
            "python3 landcover/train_model_landcover.py --output %s --name test "
            "--data_dir %s --training_states test --validation_states test "
            "--model_type unet --epochs 1 --loss superres --batch_size 1"
            % (str(temp), str(mock_data)),
            shell=True,
            check=True,
        )
        print(os.listdir(str(temp)))
        output_dir = temp / "test"
        assert output_dir.is_dir()
        assert len(list(output_dir.rglob("*"))) == 6
        model_weights = output_dir / "final_model.h5"
        assert model_weights.is_file()


def test_train_test_model_landcover():
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        mock_data = FILE_DIR / "mock_data"
        subprocess.run(
            "python3 landcover/train_model_landcover.py --output %s --name test "
            "--data_dir %s --training_states test --validation_states test "
            "--model_type unet --epochs 1 --loss jaccard --batch_size 1"
            % (str(temp), str(mock_data)),
            shell=True,
            check=True,
        )
        print(os.listdir(str(temp)))
        output_dir = temp / "test"
        assert output_dir.is_dir()
        assert len(list(output_dir.rglob("*"))) == 6
        model_weights = output_dir / "final_model.h5"
        assert model_weights.is_file()

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


def test_train_uint16_data():
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        mock_data = FILE_DIR / "mock_data"
        train = Train(
            temp,
            "ou_2018",
            mock_data,
            ["ou_2018"],
            ["ou_2018"],
            [],
            model_type="unet",
            loss="jaccard",
            epochs=1,
            batch_size=1,
            classes=5,
            lr_num_classes=11,
            hr_labels_index=4,
            lr_labels_index=5,
            hr_label_key="data/ou_to_hr_labels.txt",
            lr_label_key="",
            data_type="int16",
        )
        train.run_experiment(workers=1)
        print(os.listdir(str(temp)))
        output_dir = temp / "ou_2018"
        assert output_dir.is_dir()
        assert len(list(output_dir.rglob("*"))) == 6
        model_weights = output_dir / "final_model.h5"
        assert model_weights.is_file()
