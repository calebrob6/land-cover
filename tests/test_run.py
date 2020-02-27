import os
import subprocess
from pathlib import Path
import tempfile

FILE_DIR = Path(os.path.dirname(os.path.realpath(__file__)))


def test_run_with_subprocess():
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        mock_data = FILE_DIR / "mock_data"
        subprocess.run(
            "python3 run.py --output %s --name a_test "
            "--data-dir %s --training-states test --validation-states test "
            "--test-states test "
            "--model unet --epochs 1 --loss jaccard --batch-size 1"
            % (str(temp), str(mock_data)),
            shell=True,
            check=True,
        )

        output_dir = temp / "a_test/"
        print(os.listdir(str(output_dir)))
        assert output_dir.is_dir()
        assert len(list(output_dir.rglob("*"))) == 8


def test_run_with_subprocess_sr():
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        mock_data = FILE_DIR / "mock_data"
        subprocess.run(
            "python3 run.py --output %s --name a_test "
            "--data-dir %s --training-states test --validation-states test --superres-states test "
            "--test-states test "
            "--model unet --epochs 1 --loss superres --batch-size 1"
            % (str(temp), str(mock_data)),
            shell=True,
            check=True,
        )

        output_dir = temp / "a_test/"
        print(os.listdir(str(output_dir)))
        assert output_dir.is_dir()
        assert len(list(output_dir.rglob("*"))) == 8
