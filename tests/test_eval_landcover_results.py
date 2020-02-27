import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# pylint: disable=wrong-import-position
from landcover.eval_landcover_results import eval_landcover_results

FILE_DIR = Path(os.path.dirname(os.path.realpath(__file__)))


def test_eval_landcover_result():
    r = eval_landcover_results(FILE_DIR / "mock_data/log_acc_test.txt")
    assert len(r) == 4
    assert round(r[0], 2) == 0.83  # All acc
    assert round(r[1], 2) == 0.42  # All jac
    assert round(r[2], 2) == 0.74  # Developed acc
    assert round(r[3], 2) == 0.34  # Developed jac
