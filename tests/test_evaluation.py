from __future__ import annotations

from pathlib import Path

import pandas as pd

from nba_predictor.train.evaluate_model import save_evaluation_artifacts


def test_evaluation_artifacts_are_written(tmp_path: Path) -> None:
    artifacts = save_evaluation_artifacts(
        pd.Series([0, 1, 0, 1]),
        pd.Series([0.1, 0.8, 0.2, 0.9]),
        tmp_path,
    )
    assert Path(artifacts["roc_curve"]).exists()
    assert Path(artifacts["calibration_curve"]).exists()
    assert Path(artifacts["validation_predictions"]).exists()
    frame = pd.read_csv(artifacts["validation_predictions"])
    assert frame.shape == (4, 2)
    assert not frame.isna().any().any()
