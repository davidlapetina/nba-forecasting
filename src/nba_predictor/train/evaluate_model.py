from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import RocCurveDisplay, roc_curve

from nba_predictor.config import settings
from nba_predictor.train.train_classifier import (
    FEATURE_COLUMNS,
    blended_probabilities,
    elo_probabilities,
    evaluate_predictions,
    history_probabilities,
    load_training_data,
    time_based_split,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_evaluation_artifacts(
    y_true: pd.Series,
    probabilities: pd.Series,
    output_dir: Path | None = None,
) -> dict[str, str]:
    target_dir = output_dir or settings.data_dir / "processed" / "evaluation"
    target_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, probabilities)
    roc_path = target_dir / "roc_curve.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(roc_path)
    plt.close(fig)

    prob_true, prob_pred = calibration_curve(y_true, probabilities, n_bins=10, strategy="uniform")
    calibration_path = target_dir / "calibration_curve.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#555555")
    ax.plot(prob_pred, prob_true, marker="o")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed home win rate")
    ax.set_title("Calibration Curve")
    fig.tight_layout()
    fig.savefig(calibration_path)
    plt.close(fig)

    frame = pd.DataFrame(
        {
            "actual_home_win": y_true.astype(int).reset_index(drop=True),
            "home_win_probability": probabilities.reset_index(drop=True),
        }
    )
    predictions_path = target_dir / "validation_predictions.csv"
    frame.to_csv(predictions_path, index=False)

    return {
        "roc_curve": str(roc_path),
        "calibration_curve": str(calibration_path),
        "validation_predictions": str(predictions_path),
    }


def evaluate_saved_model() -> dict[str, float]:
    frame = load_training_data()
    _, valid_df = time_based_split(frame)
    metadata_path = settings.model_dir / "classifier" / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_path = settings.model_dir / "classifier" / f"game_winner_{metadata['model_name']}.joblib"
    model = joblib.load(model_path)
    probabilities = pd.Series(model.predict_proba(valid_df[FEATURE_COLUMNS])[:, 1], index=valid_df.index)
    y_true = valid_df["home_team_win"].astype(int)
    metrics = evaluate_predictions(y_true, probabilities)
    elo_probability = elo_probabilities(valid_df)
    history_probability = history_probabilities(valid_df)
    blend_probability = blended_probabilities(
        probabilities,
        elo_probability,
        history_probability,
        metadata.get("blend_weights", {"classifier": 0.6, "elo": 0.3, "history": 0.1}),
    )
    elo_metrics = evaluate_predictions(y_true, elo_probability)
    history_metrics = evaluate_predictions(y_true, history_probability)
    blend_metrics = evaluate_predictions(y_true, blend_probability)
    artifacts = save_evaluation_artifacts(y_true, blend_probability)
    output_dir = settings.data_dir / "processed" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latest_metrics.json").write_text(
        json.dumps(
            {
                "blend": blend_metrics,
                "classifier": metrics,
                "elo_baseline": elo_metrics,
                "history_baseline": history_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    metadata["evaluation_artifacts"] = artifacts
    metadata["metrics"] = metrics
    metadata["elo_baseline_metrics"] = elo_metrics
    metadata["history_baseline_metrics"] = history_metrics
    metadata["blend_metrics"] = blend_metrics
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "blend": blend_metrics,
                "classifier": metrics,
                "elo_baseline": elo_metrics,
                "history_baseline": history_metrics,
            },
            indent=2,
        )
    )
    return blend_metrics


def main() -> None:
    evaluate_saved_model()


if __name__ == "__main__":
    main()
