from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nba_predictor.config import settings


@dataclass
class TimesFMHandle:
    model: Any | None
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.model is not None


def load_timesfm() -> TimesFMHandle:
    try:
        import timesfm  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        return TimesFMHandle(model=None, error=f"timesfm import failed: {exc}")

    local_model_dir = settings.model_dir / "timesfm" / settings.timesfm_model_version
    model_ref: str | Path = local_model_dir if local_model_dir.exists() else "google/timesfm-2.5-200m-pytorch"
    try:
        load_kwargs = {"local_files_only": True} if local_model_dir.exists() else {}
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(str(model_ref), **load_kwargs)
        model.compile(
            timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=256,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        return TimesFMHandle(model=model)
    except Exception as exc:  # pragma: no cover - depends on local model/backend
        return TimesFMHandle(model=None, error=f"timesfm load failed: {exc}")
