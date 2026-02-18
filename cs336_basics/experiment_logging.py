from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_dir(
    base_dir: str | os.PathLike = "outputs/runs", *, name: str | None = None
) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    if name is None:
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    run_dir = base / name
    # Avoid collisions
    if run_dir.exists():
        suffix = int(time.time() * 1e6) % 1_000_000
        run_dir = base / f"{name}-{suffix:06d}"

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


@dataclass
class ExperimentLogger:
    """Minimal experiment logger.

    Writes:
    - metrics.jsonl: one JSON record per log() call
    - run.json: run metadata/config (optional)
    - notes.log: free-form text notes (optional)

    Designed to track curves vs both gradient step and wallclock time.
    """

    run_dir: Path

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._t0 = time.perf_counter()
        self._metrics_path = self.run_dir / "metrics.jsonl"
        self._notes_path = self.run_dir / "notes.log"

    @property
    def seconds_since_start(self) -> float:
        return float(time.perf_counter() - self._t0)

    def save_run_config(self, config: dict[str, Any]) -> None:
        path = self.run_dir / "run.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"created_at": _utc_now_iso(), **config}, f, indent=2, sort_keys=True
            )
            f.write("\n")

    def log(self, *, step: int, split: str, loss: float, **extra: Any) -> None:
        record: dict[str, Any] = {
            "time": _utc_now_iso(),
            "wallclock_s": self.seconds_since_start,
            "step": int(step),
            "split": str(split),
            "loss": float(loss),
        }
        record.update(extra)
        with open(self._metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")

    def note(self, message: str) -> None:
        with open(self._notes_path, "a", encoding="utf-8") as f:
            f.write(f"[{_utc_now_iso()}] {message}\n")
