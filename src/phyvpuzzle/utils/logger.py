"""
Experiment Logger Module

Provides structured logging for VLM multi-turn interactions, including
inputs, outputs, images, and errors. Directory layout:

logs/{model_name}/{task_name}/{timestamp}/
├── trial_info.json
└── rounds/
    ├── round_01/
    │   ├── input.json
    │   ├── output.json
    │   ├── pre_action.png
    │   └── post_action.png
    └── ...
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional
from PIL import Image


class ExperimentLogger:
    """Structured experiment logger for multi-turn VLM interaction."""

    def __init__(self, base_dir: str | Path = "logs") -> None:
        self.base_dir = Path(base_dir)
        self.model_name: Optional[str] = None
        self.task_name: Optional[str] = None
        self.timestamp: Optional[str] = None
        self.current_round_dir: Optional[Path] = None
        self.trial_dir: Optional[Path] = None
        self.round_index: int = 0

    @staticmethod
    def _sanitize(name: str) -> str:
        return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)[:128]

    def start_trial(self, model_name: str, task_name: str) -> Path:
        self.model_name = self._sanitize(model_name)
        self.task_name = self._sanitize(task_name)
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")

        self.trial_dir = self.base_dir / self.model_name / self.task_name / self.timestamp
        rounds_dir = self.trial_dir / "rounds"
        rounds_dir.mkdir(parents=True, exist_ok=True)
        return self.trial_dir

    def start_round(self, round_index: int) -> Path:
        assert self.trial_dir is not None, "Call start_trial() first"
        self.round_index = round_index
        self.current_round_dir = self.trial_dir / "rounds" / f"round_{round_index:02d}"
        self.current_round_dir.mkdir(parents=True, exist_ok=True)
        return self.current_round_dir

    def log_input(self, data: Dict[str, Any]) -> None:
        assert self.current_round_dir is not None, "Call start_round() first"
        path = self.current_round_dir / "input.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def log_output(self, data: Dict[str, Any]) -> None:
        assert self.current_round_dir is not None, "Call start_round() first"
        path = self.current_round_dir / "output.json"
        # Avoid dumping non-serializable raw SDK objects
        safe_data: Dict[str, Any] = {}
        for k, v in data.items():
            if k == "raw_response":
                try:
                    # Try to convert to plain dict if possible
                    safe_data[k] = getattr(v, "model_dump", lambda: str(v))()
                except Exception:
                    safe_data[k] = str(v)
            else:
                safe_data[k] = v
        with path.open("w", encoding="utf-8") as f:
            json.dump(safe_data, f, ensure_ascii=False, indent=2)

    def save_image(self, image: Image.Image, name: str) -> None:
        assert self.current_round_dir is not None, "Call start_round() first"
        image_path = self.current_round_dir / name
        image.save(str(image_path))

    def log_error(self, error: BaseException) -> None:
        assert self.trial_dir is not None, "Call start_trial() first"
        error_path = self.trial_dir / "errors.log"
        with error_path.open("a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
            f.write("\n")
            traceback.print_exc(file=f)

    def save_trial_info(self, info: Dict[str, Any]) -> None:
        assert self.trial_dir is not None, "Call start_trial() first"
        path = self.trial_dir / "trial_info.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
