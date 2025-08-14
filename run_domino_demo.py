#!/usr/bin/env python3
"""
Run a minimal domino demo to validate multi-turn logging, tool calls, and metrics.
"""
from __future__ import annotations

import os
from pathlib import Path

from src.phyvpuzzle.core.pipeline import PipelineConfig, create_pipeline
from src.phyvpuzzle.tasks.domino_task import DominoTask


def main():
    config = PipelineConfig(
        vllm_type="openai",
        vllm_model=os.getenv("VLB_MODEL", "gpt-4o"),
        translator_type="rule_based",
        environment_type="pybullet",
        max_iterations=5,
        timeout=60.0,
        enable_logging=True,
        log_level="INFO",
        feedback_history_size=5,
        retry_attempts=1,
    )

    pipeline = create_pipeline(config)
    with pipeline:
        task = DominoTask()
        result = pipeline.execute_task(task)
        print({
            "success": result.success,
            "final_score": result.final_score,
            "steps_taken": result.steps_taken,
            "time_taken": result.time_taken,
            "metrics": result.metrics,
        })


if __name__ == "__main__":
    main()
