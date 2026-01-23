"""Luban Lock benchmark runner with parallel level evaluation."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from queue import Queue
from typing import Dict, Iterable, List, Optional, Tuple

from phyvpuzzle import load_config, validate_config
from phyvpuzzle.core.base import TaskResult
from phyvpuzzle.evaluation.metrics import MetricsCalculator
from phyvpuzzle.runner import BenchmarkRunner


@dataclass
class Pricing:
    """Token pricing in USD per 1K tokens."""

    input_per_1k: float = 0.0
    output_per_1k: float = 0.0


@dataclass
class MetricSummary:
    accuracy: float
    pass_at_k: Dict[int, float]
    avg_at_k: Dict[int, float]
    avg_steps_solved: float
    distance_to_optimal: float
    tokens_per_solved: float
    tokens_per_step: float
    cost_total: float
    usd_per_solved: float
    total_tasks: int
    solved_tasks: int


def _clone_config_with_suffix(config_path: str, suffix: str):
    config = load_config(config_path)
    issues = validate_config(config)
    errors = [i for i in issues if i.startswith("ERROR")]
    if errors:
        raise RuntimeError(f"Config validation failed: {errors}")
    config.runner.experiment_name = f"{config.runner.experiment_name}_{suffix}"
    return config


def _apply_luban_overrides(config, level_index: int, env_id: int, seed: Optional[int]) -> None:
    if hasattr(config.environment, "env_id"):
        config.environment.env_id = env_id
    if hasattr(config.environment, "level_index"):
        config.environment.level_index = level_index
    if hasattr(config.task, "level_index"):
        config.task.level_index = level_index
    if seed is not None:
        if hasattr(config.task, "init_seed"):
            config.task.init_seed = seed
        if hasattr(config.environment, "init_seed"):
            config.environment.init_seed = seed


def _run_single_task(
    config_path: str,
    batch_id: int,
    run_id: int,
    level_index: int,
    env_id: int,
    seed: Optional[int] = None,
) -> TaskResult:
    suffix = f"b{batch_id}_r{run_id}_l{level_index}_e{env_id}_{int(time.time() * 1000)}"
    config = _clone_config_with_suffix(config_path, suffix)
    _apply_luban_overrides(config, level_index=level_index, env_id=env_id, seed=seed)
    runner = BenchmarkRunner(config)
    runner.setup()
    result = runner.run_single_task()
    evaluation = runner.evaluate([result])
    evaluated = evaluation.task_results[0]
    evaluated.metadata["group_key"] = f"level_{level_index}"
    evaluated.metadata["config_path"] = config_path
    evaluated.metadata["env_id"] = env_id
    evaluated.metadata["level_index"] = level_index
    
    # Export results to Excel and detailed reports (same as run_benchmark)
    # This ensures luban_task has the same logging capabilities as stacking_task
    import os
    runner.evaluator.export_results_to_excel(
        evaluation,
        os.path.join(runner.logger.run_dir, config.runner.results_excel_path),
        config.agent.model_name
    )
    
    runner.evaluator.export_detailed_report(
        evaluation,
        os.path.join(runner.logger.run_dir, "detailed_reports"),
        config.agent.model_name
    )
    
    return evaluated


def _calculate_metrics(
    task_results: List[TaskResult],
    pricing: Pricing,
    k_values: Optional[List[int]] = None,
) -> MetricSummary:
    if k_values is None:
        k_values = [1, 3, 5]

    calculator = MetricsCalculator()
    total_tasks = len(task_results)
    solved_results = [r for r in task_results if r.success]
    solved_tasks = len(solved_results)
    accuracy = calculator.calculate_accuracy(task_results)
    pass_at_k = calculator.calculate_pass_at_k(task_results, k_values)
    avg_at_k = calculator.calculate_avg_at_k(task_results, k_values)
    avg_steps_solved = calculator.calculate_step_efficiency(task_results)
    distance_to_optimal = calculator.calculate_distance_to_optimal(task_results)
    tokens_per_solved = calculator.calculate_token_efficiency(task_results)
    tokens_per_step = calculator.calculate_tokens_per_step(task_results)
    total_cost = calculator.calculate_cost(
        task_results,
        price_in=pricing.input_per_1k,
        price_out=pricing.output_per_1k,
    )
    usd_per_solved = calculator.calculate_usd_per_solved(
        task_results,
        price_in=pricing.input_per_1k,
        price_out=pricing.output_per_1k,
    )

    return MetricSummary(
        accuracy=accuracy,
        pass_at_k=pass_at_k,
        avg_at_k=avg_at_k,
        avg_steps_solved=avg_steps_solved,
        distance_to_optimal=distance_to_optimal,
        tokens_per_solved=tokens_per_solved,
        tokens_per_step=tokens_per_step,
        cost_total=total_cost,
        usd_per_solved=usd_per_solved,
        total_tasks=total_tasks,
        solved_tasks=solved_tasks,
    )


def _print_metrics(title: str, summary: MetricSummary) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("-" * 80)
    print(f"Tasks: {summary.total_tasks}, Solved: {summary.solved_tasks}, Accuracy: {summary.accuracy:.2%}")
    if summary.pass_at_k:
        pass_k = ", ".join([f"Pass@{k}: {v:.2%}" for k, v in summary.pass_at_k.items()])
        print(pass_k)
    if summary.avg_at_k:
        avg_k = ", ".join([f"Avg@{k}: {v:.2%}" for k, v in summary.avg_at_k.items()])
        print(avg_k)
    print(f"AvgSteps (solved): {summary.avg_steps_solved:.2f}")
    print(f"Distance-to-Optimal: {summary.distance_to_optimal:.2f}")
    print(f"Tokens/Solved: {summary.tokens_per_solved:.2f}")
    print(f"Tokens/Step: {summary.tokens_per_step:.2f}")
    print(f"Total Cost (USD): {summary.cost_total:.4f}")
    print(f"USD/Solved: {summary.usd_per_solved:.4f}")
    print("=" * 80)


def _build_level_list(levels: Optional[List[int]], level_range: Optional[str], default_level: int) -> List[int]:
    if levels:
        return levels
    if level_range:
        start_str, end_str = level_range.split("-", 1)
        start = int(start_str)
        end = int(end_str)
        if end < start:
            raise ValueError("level-range must be in ascending order, e.g., 0-31")
        return list(range(start, end + 1))
    return [default_level]


def run_parallel_levels(
    config_path: str,
    levels: Optional[List[int]],
    level_range: Optional[str],
    runs_per_level: int,
    pricing: Pricing,
    max_workers: Optional[int] = None,
    seed_base: Optional[int] = None,
) -> List[TaskResult]:
    config = load_config(config_path)
    level_list = _build_level_list(levels, level_range, getattr(config.task, "level_index", 0))
    workers = min(max_workers or min(8, len(level_list) * runs_per_level), 8)
    env_pool: Queue[int] = Queue()
    for env_id in range(1, workers + 1):
        env_pool.put(env_id)

    all_results: List[TaskResult] = []
    futures = []
    print(f"\nStarting parallel runs for {len(level_list)} levels with up to {workers} workers...")

    def _run_with_env(batch_id: int, run_id: int, level_index: int, seed: Optional[int]) -> TaskResult:
        env_id = env_pool.get()
        try:
            return _run_single_task(
                config_path=config_path,
                batch_id=batch_id,
                run_id=run_id,
                level_index=level_index,
                env_id=env_id,
                seed=seed,
            )
        finally:
            env_pool.put(env_id)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        batch_id = 1
        for level_index in level_list:
            for run_id in range(1, runs_per_level + 1):
                seed = seed_base + (level_index * 1000 + run_id) if seed_base is not None else None
                futures.append(
                    executor.submit(_run_with_env, batch_id, run_id, level_index, seed)
                )
        for future in as_completed(futures):
            all_results.append(future.result())

    summary = _calculate_metrics(all_results, pricing, k_values=[runs_per_level])
    _print_metrics("Overall Metrics (Luban Levels)", summary)
    return all_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parallel runner for Luban Lock tasks.")
    parser.add_argument("--config", required=True, help="Path to a Luban YAML config.")
    parser.add_argument("--levels", nargs="+", type=int, help="Level indices to run (0-31).")
    parser.add_argument("--level-range", help="Inclusive level range, e.g., 0-7.")
    parser.add_argument("--runs-per-level", type=int, default=1, help="Runs per level.")
    parser.add_argument("--workers", type=int, default=None, help="Max parallel workers (<= 8).")
    parser.add_argument("--price-in", type=float, default=0.1, help="USD per 1K input tokens.")
    parser.add_argument("--price-out", type=float, default=0.1, help="USD per 1K output tokens.")
    parser.add_argument("--seed-base", type=int, default=None, help="Base seed for deterministic runs.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    pricing = Pricing(input_per_1k=args.price_in, output_per_1k=args.price_out)
    run_parallel_levels(
        config_path=args.config,
        levels=args.levels,
        level_range=args.level_range,
        runs_per_level=max(1, args.runs_per_level),
        pricing=pricing,
        max_workers=args.workers,
        seed_base=args.seed_base,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python -m phyvpuzzle.luban_runner --config eval_configs/luban.yaml --levels 0 --runs-per-level 1