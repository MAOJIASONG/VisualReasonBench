"""Stacking-game benchmark runner with parallel evaluation and custom metrics."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

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


def _apply_seed_overrides(config, seed: Optional[int]) -> None:
    if seed is None:
        return
    if hasattr(config.task, "init_seed"):
        config.task.init_seed = seed
    if hasattr(config.environment, "init_seed"):
        config.environment.init_seed = seed


def _run_single_task(
    config_path: str,
    batch_id: int,
    run_id: int,
    seed: Optional[int] = None,
) -> TaskResult:
    suffix = f"b{batch_id}_r{run_id}_{int(time.time() * 1000)}"
    config = _clone_config_with_suffix(config_path, suffix)
    _apply_seed_overrides(config, seed)
    runner = BenchmarkRunner(config)
    runner.setup()
    result = runner.run_single_task()
    evaluation = runner.evaluate([result])
    evaluated = evaluation.task_results[0]
    evaluated.metadata["group_key"] = config.task.name
    evaluated.metadata["difficulty"] = config.task.difficulty.value if config.task.difficulty else "unknown"
    evaluated.metadata["config_path"] = config_path
    return evaluated


def _group_results(task_results: Iterable[TaskResult]) -> Dict[str, List[TaskResult]]:
    grouped: Dict[str, List[TaskResult]] = {}
    for result in task_results:
        group_key = result.metadata.get("group_key", result.task_id)
        grouped.setdefault(group_key, []).append(result)
    return grouped


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


def run_parallel_kxk(
    config_path: str,
    k: int,
    pricing: Pricing,
    max_workers: Optional[int] = None,
    seed_base: Optional[int] = None,
) -> List[TaskResult]:
    os.environ.setdefault("MPLBACKEND", "Agg")
    all_results: List[TaskResult] = []
    workers = max_workers or k
    for batch_id in range(1, k + 1):
        print(f"\nStarting batch {batch_id}/{k} with {k} parallel runs...")
        batch_results: List[TaskResult] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for run_id in range(1, k + 1):
                seed = seed_base + (batch_id * 1000 + run_id) if seed_base is not None else None
                futures.append(
                    executor.submit(_run_single_task, config_path, batch_id, run_id, seed)
                )
            for future in as_completed(futures):
                batch_results.append(future.result())

        batch_results.sort(key=lambda r: r.task_id)
        all_results.extend(batch_results)
        summary = _calculate_metrics(batch_results, pricing, k_values=[k])
        _print_metrics(f"Batch {batch_id} Metrics (K={k})", summary)

    overall = _calculate_metrics(all_results, pricing, k_values=[k])
    _print_metrics(f"Overall Metrics (KxK, K={k})", overall)
    return all_results


def run_parallel_difficulty_tasks(
    config_paths: List[str],
    pricing: Pricing,
    max_workers: Optional[int] = None,
    runs_per_config: int = 1,
    seed_base: Optional[int] = None,
) -> List[TaskResult]:
    os.environ.setdefault("MPLBACKEND", "Agg")
    workers = max_workers or len(config_paths)
    all_results: List[TaskResult] = []
    print(f"\nStarting parallel runs for {len(config_paths)} configs...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for config_index, config_path in enumerate(config_paths):
            for run_id in range(1, runs_per_config + 1):
                seed = seed_base + (config_index * 1000 + run_id) if seed_base is not None else None
                futures.append(
                    executor.submit(_run_single_task, config_path, 1, run_id, seed)
                )
        for future in as_completed(futures):
            all_results.append(future.result())

    summary = _calculate_metrics(all_results, pricing, k_values=[runs_per_config])
    _print_metrics("Overall Metrics (Different Difficulties)", summary)

    difficulty_groups: Dict[str, List[TaskResult]] = {}
    for result in all_results:
        difficulty = result.metadata.get("difficulty", "unknown")
        difficulty_groups.setdefault(difficulty, []).append(result)
    for difficulty, results in difficulty_groups.items():
        diff_summary = _calculate_metrics(results, pricing, k_values=[runs_per_config])
        _print_metrics(f"Difficulty '{difficulty}' Metrics", diff_summary)

    return all_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parallel runner for stacking_game tasks.")
    parser.add_argument("--config", help="Path to a single stacking_game YAML config.")
    parser.add_argument(
        "--configs",
        nargs="+",
        help="List of stacking_game YAML configs (different difficulties).",
    )
    parser.add_argument("--k", type=int, default=1, help="K for KxK parallel testing.")
    parser.add_argument("--workers", type=int, default=None, help="Max parallel workers.")
    parser.add_argument("--runs-per-config", type=int, default=1, help="Runs per config.")
    parser.add_argument("--price-in", type=float, default=0.1, help="USD per 1K input tokens.")
    parser.add_argument("--price-out", type=float, default=0.1, help="USD per 1K output tokens.")
    parser.add_argument("--seed-base", type=int, default=None, help="Base seed for deterministic runs.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    pricing = Pricing(input_per_1k=args.price_in, output_per_1k=args.price_out)

    if args.config:
        run_parallel_kxk(
            config_path=args.config,
            k=max(1, args.k),
            pricing=pricing,
            max_workers=args.workers,
            seed_base=args.seed_base,
        )
        return 0

    if args.configs:
        run_parallel_difficulty_tasks(
            config_paths=args.configs,
            pricing=pricing,
            max_workers=args.workers,
            runs_per_config=max(1, args.runs_per_config),
            seed_base=args.seed_base,
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
