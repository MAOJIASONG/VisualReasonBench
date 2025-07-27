"""
Environment module for physical visual reasoning.
"""

from .physics_env import (
    PhysicsEnvironment,
    PyBulletEnvironment,
    CameraConfig,
    RobotConfig,
    ObjectInfo,
    TaskDefinition,
    ExecutionStep,
    BenchmarkResult,
    create_environment
)

from .phyvpuzzle_env import (
    PhyVPuzzleEnvironment,
    PuzzleType,
    PhyVPuzzleConfig,
    PuzzleObjectConfig,
    create_luban_lock_environment,
    create_pagoda_environment
)

from .vlm_benchmark import (
    VLMBenchmarkController,
    VLMBenchmarkConfig,
    run_luban_lock_benchmark,
    run_pagoda_benchmark,
    run_full_benchmark_suite
)

__all__ = [
    # Base physics environment
    'PhysicsEnvironment',
    'PyBulletEnvironment', 
    'CameraConfig',
    'RobotConfig',
    'ObjectInfo',
    'TaskDefinition',
    'ExecutionStep',
    'BenchmarkResult',
    'create_environment',
    
    # PhyVPuzzle environment
    'PhyVPuzzleEnvironment',
    'PuzzleType',
    'PhyVPuzzleConfig', 
    'PuzzleObjectConfig',
    'create_luban_lock_environment',
    'create_pagoda_environment',
    
    # VLM Benchmark
    'VLMBenchmarkController',
    'VLMBenchmarkConfig',
    'run_luban_lock_benchmark',
    'run_pagoda_benchmark',
    'run_full_benchmark_suite'
]