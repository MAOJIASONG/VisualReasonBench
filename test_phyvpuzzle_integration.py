#!/usr/bin/env python3
"""
Test script for PhyVPuzzle integration.

This script tests the integrated PhyVPuzzle environment to ensure
all components work together correctly.
"""

import os
import sys
import traceback
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports...")
    
    try:
        from phyvpuzzle.environment import (
            PhyVPuzzleEnvironment,
            PuzzleType,
            create_luban_lock_environment,
            create_pagoda_environment,
            VLMBenchmarkController,
            VLMBenchmarkConfig
        )
        print("✓ Core environment imports successful")
        
        from phyvpuzzle.environment.success_metrics import (
            PuzzleSuccessEvaluator,
            SuccessMetrics,
            evaluate_puzzle_success
        )
        print("✓ Success metrics imports successful")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_environment_creation():
    """Test environment creation without GUI."""
    print("\nTesting environment creation...")
    
    try:
        from phyvpuzzle.environment import PuzzleType, PhyVPuzzleConfig, PhyVPuzzleEnvironment
        from phyvpuzzle.environment.physics_env import CameraConfig
        
        # Test configuration creation
        config = PhyVPuzzleConfig(
            puzzle_type=PuzzleType.LUBAN_LOCK,
            urdf_base_path="./src/phyvpuzzle/environment/phobos_models",
            meshes_path="./src/phyvpuzzle/environment/phobos_models/luban-simple-prismatic/base_link/meshes/stl",
            initial_camera_config=CameraConfig(),
            max_steps=10,
            time_limit=60.0
        )
        print("✓ Configuration created successfully")
        
        # Test environment creation (without setup)
        env = PhyVPuzzleEnvironment(config, gui=False)
        print("✓ Environment object created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        traceback.print_exc()
        return False


def test_benchmark_config():
    """Test benchmark configuration."""
    print("\nTesting benchmark configuration...")
    
    try:
        from phyvpuzzle.environment import VLMBenchmarkConfig, PuzzleType
        
        config = VLMBenchmarkConfig(
            model_name="test_model",
            puzzle_type=PuzzleType.LUBAN_LOCK,
            max_steps=10,
            time_limit=60.0,
            output_dir="./test_output"
        )
        print("✓ Benchmark configuration created successfully")
        print(f"  Model: {config.model_name}")
        print(f"  Puzzle type: {config.puzzle_type.value}")
        print(f"  Max steps: {config.max_steps}")
        
        return True
        
    except Exception as e:
        print(f"✗ Benchmark configuration failed: {e}")
        traceback.print_exc()
        return False


def test_success_metrics():
    """Test success metrics system."""
    print("\nTesting success metrics...")
    
    try:
        from phyvpuzzle.environment.success_metrics import (
            PuzzleSuccessEvaluator,
            SuccessLevel,
            SuccessMetrics
        )
        from phyvpuzzle.environment import PuzzleType
        
        # Create evaluator
        evaluator = PuzzleSuccessEvaluator(PuzzleType.LUBAN_LOCK)
        print("✓ Success evaluator created")
        
        # Test success metrics creation
        metrics = SuccessMetrics(
            success_level=SuccessLevel.PARTIAL,
            overall_score=0.7,
            component_scores={"test": 0.7},
            detailed_analysis={"test": "analysis"},
            reasoning="Test reasoning"
        )
        print("✓ Success metrics object created")
        print(f"  Success: {metrics.is_success()}")
        print(f"  Partial success: {metrics.is_partial_success()}")
        print(f"  Overall score: {metrics.overall_score}")
        
        return True
        
    except Exception as e:
        print(f"✗ Success metrics test failed: {e}")
        traceback.print_exc()
        return False


def test_file_structure():
    """Test that required files and directories exist."""
    print("\nTesting file structure...")
    
    required_files = [
        "src/phyvpuzzle/environment/__init__.py",
        "src/phyvpuzzle/environment/physics_env.py",
        "src/phyvpuzzle/environment/phyvpuzzle_env.py",
        "src/phyvpuzzle/environment/vlm_benchmark.py",
        "src/phyvpuzzle/environment/success_metrics.py",
        "src/phyvpuzzle/environment/example_usage.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - Missing")
            all_exist = False
    
    # Check for models directory
    models_path = "src/phyvpuzzle/environment/phobos_models"
    if os.path.exists(models_path):
        print(f"✓ {models_path}")
        
        # List some expected model files
        expected_models = [
            "lego-pagoda-setup-v2/urdf/test-pagoda-urdf-auto-3.urdf",
            "luban-simple-prismatic/base_link/urdf/base_link.urdf"
        ]
        
        for model in expected_models:
            model_path = os.path.join(models_path, model)
            if os.path.exists(model_path):
                print(f"  ✓ Model: {model}")
            else:
                print(f"  ✗ Model: {model} - Missing")
    else:
        print(f"✗ {models_path} - Missing")
        all_exist = False
    
    return all_exist


def main():
    """Run all tests."""
    print("PhyVPuzzle Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Environment Creation", test_environment_creation),
        ("Benchmark Configuration", test_benchmark_config),
        ("Success Metrics", test_success_metrics)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! PhyVPuzzle integration is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())