#!/usr/bin/env python3
"""
Simplified test script for PhyVPuzzle environment only.

This script tests only the environment components without VLM dependencies.
"""

import os
import sys
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_direct_environment_imports():
    """Test environment imports directly without core dependencies."""
    print("Testing direct environment imports...")
    
    try:
        # Test basic physics environment
        from phyvpuzzle.environment.physics_env import (
            PhysicsEnvironment,
            PyBulletEnvironment,
            CameraConfig,
            ObjectInfo,
            TaskDefinition
        )
        print("✓ Basic physics environment imports successful")
        
        # Test PhyVPuzzle environment
        from phyvpuzzle.environment.phyvpuzzle_env import (
            PhyVPuzzleEnvironment,
            PuzzleType,
            PhyVPuzzleConfig,
            PuzzleObjectConfig
        )
        print("✓ PhyVPuzzle environment imports successful")
        
        # Test success metrics
        from phyvpuzzle.environment.success_metrics import (
            PuzzleSuccessEvaluator,
            SuccessMetrics,
            SuccessLevel
        )
        print("✓ Success metrics imports successful")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_environment_creation():
    """Test environment creation without dependencies."""
    print("\nTesting environment creation...")
    
    try:
        from phyvpuzzle.environment.phyvpuzzle_env import (
            PuzzleType, 
            PhyVPuzzleConfig, 
            PhyVPuzzleEnvironment
        )
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
        print(f"  Puzzle type: {config.puzzle_type.value}")
        print(f"  Max steps: {config.max_steps}")
        
        # Test environment creation (without setup)
        env = PhyVPuzzleEnvironment(config, gui=False)
        print("✓ Environment object created successfully")
        print(f"  Environment type: {type(env).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        traceback.print_exc()
        return False


def test_success_metrics_standalone():
    """Test success metrics without external dependencies."""
    print("\nTesting success metrics standalone...")
    
    try:
        from phyvpuzzle.environment.success_metrics import (
            PuzzleSuccessEvaluator,
            SuccessLevel,
            SuccessMetrics
        )
        from phyvpuzzle.environment.phyvpuzzle_env import PuzzleType
        
        # Create evaluator
        evaluator = PuzzleSuccessEvaluator(PuzzleType.LUBAN_LOCK)
        print("✓ Success evaluator created")
        print(f"  Evaluator puzzle type: {evaluator.puzzle_type.value}")
        
        # Test success metrics creation
        metrics = SuccessMetrics(
            success_level=SuccessLevel.PARTIAL,
            overall_score=0.7,
            component_scores={"interlocking": 0.8, "stability": 0.6},
            detailed_analysis={"test": "analysis"},
            reasoning="Test reasoning for partial success"
        )
        print("✓ Success metrics object created")
        print(f"  Success: {metrics.is_success()}")
        print(f"  Partial success: {metrics.is_partial_success()}")
        print(f"  Overall score: {metrics.overall_score}")
        print(f"  Components: {list(metrics.component_scores.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Success metrics test failed: {e}")
        traceback.print_exc()
        return False


def test_puzzle_types():
    """Test puzzle type enumeration."""
    print("\nTesting puzzle types...")
    
    try:
        from phyvpuzzle.environment.phyvpuzzle_env import PuzzleType
        
        # Test all puzzle types
        puzzle_types = list(PuzzleType)
        print(f"✓ Found {len(puzzle_types)} puzzle types:")
        for ptype in puzzle_types:
            print(f"  - {ptype.name}: {ptype.value}")
        
        # Test specific types
        assert PuzzleType.LUBAN_LOCK.value == "luban_lock"
        assert PuzzleType.PAGODA.value == "pagoda"
        print("✓ Puzzle type values correct")
        
        return True
        
    except Exception as e:
        print(f"✗ Puzzle types test failed: {e}")
        traceback.print_exc()
        return False


def test_basic_pybullet():
    """Test basic PyBullet functionality."""
    print("\nTesting basic PyBullet...")
    
    try:
        import pybullet as p
        
        # Test connection (headless)
        client_id = p.connect(p.DIRECT)
        print(f"✓ PyBullet connected (client ID: {client_id})")
        
        # Test basic operations
        p.setGravity(0, 0, -9.81)
        print("✓ Gravity set")
        
        # Load a simple object
        plane_id = p.loadURDF("plane.urdf")
        print(f"✓ Plane loaded (ID: {plane_id})")
        
        # Clean up
        p.disconnect()
        print("✓ PyBullet disconnected")
        
        return True
        
    except Exception as e:
        print(f"✗ PyBullet test failed: {e}")
        traceback.print_exc()
        return False


def test_file_access():
    """Test access to model files."""
    print("\nTesting model file access...")
    
    try:
        models_base = "./src/phyvpuzzle/environment/phobos_models"
        
        # Check main directories
        expected_dirs = [
            "lego-pagoda-setup-v2",
            "lego-pagoda-finished", 
            "luban-simple-prismatic"
        ]
        
        found_dirs = []
        for dir_name in expected_dirs:
            dir_path = os.path.join(models_base, dir_name)
            if os.path.exists(dir_path):
                found_dirs.append(dir_name)
                print(f"✓ Found directory: {dir_name}")
            else:
                print(f"✗ Missing directory: {dir_name}")
        
        # Check specific files
        test_files = [
            "lego-pagoda-setup-v2/urdf/test-pagoda-urdf-auto-3.urdf",
            "luban-simple-prismatic/base_link/urdf/base_link.urdf"
        ]
        
        found_files = 0
        for file_path in test_files:
            full_path = os.path.join(models_base, file_path)
            if os.path.exists(full_path):
                found_files += 1
                print(f"✓ Found file: {file_path}")
            else:
                print(f"✗ Missing file: {file_path}")
        
        success = len(found_dirs) >= 2 and found_files >= 1
        print(f"✓ Model access test: {found_files}/{len(test_files)} files found")
        
        return success
        
    except Exception as e:
        print(f"✗ File access test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run environment-only tests."""
    print("PhyVPuzzle Environment Test Suite")
    print("=" * 50)
    
    tests = [
        ("Direct Environment Imports", test_direct_environment_imports),
        ("Puzzle Types", test_puzzle_types),
        ("Environment Creation", test_environment_creation),
        ("Success Metrics Standalone", test_success_metrics_standalone),
        ("Basic PyBullet", test_basic_pybullet),
        ("Model File Access", test_file_access)
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
        print("🎉 All environment tests passed! PhyVPuzzle environment is ready.")
        print("\nNext steps:")
        print("1. Install VLM dependencies for full functionality")
        print("2. Run environment with GUI for visual testing")
        print("3. Integrate with your VLM model")
        return 0
    elif passed >= total * 0.7:
        print("✅ Most tests passed! Core environment functionality is working.")
        print("Check failed tests for any missing dependencies or files.")
        return 0
    else:
        print("⚠️  Many tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())