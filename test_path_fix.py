#!/usr/bin/env python3
"""
Test script to verify path fixes for PhyVPuzzle environment.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_file_paths():
    """Test that all required files exist with correct paths."""
    print("🔍 Testing PhyVPuzzle file paths...")
    
    # Base paths
    script_dir = os.path.dirname(__file__)
    models_base = os.path.join(script_dir, "src", "phyvpuzzle", "environment", "phobos_models")
    
    print(f"Script directory: {script_dir}")
    print(f"Models base: {models_base}")
    
    # Test required paths
    test_paths = [
        # Luban lock files
        ("Luban URDF", os.path.join(models_base, "luban-simple-prismatic/base_link/urdf/base_link.urdf")),
        ("Luban meshes dir", os.path.join(models_base, "luban-simple-prismatic/base_link/meshes/stl")),
        ("Luban sample mesh", os.path.join(models_base, "luban-simple-prismatic/base_link/meshes/stl/luban-lock-sliding.stl")),
        
        # Pagoda files  
        ("Pagoda URDF", os.path.join(models_base, "lego-pagoda-setup-v2/urdf/test-pagoda-urdf-auto-3.urdf")),
        ("Pagoda meshes dir", os.path.join(models_base, "lego-pagoda-setup-v2/meshes/stl")),
    ]
    
    all_exist = True
    for name, path in test_paths:
        if os.path.exists(path):
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} - NOT FOUND")
            all_exist = False
    
    return all_exist


def test_environment_imports():
    """Test environment imports without PyBullet dependency."""
    print("\n🔧 Testing environment imports...")
    
    try:
        # Test basic imports
        from phyvpuzzle.environment.phyvpuzzle_env import PuzzleType, PhyVPuzzleConfig
        from phyvpuzzle.environment.physics_env import CameraConfig
        print("✅ Basic imports successful")
        
        # Test configuration creation
        models_base = os.path.join(os.path.dirname(__file__), "src", "phyvpuzzle", "environment", "phobos_models")
        
        config = PhyVPuzzleConfig(
            puzzle_type=PuzzleType.LUBAN_LOCK,
            urdf_base_path=models_base,
            meshes_path=os.path.join(models_base, "luban-simple-prismatic/base_link/meshes/stl"),
            initial_camera_config=CameraConfig(),
            max_steps=10,
            time_limit=60.0
        )
        print("✅ Configuration created successfully")
        print(f"   Puzzle type: {config.puzzle_type.value}")
        print(f"   URDF base: {config.urdf_base_path}")
        print(f"   Meshes path: {config.meshes_path}")
        
        # Verify paths exist
        if os.path.exists(config.urdf_base_path):
            print("✅ URDF base path exists")
        else:
            print("❌ URDF base path does not exist")
            return False
            
        if os.path.exists(config.meshes_path):
            print("✅ Meshes path exists")
        else:
            print("❌ Meshes path does not exist")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Import/config test failed: {e}")
        return False


def test_urdf_parsing():
    """Test URDF file parsing without PyBullet."""
    print("\n📄 Testing URDF parsing...")
    
    try:
        import xml.etree.ElementTree as ET
        
        models_base = os.path.join(os.path.dirname(__file__), "src", "phyvpuzzle", "environment", "phobos_models")
        urdf_path = os.path.join(models_base, "luban-simple-prismatic/base_link/urdf/base_link.urdf")
        
        if not os.path.exists(urdf_path):
            print(f"❌ URDF file not found: {urdf_path}")
            return False
        
        print(f"✅ URDF file found: {urdf_path}")
        
        # Parse URDF
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        print(f"✅ URDF parsed successfully")
        print(f"   Root tag: {root.tag}")
        
        # Count joints and links
        joints = root.findall('.//joint')
        links = root.findall('.//link')
        
        print(f"   Joints found: {len(joints)}")
        print(f"   Links found: {len(links)}")
        
        # Find object joints
        obj_joints = [j for j in joints if j.get('name', '').startswith('obj_') and j.get('name', '').endswith('_joint')]
        print(f"   Object joints: {len(obj_joints)}")
        
        if obj_joints:
            for joint in obj_joints[:3]:  # Show first 3
                print(f"     - {joint.get('name')}")
        
        return len(obj_joints) > 0
        
    except Exception as e:
        print(f"❌ URDF parsing failed: {e}")
        return False


def main():
    """Run all path tests."""
    print("🧪 PhyVPuzzle Path Fix Verification")
    print("=" * 50)
    
    tests = [
        ("File Paths", test_file_paths),
        ("Environment Imports", test_environment_imports), 
        ("URDF Parsing", test_urdf_parsing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All path tests passed! Environment setup should work now.")
        print("\nNext steps:")
        print("1. Install PyBullet: pip install pybullet")
        print("2. Run: python test_gpt4o_real_env.py")
    else:
        print("⚠️ Some tests failed. Check file paths and directory structure.")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())