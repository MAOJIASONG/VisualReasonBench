#!/usr/bin/env python3
"""测试路径修复是否正确"""

import json
import os

def test_path_construction(puzzle_size, puzzle_id):
    """测试路径构建逻辑"""
    # 这是修复后的逻辑
    if puzzle_id.startswith("puzzle_"):
        puzzle_id = puzzle_id[7:]  # Remove "puzzle_" prefix
    
    # Split by underscore: "mid_001" -> ["mid", "001"]
    parts = puzzle_id.split("_")
    if len(parts) >= 2:
        difficulty = parts[0]  # e.g., "easy", "mid", "hard"
        number = parts[1]      # e.g., "001", "002"
    else:
        # Fallback: assume the entire puzzle_id is the number
        difficulty = "easy"
        number = puzzle_id
    
    # Build correct path
    puzzle_dir = "Stacking_scaling/puzzles_full_v9"
    config_path = os.path.join(
        puzzle_dir,
        puzzle_size,
        difficulty,
        number,
        f"{puzzle_size}_{difficulty}_{number}.json"
    )
    
    return config_path, os.path.exists(config_path)

# 测试用例
test_cases = [
    ("2x2x2", "puzzle_easy_001"),
    ("2x2x2", "puzzle_mid_001"),
    ("2x2x3", "puzzle_easy_001"),
    ("2x2x3", "puzzle_easy_002"),
]

print("测试路径构建逻辑：")
print("=" * 80)

for size, puzzle_id in test_cases:
    path, exists = test_path_construction(size, puzzle_id)
    status = "✓ 存在" if exists else "✗ 不存在"
    print(f"{status} | Size: {size:8} | Puzzle: {puzzle_id:20} | Path: {path}")
    
    if exists:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                piece_count = len(data.get("pieces", []))
                print(f"      → 成功加载！包含 {piece_count} 个拼图块")
        except Exception as e:
            print(f"      → 加载失败: {e}")
    print()

print("=" * 80)
print("路径修复测试完成！")
