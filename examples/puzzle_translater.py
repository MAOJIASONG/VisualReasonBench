#!/usr/bin/env python3
"""
Puzzle Translater 测试任务演示
这是一个简单的拼图任务，包含两个步骤：
1. 将 Object #7 (ID: 2) 移动到黑色框（容器）中
2. 将 Object #6 (ID: 3) 放在 ID 2 的上面
"""
import os
from pathlib import Path

from phyvpuzzle import load_config, BenchmarkRunner, validate_config

# 尝试从.env文件加载环境变量（可选）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("注意：python-dotenv 不可用。仅使用系统环境变量。")

def check_api_keys():
    """检查API密钥是否设置，如果没有则打印警告。"""
    if not os.getenv("OPENAI_API_KEY"):
        print("\n" + "="*80)
        print("⚠️  警告：环境变量中未找到 OPENAI_API_KEY。")
        print("请在根目录下创建 .env 文件并添加你的 API 密钥：")
        print("  OPENAI_API_KEY='your-key-here'")
        print("或者将其导出为环境变量。")
        print("没有它，脚本可能会失败。")
        print("="*80 + "\n")

def main():
    """运行 Puzzle Translater 测试任务的主函数。"""
    check_api_keys()

    print("\n" + " 🧩 Puzzle Translater 测试任务 ".center(80, "="))
    
    # 从YAML文件加载配置
    config_path = Path(__file__).resolve().parent.parent / "eval_configs" / "puzzle_translater.yaml"
    
    if not config_path.exists():
        print(f"❌ 未找到配置文件：{config_path}")
        return 1
        
    try:
        config = load_config(str(config_path))
        validation_errors = validate_config(config)
        if validation_errors:
            print(f"❌ 配置无效：{validation_errors}")
            raise ValueError("配置无效")
        
        print("\n" + "📋 实验配置".center(50, "-"))
        print(f"  实验名称        : {config.runner.experiment_name}")
        print(f"  智能体（模型）  : {config.agent.model_name}")
        print(f"  任务类型        : {config.task.type} ({config.task.difficulty.value})")
        print(f"  最大步数        : {config.environment.max_steps}")
        print("-" * 50)
        
        # 初始化并设置基准测试运行器
        print("\n🚀 初始化拼图基准测试运行器...")
        runner = BenchmarkRunner(config)
        
        print("\n🎯 任务概览：")
        print("  • 步骤 1：将 Object #7 (ID: 2) 移动到黑色框（容器）中")
        print("  • 步骤 2：将 Object #6 (ID: 3) 放在 ID 2 的上面")
        print("  • 这是一个简单的堆叠任务，用于测试基本操作能力")
        print("\n" + "🎮 开始 Puzzle Translater 挑战...".center(60, "-"))
        
        # --- 运行基准测试 ---
        try:
            evaluation_result = runner.run_benchmark(num_runs=1)
        except Exception as benchmark_error:
            print(f"\n❌ 基准测试执行失败：{benchmark_error}")
            import traceback
            traceback.print_exc()
            return 1

        # --- 最终总结 ---
        print("\n" + "🏁 Puzzle Translater 挑战结果".center(80, "="))
        
        if evaluation_result.accuracy > 0.5:
            print("🎉 成功！任务已成功完成！")
        else:
            print("😔 挑战未完成。下次好运！")
            
        print(f"\n📊 性能指标：")
        print(f"  • 成功率: {evaluation_result.accuracy:.1%}")
        if evaluation_result.pass_at_k:
            for k, rate in evaluation_result.pass_at_k.items():
                print(f"  • Pass@{k}: {rate:.1%}")
        if evaluation_result.token_efficiency != float('inf'):
            print(f"  • Token 效率: {evaluation_result.token_efficiency:.0f} tokens/成功")
        if evaluation_result.distance_to_optimal != float('inf'):
            print(f"  • 步骤效率: {evaluation_result.distance_to_optimal:.2f}x 最优")
            
        print(f"\n📁 输出文件：")
        print(f"  ➡️  日志目录    : {runner.logger.run_dir}")
        print(f"  ➡️  结果Excel   : {runner.logger.run_dir}/{config.runner.results_excel_path}")
        print(f"  ➡️  完整日志    : {runner.logger.run_dir}/experiment_log.json")
        print(f"  ➡️  图像        : {runner.logger.run_dir}/images/")
        print("=" * 80)
        
        return 0 if evaluation_result.accuracy > 0.5 else 1
        
    except Exception as e:
        print(f"\n❌ 发生意外错误：{e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

