# 1. API 密钥：在运行前自行 export OPENAI_API_KEY，或写入仓库根目录的 .env
#    （不要在这里写死，否则会覆盖已有的真实密钥）

cd "$(dirname "$0")/.."

# 2. 运行示例
PYTHONPATH=src python examples/domino_quick.py

# 3. 使用CLI
PYTHONPATH=src python -m phyvpuzzle.cli run --config eval_configs/domino_quick.yaml
