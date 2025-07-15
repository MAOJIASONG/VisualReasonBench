VisualReasonBench/
├── setup.py & pyproject.toml          # 标准Python包配置
├── src/phyvpuzzle/                    # 源代码目录
│   ├── core/                          # 核心模块
│   │   ├── vllm_processor.py          # VLLM处理器
│   │   ├── action_descriptor.py       # 动作解析器
│   │   ├── translator.py              # 动作翻译器
│   │   └── pipeline.py                # 主流水线
│   ├── environment/physics_env.py     # PyBullet物理环境
│   ├── tasks/base_task.py             # 任务基类
│   ├── evaluation/metrics.py          # 评估指标
│   └── cli.py                         # 命令行界面
├── tests/                             # 测试文件
├── examples/                          # 使用示例
├── configs/                           # 配置文件
└── docs/                              # 文档目录