# CHAIN 新增难题：每种游戏 5 道

从本地已保存的 17 道 xhard 候选题中选出 15 道：FluidNet F1–F5、PipeFit U1/U2/U4/U5/U7、Domino W1–W5。每道均包含 `level.json`、`solution.txt`、开局正面图和背面图。

## 选择依据与参考结果

参考数据为 2026-11-16 的 `gpt-6.1-sol`、medium、禁止代码、40 步预算测试，每题一次。源结果表现有 78 局，其中这 17 道候选题各有一条记录；此前未完成的 U4、U5、U7 现已补齐。这里保存这 17 条参考记录，便于核对入选和未入选的题目。

| 游戏 | 入选 | 该次 Sol 通关 | 选择依据 |
|---|---|---:|---|
| FluidNet | F1–F5 | 0/5 | 五道均未通关；包含延迟生效的阀门陷阱。 |
| PipeFit | U1、U2、U4、U5、U7 | 1/5 | 优先选未通关的 U1/U2/U4/U5；第五道选搜索报告仅找到一条网络、且 Sol 经一次失败才通关的 U7。 |
| Domino | W1–W5 | 4/5 | 保留现有五道布线难题；仅 W1 未通关，尚未达到“至少两道未通关”的旧目标。 |

U3（37 步、一次失败后通关）与 U6（19 步通关）未入选。U7（32 步、一次失败后通关）与 U3 的难度无法凭单次测试严格排序；选择 U7 同时考虑搜索报告中的管路数量，而不是仅按模型步数排序。该搜索存在截断，找到一条网络不等于证明全局唯一。

## 题目清单

| 编号 | 游戏 | 题目目录 | 保存解法步数 | Sol 结果 | Sol 已用步数 |
|---|---|---|---:|---|---:|
| F1 | fluidnet | [F1_tangle_plan16_seed38](../type_3/puzzles_xhard/F1_tangle_plan16_seed38/) | 16 | 未通关 | 30 |
| F2 | fluidnet | [F2_tangle_plan15_seed18](../type_3/puzzles_xhard/F2_tangle_plan15_seed18/) | 15 | 未通关（提前结束） | 31 |
| F3 | fluidnet | [F3_tangle_plan15_seed28](../type_3/puzzles_xhard/F3_tangle_plan15_seed28/) | 15 | 未通关 | 36 |
| F4 | fluidnet | [F4_tangle_plan17_seed47](../type_3/puzzles_xhard/F4_tangle_plan17_seed47/) | 17 | 未通关 | 32 |
| F5 | fluidnet | [F5_tangle_plan17_seed28](../type_3/puzzles_xhard/F5_tangle_plan17_seed28/) | 17 | 未通关 | 32 |
| U1 | pipefit | [U1_unique_n1_min18_seed81100](../type_2_3/puzzles_xhard/U1_unique_n1_min18_seed81100/) | 19 | 未通关 | 39 |
| U2 | pipefit | [U2_unique_n3_min16_seed81057](../type_2_3/puzzles_xhard/U2_unique_n3_min16_seed81057/) | 18 | 未通关 | 38 |
| U4 | pipefit | [U4_unique_n1_min19_seed91150](../type_2_3/puzzles_xhard/U4_unique_n1_min19_seed91150/) | 20 | 未通关 | 36 |
| U5 | pipefit | [U5_unique_n1_min18_seed91138](../type_2_3/puzzles_xhard/U5_unique_n1_min18_seed91138/) | 19 | 未通关（提前结束） | 17 |
| U7 | pipefit | [U7_unique_n1_min17_seed91122](../type_2_3/puzzles_xhard/U7_unique_n1_min17_seed91122/) | 18 | 通关 | 32 |
| W1 | domino | [W1_wiring_p18_seed7085](../type_3_1/puzzles_xhard/W1_wiring_p18_seed7085/) | 20 | 未通关 | 20 |
| W2 | domino | [W2_wiring_p18_seed7204](../type_3_1/puzzles_xhard/W2_wiring_p18_seed7204/) | 20 | 通关 | 20 |
| W3 | domino | [W3_wiring_p18_seed7150](../type_3_1/puzzles_xhard/W3_wiring_p18_seed7150/) | 20 | 通关 | 20 |
| W4 | domino | [W4_wiring_p17_seed7059](../type_3_1/puzzles_xhard/W4_wiring_p17_seed7059/) | 19 | 通关 | 19 |
| W5 | domino | [W5_wiring_p17_seed7326](../type_3_1/puzzles_xhard/W5_wiring_p17_seed7326/) | 19 | 通关 | 19 |

## 验证与启动

在已安装项目依赖的环境中，从仓库根目录回放全部保存解法：

```bash
python Stacking_scaling/xhard/verify.py
```

本次验证结果为 **15/15 获胜**。验证器检查四个资产文件的 SHA-256，并用实际游戏规则逐步回放，检查每步未被拒绝、没有提前结束、最终获胜。保存的解法是可行解，不声称都是最短解。这项检查不调用模型。

启动一道题供人工或外部 agent 操作，例如：

```bash
cd Stacking_scaling
python -m chain_analysis.play_cli start \
  type_3/puzzles_xhard/F1_tangle_plan16_seed38 \
  /tmp/fluidnet-f1-session --max-actions 40
python -m chain_analysis.play_cli status /tmp/fluidnet-f1-session
```

正式模型测试只向模型提供游戏观察和动作接口；`solution.txt`、本验证器、题目原始数据和参考结果用于维护者验证，不应提供给待测模型。

## 可核对文件

- [selection.json](selection.json)：15 道题的路径、资产摘要和筛选所用结果。
- [reference_results.csv](reference_results.csv)：原实验中全部 17 道候选题的结果，字段保留源表定义。
- [reference_runs/](reference_runs/)：对应的结果 JSON 摘录，保留模型配置、退出码、原有审计标记和用量；不包含运行环境凭据。
- [validation.json](validation.json)：本次 15 道保存解法的逐步回放记录。

## 结果边界

- 这是一组依据既有结果筛选的难题，不是随机留出的测试集。单模型、单次运行的成败不能证明其他模型的难度或稳定成功率。
- `playing` 表示该次模型停止时尚未通关；不能解释为题目无解或已耗尽全部 40 步。
- 原实验文字说明与 `Session` 的计数细节有出入：被规则拒绝的游戏动作计入预算；解析失败的调用不计入；reset 不增加、也不补充已用步数，失败次数在游戏判负时累计。这里保留原记录数值，没有重写历史成绩。
- PipeFit 开局图片中的步数面板是游戏自身计数，不代表模型测试的 40 步预算。
- 参考 JSON 中的 `valid` 和审计标记来自原测试。本次核对了它们与结果表的一致性，并重新验证题目解法；没有重新宣称审核全部原始模型日志。
- 这些题作为独立 `puzzles_xhard` 目录新增，现有 `puzzles_50` 题集不变。
