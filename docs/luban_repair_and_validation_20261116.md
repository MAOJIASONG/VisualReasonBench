# Luban 修复与32关验证记录

日期：2026-11-16。编号均从0开始。

**结果：32/32关均有完整动作路径，并已在实际Unity游戏中逐步验证通关。** 每关同时通过真实场景的零件两两分离检查。这32条已验证路径合计506步；路径可复现，不声称最短。

## 修改内容

- 保留全部32关的形状和初始摆放。恢复源码中遗漏的20条条件动作，分布在8关。原始上传包未发现覆盖32关的完整标准解。
- 恢复组合平移、整数方向斜移、±45°/±90°旋转、自定义旋转中心及条件式平移旋转联动。
- 修复负坐标数值转换造成的旋转轴心偏差；修复原接口把180°请求截为90°。
- 修复截图等待后返回空行的问题；响应预算为300秒，客户端310秒，超时明确报告动作可能已经执行，不自动重放。
- 增加无显示器启动、足够的启动等待、状态读取失败处理，以及验证程序对自有进程组的完整清理。
- 求解加入精确碰撞缓存、状态去重、断点续搜、分批展开，以及可选的已分离零件/组件剪枝。
- 通关须同时满足原生胜利标记与实际场景全部零件分离，避免整组搬走造成误报。最短步数未知，不再固定记作1。

## 验证标准与证据

- 32关全部通过初始加载、原生坐标一致性及六视角截图检查。
- 每条路径均从原关卡初态执行；没有通过恢复终态快照或写入胜利标记来通关。
- 每步动作由恢复的原版规则检查；最终有理数坐标、朝向、45°状态与搜索记录精确一致。
- 实际渲染器/碰撞体的世界包围盒逐对保留正间距，且Unity原生完成信号为真。
- 2232项初始移动距离对照、3427项原生移动响应对照均无差异；运行补丁及接口14项测试、验证器4项测试通过。
- 新求解器构建和回归通过；原有求解器仍可编译。正常项目接口已自动启动、完成第3关5步解并退出。

核心规则源码指纹：`955375b81731551b04ab1d83c8f96c0e67405d7659f418cbd214df6a645eda0b`。

[完整解法与独立游戏凭据索引](../tools/luban_solver/verified_solutions/verification_index.json)。每个原始候选文件仍保留生成时的模型验证标签和字节；是否经过实际游戏验证，以此索引中绑定的SHA-256和独立凭据为准。

## 已处理的加载异常

第30关曾在120秒等待期内未完成冷加载。诊断确认是在共享CPU竞争下持续进行几何预计算，约179秒后正常完成，并非死循环。扩大合理等待预算后，最终冷加载约168.3秒，随后12步解法全部通过。历史失败记录与后续成功记录均保留。

最终运行包使用同一核心、扩展和32关规则；与前一版相比，游戏DLL仅等待常数的3个字节变化。复杂关卡首次加载在当前繁忙机器上仍可能需要约3分钟。旋转合法性沿用原始源码的离散检查，条件联动沿用原始规则；这些结果不是连续物理运动的形式证明。

[加载诊断记录](../artifacts/luban/cold_load_diagnosis/summary.json) · [32关汇总](../artifacts/luban/summary.json)

## 使用

当前工作目录内已经准备好 `artifacts/luban/runtime_final`。正常评测选择 [luban_restored.yaml](../eval_configs/luban_restored.yaml)；默认native模式仍保持兼容。无需模型调用即可复查全部解法：

```bash
python tools/luban_validation/replay_original.py \
  --package artifacts/luban/runtime_final \
  --candidates tools/luban_solver/verified_solutions/Puzzle_*.json \
  --out artifacts/luban/recheck
```

重建、枚举搜索、单关接口检查和凭据汇总见 [验证工具说明](../tools/luban_validation/README.md)。原始上传附件不是重建所必需的；恢复规则已随源码保存。

## 逐关结果

|关卡|验证步数|实际游戏|动作文件|
|---:|---:|---|---|
|0|1|通过|[Puzzle_0.json](../tools/luban_solver/verified_solutions/Puzzle_0.json)|
|1|6|通过|[Puzzle_1.json](../tools/luban_solver/verified_solutions/Puzzle_1.json)|
|2|7|通过|[Puzzle_2.json](../tools/luban_solver/verified_solutions/Puzzle_2.json)|
|3|5|通过|[Puzzle_3.json](../tools/luban_solver/verified_solutions/Puzzle_3.json)|
|4|3|通过|[Puzzle_4.json](../tools/luban_solver/verified_solutions/Puzzle_4.json)|
|5|6|通过|[Puzzle_5.json](../tools/luban_solver/verified_solutions/Puzzle_5.json)|
|6|14|通过|[Puzzle_6.json](../tools/luban_solver/verified_solutions/Puzzle_6.json)|
|7|10|通过|[Puzzle_7.json](../tools/luban_solver/verified_solutions/Puzzle_7.json)|
|8|11|通过|[Puzzle_8.json](../tools/luban_solver/verified_solutions/Puzzle_8.json)|
|9|20|通过|[Puzzle_9.json](../tools/luban_solver/verified_solutions/Puzzle_9.json)|
|10|19|通过|[Puzzle_10.json](../tools/luban_solver/verified_solutions/Puzzle_10.json)|
|11|18|通过|[Puzzle_11.json](../tools/luban_solver/verified_solutions/Puzzle_11.json)|
|12|15|通过|[Puzzle_12.json](../tools/luban_solver/verified_solutions/Puzzle_12.json)|
|13|23|通过|[Puzzle_13.json](../tools/luban_solver/verified_solutions/Puzzle_13.json)|
|14|19|通过|[Puzzle_14.json](../tools/luban_solver/verified_solutions/Puzzle_14.json)|
|15|38|通过|[Puzzle_15.json](../tools/luban_solver/verified_solutions/Puzzle_15.json)|
|16|15|通过|[Puzzle_16.json](../tools/luban_solver/verified_solutions/Puzzle_16.json)|
|17|32|通过|[Puzzle_17.json](../tools/luban_solver/verified_solutions/Puzzle_17.json)|
|18|18|通过|[Puzzle_18.json](../tools/luban_solver/verified_solutions/Puzzle_18.json)|
|19|7|通过|[Puzzle_19.json](../tools/luban_solver/verified_solutions/Puzzle_19.json)|
|20|16|通过|[Puzzle_20.json](../tools/luban_solver/verified_solutions/Puzzle_20.json)|
|21|13|通过|[Puzzle_21.json](../tools/luban_solver/verified_solutions/Puzzle_21.json)|
|22|6|通过|[Puzzle_22.json](../tools/luban_solver/verified_solutions/Puzzle_22.json)|
|23|14|通过|[Puzzle_23.json](../tools/luban_solver/verified_solutions/Puzzle_23.json)|
|24|23|通过|[Puzzle_24.json](../tools/luban_solver/verified_solutions/Puzzle_24.json)|
|25|14|通过|[Puzzle_25.json](../tools/luban_solver/verified_solutions/Puzzle_25.json)|
|26|12|通过|[Puzzle_26.json](../tools/luban_solver/verified_solutions/Puzzle_26.json)|
|27|12|通过|[Puzzle_27.json](../tools/luban_solver/verified_solutions/Puzzle_27.json)|
|28|35|通过|[Puzzle_28.json](../tools/luban_solver/verified_solutions/Puzzle_28.json)|
|29|56|通过|[Puzzle_29.json](../tools/luban_solver/verified_solutions/Puzzle_29.json)|
|30|12|通过|[Puzzle_30.json](../tools/luban_solver/verified_solutions/Puzzle_30.json)|
|31|6|通过|[Puzzle_31.json](../tools/luban_solver/verified_solutions/Puzzle_31.json)|
