# Luban 原始源码与解法资料核查

检查日期：2026-11-16（Asia/Shanghai）。本文件记录用户上传源码的核查结果；随后按用户要求继续恢复动作、搜索和实际验证。复现入口见 `tools/luban_validation/README.md`。

结论：压缩包中尚未找到覆盖32关的完整标准解、已验证通关轨迹或自动求解器；找到原版动作规则、目标布局及历史玩法代码。这些证据说明当前 API 的动作范围没有覆盖原版玩法，不能把当前求解器的搜索失败当作题目无解。

## 数据一致性

- 压缩包成功读取6855个条目，总解压大小113,134,952字节；编辑器版本2022.3.62f2。
- Assets/Resources/Puzzles 的32关，与当前Linux程序提取到 tools/luban_solver/levels 的32关逐份JSON对象比较完全一致，编号一一对应。
- Assets/Ass/Date/obj_datas.json 有36道原始题。按四个基础字段的结构内容（统一整数/浮点表示后）匹配，当前32关均有唯一来源；原始编号2、3、10、24未包含在当前32关中。

## 找到的特殊动作规则

原始36题中9题含22条非空 moveRotates 规则；其中8题、20条对应当前32关。每条包括目标零件、其他零件的前置位置/朝向条件、平移量、旋转轴/次数与是否45°等信息。它们是局部动作模板，不是覆盖整关的按序通关答案。当前运行JSON均未包含 moveRotates 字段。

|当前关卡（从0开始）|原始题号|特殊规则条数|
|---:|---:|---:|
|2|4|4|
|4|5|4|
|8|9|2|
|10|13|2|
|16|19|2|
|18|21|2|
|22|26|2|
|29|33|2|

原始第3题的另外2条规则不属于当前32关。

[完整编号映射](/tmp/luban-source-inspection-20261116/level_mapping.json) · [20条特殊动作原始记录](/tmp/luban-source-inspection-20261116/special_rules.json)

## 第4关的具体证据

当前第4关对应原始第5题（均从0计数）。`obj_datas[5].moveRotates[1]` 的三个零件的位置、朝向与半旋转状态前置条件，已按原版 ScaleFactor=10000 与当前初态逐项比较，全部一致。

该规则指定：**2号零件沿Y移动−20逻辑单位，同时绕X正向旋转90°**。旧代码在同一段动画中执行两个变化。原先API只提供分别移动或旋转，无法表达这条联动动作。这解释了此前单块动作均被阻挡的原因之一。恢复规则后，已搜索到完整3步路径，并在实际Unity中逐动作通过：最终精确状态一致、真实场景所有零件分离且游戏通关。对应凭据保存在 `artifacts/luban/live_v6_solutions_batch1/report.json` 的第4关记录中。

[前置条件核对记录](/tmp/luban-source-inspection-20261116/level4_initial_special_move.json)。

## 旧玩法与当前接口差异

- [原作者玩法提示](/tmp/luban-source-inspection-20261116/extracted/LuBanSuoEnv/Assets/Ass/OldScripts/BurrPuzzle/BurrPuzzleConstText.txt:99)明确说明特殊结构需要边平移边旋转，另一些需要45°旋转。
- [45°动作实现](/tmp/luban-source-inspection-20261116/extracted/LuBanSuoEnv/Assets/Ass/OldScripts/BurrPuzzle/Play/PlayLogic.txt:508)；[多件绕指定中心旋转](/tmp/luban-source-inspection-20261116/extracted/LuBanSuoEnv/Assets/Ass/OldScripts/BurrPuzzle/Play/PlayLogic.txt:916)；[多选组平移](/tmp/luban-source-inspection-20261116/extracted/LuBanSuoEnv/Assets/Ass/OldScripts/BurrPuzzle/Play/PlayLogic.txt:956)。
- [特殊平移旋转执行](/tmp/luban-source-inspection-20261116/extracted/LuBanSuoEnv/Assets/Ass/OldScripts/BurrPuzzle/DataCore/DataTools.txt:675)；[组旋转碰撞检查](/tmp/luban-source-inspection-20261116/extracted/LuBanSuoEnv/Assets/Ass/OldScripts/BurrPuzzle/DataCore/DataTools.txt:695)。
- 上述旧逻辑保存在 OldScripts 的 .txt 文件中，当前Unity不会把它们作为C#脚本编译使用。
- 当前 [控制接口](/tmp/luban-source-inspection-20261116/extracted/LuBanSuoEnv/Assets/Scripts/Logic/EnvControlAPI.cs:172)把非零角度截成±90°；当前操作只允许单件轴向平移/绕自身原点旋转。

## 不能当作正确解的资料

- position2 / switchAxisTypeXYZ2 是目标或参考布局。显示目标布局不提供从初态到终态的合法路径。
- EnvRecorder / ReplayManager 是用户操作录制与播放功能，未找到随包的完整通关session数据。
- [ReplayManager](/tmp/luban-source-inspection-20261116/extracted/LuBanSuoEnv/Assets/Scripts/Logic/ReplayManager.cs:160)可在动作后强制同步快照，并在196行直接改坐标、210行改朝向。验证解法时必须禁用这类状态强制恢复，逐步检查动作接受情况及最终游戏通关标志。

## 对后续修复的意义

恢复原版规则时，需要覆盖特殊平移旋转、45°状态和组合操作，并明确各自碰撞检查与前置条件。再在对应动作范围内搜索，逐关回放确认。此处没有声称32关已全部求解，也没有把参考终态或快照播放当作通关证明。

前轮已发现的固定5秒响应等待问题，在附件 SocketBridge.cs:175 也存在；这与是否有标准解是两件不同的事。
