下面给你一份**可直接落地的游戏设计文档**（3D 多联通块装箱），覆盖：数据模型、形状匹配、放置/移动判定、碰撞与“非悬空”支持、胜负判定、UI/交互、性能与可测试性。示例里默认坐标从 **(1,1,1)** 开始计（与您的例子一致），引擎内部可以 0-based 实现后对外转换。

---

# 目标与规则

* 给定一个空盒子（AxBxC 网格）与若干**联通**的 polycube 块（每块由若干单位正方体组成，面相连）。
* 玩家需要把所有块按**网格对齐**放入盒子，直至**恰好填满**且**无重叠**。
* 每次放置或移动一个块时，必须同时满足：

  1. **形状匹配**：玩家指定的 n 个目标格子与该块在**某个旋转+平移**下的体素集合**完全一致**；
  2. **无碰撞**：目标格子内没有已放置的其他方块；
  3. **非悬空**：该块**至少有一个单位立方体的一个整面**与**底部**（z=1 平面）或**其它已放置块**发生**面接触**（±x/±y/±z 方向的共享面）。
* 胜利条件：盒子被**刚好填满**（无空洞、无重叠），所有块均成功放置。

---

# 坐标系与数据模型

## 坐标系

* 世界坐标使用**整数网格**：(x,y,z)，**从 1 开始**，范围：

  * x ∈ [1..A]，y ∈ [1..B]，z ∈ [1..C]
* 重力/“底部”方向：**-z**；底面为 **z=1**。

## 基础类型

```ts
type Vec3 = { x: number; y: number; z: number };          // 1-based 对外
type Cell = Vec3;                                         // 单位格坐标
type CellSet = Set<string>;                               // 用 "x,y,z" 串键存储
type Rot24 = 0..23;                                       // 24 个正旋转之一（立方体旋群）
type Transform = { rot: Rot24; t: Vec3 };                 // 刚体变换（旋转+平移）
```

## 关卡与状态

```ts
type PieceDef = {
  id: string;
  // 本地坐标（以其最小包围盒左下前角为 (0,0,0) 的 0-based）
  // 注意：只存 shape，朝向与位置由 Transform 决定
  localVoxels: Vec3[];
  // 预处理：所有旋转形状的“规范形签名”，见下文
  rotationSignatures: string[];   // len=24（去重后≤24）
};

type LevelSpec = {
  box: { A: number; B: number; C: number };   // 盒子尺寸
  pieces: PieceDef[];                          // 全部块定义
};

type PlacedPiece = {
  id: string;
  transform: Transform;   // 已放置朝向+平移（世界坐标 1-based）
  worldCells: Cell[];     // 此变换后占用的世界格子（缓存）
};

type GameState = {
  spec: LevelSpec;
  occupied: CellSet;                // 已占用格集合（世界）
  byCell: Map<string, string>;      // "x,y,z" -> pieceId（用于 O(1) 碰撞/查找）
  placed: Map<string, PlacedPiece>; // 已放置
  unplaced: Set<string>;            // 未放置 id
  // 可选：支持图/邻接缓存、撤销栈等
};
```
关卡导入：/mnt/moonfs/wuyuhao-m2/wyh/VisualReasonBench/Stacking_scaling/puzzles_full_v9 
---

# 形状匹配（核心一）

**目标**：给定玩家指定的 n 个目标格子 `T = {t1..tn}`（世界 1-based），判定这些格子能否放置某个待放块 `P`。

## 规范化与旋转等价

1. **块的规范形**（预处理）

   * 将 `P.localVoxels` 平移使最小坐标为 (0,0,0)，排序后序列化（如 `"x_y_z;"` 连接）得到 `sig0`。
   * 对 24 个旋转 `r∈Rot24`，把每个本地点 `v` 旋转得到 `r(v)`，再**规范化**（平移使 min=0），序列化得到 `sig_r`。
   * 收集所有 `sig_r` 去重，即 `rotationSignatures`。这一步**关卡加载时预处理一次**。

2. **玩家目标集合的规范形**（实时）

   * 输入 `T`（世界 1-based），先转换为 0-based（减 1），再平移使最小坐标为 (0,0,0)，排序序列化得 `sigT`。
   * 若 `sigT ∈ P.rotationSignatures`，则**形状匹配成功**；并且还能**反推出**对应的 `rot` 与 `t`：

     * 找到与 `sigT` 相等的那一个旋转 `r` 的规范形 `R`（0-based）。
     * 取 `T` 中任意一点 `t0` 与 `R` 中对应的一点 `r0`，有 `t = t0 - r0`（注意 0/1-based 转换）。
       这给出了**唯一的刚体变换** `Transform { rot: r, t }`。

> 复杂度：预处理 O(|P|×24)；匹配 O(n log n)（排序），哈希查找 O(1)。

---

# 放置判定流水线（核心二）

给定 `pieceId` 与玩家给出的目标格子 `T`，执行以下流程：

1. **形状匹配**

   * 若 `|T| ≠ |P|` → 失败：`WrongCount`。
   * 计算 `sigT` 并比对 `rotationSignatures` → 不匹配 → 失败：`ShapeMismatch`。
   * 推导 `Transform`（`rot`,`t`）与对应的 `worldCells`。

2. **边界检查**

   * 确保 `worldCells` 全部落在 `[1..A]×[1..B]×[1..C]` → 否则失败：`OutOfBounds`。

3. **碰撞检查**

   * 对每个 `c ∈ worldCells`，若 `occupied` 已包含 `c` → 失败：`Collision`。

4. **非悬空检查（支撑）**

   * 定义**支撑成立**的条件：存在某个 `c=(x,y,z) ∈ worldCells`，满足以下两者之一：
     a) **底面支撑**：`z==1`（接触底部面）；
     b) **相邻支撑**：存在 `c'` 为 `c` 在六邻域之一（±x/±y/±z）且 `c'` 已被**其他块**占据（`byCell.get(c') != pieceId`）。
   * 若**所有**体素都不满足 → 失败：`Floating`。

5. **提交**

   * 将 `worldCells` 写入 `occupied` 与 `byCell`；
   * 写入/更新 `placed`，从 `unplaced` 移除；
   * 返回 `OK`，并附上 `Transform` 与任何 UI 提示（如支撑点高亮）。

6. **胜利检测**（可在每次成功放置后检查）

   * 若 `occupied.size === A*B*C` 且 `unplaced` 为空 → `Win`。

> 注意：**支撑检查**是“至少**一处**面接触”，不是“对每个体素都有支撑”。这样避免“完全悬空的整体位姿”。

---

# 移动/撤销（核心三）

## 移动到新位置

* `movePieceByCells(pieceId, newCells: Cell[])`：与放置流程一致，但**先临时移除**该块在 `occupied/byCell` 的占用再检查。
* 或 `movePieceByTransform(pieceId, rot:Rot24, t:Vec3)`：直接由变换生成 `worldCells`，走**边界/碰撞/支撑**判定。

## 取出（撤销）

* `pickupPiece(pieceId)`：

  * 从 `placed` 读取 `worldCells`，对每个单元从 `occupied/byCell` 删除；
  * pieceId 回到 `unplaced`；
  * 用于“悔棋/重摆”。

## 撤销/重做（可选）

* 维护栈：`Action { type: 'place' | 'move' | 'pickup', payload... }`
* 每次操作后 push，撤销时反向操作。

---

# 渲染与交互

## 初始化与初始显示

* 生成盒子线框：A×B×C 的 3D 网格；
* **各块单独上色**（固定色表），显示在**托盘**区域（盒子外）或列表；
* 在盒子内渲染**占位预览层**（ghost）：

  * 玩家正在放置/移动时，实时显示当前选中块按候选变换得到的 `worldCells`：

    * 若合法：显示半透明绿色，支撑点描边；
    * 若非法：半透明红色，并给出原因提示（碰撞/越界/悬空/形状不符）。

## 放置交互（两种模式，可同时支持）

1. **按“目标格子集合”选择**

   * 玩家在盒内点选 n 个单元格作为目标集合 `T`（支持框选/连选）。
   * 调用 `placePieceByCells(pieceId, T)`，走判定流水线。
   * 适合你说的例子：把 1×3 的块放到 `(1,1,1)-(1,1,3)`。

2. **按“锚点+朝向”**

   * 玩家先选块，再选一个**锚点**（如本地最小点或质心投影），在盒内点击目标锚单元；
   * 通过旋转键/手势切换 24 个朝向，实时 ghost；
   * 点击确认时调用 `placePieceByTransform`。
   * 这种方式更快，但**不检查形状等价**（因为变换已经给定），仍需**越界/碰撞/支撑**判定。

## 取出/移动

* 点击已放置块 → 浮动工具条：**移动**/**旋转**/**取出**；
* 移动使用与放置相同的 ghost + 判定流程；
* 取出后块回托盘。

---

# 关键算法细节

## 24 个正旋转（Rot24）

* 立方体的旋转群（SO(3) 的离散子群）可用**轴置换+符号**构造，或枚举“把 +Z 指向 6 个方向，再绕该轴旋转 0/90/180/270”。
* 预先生成 24 个 3×3 整矩阵（行列式 +1），缓存为 `rotMats[24]`。

## 形状签名（去序、平移、旋转不变）

* `normalize(pts)`：整体平移使 `min(x)=min(y)=min(z)=0`，然后按字典序排序序列化。
* `signature(pts)`：对 24 个旋转后的 `normalize` 取**字典序最小**串作为**规范签名**。
* 块的 `rotationSignatures`：把每个旋转形状的 `normalize` 序列化后去重集合（匹配时查集合；规范签名用于去重与比较）。
* 玩家给的 `T` 用 `normalize` 得到 `sigT`，与集合作包含测试即可。

## 碰撞与越界

* 维护 `occupied: Set` 与 `byCell: Map<string, pieceId>`；
* 越界：检查坐标范围；
* 碰撞：对每个世界格子查 `occupied`。

## 非悬空（支撑）

* **底部判定**：`z==1` 即成立；
* **相邻判定**：六邻域 `(±1,0,0),(0,±1,0),(0,0,±1)` 中是否存在属于**其他块**的格子；
* **至少一个体素**满足即通过。
* 可选更严格版本：要求**总接触面数 ≥ 1**（同上等价，因为单位面面积=1）。

## 完成判定

* `occupied.size === A*B*C` 且 `unplaced.size === 0` 即胜利；
* 或额外校验：所有世界格子均被某个 `pieceId` 占据（冗余一致性检查）。

---

# 接口与伪代码

## 预处理

```ts
function preprocessPiece(p: PieceDef): PieceDef {
  const rots = all24Rotations();
  const seen = new Set<string>();
  p.rotationSignatures = [];
  for (const R of rots) {
    const pts = p.localVoxels.map(v => matMul(R, v));
    const sig = normalizeAndStringify(pts);
    if (!seen.has(sig)) { seen.add(sig); p.rotationSignatures.push(sig); }
  }
  return p;
}
```

## 形状匹配与放置（按格子集合）

```ts
function placePieceByCells(gs: GameState, pieceId: string, targetCells: Cell[]): Result {
  const p = getPieceDef(gs, pieceId);
  if (targetCells.length !== p.localVoxels.length) return Err('WrongCount');

  // 1) 形状匹配
  const sigT = normalizeAndStringify(toZeroBased(targetCells));
  if (!p.rotationSignatures.includes(sigT)) return Err('ShapeMismatch');

  const { rot, t } = inferTransformFromTarget(p, targetCells); // 通过 sigT 对应旋转+位移恢复
  const worldCells = applyTransform(p.localVoxels, rot, t, +1 /*to 1-based*/);

  // 2) 越界
  if (!withinBox(worldCells, gs.spec.box)) return Err('OutOfBounds');

  // 3) 碰撞
  for (const c of worldCells) if (gs.occupied.has(key(c))) return Err('Collision');

  // 4) 非悬空
  if (!hasSupport(worldCells, gs.byCell)) return Err('Floating');

  // 5) 提交
  commitPlacement(gs, pieceId, { rot, t }, worldCells);
  return Ok({ transform: { rot, t }, worldCells });
}
```

## 支撑判定

```ts
function hasSupport(worldCells: Cell[], byCell: Map<string,string>): boolean {
  for (const c of worldCells) {
    if (c.z === 1) return true; // 底部支撑
    const neigh = [
      {x:c.x+1,y:c.y,z:c.z}, {x:c.x-1,y:c.y,z:c.z},
      {x:c.x,y:c.y+1,z:c.z}, {x:c.x,y:c.y-1,z:c.z},
      {x:c.x,y:c.y,z:c.z+1}, {x:c.x,y:c.y,z:c.z-1},
    ];
    for (const n of neigh) {
      const pid = byCell.get(key(n));
      if (pid) return true; // 面接触到其他块
    }
  }
  return false;
}
```

## 移动与取出

```ts
function movePieceByCells(gs, pieceId, targetCells) {
  const prev = gs.placed.get(pieceId);
  uncommit(gs, prev.worldCells);
  const res = placePieceByCells(gs, pieceId, targetCells);
  if (!res.ok) commitPlacement(gs, pieceId, prev.transform, prev.worldCells); // 回滚
  return res;
}

function pickupPiece(gs, pieceId) {
  const prev = gs.placed.get(pieceId);
  uncommit(gs, prev.worldCells);
  gs.placed.delete(pieceId);
  gs.unplaced.add(pieceId);
}
```

---

# UI/UX 方案

* **视图**：正交 3D + 可切换俯视/前视/侧视剖切，格子对齐显示；
* **托盘**：未放置块平铺展示，鼠标悬停预览 24 个朝向缩略图；
* **放置模式**：

  * **选格放置**：在盒内点亮 n 个格子（与块体素数相同），实时显示**形状匹配状态**；
  * **锚点放置**：选块→点锚→旋转（Q/E 或滚轮）→确认；
* **反馈**：

  * 绿色 ghost = 合法；红色 ghost = 非法；
  * 非法提示码（越界/碰撞/悬空/形状不符/数量不符）；
  * 支撑触点高亮小方框；
* **操作**：点击已放置块 → 工具条（移动/旋转/取出）；
* **辅助**：自动“**吸附**”到最近合法位姿（可选）；撤销/重做；
* **完成**：满格闪烁+统计信息（用时、移动次数）。

---

# 性能与工程实现

* **预处理缓存**：块的 `rotationSignatures`、24 旋转矩阵常量；
* **O(1) 查找**：`occupied: Set<string>` 与 `byCell: Map`；
* **批量操作**：`worldCells` 以数组缓存，避免每次再次旋转；
* **可选 Bitset**：小盒子（≤64 格）可用位集（`BigInt`）做越界/碰撞/支撑并行位运算；
* **单元测试**：

  * `normalize/rot/signature` 等价性；
  * 放置三步（越界/碰撞/支撑）独立测试；
  * 典型案例：

    * 1×3 放在 (1,1,1)-(1,1,3)；
    * 悬空块（仅顶层）→ 应失败；
    * 与底面仅顶点接触→ 失败（非面接触）；
    * 与已放块边/角接触→ 失败（仍非面）。

---

# 关卡与存档格式（示例）

```json
{
  "box": { "A": 4, "B": 4, "C": 4 },
  "pieces": [
    {
      "id": "A",
      "localVoxels": [[0,0,0],[0,0,1],[0,0,2]],    // 1x1x3
      "rotationSignatures": ["...预处理填充..."]
    },
    { "id": "B", "localVoxels": [[0,0,0],[1,0,0],[0,1,0],[0,0,1]], "rotationSignatures": ["..."] }
  ]
}
```

**存档**：在此基础上追加 `placed`（pieceId→transform/worldCells）、`occupied`（可重建）与操作历史。

---

# 错误码建议

* `OK`
* `WrongCount`（目标格子数量与块体素数不一致）
* `ShapeMismatch`（目标格子构成的形状与块的任一旋转不一致）
* `OutOfBounds`
* `Collision`
* `Floating`（未满足非悬空支撑）

---

# 可选扩展

* **装配物理模式**：保留你之前“直线可拆/可装”的检测作为高难模式；
* **帮手提示**：给出与当前 ghost 最相近的合法变换（最少冲突格+最近支撑）；
* **谜题验证器**：离线用 DLX/精确覆盖检查“至少 1 解/唯一解”；
* **皮肤**：单位立方体加倒角与阴影、每块随机配色但固定种子。

---

提供一份**Python 原型**（仅逻辑+CLI/简单可视化）帮你验证判定与交互流程。
