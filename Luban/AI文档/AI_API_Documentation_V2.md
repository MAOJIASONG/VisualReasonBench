# LuBanSuo External AI Control API (外部 AI 控制接口)

本文档定义了外部 AI 智能体通过 Socket 请求与 Unity 环境交互的协议和 API 接口。

## 1. 通信协议 (Communication Protocol)
- **协议类型**: TCP (Socket)
- **默认端口**: 9999
- **数据格式**: JSON

---

## 2. 指令列表 (Command List)

所有请求必须包含 `command` 字段。参数放在 `params` 字段中。
- **env_id**: 范围 **0 - 8**。
    - `0`: **用户/回放环境**。主视角控制，支持 UI 操作与回放重演。
    - `1 - 8`: **AI 专用环境**。独立运行，不干扰主界面，专门用于控制与数据采集。
- **执行方式**: 指令是**串行处理**的，Unity 会在处理完当前指令后才接受下一个请求。请避免多线程并发对同一环境发送指令。

### 2.1 重置与开始 (Start/Reset)
重置指定环境的关卡进度。

```json
  "command": "start_level", "env_id": 1,
  "params": { "level_index": 0 }
}
```
- **level_index**: 范围 **0 - 31**。

### 2.2 移动零件 (Move Piece)
沿指定轴向移动零件。使用**逻辑单位**（1单位 = 0.0001 Unity单位）。

```json
{
  "command": "move", "env_id": 0,
  "params": {
    "piece_id": 1,
    "axis": "x",
    "distance": 10.0,
    "is_exploratory": false 
  }
}
```
- `is_exploratory`: 若为 `true`，零件会尝试朝指定方向移动直到被阻挡。

### 2.3 旋转零件 (Rotate Piece)
旋转零件。角度必须是 **90 的倍数**（如 90, -90, 180）。不支持任意角度（如 45 度）。

```json
{
  "command": "rotate", "env_id": 0,
  "params": {
    "piece_id": 1,
    "axis": "y",
    "angle": 90.0
  }
}
```

### 2.4 获取状态 (Get State)
获取当前零件的数值状态（坐标、旋转、是否解决、步数）。**注意：此指令不触发截图**。

```json
{
  "command": "get_state", "env_id": 0
}
```

### 2.5 请求截图 (Capture)
显式请求环境的所有视角（Camera）截图。

```json
{
  "command": "capture", "env_id": 0
}
```
- **返回结果**: 包含所有截图的绝对路径，路径之间用 `|` 分隔。

---

## 3. 返回数据格式 (Common Response)

```json
{
  "status": "success",
  "env_id": 0,
  "data": {
    "puzzle_index": 1,
    "step_count": 5,
    "is_solved": false,
    "pieces": [
      {
        "id": 1,
        "pos": {"x": 10.0, "y": 0.0, "z": -20.0},
        "rot": "XPositive_YPositive_ZPositive",
        "is_finished": false
      }
    ],
    "screenshot_path": "C:/Snapshots/Run_01/cam_0.png|C:/Snapshots/Run_01/cam_1.png"
  }
}
```

---

## 4. 关键原则 (Key Principles)

1.  **数值单位**: `distance` 和 `pos` 的数值采用逻辑单位（整数倍）。例如 API 下发的 10 代表 Unity 中的 0.001f。获取状态时也会返回此逻辑单位。
2.  **异步截图**: `get_state` 是瞬时返回的纯数据请求；`capture` 是异步请求，仅在所有视角截取完成后才返回路径。
3.  **独立性**: 对 `env_id: 1` 的操作不会影响 `env_id: 0` 的渲染或逻辑。
4.  **延迟录制**: 只有在发生第一次有效操作（move/rotate）后，才会真正创建对应的文件目录。
5.  **零件 ID 发现器**: 每一关的零件 ID 是固定的。建议在 `start_level` 后调用一次 `get_state`，通过返回列表获取该关卡内所有零件的初始 ID 和坐标。
6.  **坐标系说明**:
    - **Y 轴**: 垂直向上（高度）。
    - **X/Z 轴**: 水平平面。
    - **单位**: 全部为逻辑单位（Integer Scaling）。
7.  **碰撞处理**: 
    - `move` 指令如果返回值 `success: false`，说明路径被阻挡。
    - `is_exploratory: true` 会尝试移动至阻挡点。

## 5. 24 种零件方向状态 (Orientation States)
零件在空间中有 24 种可能的朝向（由 X/Y/Z 轴的指向决定）。返回格式为 `X朝向_Y朝向_Z朝向`。

| 序号 | 旋转状态字符串 (rot) | 说明 |
| :--- | :--- | :--- |
| 1 | `XPositive_YPositive_ZPositive` | 原始/默认姿态 |
| 2 | `XPositive_YNegative_ZNegative` | |
| 3 | `XNegative_YPositive_ZNegative` | |
| 4 | `XNegative_YNegative_ZPositive` | |
| 5 | `XPositive_ZPositive_YNegative` | |
| 6 | `XPositive_ZNegative_YPositive` | |
| 7 | `XNegative_ZPositive_YPositive` | |
| 8 | `XNegative_ZNegative_YNegative` | |
| 9 | `YPositive_ZPositive_XPositive` | |
| 10 | `YPositive_ZNegative_XNegative` | |
| 11 | `YNegative_ZPositive_XNegative` | |
| 12 | `YNegative_ZNegative_XPositive` | |
| 13 | `YPositive_XPositive_ZNegative` | |
| 14 | `YPositive_XNegative_ZPositive` | |
| 15 | `YNegative_XPositive_ZPositive` | |
| 16 | `YNegative_XNegative_ZNegative` | |
| 17 | `ZPositive_XPositive_YPositive` | |
| 18 | `ZPositive_XNegative_YNegative` | |
| 19 | `ZNegative_XPositive_YNegative` | |
| 20 | `ZNegative_XNegative_YPositive` | |
| 21 | `ZPositive_YPositive_XNegative` | |
| 22 | `ZPositive_YNegative_XPositive` | |
| 23 | `ZNegative_YPositive_XPositive` | |
| 24 | `ZNegative_YNegative_XNegative` | |
