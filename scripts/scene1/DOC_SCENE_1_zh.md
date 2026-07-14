# DOC SCENE 1 — 视觉与抓取（团队）

**视觉作者：** Williams  
**更新日期：** 2026-07-15  
**重点：** 头部检测 + 腕部相机微调；抓取 / 称重 / 交接沿用主办动作结构。

**French：** [`DOC_SCENE_1.md`](./DOC_SCENE_1.md)  
**仓库：** https://github.com/williams73-oss/leju-competiton

---

## 0. 别人能不能下载后直接跑？

**可以**，前提是：对方已经有主办的 **挑战杯仿真工作空间 / Docker 镜像**（和官方选手环境一样）。  
本仓库 **不是** 整套仿真，而是要放进工作空间的 **任务包**（`challenge_cup_task_template`）：Scene1 代码 + 扩展后的 `robot_api`。

按下面 **§1 从零运行** 操作即可。

---

## 1. 从零运行（克隆 → 安装 → 启动）

### 1.1 前置条件

- 已安装主办环境：`kuavo_challenge_cup_2026` Docker 镜像（或等价 `kuavo_ws`）
- 能正常启动仿真并跑官方 `challenge_task.py`
- 有 GPU / 显示（与官方一样）

### 1.2 下载本仓库代码

```bash
cd /tmp
git clone https://github.com/williams73-oss/leju-competiton.git
```

本仓库根目录 ≈ ROS 包 `challenge_cup_task_template`，关键文件：

| 路径 | 作用 |
|------|------|
| `scripts/challenge_task.py` | 三场景统一入口 |
| `scripts/scene1_task.py` | Scene1 入口 |
| `scripts/scene1/` | 感知 / 动作 / 配置 |
| `src/robot_api.py` | 手臂/夹爪（含 FK、单臂 IK、hold…） |
| `src/perception_api.py` | 相机 + LiDAR + TF |

### 1.3 装进工作空间（二选一）

**方式 A — 整包替换（推荐）**

把现有包备份后，用本仓库覆盖：

```bash
# 例：主办工作空间路径按你的机器改
WS=~/leju-kuavo-challenge-cup-2026   # 或 /root/kuavo_ws（容器内）
PKG=$WS/src/challenge_cup_task_template

mv "$PKG" "${PKG}.bak_$(date +%Y%m%d)"   # 备份
cp -a /tmp/leju-competiton "$PKG"
```

**方式 B — 只覆盖 Scene1 相关文件**

```bash
PKG=~/leju-kuavo-challenge-cup-2026/src/challenge_cup_task_template
SRC=/tmp/leju-competiton

cp -a "$SRC/scripts/scene1" "$PKG/scripts/"
cp -a "$SRC/scripts/scene1_task.py" "$PKG/scripts/"
cp -a "$SRC/src/robot_api.py" "$PKG/src/"
cp -a "$SRC/src/perception_api.py" "$PKG/src/"
```

若改过 `CMakeLists` / `package.xml`，在工作空间里重新 `catkin build challenge_cup_task_template`（或团队惯用编译命令）。

### 1.4 配置模式（`scripts/scene1/config.py`）

一次只开一种：

| 设置 | 含义 |
|------|------|
| `PERCEPTION_ONLY = True` | 只跑检测，手臂不动（视觉调试） |
| `TOUCH_TEST = True`（且上面 False） | 检测后点触 |
| **两个都 False** | **完整任务**：抓 → 称 → 交接 → 投箱 |

当前团队默认：**两个都 `False`（完整任务）**，`FORCE_PARCEL_NAME = None`（四个包裹）。

### 1.5 启动 Scene1

在 **已 source 的仿真环境**里（容器或主机，与官方相同）：

```bash
source /root/kuavo_ws/devel/setup.zsh   # 路径按实际改
# 或: source ~/leju-kuavo-challenge-cup-2026/devel/setup.zsh

rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 30
```

换种子：改 `--seed`（例：`0`、`30`、`400`）。

### 1.6 日志里看什么

```text
DETECT | COLOR | FUSE | WRIST | VISION | Grabbed | DONE | claw
```

成功抓取示例：`VISION OK`、`claw R=3`（Grabbed）、然后称重 / 交接。

### 1.7 可选：本机 monorepo 里的 Docker 快捷脚本

若你有完整仓库 `leju-kuavo-challenge-cup-2026` **并且**里面有 `docker/run_scene1_*.sh`：

```bash
cd ~/leju-kuavo-challenge-cup-2026
bash docker/stop_scene1.sh
bash docker/run_scene1_mission.sh 30 900
```

这些脚本 **不在** GitHub `leju-competiton` 里；没有它们时用上面的 **`rosrun`** 即可。

`Ctrl+C` 只停终端，不一定停 Docker — 有 `stop_scene1.sh` 时再用它。

---

## 2. 项目思路

**视觉目标：** 找到 4 个包裹（谁 / 在哪），**不作弊**（不用 MuJoCo GT 控臂）。

动作结构参考主办：  
`challenge_cup_simulator/.../collect_scene1_handoff_dataset.py`  
（检测 → 右手抓 → 称重 → 再抓 → 交接 → 投箱）。

**视觉核心：** `perception.py` + `wrist_vision.py` + `config.py`。  
**反作弊：** 禁止 `/mujoco/qpos` / GT 控臂；`GT_COMPARE` 仅实验室。

---

## 3. 场景目标

4 包裹 → 右手抓 → 称重 → 再抓 → 左手 → 投箱。

---

## 4. 关键文件

| 文件 | 作用 |
|------|------|
| `perception.py` | 头：LiDAR + RGB LAB/HSV + depth → 4 包裹 |
| `wrist_vision.py` | 腕：颜色/深度 blob + 小步 Δxy |
| `config.py` | 模式、阈值、WRIST_*、FORCE_PARCEL |
| `actions.py` | 抓取/称重/交接 |
| `../scene1_task.py` | 场景入口 |
| `../../src/perception_api.py` | 相机 + LiDAR + TF |
| `../../src/robot_api.py` | 手臂/夹爪（含 FK / 单臂 IK 等） |

---

## 5. 架构

```
HEAD detect → 接近 → HAND refine → GRASP close
```

---

## 6. 已验证

| Seed / 包裹 | 结果 |
|-------------|------|
| Seed **30**, `parcel_1` | VISION OK + Grabbed |
| Seed **0** 头部 | 4/4，`err_structure_2x2 ≈ 0.009` m |
| Seed **400**, 黄 `parcel_2` | 按需复测 |

---

## 7. 配置速查

```python
PERCEPTION_ONLY = False
TOUCH_TEST = False
FORCE_PARCEL_NAME = None   # 或 "parcel_2" 强制黄箱
```

| Name | 颜色 |
|------|------|
| `parcel_1` | 棕/灰 |
| `parcel_2` | 黄 |
| `parcel_3` | 橙 |
| `parcel_4` | 蓝 |

---

## 8. robot_api 说明（必读）

`actions.py` 需要相对 **原版模板** 多出的接口，本仓库 **已写入** `src/robot_api.py`：

| 接口 | 作用 |
|------|------|
| `_read_arm_joints_rad()` | 读 14 臂关节（传感器） |
| `call_fk()` | 正运动学 |
| `solve_ik_one_hand()` | 单臂 IK（另一手锁定） |
| `_last_cmd_deg` | 上次下发的关节角 |
| `right_holding()` | 右爪是否抓稳 |
| `describe_right()` | 右爪状态文字 |
| imports `fkSrv` / `sensorsData` | FK 服务与传感器消息 |

同事若仍用旧 `robot_api`，运行到抓取/交接会缺方法。  
**请使用本仓库的 `src/robot_api.py`。**

---

## 9. 链接

- 主办参考：`collect_scene1_handoff_dataset.py`
- 团队仓库：https://github.com/williams73-oss/leju-competiton
