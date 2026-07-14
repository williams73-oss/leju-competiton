# DOC SCENE 1 — 视觉（团队）

**视觉作者：** Williams  
**更新日期：** 2026-07-14  
**重点：** 头部检测 + 腕部相机微调。  
手臂 / 交接 = 非视觉主责（但沿用主办结构）。

**French：** [`DOC_SCENE_1.md`](./DOC_SCENE_1.md)

---

## 0. 项目思路（先读）

**视觉目标：** 找到 4 个包裹（谁/在哪），**不作弊**（不用 MuJoCo GT 驱动手臂）。

**动作结构灵感：** 主办脚本  
`challenge_cup_simulator/test/collect_scene1_dataset/collect_scene1_handoff_dataset.py`  
（检测 → 右手抓 → 称重 → 再抓 → 交接 → 投箱）。

本仓库以 **视觉优先**：`actions.py` 沿用/适配主办动作流程；  
`robot_api.py` 少量扩展（FK、单臂 IK、右爪 hold…）以匹配该流程。  
**视觉评分核心：** `perception.py` + `wrist_vision.py` + `config.py` 阈值。

**反作弊：** 禁止用 `/mujoco/qpos` / GT 控臂。`GT_COMPARE` 仅实验室。

---

## 1. 场景目标

4 包裹 → 右手抓 → 称重 → 再抓 → 左手 → 投箱。

---

## 2. 关键文件

| 文件 | 作用 |
|------|------|
| `perception.py` | 头：LiDAR + RGB LAB/HSV + depth → 4 包裹 |
| `wrist_vision.py` | 腕：颜色/深度 blob + 小步 Δxy |
| `config.py` | 模式、阈值、WRIST_*、FORCE_PARCEL |
| `actions.py` | 抓取/称重/交接（主办风格） |
| `../scene1_task.py` | 场景入口 |
| `../../src/perception_api.py` | 相机 + LiDAR + TF |
| `../../src/robot_api.py` | 手臂/夹爪（含 FK/单臂 IK 等） |

---

## 3. 架构

```
HEAD detect → 接近 → HAND refine → GRASP close
```

---

## 4. 已验证

| Seed / 包裹 | 结果 |
|-------------|------|
| Seed **30**, `parcel_1` | VISION OK + Grabbed |
| Seed **0** 头部 | 4/4，`err_structure_2x2 ≈ 0.009` m |
| Seed **400**, 黄 `parcel_2` | 跑完后填 |

---

## 5. 配置

```python
PERCEPTION_ONLY = True
FORCE_PARCEL_NAME = None  # 或 "parcel_2" 强制黄色
```

| Name | 颜色 |
|------|------|
| `parcel_1` | 棕/灰 |
| `parcel_2` | 黄 |
| `parcel_3` | 橙 |
| `parcel_4` | 蓝 |

---

## 6. 运行

```bash
bash docker/stop_scene1.sh
SEED=400 bash docker/run_scene1_local.sh
bash docker/run_scene1_mission.sh 30 900
```

---

## 7. robot_api 说明

`actions.py` 需要相对原版模板多出的接口：  
`_read_arm_joints_rad` / `call_fk` / `solve_ik_one_hand` / `right_holding` / `describe_right`  
已写入 `src/robot_api.py`（学自主办，**非**包裹 GT）。  
同事若只有旧 `robot_api`，actions 会缺方法。

---

## 链接

- 主办参考：`collect_scene1_handoff_dataset.py`  
- 团队仓库：https://github.com/williams73-oss/leju-competiton  
