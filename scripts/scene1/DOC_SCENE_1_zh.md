# DOC SCENE 1 — 视觉与抓取（团队说明 · 中文）

**视觉作者：** Williams  
**最后更新：** 2026年7月13日  
**职责：** 头部检测 + 手腕相机精修。交接/掉落属于控制，不在视觉范围内。

本文说明：**做了什么、为什么、踩过的坑、如何复现**。  
文档与代码同目录：`scripts/scene1/`。

法文版（同内容）：[`DOC_SCENE_1.md`](./DOC_SCENE_1.md)

---

## 1. 目标

主办方流程：检测 4 个快递 → 右手抓取 → 称重 → 二次抓取 → 交左手 → 放入分拣箱。

**反作弊：** 机械臂**禁止**用 `/mujoco/qpos` 或 `/ground_truth/state` 驱动。  
任务中 `GT_COMPARE = False`。MuJoCo GT 仅用于实验室 CSV 对比。

---

## 2. 需要了解的文件（本目录）

| 文件 | 作用 |
|------|------|
| `perception.py` | 头部：LiDAR + RGB LAB/HSV + 深度 → 4 个快递 |
| `wrist_vision.py` | 手部：颜色/深度 blob + 小步 Δxy 伺服 |
| `config.py` | 模式、阈值、指尖偏移、WRIST_*、FORCE_PARCEL |
| `actions.py` | 抓取 / 称重 / 交接（调用视觉） |
| `../scene1_task.py` | 场景 1 入口 |
| `../../src/perception_api.py` | 相机 `cam_h` / `cam_l` / `cam_r` + LiDAR + TF |

**不必为运行时推送：** `labo/scene1/**`（csv、logs、reports）、仓库根目录的日志/CSV。

实验产物 → 仓库根目录 `labo/scene1/`。

---

## 3. 架构（工作方式）

```
1. 头部  detect_parcels()
   LiDAR 聚类（位置）+ 头部颜色 LAB/HSV（身份）+ 深度
   → 融合 → 相对 2×2 网格
   → [{name, color, center, source}, ...]

2. 右手接近到检测中心上方

3. 手部  observe_hand() / refine_target_with_wrist()
   腕部相机颜色+深度 blob
   小步 Δxy（每步最多约 2 cm，最多 4 次）
   门控：关闭前需「see-check OK」

4. 抓取  浅下降 + 闭合
   指尖浅进（避免过深导致 IK 失败）
```

**视觉优先策略：** 用手腕相机决定是否允许闭合（`WRIST_VISION_ONLY_GATE`、`WRIST_SKIP_CLAW_HOLD_CHECK`）。  
仿真常在空爪时仍报 `REACHED` ≈88% —— **不要**用 hold 状态判断视觉好坏。

---

## 4. 已验证结果

| Seed / 快递 | 视觉 / 抓取结果 |
|-------------|----------------|
| Seed **30**，`parcel_1`（灰/棕） | **最佳跑次**：VISION OK（frac≈0.10），`claw R=3` Grabbed，称重 + 二次抓取 + 交接 OK（`DONE 1/1`）。 |
| Seed 0，`parcel_4` 蓝 | 上方到位 OK；爪常空（夹爪/指尖问题，不是头部检测失败）。 |
| Seed 0 / 50，`parcel_2` 黄 | 接近 OK；hold / tip 另论。 |

头部检测多 seed 0–3（早期实验）：4/4，结构误差约 0.2–3.7 cm —— 见 `labo/scene1/reports/`。

---

## 5. 遇到的问题与解决

| 问题 | 现象 | 修复 / 经验 |
|------|------|-------------|
| 下降过深 | `z≈-0.057` → see-check 后 IK 失败 → **从未闭合** | 浅 tip：`RIGHT_PICK_IK_Z ≈ -0.005`，`RIGHT_CLAW_TIP_OFFSET ≈ [0.02, 0.01, -0.005]`；`WRIST_CLOSE_EVEN_IF_IK_FAIL = True` |
| 假「holding」 | `MOVING` + effort → 误判已抓住 | `right_holding()` **忽略 MOVING** |
| 腕部伺服过大 | Δxy 6–29 cm → 发散 | `WRIST_MAX_DELTA_XY = 0.02`，gain 0.5，最多 4 次 |
| 偏航 0°/90°「方轴」 | 抓取中途打乱手臂 | **已移除**；`WRIST_YAW_ENABLE = False`；保持主办方固定 `RIGHT_PICK_QUAT` |
| YOLO | Docker 内 Torch 过旧 | 放弃 → LAB/HSV + LiDAR |
| 用 GT 控臂 | 反作弊 | 禁止；`GT_COMPARE` 仅实验室 |

**以后可考虑（不要在下降伺服中途做）：** 夹爪平行/垂直于快递边 —— 但不在伺服循环里改 yaw。

---

## 6. 当前配置（要点）

同目录 `config.py`：

```python
PERCEPTION_ONLY = False   # False = 完整任务
TOUCH_TEST = False
GT_COMPARE = False        # 仅实验室 CSV 时为 True

FORCE_PARCEL_NAME = "parcel_1"  # 单测；团队联调请设 None

RIGHT_CLAW_TIP_OFFSET = [0.02, 0.01, -0.005]
RIGHT_PICK_IK_Z = -0.005

WRIST_SERVO_ITERS = 4
WRIST_MAX_DELTA_XY = 0.02
WRIST_YAW_ENABLE = False
WRIST_REQUIRE_SEE_BEFORE_CLOSE = True
WRIST_VISION_ONLY_GATE = True
WRIST_SKIP_CLAW_HOLD_CHECK = True   # 聚焦视觉；夹爪可由控制侧再调
```

**快递名 ↔ 颜色（约）：**

| Name | 颜色 |
|------|------|
| `parcel_1` | 棕 / 灰 |
| `parcel_2` | 黄 |
| `parcel_3` | 橙 |
| `parcel_4` | 蓝 |

推送给全队前，建议 `FORCE_PARCEL_NAME = None`。

---

## 7. 如何重跑（复现）

```bash
cd ~/leju-kuavo-challenge-cup-2026

# 新跑前务必先停
bash docker/stop_scene1.sh

# 完整任务（seed，超时秒）
bash docker/run_scene1_mission.sh 30 900

# 仅感知（不动臂）
# → config.py 中 PERCEPTION_ONLY = True
bash docker/run_scene1_local.sh

# 多 seed 头部检测
bash docker/run_scene1_multiseed_perception.sh 0 1 2 3
```

**常用日志：**

```bash
# 任务
grep -E 'DETECT|COLOR|FUSE|WRIST|VISION|Grabbed|DONE|claw' scene1_mission_run.log

# 仅感知
grep -E 'DETECT|COLOR|FUSE|REPORT|DONE' scene1_local_run.log
```

视觉抓取成功（例 seed 30）：出现 `VISION OK`、`claw R=3`、Grabbed，然后称重。  
交接掉落**不否定**头部/手腕检测。

终端 **Ctrl+C** 不会停 Docker → 用 `stop_scene1.sh`。

---

## 8. 感知技术 / 分层

1. **LiDAR** — 位置（桌面 XY 聚类，2×2 网格兜底）  
2. **头部 RGB（LAB/HSV）** — 身份（颜色 → 名称）  
3. **头部深度** — 相机 3D / 射线  
4. **融合** — 颜色 ↔ 聚类；`err_structure_2x2` = 相对几何（非 seed0 绝对坐标）  
5. **腕部相机** — 闭合前局部精修（禁止大跳）

实验室「检测 OK」标准（头部）：

1. `named == 4/4`  
2. `err_structure_2x2 < 0.05` m  
3. `colorish == 4`  
4. landmarks `lm=2/2`（加分）

---

## 9. 分工建议

| 视觉 | 控制 / 任务 |
|------|-------------|
| 多 seed 质量、颜色、腕部伺服（降低 frac） | 真实 hold、按色 tip、交接不掉 |
| `perception.py`、`wrist_vision.py`、`config` 阈值 | `actions.py`（夹爪 / 交接 / 箱） |
| 不要把 GT 接回手臂 | 不要无故拆掉视觉门控 |

---

## 10. 视觉后续方向

- 去掉 `FORCE_PARCEL_NAME` 后的多 seed 鲁棒性  
- 标定腕部伺服符号（`WRIST_SERVO_SIGN_*`），使 `frac` / Δpx **持续变好**  
- 减少假 blob（天空 / 手指）— depth + ROI 已过滤一部分  
- 方轴 yaw：仅在下降环之外，如有需要再做  

---

## 链接

- 实验产物：`labo/scene1/`（仓库根）  
- 法文版：[`DOC_SCENE_1.md`](./DOC_SCENE_1.md)
