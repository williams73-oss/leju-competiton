#!/usr/bin/env python3
"""
Actions Scene 1 — touch, saisie, balance, handoff, bac.

Pick / weigh / regrasp : main droite seule (orga `_move_hand` + FK lock gauche).
Handoff / bac : double bras quand la gauche doit bouger.
"""
from __future__ import print_function
import math
import os
import sys

import rospy

_scripts = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)
_pkg = os.path.dirname(_scripts)
sys.path.insert(0, os.path.join(_pkg, "src"))

from perception_api import CameraReader, LidarReader, TFReader

from scene1.config import *  # noqa: F401,F403
from scene1.config import _quat_from_ypr_deg  # import * n'exporte pas les _
from scene1.perception import (
    detect_parcels,
    detect_all_touch_targets,
    log_touch_targets_report,
    log_scene_landmarks,
    inspect_table_depth,
    _is_safe_touch_target,
    _targets_for_touch,
    _expected_parcel_positions,
    _landmark_ref_positions,
    _is_rgb_backed,
    _is_lidar_backed,
)
from scene1.wrist_vision import (
    observe_hand,
    refine_target_with_wrist,
    wrist_sees_centered,
)


def _sanitize_touch_target(target, log):
    """Corrige z/y aberrants (rgb-depth) avant IK — évite poses instables."""
    t = dict(target)
    cx, cy, cz = t["center"]
    name = t["name"]
    src = t.get("source", "")

    if t["kind"] == "parcel" or name == "weighing_area":
        if cz > TOUCH_TABLE_Z_MAX or cz < TOUCH_TABLE_Z_MIN:
            log("[TOUCH] %s z=%.3f → %.3f (corr. table, was %s)",
                name, cz, TABLE_PARCEL_Z, src)
            cz = TABLE_PARCEL_Z
        if cx < TOUCH_MIN_X:
            refs = _expected_parcel_positions() if name in PARCEL_NAMES else _landmark_ref_positions()
            if name in refs:
                cx = refs[name][0]
                log("[TOUCH] %s x trop bas → %.3f (ref)", name, cx)

    if name == "weighing_area":
        _, ref_y, ref_z = _landmark_ref_positions()["weighing_area"]
        if cy > -0.25:
            log("[TOUCH] weighing_area y=%.3f → %.3f", cy, ref_y)
            cy = ref_y
        cz = min(cz, ref_z + 0.04)

    if name == "sorting_box":
        ref_x, ref_y, ref_z = _landmark_ref_positions()["sorting_box"]
        if cz > 0.15:
            cz = ref_z + 0.06
            log("[TOUCH] sorting_box z → %.3f", cz)
        if cy < 0.08:
            cy = ref_y
            cx = max(cx, ref_x)

    t["center"] = (float(cx), float(cy), float(cz))
    return t


def _stable_between_touches(arm, robot, log):
    """Repos stable entre deux contacts (pas go_ready — évite chute)."""
    robot.stop()
    robot.switch_to_stance()
    arm.go_home()
    rospy.sleep(TOUCH_ARM_SETTLE)


def _parcels_for_touch(all_parcels):
    """Colis nommés triés parcel_1 … parcel_4."""
    by_name = {p["name"]: p for p in all_parcels if p.get("name") in PARCEL_NAMES}
    return [by_name[n] for n in PARCEL_NAMES if n in by_name]


def _solve_and_move_touch(arm, side, pos_xyz, quat, log, label):
    """Touch = une main seule (orga-style), l'autre reste figée via FK."""
    ok = _move_hand(
        arm, side, pos_xyz, quat, log, label,
        constraint_mode=RIGHT_PICK_IK_MODE, settle=TOUCH_ARM_SETTLE,
    )
    return ok


def _right_pick_offset(name, cy):
    """Offsets prise orga (par nom ou rangée près/loin)."""
    if name in RIGHT_PICK_OFFSET_BY_PARCEL:
        return list(RIGHT_PICK_OFFSET_BY_PARCEL[name])
    if cy > RIGHT_PICK_NEAR_FAR_Y_THRESHOLD:
        return list(RIGHT_PICK_OFFSET_NEAR_ROW)
    return list(RIGHT_PICK_OFFSET_FAR_ROW)


def _action_center_xyz(target, prefer_raw=False):
    """
    Centre pour bras.
    prefer_raw=True : ancien LiDAR avant grid (touch only).
    Mission grasp : toujours `center` (même point que SELECT) —
    center_raw après grid-x divergeait (2/4: SELECT 0.30 vs grasp 0.41).
    """
    if prefer_raw:
        raw = target.get("center_raw")
        if raw is not None:
            return [float(raw[0]), float(raw[1]), float(raw[2])]
    cx, cy, cz = target["center"]
    return [float(cx), float(cy), float(cz)]


def touch_with_right_hand(arm, claw, target, log, cam=None, tf_reader=None):
    """
    Approche et contact léger : pince droite ouverte, pas de saisie.
    Droite seule (comme pick orga) — gauche figée après home/preset.
    """
    target = _sanitize_touch_target(target, log)
    cx, cy, cz = _action_center_xyz(target, prefer_raw=True)
    name = target["name"]
    ox, oy, oz = _right_pick_offset(name, cy)
    pick_x = cx + ox
    pick_y = cy + oy + TOUCH_Y_OFFSET
    pick_z = RIGHT_PICK_IK_Z + oz + TOUCH_Z_ABOVE_CENTER
    approach_z = max(RIGHT_PICK_TRANSIT_IK_Z, pick_z + 0.08)

    log("[TOUCH] %s (%s) LiDAR=(%.3f,%.3f,%.3f) → pick≈(%.3f,%.3f,%.3f)",
        name, target.get("color", "?"), cx, cy, cz,
        pick_x, pick_y, pick_z)

    rq = RIGHT_PICK_QUAT
    clear_z = max(ARM_CLEAR_TABLE_Z, approach_z)
    if not _move_hand(
        arm, "right", [pick_x, pick_y, clear_z], rq, log, "TOUCH 右臂抬高",
        constraint_mode=RIGHT_PICK_IK_MODE, settle=TOUCH_ARM_SETTLE,
    ):
        log("[TOUCH] %s IK 失败 (haute)", name)
        return False

    target = refine_target_with_wrist(
        cam, tf_reader, target, hand="right", log=log)
    cx, cy, cz = _action_center_xyz(target)
    ox, oy, oz = _right_pick_offset(name, cy)
    pick_x = cx + ox
    pick_y = cy + oy + TOUCH_Y_OFFSET
    pick_z = RIGHT_PICK_IK_Z + oz + TOUCH_Z_ABOVE_CENTER
    approach_z = max(RIGHT_PICK_TRANSIT_IK_Z, pick_z + 0.08)
    if target.get("wrist_refined"):
        if not _solve_and_move_touch(
            arm, "right", [pick_x, pick_y, clear_z], rq, log, "TOUCH 腕部校正",
        ):
            log("[TOUCH] %s IK 失败 (wrist realign)", name)
            return False

    claw.open()
    claw.right_open()
    rospy.sleep(GRIPPER_SETTLE_TIME)

    right_above = [pick_x, pick_y, approach_z]
    if not _solve_and_move_touch(
        arm, "right", right_above, rq, log, "TOUCH 预接近",
    ):
        log("[TOUCH] %s IK 失败 (approche)", name)
        return False

    claw.open()
    claw.right_open()
    rospy.sleep(GRIPPER_SETTLE_TIME)

    right_touch = [pick_x, pick_y, pick_z]
    if not _solve_and_move_touch(
        arm, "right", right_touch, rq, log, "TOUCH 下降接触",
    ):
        log("[TOUCH] %s IK 失败 (contact)", name)
        return False

    log("[TOUCH] %s 接触 — 观察 simu (%.1f s)", name, TOUCH_DWELL)
    rospy.sleep(TOUCH_DWELL)

    if not _solve_and_move_touch(
        arm, "right", right_above, rq, log, "TOUCH 抬起",
    ):
        log("[TOUCH] %s IK 失败 (retrait)", name)
        return False

    log("[TOUCH] %s terminé OK", name)
    return True


def touch_with_left_hand(arm, claw, target, log, cam=None, tf_reader=None):
    """Contact léger main gauche (bac) — gauche seule, droite figée via FK."""
    target = _sanitize_touch_target(target, log)
    _ = (cam, tf_reader)  # refine colis seulement (cam_r) ; bac = landmark fixe
    cx, cy, cz = target["center"]
    touch_z = cz + TOUCH_Z_ABOVE_BOX
    name = target["name"]
    log("[TOUCH] %s (%s) main gauche → (%.3f, %.3f, %.3f)",
        name, target.get("color", "?"), cx, cy, touch_z)

    left_above = [cx, cy, touch_z + APPROACH_Z_OFFSET]
    if not _solve_and_move_touch(
        arm, "left", left_above, GRASP_QUAT, log, "TOUCH 左臂预接近",
    ):
        log("[TOUCH] %s IK 失败 (approche gauche)", name)
        return False

    claw.open()
    claw.left_open()
    rospy.sleep(GRIPPER_SETTLE_TIME)

    left_touch = [cx, cy, touch_z]
    if not _solve_and_move_touch(
        arm, "left", left_touch, GRASP_QUAT, log, "TOUCH 左臂接触",
    ):
        log("[TOUCH] %s IK 失败 (contact gauche)", name)
        return False

    log("[TOUCH] %s 接触 — 观察 simu (%.1f s)", name, TOUCH_DWELL)
    rospy.sleep(TOUCH_DWELL)

    if not _solve_and_move_touch(
        arm, "left", left_above, GRASP_QUAT, log, "TOUCH 左臂抬起",
    ):
        return False

    log("[TOUCH] %s terminé OK (gauche)", name)
    return True


def touch_target(robot, arm, claw, target, log, cam=None, tf_reader=None):
    """Route vers la bonne main selon la cible."""
    if TOUCH_USE_FORWARD:
        cx, _, _ = target["center"]
        if cx > 0.55 and target.get("hand") == "right":
            forward_duration = min(2.0, max(0.0, (cx - 0.45) / 0.05))
            if forward_duration > 0.1:
                log("[TOUCH] 前进 %.1f s 靠近 %s", forward_duration, target["name"])
                robot.move_forward(0.05, duration=forward_duration)
                robot.stop()
                rospy.sleep(0.5)
    if target.get("hand") == "left":
        return touch_with_left_hand(
            arm, claw, target, log, cam=cam, tf_reader=tf_reader)
    return touch_with_right_hand(
        arm, claw, target, log, cam=cam, tf_reader=tf_reader)


def approach_and_touch(robot, arm, claw, target, log, cam=None, tf_reader=None):
    """Toucher une cible depuis posture stable (home)."""
    return touch_target(
        robot, arm, claw, target, log, cam=cam, tf_reader=tf_reader)


def run_scene1_touch_test(robot, arm, claw, head, log):
    """Perception puis toucher chaque élément détecté (colis + balance + bac)."""
    log("=" * 50)
    log("场景一：触摸测试 — colis + balance + bac (TOUCH_TEST)")
    log("=" * 50)

    log("[STEP 1] 切换手臂到外部控制模式")
    arm.switch_to_external_control()
    rospy.sleep(0.5)
    robot.stop()
    robot.switch_to_stance()
    arm.go_home()
    rospy.sleep(TOUCH_ARM_SETTLE)
    claw.open()
    claw.right_open()
    claw.left_open()
    rospy.sleep(GRIPPER_SETTLE_TIME)
    log_scene_landmarks(log)

    lidar = LidarReader()
    cam = CameraReader()
    tf_reader = TFReader()
    rospy.sleep(0.5)

    log("[STEP 2] 低头感知 — colis, balance, bac (tête orga, fixe)")
    head.look_at(HEAD_LOOK_YAW, HEAD_LOOK_PITCH)
    rospy.sleep(HEAD_SETTLE_SEC)
    targets = detect_all_touch_targets(lidar, cam, tf_reader, log)
    # pas de look_forward : on reste comme l'orga pendant la détection

    log_touch_targets_report(targets, log)
    touch_list = _targets_for_touch(targets)
    touch_list = [t for t in touch_list if _is_safe_touch_target(t, log)]
    n_color = sum(
        1 for t in targets
        if t.get("kind") == "parcel" and (
            _is_rgb_backed(t.get("source", "")) or "color" in (t.get("source") or "")))
    log("[TOUCH] %d cibles safe (max_parcels=%d landmarks=%s) colorish=%d",
        len(touch_list), TOUCH_MAX_PARCELS, TOUCH_LANDMARKS, n_color)
    for t in touch_list:
        cx, cy, cz = t["center"]
        log("[TOUCH]   planifié %s (%s) main=%s → (%.3f, %.3f, %.3f)",
            t["name"], t.get("color", "?"), t.get("hand", "?"), cx, cy, cz)

    if not touch_list:
        log("[TOUCH] aucune cible safe — 中止 (évite chute)")
        return
    if n_color < 1:
        # LiDAR seul suffit pour TOUCH (couleur sert surtout au nommage)
        log("[TOUCH] couleur 0 — continue quand même (LiDAR OK, %d cibles)",
            len(touch_list))

    log("[STEP 3] contact — bras depuis home (stable)")
    _stable_between_touches(arm, robot, log)

    touched = 0
    failed = 0
    for i, target in enumerate(touch_list, 1):
        log("[TOUCH] --- %d/%d : %s ---", i, len(touch_list), target["name"])
        if approach_and_touch(
                robot, arm, claw, target, log, cam=cam, tf_reader=tf_reader):
            touched += 1
        else:
            failed += 1
        _stable_between_touches(arm, robot, log)

    log("[DONE] 触摸测试完成：成功 %d/%d  失败 %d",
        touched, len(touch_list), failed)
    log("场景一：触摸测试结束 — simu : pince sur chaque élément détecté ?")


# =============================================================================
# DÉCISION — choisir quel colis saisir en premier
# =============================================================================

def _parcel_xy_distance(parcel):
    """Distance horizontale du colis à l'origine du robot (approximation : le plus proche)."""
    cx, cy, _ = parcel["center"]
    return math.hypot(cx, cy)


def _parcel_select_score(parcel):
    """Score bas = meilleur. Pénalise fort grid/row-infer (cible vide / chute)."""
    score = _parcel_xy_distance(parcel)
    src = parcel.get("source") or ""
    if "row-infer" in src or src == "layout":
        score += 0.45
    elif "grid-2x2" in src:
        score += 0.35
    elif "grid-x" in src or "grid-y" in src:
        score += 0.15
    elif not _is_lidar_backed(src) and not _is_rgb_backed(src):
        score += 0.25
    # Préférer colis fiables LiDAR (souvent parcel_1 / 3)
    if _is_lidar_backed(src) and "grid" not in src:
        score -= 0.05
    return score


def select_nearest_parcel(parcels, log, exclude_names=None, skip_failures=None):
    """Choisit le colis le plus accessible (distance + fiabilité perception)."""
    exclude = set(exclude_names or ())
    failures = skip_failures or {}
    force = FORCE_PARCEL_NAME
    if force:
        log("[SELECT] mode FOCUS → seulement %s", force)
    candidates = []
    for p in parcels:
        name = p.get("name")
        if force and name != force:
            continue
        if name in exclude or failures.get(name, 0) >= 2:
            continue
        cx, cy, cz = p["center"]
        # Hors table → ne PAS saisir (logs: y=+0.04 → chute)
        # FOCUS: marge un peu plus large pour ne pas rater l'orange
        margin = 0.08 if force else 0.05
        if not (TABLE_X_RANGE[0] - margin <= cx <= TABLE_X_RANGE[1] + margin
                and TABLE_Y_RANGE[0] - margin <= cy <= TABLE_Y_RANGE[1] + margin):
            log("[SELECT] skip %s hors table (%.3f, %.3f)", name, cx, cy)
            continue
        src = p.get("source") or ""
        # FOCUS: accepter rgb-ray / couleur si LiDAR faible (priorité: trouver orange)
        if not force:
            if ("couleur-uv" in src or src == "couleur") and "lidar" not in src:
                if "rgb-depth" not in src and "rgb-ray" not in src:
                    log("[SELECT] skip %s source faible [%s]", name, src)
                    continue
        candidates.append(p)
    if not candidates:
        if force:
            log("[SELECT] FOCUS %s introuvable / hors table", force)
            for p in parcels:
                if p.get("name") == force:
                    cx, cy, cz = p["center"]
                    log("[SELECT]   (détecté mais filtré) center=(%.3f,%.3f,%.3f) [%s]",
                        cx, cy, cz, p.get("source", "?"))
        else:
            log("[SELECT] 无可用快递 (exclus=%s)", sorted(exclude) or "—")
        return None
    target = min(candidates, key=_parcel_select_score)
    dist = _parcel_xy_distance(target)
    score = _parcel_select_score(target)
    cx, cy, cz = target["center"]
    log("[SELECT] 目标 %s (%s) [%s] center=(%.3f, %.3f, %.3f) dist=%.3f score=%.3f",
        target["name"], target.get("color", "?"), target.get("source", "?"),
        cx, cy, cz, dist, score)
    return target


# =============================================================================
# BRAS — helpers IK et déplacement
# =============================================================================

def _with_ik_z(pos, z):
    """Garde x et y d'un point IK, remplace seulement z (utile pour monter/descendre)."""
    return [float(pos[0]), float(pos[1]), float(z)]


def _box_drop_ik(parcel_name):
    """Point de dépose dans le bac = position de base + offset selon le numéro de colis."""
    offset = BOX_DROP_OFFSET_BY_PARCEL.get(parcel_name, [0.0, 0.0, 0.0])
    return [
        BOX_DROP_BASE_IK[0] + offset[0],
        BOX_DROP_BASE_IK[1] + offset[1],
        BOX_DROP_BASE_IK[2] + offset[2],
    ]


def _current_right_xyz(arm):
    """Pose FK courante de la main droite (orga `_call_fk(...).right_pose`)."""
    q0 = arm._read_arm_joints_rad(timeout=2.0)
    if len(q0) != 14:
        return None
    fk = arm.call_fk(q0, timeout=5.0)
    if fk is None:
        return None
    return [float(v) for v in fk.right_pose.pos_xyz]


def _move_hand(arm, side, pos_xyz, quat_xyzw, log, label,
               constraint_mode=None, settle=None, pos_cost_weight=0.0):
    """
    Une seule main — orga `_move_hand` / `_call_one_hand_ik`.
    L'autre bras reste sur joints courants (FK lock).
    """
    modes = []
    if constraint_mode is not None:
        modes.append(constraint_mode)
    for m in (IK_MODE_POS_HARD_ORI_SOFT, None):
        if m not in modes:
            modes.append(m)
    ok, joints = False, []
    used_mode = constraint_mode
    for mode in modes:
        try:
            ok, joints = arm.solve_ik_one_hand(
                side, pos_xyz, quat_xyzw,
                constraint_mode=mode,
                pos_cost_weight=pos_cost_weight,
                major_iterations_limit=IK_MAJOR_ITERATIONS,
            )
        except rospy.exceptions.ROSException as exc:
            log("[MOVE] %s IK 服务不可用: %s", label, exc)
            return False
        used_mode = mode
        if ok:
            if mode != constraint_mode:
                log("[MOVE] %s IK OK via fallback mode=%s", label, mode)
            break
    if not ok:
        log("[MOVE] %s IK 求解失败 (mode=%s side=%s)", label, used_mode, side)
        return False
    arm.go_to_joints(joints)
    rospy.sleep(ARM_SETTLE_TIME if settle is None else settle)
    return True


def _solve_and_move(arm, left_xyz, right_xyz, log, label,
                    left_quat=None, right_quat=None,
                    constraint_mode=None, settle=None):
    """
    IK double bras — uniquement pour handoff / bac (les deux mains bougent).
    Pick / weigh / regrasp → `_move_hand("right")` (orga).
    """
    lq = GRASP_QUAT if left_quat is None else left_quat
    rq = GRASP_QUAT if right_quat is None else right_quat
    modes = []
    if constraint_mode is not None:
        modes.append(constraint_mode)
    for m in (IK_MODE_POS_HARD_ORI_SOFT, None):
        if m not in modes:
            modes.append(m)
    ok, joints = False, []
    used_mode = constraint_mode
    for mode in modes:
        try:
            ok, joints = arm.solve_ik(
                left_xyz, lq, right_xyz, rq,
                constraint_mode=mode,
                major_iterations_limit=IK_MAJOR_ITERATIONS,
            )
        except rospy.exceptions.ROSException as exc:
            log("[MOVE] %s IK 服务不可用: %s", label, exc)
            return False
        used_mode = mode
        if ok:
            if mode != constraint_mode:
                log("[MOVE] %s IK OK via fallback mode=%s", label, mode)
            break
    if not ok:
        log("[MOVE] %s IK 求解失败 (mode=%s)", label, used_mode)
        return False
    arm.go_to_joints(joints)
    rospy.sleep(ARM_SETTLE_TIME if settle is None else settle)
    return True


def _right_pick_yz_align_ik(current_right_xyz, right_pick_pre_ik):
    """Orga: aligner y/z d'abord, x prudent — évite de raser la table."""
    current_x = float(current_right_xyz[0])
    target_x = float(right_pick_pre_ik[0])
    safe_x = float(RIGHT_PICK_YZ_ALIGN_SAFE_IK_X)
    if current_x <= target_x:
        align_x = min(target_x, max(current_x, safe_x))
    else:
        align_x = max(target_x, min(current_x, safe_x))
    align_z = max(float(current_right_xyz[2]), float(right_pick_pre_ik[2]))
    return [align_x, float(right_pick_pre_ik[1]), align_z]


def _move_right_cartesian_to(arm, target_ik, quat, log, label,
                             n_points=None, seg_sleep=None, constraint_mode=None):
    """
    Droite seule, ligne cartésienne — orga `_move_right_cartesian_to`.
    Gauche = joints courants (jamais re-résolus).
    """
    n = int(n_points or CARTESIAN_LIFT_POINTS)
    dt = CARTESIAN_LIFT_SEG_SLEEP if seg_sleep is None else seg_sleep
    mode = RIGHT_GRASP_FINAL_IK_MODE if constraint_mode is None else constraint_mode
    start = _current_right_xyz(arm)
    if start is None:
        log("[MOVE] %s FK droite indisponible", label)
        return False
    end = [float(v) for v in target_ik]
    for i in range(1, n + 1):
        a = float(i) / float(n)
        pt = [start[j] + (end[j] - start[j]) * a for j in range(3)]
        if not _move_hand(
            arm, "right", pt, quat, log, "%s_%d/%d" % (label, i, n),
            constraint_mode=mode, settle=dt,
        ):
            if mode == RIGHT_GRASP_FINAL_IK_MODE:
                if not _move_hand(
                    arm, "right", pt, quat, log,
                    "%s_%d/%d_fb" % (label, i, n),
                    constraint_mode=RIGHT_PICK_IK_MODE, settle=dt,
                ):
                    return False
            else:
                return False
    return True


def _run_arm_raise_preset(arm, log):
    """Preset bras orga (move_home) — gauche reste figée ensuite jusqu'au handoff."""
    log("[ARM] preset orga (raise) — %d étapes", len(ARM_RAISE_PRESET_DEG))
    for i, deg in enumerate(ARM_RAISE_PRESET_DEG):
        arm.go_to_joints(list(deg))
        log("[ARM] preset %d/%d", i + 1, len(ARM_RAISE_PRESET_DEG))
        rospy.sleep(ARM_RAISE_STEP_SLEEP)
    return True


# =============================================================================
# Vérif saisie réelle — LiDAR/RGB (PAS /mujoco/qpos : anti-triche)
# =============================================================================

def _verify_parcel_lifted_perception(lidar, cam, tf_reader, parcel_name,
                                     pick_xy, log):
    """
    Indique si le colis a quitté la table (preuve soft).
    Ne guide jamais le bras. /mujoco/qpos interdit.
    """
    if not GRASP_VERIFY_ENABLED:
        return True
    if lidar is None or cam is None or tf_reader is None:
        log("[VERIFY] sensors absents — skip")
        return True
    rospy.sleep(0.3)
    try:
        # Bras en l'air → LiDAR table souvent vide : ne pas bloquer / faux EMPTY
        pts = None
        try:
            pts = lidar.get_points() if hasattr(lidar, "get_points") else None
        except Exception:
            pts = None
        parcels = detect_parcels(lidar, cam, tf_reader, log)
    except Exception as exc:
        log("[VERIFY] detect échoué: %s — skip HOLD", exc)
        return True
    if not parcels:
        log("[VERIFY] aucune détection après lift → HOLD")
        return True
    # Si toutes sources sont couleur-uv / faibles → skip (LiDAR occulté)
    weak = 0
    for p in parcels:
        src = p.get("source") or ""
        if "lidar" not in src and "rgb-depth" not in src:
            weak += 1
    if weak >= len(parcels):
        log("[VERIFY] perception faible post-lift (LiDAR occulté?) — skip HOLD")
        return True
    px, py = float(pick_xy[0]), float(pick_xy[1])
    for p in parcels:
        if p.get("name") != parcel_name:
            continue
        cx, cy, cz = p["center"]  # pas center_raw
        cx, cy, cz = float(cx), float(cy), float(cz)
        dxy = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        # Colis encore près du point de prise = MAIN VIDE (même si z foireux)
        still_near_pick = dxy <= float(GRASP_VERIFY_STILL_ON_TABLE_XY)
        on_table = still_near_pick and cz <= float(GRASP_VERIFY_TABLE_Z_MAX)
        # z haut mais toujours au pick XY = bousculé / pas saisi
        bumped_not_held = still_near_pick and cz < 0.40
        empty = on_table or bumped_not_held
        log(
            "[VERIFY] %s après lift det=(%.3f,%.3f,%.3f) dxy=%.3f → %s",
            parcel_name, cx, cy, cz, dxy,
            "EMPTY" if empty else "HOLD",
        )
        if empty and GRASP_VERIFY_ABORT_ON_EMPTY:
            return False
        return True
    log("[VERIFY] %s plus près du pick → HOLD", parcel_name)
    return True


def _move_left_keep_right(arm, left_xyz, left_quat, locked_right_deg, log, label,
                          constraint_mode=None, settle=None):
    """Gauche seule, joints droite figés (orga `_move_left_with_locked_right_joints`)."""
    modes = []
    if constraint_mode is not None:
        modes.append(constraint_mode)
    for m in (IK_MODE_THREE_POINT_MIXED, IK_MODE_POS_HARD_ORI_SOFT, None):
        if m not in modes:
            modes.append(m)
    ok, joints = False, []
    for mode in modes:
        try:
            ok, joints = arm.solve_ik_one_hand(
                "left", left_xyz, left_quat,
                constraint_mode=mode,
                major_iterations_limit=IK_MAJOR_ITERATIONS,
            )
        except rospy.exceptions.ROSException as exc:
            log("[MOVE] %s IK fail: %s", label, exc)
            return False
        if ok:
            break
    if not ok:
        log("[MOVE] %s IK gauche échoué", label)
        return False
    full = list(joints[0:7]) + [float(v) for v in locked_right_deg]
    arm.go_to_joints(full)
    rospy.sleep(ARM_SETTLE_TIME if settle is None else settle)
    return True


# =============================================================================
# ACTION — saisie main droite sur la table
# =============================================================================

def grasp_parcel_right(arm, claw, parcel, log, cam=None, tf_reader=None,
                       lidar=None):
    """
    Saisie droite seule :
      LiDAR/tête = zone → pre → caméra poignet affine XY → descente → lift.
    """
    cx, cy, cz = _action_center_xyz(parcel)
    name = parcel["name"]
    if not (TABLE_X_RANGE[0] - 0.05 <= cx <= TABLE_X_RANGE[1] + 0.05
            and TABLE_Y_RANGE[0] - 0.05 <= cy <= TABLE_Y_RANGE[1] + 0.05):
        log("[GRASP] ABORT %s hors table det=(%.3f,%.3f)", name, cx, cy)
        return False
    ox, oy, oz = _right_pick_offset(name, cy)
    tip = RIGHT_CLAW_TIP_OFFSET
    pick_x = cx + ox + float(tip[0])
    pick_y = cy + oy + float(tip[1])
    pick_z = RIGHT_PICK_IK_Z + oz + float(tip[2])
    right_pre = [pick_x, pick_y, float(RIGHT_PICK_TRANSIT_IK_Z)]
    right_grasp = [pick_x, pick_y, pick_z]
    rq = RIGHT_PICK_QUAT  # orga fixe — pas de rotation yaw (dérègle le bras)
    log("[GRASP] zone %s (%s) det=(%.3f,%.3f) tip→(%.3f,%.3f,%.3f) tip_off=%s",
        name, parcel.get("color", "?"), cx, cy, pick_x, pick_y, pick_z, tip)

    claw.open()
    claw.right_open()
    rospy.sleep(GRIPPER_SETTLE_TIME)

    cur = _current_right_xyz(arm)
    if cur is None:
        log("[GRASP] FK droite indisponible")
        return False
    yz_align = _right_pick_yz_align_ik(cur, right_pre)
    if not _move_hand(
        arm, "right", yz_align, rq, log, "right_yz_align",
        constraint_mode=IK_MODE_THREE_POINT_MIXED, settle=PICK_ALIGN_MOVE_SLEEP,
    ):
        return False

    if not _move_hand(
        arm, "right", right_pre, rq, log, "right_x_to_pick_pre",
        constraint_mode=RIGHT_PICK_IK_MODE, settle=PICK_ALIGN_MOVE_SLEEP,
    ):
        return False

    # =========================================================================
    # VISION MAIN (essentiel) — tête = zone ; main peaufine en boucle
    # Pince hold check assoupli (WRIST_SKIP_CLAW_HOLD_CHECK) — focus vision
    # =========================================================================
    clear_z = float(max(RIGHT_PICK_TRANSIT_IK_Z, 0.22))
    shallow = float(globals().get("WRIST_SHALLOW_PLUNGE", 0.045) or 0.045)
    approach_z = float(pick_z) + max(shallow, 0.04)
    right_pre = [pick_x, pick_y, clear_z]
    right_approach = [pick_x, pick_y, approach_z]
    right_grasp = [pick_x, pick_y, pick_z]
    working = dict(parcel)

    claw.right_open()
    if not _move_right_cartesian_to(
        arm, right_approach, rq, log, "hand_approach",
        n_points=max(4, int(CARTESIAN_LIFT_POINTS)),
        constraint_mode=RIGHT_PICK_IK_MODE,
    ):
        if not _move_hand(
            arm, "right", right_approach, rq, log, "hand_approach_fb",
            constraint_mode=RIGHT_PICK_IK_MODE, settle=PICK_GRASP_MOVE_SLEEP,
        ):
            return False

    vision_ok = False
    if cam is not None and tf_reader is not None and not parcel.get("skip_wrist"):
        n_iters = max(1, int(globals().get("WRIST_SERVO_ITERS", 4) or 4))
        step_max = float(globals().get("WRIST_MAX_DELTA_XY", 0.02) or 0.02)
        for wi in range(n_iters):
            obs = observe_hand(cam, working, hand="right", log=log)
            if obs["seen"] and obs["centered"]:
                vision_ok = True
                log("[GRASP] VISION OK %d/%d frac=%.2f area=%d",
                    wi + 1, n_iters, obs["frac"], obs["area"])
                break
            if not obs["seen"]:
                log("[GRASP] VISION %d/%d — pas vu, garde tête", wi + 1, n_iters)
                continue
            refined = refine_target_with_wrist(
                cam, tf_reader, working, hand="right", log=log,
                max_delta_xy=step_max)
            if not refined.get("wrist_refined"):
                continue
            dxy = float(refined.get("wrist_delta_xy", 0.0))
            working = refined
            if dxy < 0.003 and refined.get("wrist_centered"):
                vision_ok = True
                break
            rcx, rcy, _rcz = _action_center_xyz(refined)
            ox, oy, oz = _right_pick_offset(name, rcy)
            tip = RIGHT_CLAW_TIP_OFFSET
            pick_x = rcx + ox + float(tip[0])
            pick_y = rcy + oy + float(tip[1])
            pick_z = RIGHT_PICK_IK_Z + oz + float(tip[2])
            approach_z = float(pick_z) + max(shallow, 0.04)
            right_approach = [pick_x, pick_y, approach_z]
            right_grasp = [pick_x, pick_y, pick_z]
            right_pre = [pick_x, pick_y, clear_z]
            log("[GRASP] VISION peaufine %d/%d Δ=%.1fcm → (%.3f,%.3f)",
                wi + 1, n_iters, 100.0 * dxy, pick_x, pick_y)
            _move_hand(
                arm, "right", right_approach, rq, log, "hand_servo_%d" % (wi + 1),
                constraint_mode=RIGHT_PICK_IK_MODE, settle=0.50,
            )

        if not vision_ok:
            obs = observe_hand(cam, working, hand="right", log=log)
            vision_ok = bool(obs.get("seen") and obs.get("centered"))

        if WRIST_REQUIRE_SEE_BEFORE_CLOSE and not vision_ok:
            log("[GRASP] VISION FAIL — pas centré, remonte OUVERT")
            _move_right_cartesian_to(
                arm, right_pre, rq, log, "vision_abort_raise",
                n_points=3, constraint_mode=RIGHT_PICK_IK_MODE,
            )
            claw.right_open()
            return False
        log("[GRASP] VISION LOCK — descente + fermer")

    plunged = _move_right_cartesian_to(
        arm, right_grasp, rq, log, "hand_shallow_grasp",
        n_points=2, constraint_mode=RIGHT_GRASP_FINAL_IK_MODE,
    )
    if not plunged:
        plunged = _move_hand(
            arm, "right", right_grasp, rq, log, "hand_grasp_fb",
            constraint_mode=RIGHT_PICK_IK_MODE, settle=0.6,
        )
    if not plunged:
        if globals().get("WRIST_CLOSE_EVEN_IF_IK_FAIL", True) and vision_ok:
            log("[GRASP] IK fail — CLOSE (vision OK)")
        else:
            return False

    log("[GRASP] tip CLOSE (vision confirmée)")
    claw.right_close()
    if not claw.wait_until_done(timeout=3.0):
        log("[GRASP] 夹爪动作超时")
    rospy.sleep(GRIPPER_CLOSE_HOLD)
    claw.right_close()
    claw.wait_until_done(timeout=2.5)
    for _ in range(25):
        if int(getattr(claw, "_right_state", -1)) != 1:
            break
        rospy.sleep(0.08)
    rospy.sleep(0.25)
    log("[GRASP] claw state: %s", claw.describe_right())

    skip_claw = bool(globals().get("WRIST_SKIP_CLAW_HOLD_CHECK", False))
    if (not skip_claw) and (not claw.right_holding()):
        log("[GRASP] PINCE VIDE — ouvre, PAS de pesée")
        claw.right_open()
        claw.wait_until_done(timeout=2.0)
        _move_right_cartesian_to(
            arm, right_pre, rq, log, "claw_empty_raise",
            n_points=3, constraint_mode=RIGHT_PICK_IK_MODE,
        )
        return False
    if skip_claw:
        log("[GRASP] claw-hold skip (focus VISION)")

    proof_z = float(pick_z) + 0.08
    proof_pt = [pick_x, pick_y, proof_z]
    if not _move_right_cartesian_to(
        arm, proof_pt, rq, log, "tip_proof_lift",
        n_points=2, constraint_mode=RIGHT_PICK_IK_MODE,
    ):
        _move_hand(
            arm, "right", proof_pt, rq, log, "tip_proof_fb",
            constraint_mode=RIGHT_PICK_IK_MODE, settle=0.4,
        )
    rospy.sleep(0.35)

    if cam is not None:
        obs_lift = observe_hand(cam, working, hand="right", log=log, settle=0.15)
        log("[GRASP] VISION post-lift seen=%s frac=%.2f",
            obs_lift.get("seen"), obs_lift.get("frac", 1.0))

    if not skip_claw:
        if not _verify_parcel_lifted_perception(
            lidar, cam, tf_reader, name, (pick_x, pick_y), log
        ):
            log("[GRASP] %s EMPTY après close — PAS de pesée", name)
            claw.right_open()
            claw.wait_until_done(timeout=2.0)
            _move_right_cartesian_to(
                arm, right_pre, rq, log, "empty_raise_open",
                n_points=3, constraint_mode=RIGHT_PICK_IK_MODE,
            )
            return False
    else:
        _verify_parcel_lifted_perception(
            lidar, cam, tf_reader, name, (pick_x, pick_y), log
        )

    carry_quat = RIGHT_WEIGH_RELEASE_QUAT
    right_lift = [pick_x, pick_y, float(WEIGH_TRANSIT_Z)]
    if not _move_right_cartesian_to(
        arm, right_lift, carry_quat, log, "lift_straight_up",
        n_points=CARTESIAN_LIFT_POINTS,
        constraint_mode=RIGHT_GRASP_FINAL_IK_MODE,
    ):
        if not _move_hand(
            arm, "right", right_lift, carry_quat, log, "lift_fallback",
            constraint_mode=RIGHT_PICK_IK_MODE,
        ):
            claw.right_open()
            return False

    claw.wait_until_done(timeout=2.0)
    rs = int(getattr(claw, "_right_state", -1))
    log("[GRASP] %s claw R=%d", name, rs)
    if not skip_claw:
        if not _verify_parcel_lifted_perception(
            lidar, cam, tf_reader, name, (pick_x, pick_y), log
        ):
            log("[GRASP] %s EMPTY au transport — abandon", name)
            claw.right_open()
            claw.wait_until_done(timeout=2.0)
            return False
    log("[GRASP] %s 抓取成功 (vision-first)", name)
    return True


# =============================================================================
# ACTION — balance, reprise, passation, bac
# =============================================================================

def place_on_weighing_area(arm, claw, log):
    """
    Pesée orga `_right_weigh_and_regrasp` (partie release) :
      cartésien → pre (z transit) → descente droite seule → ouvrir.
    """
    log("[WEIGH] 搬运到称重区 (orga right-only)")
    rq = RIGHT_WEIGH_RELEASE_QUAT
    carry_quat = RIGHT_WEIGH_RELEASE_QUAT
    weigh = list(WEIGH_RELEASE_IK)
    right_transit = _with_ik_z(weigh, WEIGH_TRANSIT_Z)
    if not _move_right_cartesian_to(
        arm, right_transit, carry_quat, log, "right_weigh_transit",
        n_points=6, constraint_mode=RIGHT_GRASP_FINAL_IK_MODE,
    ):
        if not _move_hand(
            arm, "right", right_transit, carry_quat, log, "right_weigh_transit_fb",
            constraint_mode=RIGHT_PICK_IK_MODE, settle=PICK_ALIGN_MOVE_SLEEP,
        ):
            return False

    if not _move_hand(
        arm, "right", weigh, rq, log, "right_weigh_down",
        constraint_mode=RIGHT_PICK_IK_MODE, settle=PICK_GRASP_MOVE_SLEEP,
    ):
        return False

    rospy.sleep(WEIGH_RELEASE_SETTLE)
    claw.right_open()
    claw.wait_until_done(timeout=3.0)
    rospy.sleep(GRIPPER_SETTLE_TIME)
    rospy.sleep(WEIGH_DWELL)
    log("[WEIGH] 已释放，等待称重")
    return True


def regrasp_from_weighing(arm, claw, log):
    """
    Reprise orga : pre à z=WEIGH_RELEASE_IK[2], descente one-shot à WEIGH_REGRASP_IK,
    close, remonter aux joints du pre (pas de clamp z / cartésien modifié).
    """
    log("[WEIGH] 二次抓取 (orga right-only)")
    rq = RIGHT_WEIGH_REGRASP_QUAT
    regrasp = list(WEIGH_REGRASP_IK)
    regrasp_pre = _with_ik_z(regrasp, float(WEIGH_RELEASE_IK[2]))

    if not _move_hand(
        arm, "right", regrasp_pre, rq, log, "right_regrasp_xy_ori_align",
        constraint_mode=RIGHT_PICK_IK_MODE, settle=0.0,
    ):
        return False
    # Mémoriser joints pre pour remonter sans re-IK (orga)
    pre_joints = list(arm._last_cmd_deg)

    if not _move_hand(
        arm, "right", regrasp, rq, log, "right_regrasp_from_weigh",
        constraint_mode=RIGHT_PICK_IK_MODE, settle=PICK_GRASP_MOVE_SLEEP,
    ):
        return False

    claw.right_close()
    claw.wait_until_done(timeout=3.0)
    rospy.sleep(GRIPPER_CLOSE_HOLD)
    # Orga: pas de check is_grabbed ici — sim ne publie souvent que Moving/Reached
    claw.wait_until_done(timeout=2.0)
    rs = int(getattr(claw, "_right_state", -1))
    log("[WEIGH] 二次抓取 claw R=%d (sim OK si 1/2)", rs)

    # Orga: rejouer joints du pre (évite IK fail sous charge)
    arm.go_to_joints(pre_joints)
    rospy.sleep(ARM_SETTLE_TIME)
    log("[WEIGH] 二次抓取成功")
    return True


def handoff_to_left(arm, claw, log):
    """
    Passation orga :
      droite raise → xy align → handoff IK (lock joints droite)
      gauche XZ puis Y avec droite figée → close L / open R → retract.
    """
    log("[HANDOFF] 右手交给左手 (orga lock-right)")
    claw.left_open()
    rospy.sleep(0.2)
    rq = RIGHT_HANDOFF_QUAT
    right_final = list(RIGHT_HANDOFF_IK)

    transit_zs = [float(RIGHT_HANDOFF_TRANSIT_Z)] + [
        float(z) for z in RIGHT_HANDOFF_TRANSIT_FALLBACK_ZS]
    moved = False
    for tz in transit_zs:
        cur = _current_right_xyz(arm)
        if cur is None:
            log("[HANDOFF] FK droite indisponible")
            return False
        # 1) lever à z transit en gardant xy courant (orga)
        raise_pt = [cur[0], cur[1], float(tz)]
        if not _move_hand(
            arm, "right", raise_pt, rq, log, "right_handoff_raise_z%.2f" % tz,
            constraint_mode=IK_MODE_THREE_POINT_MIXED, settle=PICK_ALIGN_MOVE_SLEEP,
        ):
            log("[HANDOFF] raise z=%.2f fail → retry", tz)
            continue
        # 2) aligner xy+ori à hauteur transit
        align_pt = [right_final[0], right_final[1], float(tz)]
        if not _move_hand(
            arm, "right", align_pt, rq, log, "right_handoff_xy_z%.2f" % tz,
            constraint_mode=IK_MODE_THREE_POINT_MIXED, settle=PICK_ALIGN_MOVE_SLEEP,
        ):
            log("[HANDOFF] xy align z=%.2f fail → retry", tz)
            continue
        moved = True
        break
    if not moved:
        return False

    if not _move_hand(
        arm, "right", right_final, rq, log, "right_handoff_to_left",
        constraint_mode=IK_MODE_THREE_POINT_MIXED, settle=PICK_ALIGN_MOVE_SLEEP,
    ):
        return False

    # Verrouiller les 7 joints droite du solve handoff (orga : pas le sag FK)
    locked_right = list(arm._last_cmd_deg[7:14])
    log("[HANDOFF] droite verrouillée — approche gauche")

    left_xz = list(LEFT_HANDOFF_RECEIVE_XZ_READY_IK)
    left_recv = list(LEFT_HANDOFF_RECEIVE_IK)
    llq = LEFT_HANDOFF_RECEIVE_QUAT
    left_ok = False
    # Fallbacks si bras déjà déplacé (2e colis) — IK gauche hors workspace
    left_offsets = [
        (0.0, 0.0, 0.0),
        (0.0, 0.03, 0.03),
        (0.02, 0.05, 0.05),
        (-0.02, 0.04, 0.02),
        (0.0, 0.08, 0.08),
    ]
    for dx, dy, dz in left_offsets:
        xz = [left_xz[0] + dx, left_xz[1] + dy, left_xz[2] + dz]
        recv = [left_recv[0] + dx, left_recv[1] + dy, left_recv[2] + dz]
        if dx or dy or dz:
            log("[HANDOFF] left retry dxyz=(%.2f,%.2f,%.2f)", dx, dy, dz)
        if not _move_left_keep_right(
            arm, xz, llq, locked_right, log, "left_receive_xz",
            constraint_mode=IK_MODE_THREE_POINT_MIXED, settle=PICK_ALIGN_MOVE_SLEEP,
        ):
            continue
        if not _move_left_keep_right(
            arm, recv, llq, locked_right, log, "left_receive_y",
            constraint_mode=IK_MODE_THREE_POINT_MIXED, settle=PICK_ALIGN_MOVE_SLEEP,
        ):
            continue
        left_ok = True
        break
    if not left_ok:
        log("[HANDOFF] approche gauche IK fail")
        return False

    claw.left_close()
    claw.wait_until_done(timeout=3.0)
    rospy.sleep(GRIPPER_CLOSE_HOLD)
    ls = int(getattr(claw, "_left_state", -1))
    log("[HANDOFF] left claw L=%d", ls)

    claw.right_open()
    claw.wait_until_done(timeout=3.0)
    rospy.sleep(GRIPPER_CLOSE_HOLD)

    # Retract : FK courant + Δy (orga — pas forcer z théorique)
    cur = _current_right_xyz(arm)
    if cur is None:
        right_retract = [
            right_final[0],
            right_final[1] + RIGHT_HANDOFF_RETRACT_Y,
            right_final[2],
        ]
    else:
        right_retract = [
            cur[0], cur[1] + RIGHT_HANDOFF_RETRACT_Y, cur[2],
        ]
    if not _move_hand(
        arm, "right", right_retract, rq, log, "right_retract",
        constraint_mode=IK_MODE_THREE_POINT_MIXED, settle=PICK_ALIGN_MOVE_SLEEP,
    ):
        return False

    log("[HANDOFF] 交接完成")
    return True


def place_in_box(arm, claw, parcel_name, log):
    """
    Lâcher au-dessus du bac — NE PAS descendre / toucher la box
    (sinon elle bascule). Main gauche au hover, ouvrir, laisser tomber.
    """
    log("[BOX] drop-from-above %s (hover z=%.2f, no contact)",
        parcel_name, BOX_DROP_HOVER_Z)
    base = _box_drop_ik(parcel_name)
    hover_z = float(BOX_DROP_HOVER_Z)
    # Priorité: centre bac, puis petits offsets (viser le trou, pas les bords)
    x_tries = [0.0, -0.03, 0.03, -0.06]
    y_tries = [0.0, 0.03, -0.03, 0.06]
    llq = LEFT_BOX_DROP_QUAT

    locked_right = list(arm._last_cmd_deg[7:14]) if len(getattr(arm, "_last_cmd_deg", [])) >= 14 else None
    if locked_right is None:
        log("[BOX] pas de joints droite — fallback dual IK hover")
        return _place_in_box_dual(arm, claw, parcel_name, log)

    for dx in x_tries:
        for dy in y_tries:
            hover_ik = [base[0] + dx, base[1] + dy, hover_z]
            log("[BOX] hover try xyz=(%.3f, %.3f, %.3f)", hover_ik[0], hover_ik[1], hover_ik[2])
            if not _move_left_keep_right(
                arm, hover_ik, llq, locked_right, log, "box_hover",
                constraint_mode=IK_MODE_THREE_POINT_MIXED,
                settle=PICK_ALIGN_MOVE_SLEEP,
            ):
                continue
            claw.left_open()
            claw.wait_until_done(timeout=3.0)
            rospy.sleep(GRIPPER_SETTLE_TIME)
            rospy.sleep(PLACE_DWELL)
            log("[BOX] 已释放 (hover drop @ %.2f,%.2f,%.2f — no box contact)",
                hover_ik[0], hover_ik[1], hover_ik[2])
            return True

    log("[BOX] hover lock-right fail → dual fallback")
    return _place_in_box_dual(arm, claw, parcel_name, log)


def _place_in_box_dual(arm, claw, parcel_name, log):
    """Dual-arm hover drop (fallback) — toujours au-dessus, pas dans le bac."""
    base = _box_drop_ik(parcel_name)
    hover_z = float(BOX_DROP_HOVER_Z)
    x_tries = [0.0] + [float(d) for d in BOX_DROP_IK_X_FALLBACK_DELTAS]
    llq = LEFT_BOX_DROP_QUAT
    rq = RIGHT_HANDOFF_QUAT
    right_park = list(RIGHT_HANDOFF_IK)
    for dx in x_tries:
        hover_ik = [base[0] + dx, base[1], hover_z]
        if dx != 0.0:
            log("[BOX] dual hover x+=%.3f", dx)
        if not _solve_and_move(
            arm, hover_ik, right_park, log, "箱上方投放",
            left_quat=llq, right_quat=rq,
            constraint_mode=IK_MODE_POS_HARD_ORI_SOFT, settle=PICK_ALIGN_MOVE_SLEEP,
        ):
            continue
        claw.left_open()
        claw.wait_until_done(timeout=3.0)
        rospy.sleep(GRIPPER_SETTLE_TIME)
        rospy.sleep(PLACE_DWELL)
        log("[BOX] 已释放 (dual hover)")
        return True
    log("[BOX] 放箱 IK 全部失败")
    return False


def process_parcel_after_grasp(arm, claw, parcel, log):
    """Après saisie : train = handoff→bac ; sinon pesée→reprise→handoff→bac."""
    name = parcel["name"]
    if SKIP_WEIGH or TRAIN_PICK_BOX:
        log("[TRAIN] skip weigh — handoff → bac (%s)", name)
        if not handoff_to_left(arm, claw, log):
            return False
        if not place_in_box(arm, claw, name, log):
            return False
        log("[PARCEL] %s train pick→box OK", name)
        return True
    if not place_on_weighing_area(arm, claw, log):
        return False
    if not regrasp_from_weighing(arm, claw, log):
        return False
    if not handoff_to_left(arm, claw, log):
        return False
    if not place_in_box(arm, claw, name, log):
        return False
    log("[PARCEL] %s 全流程完成", name)
    return True


def approach_and_grasp(robot, arm, claw, target, head, lidar, cam, tf_reader, log):
    """
    Approche du colis :
      - pas de marche (risque chute)
      - levée bras safe + refine cam_r + saisie
    """
    robot.stop()
    robot.switch_to_stance()
    rospy.sleep(0.3)
    # stance peut faire perdre le mode bras externe → re-arm avant IK
    try:
        arm.switch_to_external_control()
        rospy.sleep(0.2)
    except Exception as exc:
        log("[ARM] external control skip: %s", exc)
    return grasp_parcel_right(
        arm, claw, target, log, cam=cam, tf_reader=tf_reader, lidar=lidar)


def run_scene1_mission(robot, arm, claw, head, log):
    """Mission complète : saisie → balance → handoff → bac (4 colis)."""
    log("=" * 50)
    if TRAIN_PICK_BOX or SKIP_WEIGH:
        log("场景一 TRAIN：prise → lift → bac (pas de pesée)")
    else:
        log("场景一：快递称重与摆放 — 任务开始")
    if FORCE_PARCEL_NAME:
        log("FOCUS MODE: seulement %s — 1 colis", FORCE_PARCEL_NAME)
    log("=" * 50)

    log("[STEP 1] 切换手臂到外部控制模式")
    for attempt in range(1, 6):
        try:
            arm.switch_to_external_control()
            log("[STEP 1] external control OK (try %d)", attempt)
            break
        except Exception as exc:
            log("[STEP 1] external control fail try %d/5: %s", attempt, exc)
            if attempt >= 5:
                log("[STEP 1] abort — service bras indisponible")
                return
            rospy.sleep(3.0 * attempt)
    rospy.sleep(0.5)
    log("[STEP 1b] 等待 IK 服务就绪")
    try:
        rospy.wait_for_service("/ik/two_arm_hand_pose_cmd_srv", timeout=60.0)
        rospy.wait_for_service("/ik/fk_srv", timeout=30.0)
        log("[STEP 1b] IK + FK 服务就绪")
    except rospy.exceptions.ROSException:
        log("[STEP 1b] IK/FK 服务超时 — 任务中止")
        return
    arm.go_home()
    rospy.sleep(0.5)
    # Preset orga (move_home) : ensuite pick = droite seule, gauche inchangée
    _run_arm_raise_preset(arm, log)

    log_scene_landmarks(log)

    lidar = LidarReader()
    cam = CameraReader()
    tf_reader = TFReader()
    rospy.sleep(0.5)

    completed = 0
    failed = 0
    completed_names = set()
    grasp_failures = {}

    while completed < MAX_PARCELS:
        if failed >= MAX_MISSION_FAILURES:
            log("[LOOP] trop d'échecs (%d) — arrêt anti-blocage", failed)
            break
        log("[LOOP] 第 %d/%d 个快递 (失败 %d)", completed + 1, MAX_PARCELS, failed)

        log("[STEP 2] 低头观察桌面 (tête orga, fixe)")
        head.look_at(HEAD_LOOK_YAW, HEAD_LOOK_PITCH)
        rospy.sleep(HEAD_SETTLE_SEC)

        log("[STEP 3] LiDAR + RGB 融合检测")
        target = None
        parcels = []
        for det_try in range(1, 4):
            parcels = detect_parcels(lidar, cam, tf_reader, log)
            parcels = [p for p in parcels if p.get("name") not in completed_names]
            if FORCE_PARCEL_NAME:
                focused = [p for p in parcels if p.get("name") == FORCE_PARCEL_NAME]
                log("[STEP 3] FOCUS try %d/3 : %s parmi %d détectés",
                    det_try, "OK" if focused else "MANQUE", len(parcels))
                for p in parcels:
                    if p.get("name") == FORCE_PARCEL_NAME:
                        cx, cy, cz = p["center"]
                        log("[STEP 3]   %s det=(%.3f,%.3f,%.3f) [%s]",
                            FORCE_PARCEL_NAME, cx, cy, cz, p.get("source", "?"))
            if len(parcels) < max(1, MAX_PARCELS - completed):
                log("[STEP 3] 检测数量偏少，补充深度图检查")
                inspect_table_depth(cam, log)
            if not parcels:
                log("[LOOP] 未检测到剩余快递 (try %d)", det_try)
                rospy.sleep(0.5)
                continue
            target = select_nearest_parcel(
                parcels, log, exclude_names=completed_names, skip_failures=grasp_failures)
            if target is not None:
                break
            log("[STEP 3] pas de cible FOCUS — re-detect")
            rospy.sleep(0.6)

        if target is None:
            log("[LOOP] 无可用目标，提前结束")
            break
        log("[STEP 4] 目标: %s", target["name"])

        log("[STEP 5] 抓取")
        grasped = approach_and_grasp(
            robot, arm, claw, target, head, lidar, cam, tf_reader, log)
        if not grasped:
            failed += 1
            name = target["name"]
            grasp_failures[name] = grasp_failures.get(name, 0) + 1
            log("[LOOP] 抓取失败 %s (%d/2)，跳过本轮",
                name, grasp_failures[name])
            # Remonter droite seule (gauche reste au preset orga)
            _move_hand(
                arm, "right", [0.35, -0.20, ARM_CLEAR_TABLE_Z],
                RIGHT_PICK_QUAT, log, "grasp_fail_raise",
                constraint_mode=RIGHT_PICK_IK_MODE, settle=0.5,
            )
            continue

        log("[STEP 6] %s",
            "handoff → bac (TRAIN)" if (SKIP_WEIGH or TRAIN_PICK_BOX)
            else "称重 → 二次抓取 → 交接 → 入箱")
        if process_parcel_after_grasp(arm, claw, target, log):
            completed += 1
            completed_names.add(target["name"])
            log("[LOOP] 成功 %d/%d", completed, MAX_PARCELS)
        else:
            failed += 1
            log("[LOOP] 后处理失败")

        if completed < MAX_PARCELS:
            # go_home entre colis → états instables / detect hang (vu 2× au 2/4)
            log("[LOOP] pause courte avant prochain colis (skip go_home)")
            try:
                arm.switch_to_external_control()
            except Exception as exc:
                log("[LOOP] re-external skip: %s", exc)
            rospy.sleep(0.8)
        else:
            log("[LOOP] dernier colis — fin")

    log("[DONE] 完成 %d/%d 个快递，失败 %d 次", completed, MAX_PARCELS, failed)
    log("场景一：任务结束")
