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
import time

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
    assess_grasp_manner,
    is_excellent_grasp,
    log_wrist_event,
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
    Mission grasp : `center`, sauf si reshape grille/row a bougé fort → center_raw.
    """
    if prefer_raw:
        raw = target.get("center_raw")
        if raw is not None:
            return [float(raw[0]), float(raw[1]), float(raw[2])]
    cx, cy, cz = target["center"]
    cx, cy, cz = float(cx), float(cy), float(cz)
    raw = target.get("center_raw")
    src = target.get("source") or ""
    reshaped = any(k in src for k in ("grid-x", "grid-y", "row-lift", "row-infer"))
    if raw is not None and reshaped:
        rx, ry, rz = float(raw[0]), float(raw[1]), float(raw[2])
        max_d = float(globals().get("FUSE_MAX_RESHAPE_XY", 0.04) or 0.04)
        if abs(rx - cx) > max_d or abs(ry - cy) > max_d:
            return [rx, ry, rz if abs(rz) < 0.2 else cz]
    return [cx, cy, cz]


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
    elif "grid-2x2" in src or "row-lift" in src:
        score += 0.40
    elif "grid-x" in src or "grid-y" in src:
        score += 0.35
    elif not _is_lidar_backed(src) and not _is_rgb_backed(src):
        score += 0.25
    # Préférer colis fiables LiDAR (souvent parcel_1 / 3)
    if _is_lidar_backed(src) and "grid" not in src and "row" not in src:
        score -= 0.08
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
        max_fail = int(globals().get("GRASP_PARCEL_MAX_FAILS", 5) or 5)
        if force:
            max_fail = max(max_fail, 5)
        if name in exclude or failures.get(name, 0) >= max_fail:
            continue
        cx, cy, cz = p["center"]
        # Hors table → ne PAS saisir (logs: y=+0.04 → chute)
        # FOCUS: marge un peu plus large pour ne pas rater l'orange
        margin = 0.10 if force else 0.05
        if not (TABLE_X_RANGE[0] - margin <= cx <= TABLE_X_RANGE[1] + margin
                and TABLE_Y_RANGE[0] - margin <= cy <= TABLE_Y_RANGE[1] + margin):
            log("[SELECT] skip %s hors table (%.3f, %.3f)", name, cx, cy)
            continue
        # parcel_1 FOCUS : rejeter seulement les merges absurdes (x≈0.53 / y≈0)
        if force == "parcel_1":
            if cx > 0.52:
                log("[SELECT] skip parcel_1 x=%.3f (colonne absurde)", cx)
                continue
            if cy > -0.05:
                log("[SELECT] skip parcel_1 y=%.3f (trop près robot)", cy)
                continue
        # z flottant (bras/fantôme) → clamp table pour la saisie
        if cz > 0.10 or cz < -0.12:
            log("[SELECT] %s z=%.3f aberrant → clamp table", name, cz)
            p = dict(p)
            p["center"] = (cx, cy, float(TABLE_PARCEL_Z))
            p["center_raw"] = p.get("center_raw") or (cx, cy, float(TABLE_PARCEL_Z))
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
            log("[SELECT] FOCUS %s introuvable / hors table / trop d'échecs", force)
            for p in parcels:
                if p.get("name") == force:
                    cx, cy, cz = p["center"]
                    log("[SELECT]   (détecté mais filtré) center=(%.3f,%.3f,%.3f) [%s] fails=%d",
                        cx, cy, cz, p.get("source", "?"), failures.get(force, 0))
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


_FIRST_IK_DEBUG_DONE = False


def _log_pose_snapshot(arm, log, label):
    """Log joints commandés + FK mains (repère IK base_link)."""
    deg = list(getattr(arm, "_last_cmd_deg", []) or [])
    if len(deg) == 14:
        log("[DEBUG_POSE] %s joints_deg L=[%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f]",
            label, deg[0], deg[1], deg[2], deg[3], deg[4], deg[5], deg[6])
        log("[DEBUG_POSE] %s joints_deg R=[%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f]",
            label, deg[7], deg[8], deg[9], deg[10], deg[11], deg[12], deg[13])
    try:
        q0 = arm._read_arm_joints_rad(timeout=2.0)
    except Exception as exc:
        log("[DEBUG_POSE] %s sensors FK skip: %s", label, exc)
        return
    if q0 is None or len(q0) != 14:
        log("[DEBUG_POSE] %s sensors joints indisponibles", label)
        return
    try:
        fk = arm.call_fk(q0, timeout=5.0)
    except Exception as exc:
        log("[DEBUG_POSE] %s FK skip: %s", label, exc)
        return
    if fk is None:
        return
    lx, ly, lz = [float(v) for v in fk.left_pose.pos_xyz]
    rx, ry, rz = [float(v) for v in fk.right_pose.pos_xyz]
    log("[DEBUG_POSE] %s FK left=(%.4f,%.4f,%.4f) z=%.4f",
        label, lx, ly, lz, lz)
    log("[DEBUG_POSE] %s FK right=(%.4f,%.4f,%.4f) z=%.4f",
        label, rx, ry, rz, rz)


def _maybe_stop_after_first_ik(arm, side, target, quat, joints_deg, log, label):
    global _FIRST_IK_DEBUG_DONE
    stop_label = str(globals().get("DEBUG_STOP_IK_LABEL", "") or "").strip()
    if stop_label:
        if label != stop_label:
            return
    elif not bool(globals().get("DEBUG_STOP_AFTER_FIRST_IK", False)):
        return
    if _FIRST_IK_DEBUG_DONE:
        return
    _FIRST_IK_DEBUG_DONE = True
    tx, ty, tz = [float(v) for v in target]
    log("[DEBUG_IK] ========== ARRET DEBUG (%s) — avant descente ==========",
        label)
    log("[DEBUG_IK] label=%s side=%s target=(%.4f,%.4f,%.4f) z=%.4f",
        label, side, tx, ty, tz, tz)
    log("[DEBUG_IK] quat=(%.4f,%.4f,%.4f,%.4f)",
        float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    if side == "right" and len(joints_deg) == 14:
        log("[DEBUG_IK] solved_deg R=[%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f]",
            joints_deg[7], joints_deg[8], joints_deg[9],
            joints_deg[10], joints_deg[11], joints_deg[12], joints_deg[13])
        log("[DEBUG_IK] POIGNET R yaw(r5)=%.2f° pitch(r6)=%.2f° roll(r7)=%.2f°",
            joints_deg[11], joints_deg[12], joints_deg[13])
    elif side == "left" and len(joints_deg) == 14:
        log("[DEBUG_IK] solved_deg L=[%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f]",
            joints_deg[0], joints_deg[1], joints_deg[2],
            joints_deg[3], joints_deg[4], joints_deg[5], joints_deg[6])
    _log_pose_snapshot(arm, log, "after_%s" % label)
    log("[DEBUG_IK] REF hauteurs: TRANSIT=%.3f PICK_Z=%.3f TABLE_Z=%.3f WEIGH=%.3f",
        float(RIGHT_PICK_TRANSIT_IK_Z), float(RIGHT_PICK_IK_Z),
        float(TABLE_PARCEL_Z), float(WEIGH_TRANSIT_Z))
    log("[DEBUG_IK] ========== FIN DEBUG — shutdown ==========")
    rospy.signal_shutdown("DEBUG_STOP_IK_%s" % label)


def _current_right_xyz(arm):
    """Pose FK courante de la main droite (orga `_call_fk(...).right_pose`)."""
    try:
        q0 = arm._read_arm_joints_rad(timeout=2.0)
    except Exception:
        return None
    if q0 is None or len(q0) != 14:
        return None
    try:
        fk = arm.call_fk(q0, timeout=5.0)
    except Exception:
        return None
    if fk is None:
        return None
    return [float(v) for v in fk.right_pose.pos_xyz]


def _current_left_xyz(arm):
    """Pose FK courante de la main gauche."""
    try:
        q0 = arm._read_arm_joints_rad(timeout=2.0)
    except Exception:
        return None
    if q0 is None or len(q0) != 14:
        return None
    try:
        fk = arm.call_fk(q0, timeout=5.0)
    except Exception:
        return None
    if fk is None:
        return None
    return [float(v) for v in fk.left_pose.pos_xyz]


def _quat_mult(q1, q2):
    x1, y1, z1, w1 = [float(v) for v in q1]
    x2, y2, z2, w2 = [float(v) for v in q2]
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def _quat_inv(q):
    x, y, z, w = [float(v) for v in q]
    n2 = x * x + y * y + z * z + w * w
    if n2 < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [-x / n2, -y / n2, -z / n2, w / n2]


def _current_right_pose(arm):
    """FK main droite : (xyz, quat_xyzw) ou (None, None)."""
    try:
        q0 = arm._read_arm_joints_rad(timeout=2.0)
    except Exception:
        return None, None
    if q0 is None or len(q0) != 14:
        return None, None
    try:
        fk = arm.call_fk(q0, timeout=5.0)
    except Exception:
        return None, None
    if fk is None:
        return None, None
    xyz = [float(v) for v in fk.right_pose.pos_xyz]
    quat = [float(v) for v in fk.right_pose.quat_xyzw]
    return xyz, quat


def _mirror_left_target_from_right(right_xyz, right_quat, log=None):
    """
    Symétrie sagittale D → G (ce que tu demandais) :
      x_L ≈ x_R (+ ox),  y_L = -y_R,  z_L ≈ z_R (+ oz)
      quat_L = miroir orga (LEFT vs RIGHT handoff quat)
    Pré-approche : même XZ, Y un peu plus + avant de serrer.
    """
    ox = float(globals().get("HANDOFF_MIRROR_X_OFFSET", 0.02) or 0.02)
    oz = float(globals().get("HANDOFF_MIRROR_Z_OFFSET", -0.02) or -0.02)
    y_pre = float(globals().get("HANDOFF_MIRROR_Y_PRE_APPROACH", 0.10) or 0.10)
    rx, ry, rz = [float(v) for v in right_xyz]
    # Vraie symétrie : y_L = -y_R (D à -0.045 → G à +0.045)
    left_y = -float(ry)
    left_recv = [rx + ox, left_y, rz + oz]
    left_xz = [left_recv[0], left_recv[1] + y_pre, left_recv[2]]
    rq = list(RIGHT_HANDOFF_QUAT)
    lq_ref = list(LEFT_HANDOFF_RECEIVE_QUAT)
    m_fix = _quat_mult(lq_ref, _quat_inv(rq))
    left_quat = _quat_mult(m_fix, list(right_quat))
    if log:
        log("[HANDOFF] SYMMETRY R=(%.3f,%.3f,%.3f) → L recv=(%.3f,%.3f,%.3f) "
            "y_L=-y_R xz_y=%.3f",
            rx, ry, rz, left_recv[0], left_recv[1], left_recv[2], left_xz[1])
    return left_xz, left_recv, left_quat


def _verify_right_handoff_pose(arm, log):
    """True si FK droite ≈ RIGHT_HANDOFF_IK (position habituelle passation)."""
    xyz, quat = _current_right_pose(arm)
    if xyz is None:
        log("[HANDOFF] verify R — FK indisponible")
        return False, xyz, quat
    tgt = list(RIGHT_HANDOFF_IK)
    tol_xy = float(globals().get("HANDOFF_RIGHT_TOL_XY", 0.035) or 0.035)
    tol_z = float(globals().get("HANDOFF_RIGHT_TOL_Z", 0.05) or 0.05)
    dx = abs(float(xyz[0]) - float(tgt[0]))
    dy = abs(float(xyz[1]) - float(tgt[1]))
    dz = abs(float(xyz[2]) - float(tgt[2]))
    ok = dx <= tol_xy and dy <= tol_xy and dz <= tol_z
    log("[HANDOFF] verify R act=(%.3f,%.3f,%.3f) tgt=(%.3f,%.3f,%.3f) "
        "Δ=(%.0f,%.0f,%.0f)mm → %s",
        xyz[0], xyz[1], xyz[2], tgt[0], tgt[1], tgt[2],
        dx * 1000, dy * 1000, dz * 1000, "OK" if ok else "FAIL")
    return ok, xyz, quat


def _ensure_right_at_handoff(arm, claw, log):
    """
    Force D à RIGHT_HANDOFF_IK avant approche G.
    Retourne (ok, locked_right_joints).
    """
    rq = RIGHT_HANDOFF_QUAT
    right_final = list(RIGHT_HANDOFF_IK)
    transit_zs = [float(RIGHT_HANDOFF_TRANSIT_Z)] + [
        float(z) for z in RIGHT_HANDOFF_TRANSIT_FALLBACK_ZS]
    retries = int(globals().get("HANDOFF_RIGHT_PLACE_RETRIES", 3) or 3)

    for attempt in range(1, retries + 1):
        cur = _current_right_xyz(arm)
        if cur is None:
            log("[HANDOFF] ensure R — FK indisponible")
            return False, None
        moved = False
        for tz in transit_zs:
            raise_pt = [cur[0], cur[1], float(tz)]
            if not _move_hand(
                arm, "right", raise_pt, rq, log,
                "right_handoff_raise_z%.2f_a%d" % (tz, attempt),
                constraint_mode=IK_MODE_THREE_POINT_MIXED,
                settle=PICK_ALIGN_MOVE_SLEEP,
            ):
                continue
            align_pt = [right_final[0], right_final[1], float(tz)]
            if not _move_hand(
                arm, "right", align_pt, rq, log,
                "right_handoff_xy_z%.2f_a%d" % (tz, attempt),
                constraint_mode=IK_MODE_THREE_POINT_MIXED,
                settle=PICK_ALIGN_MOVE_SLEEP,
            ):
                continue
            moved = True
            break
        if not moved:
            log("[HANDOFF] ensure R transit fail attempt %d/%d", attempt, retries)
            continue
        if not _move_hand(
            arm, "right", right_final, rq, log,
            "right_handoff_to_left_a%d" % attempt,
            constraint_mode=IK_MODE_THREE_POINT_MIXED,
            settle=PICK_ALIGN_MOVE_SLEEP,
        ):
            log("[HANDOFF] ensure R final fail attempt %d/%d", attempt, retries)
            continue
        if bool(globals().get("HANDOFF_RIGHT_VERIFY", True)):
            ok, _, _ = _verify_right_handoff_pose(arm, log)
            if ok:
                locked = list(arm._last_cmd_deg[7:14])
                return True, locked
            log("[HANDOFF] ensure R — pas à handoff, retry %d/%d",
                attempt, retries)
            try:
                claw.right_close()
                claw.wait_until_done(timeout=2.0)
            except Exception:
                pass
            continue
        locked = list(arm._last_cmd_deg[7:14])
        return True, locked
    return False, None


def _left_handoff_light_correct(arm, locked_right, llq, cam, parcel_name,
                                left_recv, log, max_iters=2, dpix_thresh=70.0):
    """
    Petite correction cam poignet G seulement si Δpx > seuil (pas full servo).
    """
    if cam is None or not parcel_name:
        return True, list(left_recv)
    right_xyz = _current_right_xyz(arm) or list(RIGHT_HANDOFF_IK)
    working = {
        "name": parcel_name,
        "center": (float(right_xyz[0]), float(right_xyz[1]), float(right_xyz[2])),
    }
    left_xyz = list(left_recv)
    sx = float(globals().get("HANDOFF_LEFT_SERVO_SIGN_X", 1.0) or 1.0)
    sy = float(globals().get("HANDOFF_LEFT_SERVO_SIGN_Y", -1.0) or -1.0)
    step_max = float(globals().get("HANDOFF_LEFT_MAX_DELTA_XY", 0.02) or 0.02)
    gain = float(globals().get("WRIST_SERVO_GAIN", 0.45) or 0.45)

    for wi in range(max(1, int(max_iters))):
        obs = observe_hand(cam, working, hand="left", log=log, settle=0.15)
        dpix = float(obs.get("dpix", 999))
        if not obs.get("seen") or dpix <= float(dpix_thresh):
            if obs.get("seen"):
                log("[HANDOFF] L-light OK Δpx=%.0f — pas de correction", dpix)
            return True, left_xyz
        gpx = 0.00018 * gain
        dx = sx * (float(obs["u"]) - float(obs["cx"])) * gpx
        dy = sy * (float(obs["v"]) - float(obs["cy"])) * gpx
        dxy = math.hypot(dx, dy)
        if dxy > step_max and dxy > 1e-6:
            s = step_max / dxy
            dx *= s
            dy *= s
        left_xyz[0] += dx
        left_xyz[1] += dy
        log("[HANDOFF] L-light %d Δpx=%.0f → Δxy=(%+.1f,%+.1f)cm",
            wi + 1, dpix, dx * 100, dy * 100)
        if not _move_left_keep_right(
            arm, left_xyz, llq, locked_right, log, "left_light_%d" % (wi + 1),
            settle=0.35,
        ):
            break
    obs = observe_hand(cam, working, hand="left", log=log, settle=0.15)
    dpix = float(obs.get("dpix", 999))
    seen = bool(obs.get("seen"))
    zone = str(obs.get("grip_zone", "") or "")
    max_dpix = float(globals().get("HANDOFF_LEFT_CLOSE_MAX_DPIX", 120) or 120)
    ok = seen and dpix <= max_dpix and zone not in ("outside", "")
    if not ok:
        log("[HANDOFF] L-light final FAIL seen=%s Δpx=%.0f zone=%s (max=%.0f)",
            seen, dpix, zone, max_dpix)
    else:
        log("[HANDOFF] L-light final OK Δpx=%.0f zone=%s", dpix, zone)
    return ok, left_xyz


def _left_handoff_wrist_lock(arm, locked_right, llq, cam, tf_reader,
                             parcel_name, left_recv, log):
    """
    Même logique que saisie DROITE avant pesée :
      caméra poignet GAUCHE → servo Δxy → lock tip sur colis (tenu par D).
    Colis en l'air (pas table) → depth / pixel, PAS ray table.
    Retourne (vision_ok, left_xyz_final).
    """
    if cam is None or tf_reader is None or not parcel_name:
        return False, list(left_recv)

    right_xyz = _current_right_xyz(arm) or list(RIGHT_HANDOFF_IK)
    working = {
        "name": parcel_name,
        "center": (float(right_xyz[0]), float(right_xyz[1]), float(right_xyz[2])),
        "center_raw": (float(right_xyz[0]), float(right_xyz[1]), float(right_xyz[2])),
    }
    left_xyz = list(left_recv)
    n_iters = max(1, int(globals().get("HANDOFF_LEFT_SERVO_ITERS",
                                       globals().get("WRIST_SERVO_ITERS", 4)) or 4))
    step_max = float(globals().get("HANDOFF_LEFT_MAX_DELTA_XY",
                                   globals().get("WRIST_MAX_DELTA_XY", 0.02)) or 0.02)
    require_vis = bool(globals().get("HANDOFF_LEFT_REQUIRE_VISION", True))
    sx = float(globals().get("HANDOFF_LEFT_SERVO_SIGN_X",
                             globals().get("WRIST_SERVO_SIGN_X", 1.0)) or 1.0)
    sy = float(globals().get("HANDOFF_LEFT_SERVO_SIGN_Y",
                             globals().get("WRIST_SERVO_SIGN_Y", -1.0)) or -1.0)
    gain = float(globals().get("WRIST_SERVO_GAIN", 0.45) or 0.45)

    def _lock_ok(o):
        if not o.get("seen"):
            return False
        if o.get("centered") or o.get("grip_zone") == "center" or o.get("tip_in_core"):
            return True
        area = float(o.get("area", 0))
        dpix = float(o.get("dpix", 1e9))
        accept = float(globals().get("WRIST_ACCEPT_PX", 90) or 90)
        uh_d = float(globals().get("WRIST_UNDER_HAND_MAX_DPIX", 110) or 110)
        uh_area = float(globals().get("WRIST_UNDER_HAND_AREA", 12000) or 12000)
        min_area = float(globals().get("WRIST_LOCK_MIN_AREA", 3000) or 3000)
        if area >= min_area and dpix <= max(accept, uh_d):
            return True
        if area >= uh_area and dpix <= uh_d:
            return True
        return False

    vision_ok = False
    for wi in range(n_iters):
        obs = observe_hand(cam, working, hand="left", log=log, settle=0.20)
        log("[HANDOFF] L-wrist %d/%d seen=%s zone=%s Δpx=%.0f centered=%s",
            wi + 1, n_iters, obs.get("seen"), obs.get("grip_zone"),
            float(obs.get("dpix", -1)), obs.get("centered"))
        log_wrist_event(
            "handoff_left_servo", log=log, name=parcel_name, iter=wi + 1,
            seen=bool(obs.get("seen")), aim=obs.get("aim"),
            grip_zone=obs.get("grip_zone"),
            centered=bool(obs.get("centered")),
            dpix=float(obs.get("dpix", -1)),
            area=int(obs.get("area", 0)),
        )
        if _lock_ok(obs):
            conf = observe_hand(cam, working, hand="left", log=log, settle=0.18)
            if _lock_ok(conf):
                vision_ok = True
                log("[HANDOFF] L-wrist VISION LOCK frac=%.2f area=%d Δpx=%.0f",
                    float(conf.get("frac", 1.0)), int(conf.get("area", 0)),
                    float(conf.get("dpix", -1)))
                break
            log("[HANDOFF] L-wrist lock instable — continue")

        if not obs.get("seen"):
            continue

        dx = dy = 0.0
        depth = obs.get("depth")
        blob = obs.get("blob")
        used = "pixel"
        if (depth is not None and blob is not None
                and hasattr(cam, "median_depth_in_mask")
                and hasattr(cam, "pixel_to_base_link")):
            zmin = float(globals().get("WRIST_DEPTH_Z_GRASP_MIN", 0.06) or 0.06)
            zmax = float(globals().get("WRIST_DEPTH_Z_GRASP_MAX", 0.55) or 0.55)
            dm = cam.median_depth_in_mask(depth, blob, z_min=zmin, z_max=zmax)
            if dm is not None:
                pt_blob = cam.pixel_to_base_link(
                    tf_reader, obs["cam_key"], obs["u"], obs["v"], dm)
                pt_aim = cam.pixel_to_base_link(
                    tf_reader, obs["cam_key"], obs["cx"], obs["cy"], dm)
                if pt_blob is not None and pt_aim is not None:
                    dx = (float(pt_blob[0]) - float(pt_aim[0])) * gain * sx
                    dy = (float(pt_blob[1]) - float(pt_aim[1])) * gain * sy
                    used = "depth3d"

        if used == "pixel":
            # Fallback : erreur pixel → petit Δxy (colis en l'air)
            gpx = 0.00018 * gain
            dx = sx * (float(obs["u"]) - float(obs["cx"])) * gpx
            dy = sy * (float(obs["v"]) - float(obs["cy"])) * gpx

        dxy = math.hypot(dx, dy)
        if dxy < 1e-4:
            continue
        if dxy > step_max:
            s = step_max / dxy
            dx *= s
            dy *= s
            dxy = step_max

        left_xyz = [
            float(left_xyz[0]) + dx,
            float(left_xyz[1]) + dy,
            float(left_recv[2]),
        ]
        log("[HANDOFF] L-wrist APPLY[%s] Δxy=(%+.1f,%+.1f)cm → (%.3f,%.3f,%.3f)",
            used, dx * 100.0, dy * 100.0, left_xyz[0], left_xyz[1], left_xyz[2])
        if not _move_left_keep_right(
            arm, left_xyz, llq, locked_right, log, "left_wrist_servo_%d" % (wi + 1),
            constraint_mode=IK_MODE_THREE_POINT_MIXED, settle=0.45,
        ):
            log("[HANDOFF] L-wrist move IK fail — garde pose")

    if not vision_ok and not require_vis:
        log("[HANDOFF] L-wrist pas lock — close quand même (require_vis=False)")
        return True, left_xyz
    return vision_ok, left_xyz


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
    _maybe_stop_after_first_ik(arm, side, pos_xyz, quat_xyzw, joints, log, label)
    _log_pose_snapshot(arm, log, "moved_%s" % label)
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
    _log_pose_snapshot(arm, log, "preset_final")
    return True


# =============================================================================
# Pince — confirmation hold (anti drop en l'air sur faux EMPTY)
# =============================================================================

def _await_right_hold(claw, log, label, timeout=1.5):
    """
    Attend une prise confirmée après close.
    Si hold=True ou R=GRABBED une fois → latch (ne jamais right_open en l'air).
    """
    hits = 0
    latched = False
    t0 = time.time()
    while time.time() - t0 < float(timeout):
        rs = int(getattr(claw, "_right_state", -1))
        if rs == 3 or claw.right_holding():
            latched = True
            hits += 1
            if hits >= 2:
                log("[GRASP] hold confirm %s — %s", label, claw.describe_right())
                return True, True
        else:
            hits = max(0, hits - 1)
        rospy.sleep(0.08)
    if latched:
        log("[GRASP] hold latched %s (flicker) — %s",
            label, claw.describe_right())
        return True, True
    return False, False


def _maintain_right_close(claw, log, label):
    """Re-serre D — ne jamais lâcher tant que G n'a pas Grabbed."""
    try:
        claw.right_close()
        claw.wait_until_done(timeout=2.0)
        if not claw.right_holding():
            log("[HANDOFF] WARN D hold faible après %s — %s",
                label, claw.describe_right())
            return False
        return True
    except Exception as exc:
        log("[HANDOFF] maintain R skip %s: %s", label, exc)
        return False


def _await_left_grabbed_strict(claw, log, label, timeout=3.5):
    """
    Handoff : exige L=GRABBED (state 3) — REACHED 90% vide ne compte pas.
    """
    hits = 0
    need = int(globals().get("HANDOFF_LEFT_GRABBED_HITS", 3) or 3)
    latched = False
    t0 = time.time()
    timeout = float(timeout)
    while time.time() - t0 < timeout:
        ls = int(getattr(claw, "_left_state", -1))
        if ls == 3:
            latched = True
            hits += 1
            if hits >= need:
                log("[HANDOFF] L=GRABBED strict confirm %s — %s",
                    label, claw.describe_left())
                return True, True
        elif ls == 1:
            hits = max(0, hits - 1)
        else:
            hits = 0
        rospy.sleep(0.08)
    log("[HANDOFF] L=GRABBED strict FAIL %s — %s (need state 3)",
        label, claw.describe_left())
    return False, latched


def _handoff_dual_squeeze(claw, log, duration):
    """Maintient D+F fermées pendant la passation — évite lâcher prématuré."""
    duration = float(duration)
    log("[HANDOFF] dual squeeze %.1fs (D+F fermées)", duration)
    t0 = time.time()
    while time.time() - t0 < duration:
        try:
            claw.right_close()
            claw.left_close()
            claw.wait_until_done(timeout=1.5)
        except Exception as exc:
            log("[HANDOFF] dual squeeze skip: %s", exc)
        if not claw.right_holding():
            log("[HANDOFF] ABORT squeeze — DROITE a lâché: %s",
                claw.describe_right())
            return False
        rospy.sleep(0.12)
    log("[HANDOFF] dual squeeze OK — %s | %s",
        claw.describe_right(), claw.describe_left())
    return True


def _safe_open_right_after_left(claw, log, left_grabbed_ok):
    """Ouvre D SEULEMENT si G = state 3 (Grabbed) confirmé — jamais sur hold flou."""
    if not left_grabbed_ok:
        log("[HANDOFF] REFUSE open R — gauche pas Grabbed confirmé")
        return False
    claw.left_close()
    claw.wait_until_done(timeout=2.0)
    rospy.sleep(0.25)
    ls = int(getattr(claw, "_left_state", -1))
    if ls != 3:
        log("[HANDOFF] REFUSE open R — L state≠3 (pas Grabbed): %s",
            claw.describe_left())
        return False
    if not claw.right_holding():
        log("[HANDOFF] REFUSE open R — droite ne tient plus: %s",
            claw.describe_right())
        return False
    log("[HANDOFF] === OPEN DROITE (L=GRABBED state=3, D tenait) ===")
    claw.right_open()
    claw.wait_until_done(timeout=3.0)
    rospy.sleep(GRIPPER_CLOSE_HOLD)
    return True


def _await_left_hold(claw, log, label, timeout=2.5):
    """
    Attend Grabbed / left_holding après close GAUCHE.
    Ne PAS ouvrir la droite tant que ça return False.
    """
    hits = 0
    latched = False
    t0 = time.time()
    timeout = float(timeout)
    while time.time() - t0 < timeout:
        ls = int(getattr(claw, "_left_state", -1))
        holding = False
        try:
            holding = bool(claw.left_holding())
        except Exception:
            holding = ls == 3
        if ls == 3 or holding:
            latched = True
            hits += 1
            if hits >= 2:
                try:
                    desc = claw.describe_left()
                except Exception:
                    desc = "L=%d" % ls
                log("[HANDOFF] left hold confirm %s — %s", label, desc)
                return True, True
        else:
            hits = max(0, hits - 1)
        rospy.sleep(0.08)
    try:
        desc = claw.describe_left()
    except Exception:
        desc = "L=%d pos=%.0f" % (
            int(getattr(claw, "_left_state", -1)),
            float(getattr(claw, "_left_pos", -1.0)),
        )
    if latched:
        log("[HANDOFF] left hold latched %s (flicker) — %s", label, desc)
        return True, True
    log("[HANDOFF] left hold FAIL %s — %s (pas Grabbed)", label, desc)
    return False, False


def _squeeze_probe_right(claw, log, label="squeeze"):
    """
    Fait sentir la prise : close → pause → re-serre (pulses) → lit state/pos/effort.
    Retourne (holding, rs, pos, effort, reason).
      holding True  = GRABBED ou pos bloquée 25–82% ou effort élevé
      holding False = REACHED ≥85% (ferme dans le vide)
    """
    pulses = int(globals().get("GRASP_SQUEEZE_PULSES", 2) or 2)
    pause = float(globals().get("GRASP_SQUEEZE_PAUSE", 0.35) or 0.35)
    eff_min = float(globals().get("GRASP_SQUEEZE_EFFORT_MIN", 0.5) or 0.5)

    claw.right_close()
    claw.wait_until_done(timeout=3.0)
    rospy.sleep(0.2)

    for i in range(max(1, pulses)):
        # Re-commande close = re-applique effort (la pince "re-sent" la résistance)
        claw.right_close()
        claw.wait_until_done(timeout=2.5)
        # Attendre fin MOVING
        for _ in range(30):
            if int(getattr(claw, "_right_state", -1)) != 1:
                break
            rospy.sleep(0.08)
        rospy.sleep(pause)
        rs = int(getattr(claw, "_right_state", -1))
        pos = float(getattr(claw, "_right_pos", -1.0))
        eff = float(getattr(claw, "_right_effort", 0.0))
        log("[GRASP] squeeze pulse %d/%d — %s",
            i + 1, pulses, claw.describe_right())
        # Décision précoce si déjà clair
        if rs == 3 and 20.0 <= pos <= 82.0:
            log("[GRASP] PREUVE ATTRAPE (%s) — R=GRABBED pos=%.0f%% eff=%.2f "
                "après re-serrage", label, pos, eff)
            return True, rs, pos, eff, "grabbed"
        if rs == 2 and pos >= 85.0 and abs(eff) < eff_min:
            log("[GRASP] PREUVE VIDE (%s) — R=REACHED pos=%.0f%% eff=%.2f "
                "(air, même après re-serrage)", label, pos, eff)
            return False, rs, pos, eff, "empty"

    rs = int(getattr(claw, "_right_state", -1))
    pos = float(getattr(claw, "_right_pos", -1.0))
    eff = float(getattr(claw, "_right_effort", 0.0))
    holding = bool(claw.right_holding())
    if rs == 3 or (25.0 <= pos <= 82.0) or abs(eff) >= eff_min:
        holding = True
        reason = "grabbed" if rs == 3 else ("effort" if abs(eff) >= eff_min else "pos_block")
        log("[GRASP] PREUVE ATTRAPE (%s) — %s reason=%s",
            label, claw.describe_right(), reason)
        return True, rs, pos, eff, reason
    log("[GRASP] PREUVE VIDE/INCERTAINE (%s) — %s", label, claw.describe_right())
    return False, rs, pos, eff, "uncertain"


def _claw_is_holding(claw, hold_latched=False):
    """True si la pince droite tient (état + latch)."""
    if hold_latched:
        return True
    rs = int(getattr(claw, "_right_state", -1))
    return rs == 3 or bool(claw.right_holding())


def _maintain_grasp_close(claw, log, label):
    """Re-serre après détection prise — évite relâchement pendant lift."""
    if not bool(globals().get("GRASP_MAINTAIN_CLOSE", True)):
        return
    try:
        claw.right_close()
        claw.wait_until_done(timeout=2.0)
        rospy.sleep(0.15)
        log("[GRASP] maintain close %s — %s", label, claw.describe_right())
    except Exception as exc:
        log("[GRASP] maintain close skip: %s", exc)


def _safe_abort_grasp_open(claw, arm, right_pre, rq, log, label, hold_latched):
    """N'ouvre PAS en l'air si la prise a été confirmée (évite chute colis)."""
    if hold_latched:
        log("[GRASP] %s — prise latched, PAS d'ouverture en l'air", label)
        try:
            _move_right_cartesian_to(
                arm, right_pre, rq, log, "latched_abort_lower",
                n_points=3, constraint_mode=RIGHT_PICK_IK_MODE,
            )
        except Exception as exc:
            log("[GRASP] latched lower skip: %s", exc)
        return False
    claw.right_open()
    claw.wait_until_done(timeout=2.0)
    try:
        _move_right_cartesian_to(
            arm, right_pre, rq, log, label,
            n_points=3, constraint_mode=RIGHT_PICK_IK_MODE,
        )
    except Exception as exc:
        log("[GRASP] abort raise skip: %s", exc)
    return False


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
    # Clamp X workspace (seed30 : x=0.16 → IK fail)
    min_x = float(globals().get("MIN_PICK_IK_X", 0.22) or 0.22)
    if cx < min_x:
        log("[GRASP] clamp x %.3f → %.3f (workspace)", cx, min_x)
        cx = min_x
    ox, oy, oz = _right_pick_offset(name, cy)
    tip = RIGHT_CLAW_TIP_OFFSET
    pick_x = cx + ox + float(tip[0])
    pick_y = cy + oy + float(tip[1])
    pick_z = RIGHT_PICK_IK_Z + oz + float(tip[2])
    z_min = float(globals().get("GRASP_PICK_Z_MIN", -0.065) or -0.065)
    if pick_z < z_min:
        log("[GRASP] clamp pick_z %.3f → %.3f (plancher)", pick_z, z_min)
        pick_z = z_min
    right_pre = [pick_x, pick_y, float(RIGHT_PICK_TRANSIT_IK_Z)]
    right_grasp = [pick_x, pick_y, pick_z]
    rq = list(RIGHT_PICK_QUAT)
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

    # Yaw // faces du carré : 1 seule mesure au-dessus, puis quat verrouillé
    if (cam is not None and bool(globals().get("WRIST_YAW_ENABLE", False))
            and not parcel.get("skip_wrist")):
        obs_yaw = observe_hand(cam, working, hand="right", log=log, settle=0.25)
        if obs_yaw.get("seen"):
            axis = int(obs_yaw.get("square_axis", 0) or 0)
            yaw_raw = float(obs_yaw.get("yaw_raw_deg", 0.0) or 0.0)
            yaw_snap = float(obs_yaw.get("yaw_deg", 0.0) or 0.0)
            if axis == 90:
                rq = list(globals().get(
                    "RIGHT_PICK_QUAT_AXIS90", RIGHT_PICK_QUAT) or RIGHT_PICK_QUAT)
            else:
                rq = list(RIGHT_PICK_QUAT)
                axis = 0
            log("[GRASP] YAW LOCK raw=%+.0f° snap=%+.0f° → axis=%d (faces //) — figé",
                yaw_raw, yaw_snap, axis)
            if not _move_hand(
                arm, "right", right_approach, rq, log, "hand_yaw_lock",
                constraint_mode=RIGHT_PICK_IK_MODE, settle=0.55,
            ):
                log("[GRASP] YAW LOCK IK fail — garde quat précédent")
                rq = list(RIGHT_PICK_QUAT)
        else:
            log("[GRASP] YAW: blob non vu — RIGHT_PICK_QUAT défaut")

    vision_ok = False
    best_frac = 1.0
    if cam is not None and tf_reader is not None and not parcel.get("skip_wrist"):
        n_iters = max(1, int(globals().get("WRIST_SERVO_ITERS", 3) or 3))
        step_max = float(globals().get("WRIST_MAX_DELTA_XY", 0.02) or 0.02)
        soft = float(globals().get("WRIST_CLOSE_MAX_PIXEL_FRAC", 0.12) or 0.12)
        min_area = float(globals().get("WRIST_LOCK_MIN_AREA", 2500) or 2500)
        max_dpix = float(globals().get("WRIST_ACCEPT_PX", 110.0) or 110.0)
        uh_area = float(globals().get("WRIST_UNDER_HAND_AREA", 15000) or 15000)
        uh_frac = float(globals().get("WRIST_UNDER_HAND_FRAC", 0.14) or 0.14)
        uh_dpix = float(globals().get("WRIST_UNDER_HAND_MAX_DPIX", 130) or 130)

        best_snap = {
            "working": dict(working),
            "pick_x": pick_x, "pick_y": pick_y, "pick_z": pick_z,
            "right_approach": list(right_approach),
            "right_grasp": list(right_grasp),
            "right_pre": list(right_pre),
            "frac": 1.0, "area": 0, "dpix": 1e9,
        }

        def _apply_snap(snap):
            nonlocal working, pick_x, pick_y, pick_z
            nonlocal right_approach, right_grasp, right_pre
            working = dict(snap["working"])
            pick_x = snap["pick_x"]
            pick_y = snap["pick_y"]
            pick_z = snap["pick_z"]
            right_approach = list(snap["right_approach"])
            right_grasp = list(snap["right_grasp"])
            right_pre = list(snap["right_pre"])

        def _lock_ok(o):
            """Contrat main : tip sur le blob — sinon ABORT (pas faux CENTRÉ)."""
            if not o.get("seen"):
                return False
            area = float(o.get("area", 0))
            frac = float(o.get("frac", 1.0))
            dpix = float(o.get("dpix", 1e9))
            if area < min_area:
                return False
            # Tip OK seulement si centered() du poignet (déjà strict) + Δpx
            accept = float(globals().get("WRIST_ACCEPT_PX", 90) or 90)
            uh_d = float(globals().get("WRIST_UNDER_HAND_MAX_DPIX", 110) or 110)
            if dpix > max(accept, uh_d):
                return False
            if o.get("centered"):
                return True
            if area >= uh_area and frac <= uh_frac and dpix <= uh_d:
                return True
            return False

        def _remember(o):
            nonlocal best_frac, best_snap
            if not o.get("seen"):
                return
            frac = float(o.get("frac", 1.0))
            if frac <= best_frac + 1e-6:
                best_frac = frac
                best_snap = {
                    "working": dict(working),
                    "pick_x": pick_x, "pick_y": pick_y, "pick_z": pick_z,
                    "right_approach": list(right_approach),
                    "right_grasp": list(right_grasp),
                    "right_pre": list(right_pre),
                    "frac": frac,
                    "area": int(o.get("area", 0)),
                    "dpix": float(o.get("dpix", 1e9)),
                }

        for wi in range(n_iters):
            obs = observe_hand(cam, working, hand="right", log=log)
            _remember(obs)
            if _lock_ok(obs):
                # Anti faux CENTRÉ (log: Δpx 376→96 sans move) — 2 frames
                conf = observe_hand(cam, working, hand="right", log=log, settle=0.20)
                _remember(conf)
                if _lock_ok(conf) and float(conf.get("dpix", 1e9)) <= 150.0:
                    vision_ok = True
                    log("[GRASP] VISION OK %d/%d frac=%.2f area=%d Δpx=%.0f (confirmé)",
                        wi + 1, n_iters, conf["frac"], conf["area"],
                        conf.get("dpix", 0))
                    break
                log("[GRASP] VISION lock instable Δpx=%.0f→%.0f — continue",
                    obs.get("dpix", 0), conf.get("dpix", 0))
            if not obs.get("seen"):
                log("[GRASP] VISION %d/%d — pas vu couleur, garde tête",
                    wi + 1, n_iters)
                continue

            frac_before = float(obs.get("frac", 1.0))
            area_before = int(obs.get("area", 0))
            pre_snap = {
                "working": dict(working),
                "pick_x": pick_x, "pick_y": pick_y, "pick_z": pick_z,
                "right_approach": list(right_approach),
                "right_grasp": list(right_grasp),
                "right_pre": list(right_pre),
            }

            refined = refine_target_with_wrist(
                cam, tf_reader, working, hand="right", log=log,
                max_delta_xy=step_max)
            if not refined.get("wrist_refined"):
                continue
            if refined.get("wrist_centered"):
                vision_ok = True
                working = refined
                break
            dxy = float(refined.get("wrist_delta_xy", 0.0))
            working = refined
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
                constraint_mode=RIGHT_PICK_IK_MODE, settle=0.55,
            )
            obs_after = observe_hand(cam, working, hand="right", log=log, settle=0.15)

            # Si le move empiri / disparaît → RESTAURE (ne pas “partir ailleurs”)
            worse = False
            if not obs_after.get("seen"):
                worse = True
                log("[GRASP] VISION perdu après move — restaure meilleure pose")
            else:
                frac_after = float(obs_after.get("frac", 1.0))
                area_after = int(obs_after.get("area", 0))
                if frac_after > frac_before + 0.02 or area_after < 0.45 * max(area_before, 1):
                    worse = True
                    log("[GRASP] VISION empire frac %.2f→%.2f area %d→%d — restaure",
                        frac_before, frac_after, area_before, area_after)

            if worse:
                _apply_snap(best_snap if best_snap["area"] > 0 else pre_snap)
                _move_hand(
                    arm, "right", right_approach, rq, log, "hand_restore_best",
                    constraint_mode=RIGHT_PICK_IK_MODE, settle=0.45,
                )
                # Si best avait déjà un vrai lock tip → lock et descend
                if (best_snap["area"] >= uh_area
                        and best_snap["frac"] <= uh_frac
                        and best_snap.get("dpix", 1e9) <= float(
                            globals().get("WRIST_UNDER_HAND_MAX_DPIX", 110) or 110)):
                    vision_ok = True
                    log("[GRASP] VISION LOCK sur best frac=%.2f area=%d Δpx=%.0f (stop diverge)",
                        best_snap["frac"], best_snap["area"], best_snap.get("dpix", 0))
                break

            _remember(obs_after)
            if _lock_ok(obs_after):
                conf = observe_hand(cam, working, hand="right", log=log, settle=0.18)
                _remember(conf)
                if _lock_ok(conf) and float(conf.get("dpix", 1e9)) <= 150.0:
                    vision_ok = True
                    log("[GRASP] VISION OK après servo frac=%.2f area=%d Δpx=%.0f",
                        conf["frac"], conf["area"], conf.get("dpix", 0))
                    break
                log("[GRASP] VISION servo lock instable — continue")

        if not vision_ok:
            # Dernière chance : best connu (vrai tip) plutôt que poser à vide
            if (best_snap["area"] >= uh_area
                    and best_snap["frac"] <= uh_frac
                    and best_snap.get("dpix", 1e9) <= float(
                        globals().get("WRIST_UNDER_HAND_MAX_DPIX", 110) or 110)):
                _apply_snap(best_snap)
                vision_ok = True
                log("[GRASP] VISION ACCEPT best frac=%.2f area=%d Δpx=%.0f",
                    best_snap["frac"], best_snap["area"], best_snap.get("dpix", 0))
            else:
                obs = observe_hand(cam, working, hand="right", log=log)
                vision_ok = _lock_ok(obs)
                if obs.get("seen") and not vision_ok:
                    log("[GRASP] VISION final décalé frac=%.2f area=%d Δpx=%.0f (best=%.2f)",
                        obs.get("frac", 1.0), obs.get("area", 0),
                        obs.get("dpix", 0.0), best_frac)

        if WRIST_REQUIRE_SEE_BEFORE_CLOSE and not vision_ok:
            log("[GRASP] VISION FAIL — pas au-dessus du colis, remonte OUVERT")
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

    # Check vision après plongée — évite close vide (pos≈88% hold=False).
    # Assoupli : seed0 mission LOCK Δpx≈86 → post-plongée Δpx≈154 (parallaxe)
    # abortait systématiquement avant close. Accepter si encore vu + tip proche.
    if cam is not None and WRIST_REQUIRE_SEE_BEFORE_CLOSE:
        obs_pl = observe_hand(cam, working, hand="right", log=log, settle=0.20)
        accept = float(globals().get("WRIST_ACCEPT_PX", 90) or 90)
        uh_d = float(globals().get("WRIST_UNDER_HAND_MAX_DPIX", 100) or 100)
        post_d = float(globals().get("WRIST_POST_PLUNGE_MAX_DPIX", 170) or 170)
        max_d = max(accept, uh_d, post_d)
        min_area = int(globals().get("WRIST_LOCK_MIN_AREA", 3000) or 3000)
        frac_pl = float(obs_pl.get("frac", 1.0))
        area_pl = int(obs_pl.get("area", 0))
        dpix_pl = float(obs_pl.get("dpix", 1e9))
        # Pré-LOCK solide : plongée OK même si Δpx montee (cam tip change FOV)
        if vision_ok and obs_pl.get("seen") and frac_pl <= 0.22:
            pl_ok = True
        else:
            pl_ok = (
                obs_pl.get("seen")
                and bool(obs_pl.get("centered") or (
                    area_pl >= min_area and dpix_pl <= max_d))
                and frac_pl <= 0.18
                and dpix_pl <= max_d
            )
        if not pl_ok:
            log("[GRASP] VISION post-plongée KO seen=%s frac=%.2f area=%d Δpx=%.0f — remonte OUVERT",
                obs_pl.get("seen"), frac_pl, area_pl, dpix_pl)
            _move_right_cartesian_to(
                arm, right_pre, rq, log, "plunge_vision_abort",
                n_points=3, constraint_mode=RIGHT_PICK_IK_MODE,
            )
            claw.right_open()
            return False
        log("[GRASP] VISION post-plongée OK frac=%.2f area=%d Δpx=%.0f",
            frac_pl, area_pl, dpix_pl)
        obs_before_close = obs_pl
    else:
        obs_before_close = None

    def _do_close_and_assess(label):
        log("[GRASP] tip CLOSE + squeeze probe (%s)", label)
        if bool(globals().get("GRASP_SQUEEZE_PROBE", True)):
            holding, rs, pos, eff, reason = _squeeze_probe_right(
                claw, log, label=label)
        else:
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
            rs = int(getattr(claw, "_right_state", -1))
            pos = float(getattr(claw, "_right_pos", -1.0))
            eff = float(getattr(claw, "_right_effort", 0.0))
            holding = bool(claw.right_holding())
            reason = "legacy"
            log("[GRASP] claw state: %s", claw.describe_right())
            if rs == 3 and 25.0 <= pos <= 82.0:
                log("[GRASP] PREUVE ATTRAPE — R=GRABBED pos=%.0f%%", pos)
            elif rs == 2 and pos >= 85.0:
                log("[GRASP] PREUVE VIDE — R=REACHED pos=%.0f%%", pos)

        obs_now = obs_before_close
        if cam is not None:
            try:
                obs_now = observe_hand(
                    cam, working, hand="right", log=log, settle=0.12)
            except Exception:
                pass
        assess = assess_grasp_manner(
            obs=obs_now, claw_pos=pos, claw_state=rs, holding=holding)
        log("[GRASP] manner=%s aim=%s hold=%s pos=%.0f%% eff=%.2f "
            "ok_weigh=%s reason=%s (%s)",
            assess["manner"], assess["aim"], assess["hold"], pos, eff,
            assess["ok_for_weigh"], reason, label)
        log_wrist_event(
            "grasp_assess", log=log, name=name, phase=label,
            manner=assess["manner"], aim=assess["aim"], hold=assess["hold"],
            ok_for_weigh=bool(assess["ok_for_weigh"]),
            claw_pos=pos, claw_state=rs, holding=holding,
            effort=eff, squeeze_reason=reason,
            dpix=float(assess.get("dpix", -1)),
            frac=float(assess.get("frac", -1)),
            area=int(assess.get("area", 0)),
        )
        return holding, assess, rs, pos

    skip_claw = bool(globals().get("WRIST_SKIP_CLAW_HOLD_CHECK", False))
    hold_latched = False
    held = True
    assess = {"manner": "unknown", "ok_for_weigh": True, "aim": "?", "hold": "?"}
    recovery_max = int(globals().get("GRASP_RECOVERY_MAX", 2) or 2)

    held, assess, rs, pos = _do_close_and_assess("vision-confirmed")
    if not skip_claw:
        held2, latched = _await_right_hold(claw, log, "post-close")
        hold_latched = bool(latched)
        held = bool(held or held2 or hold_latched)

        # Très bonne 1ère prise → LOCK (ne jamais rouvrir ensuite)
        excellent = False
        if bool(globals().get("GRASP_LOCK_EXCELLENT_FIRST", True)):
            excellent = is_excellent_grasp(
                obs=obs_before_close,
                assess=assess,
                claw_state=rs,
                claw_pos=pos,
                holding=held,
                vision_locked=bool(vision_ok),
            )
            if excellent:
                hold_latched = True
                log("[GRASP] EXCELLENT 1er essai manner=%s hold=%s pos=%.0f%% "
                    "vision_lock=%s — LOCK (ne rouvre JAMAIS)",
                    assess.get("manner"), assess.get("hold"), float(pos),
                    bool(vision_ok))
                log_wrist_event(
                    "grasp_excellent_lock", log=log, name=name,
                    manner=assess.get("manner"), hold=assess.get("hold"),
                    claw_pos=float(pos), claw_state=int(rs),
                    vision_locked=bool(vision_ok),
                )
                _maintain_grasp_close(claw, log, "excellent-lock")

        keep_if_holding = bool(globals().get("GRASP_KEEP_IF_HOLDING", True))
        # NE PAS rouvrir si : excellent lock, OU pince tient déjà
        need_recovery = (not excellent) and (not _claw_is_holding(claw, hold_latched))
        if need_recovery:
            for rec in range(1, recovery_max + 1):
                log("[GRASP] RECOVERY %d/%d manner=%s — ouvrir ↑ wrist ↓ close",
                    rec, recovery_max, assess.get("manner"))
                log_wrist_event(
                    "grasp_recovery", log=log, name=name, try_i=rec,
                    reason=assess.get("manner"), aim=assess.get("aim"),
                    hold=assess.get("hold"),
                )
                try:
                    claw.right_open()
                    claw.wait_until_done(timeout=2.0)
                except Exception as exc:
                    log("[GRASP] recovery open skip: %s", exc)
                try:
                    _move_right_cartesian_to(
                        arm, right_pre, rq, log, "recovery_raise_%d" % rec,
                        n_points=3, constraint_mode=RIGHT_PICK_IK_MODE,
                    )
                except Exception:
                    _move_hand(
                        arm, "right", right_pre, rq, log, "recovery_raise_fb_%d" % rec,
                        constraint_mode=RIGHT_PICK_IK_MODE, settle=0.4,
                    )
                if cam is not None:
                    working = refine_target_with_wrist(
                        cam, tf_reader, working, hand="right", log=log)
                    cx, cy, cz = working["center"]
                    ox, oy, oz = _right_pick_offset(name, cy)
                    tip = RIGHT_CLAW_TIP_OFFSET
                    pick_x = float(cx) + float(ox) + float(tip[0])
                    pick_y = float(cy) + float(oy) + float(tip[1])
                    # Même formule que prise initiale + plongée plus profonde si vide
                    pick_z = float(RIGHT_PICK_IK_Z) + float(oz) + float(tip[2])
                    deeper = float(globals().get("GRASP_EMPTY_DEEPER_Z", 0.025) or 0.025)
                    z_min = float(globals().get("GRASP_PICK_Z_MIN", -0.065) or -0.065)
                    pick_z = max(z_min, pick_z - deeper * float(rec))
                    log("[GRASP] recovery deeper plunge z=%.3f (rec=%d empty→mordre)",
                        pick_z, rec)
                    right_approach = [
                        pick_x, pick_y,
                        float(pick_z) + max(float(globals().get(
                            "WRIST_SHALLOW_PLUNGE", 0.045) or 0.045), 0.04),
                    ]
                    right_grasp = [pick_x, pick_y, pick_z]
                    right_pre = list(right_approach)
                    _move_hand(
                        arm, "right", right_approach, rq, log,
                        "recovery_approach_%d" % rec,
                        constraint_mode=RIGHT_PICK_IK_MODE, settle=0.45,
                    )
                    _move_right_cartesian_to(
                        arm, right_grasp, rq, log, "recovery_plunge_%d" % rec,
                        n_points=3, constraint_mode=RIGHT_GRASP_FINAL_IK_MODE,
                    ) or _move_hand(
                        arm, "right", right_grasp, rq, log,
                        "recovery_plunge_fb_%d" % rec,
                        constraint_mode=RIGHT_PICK_IK_MODE, settle=0.5,
                    )
                    obs_before_close = observe_hand(
                        cam, working, hand="right", log=log, settle=0.15)
                held, assess, rs, pos = _do_close_and_assess("recovery_%d" % rec)
                held2, latched = _await_right_hold(
                    claw, log, "recovery-%d" % rec)
                hold_latched = bool(latched or hold_latched)
                held = bool(held or held2 or hold_latched)
                if _claw_is_holding(claw, hold_latched):
                    log("[GRASP] RECOVERY OK manner=%s", assess.get("manner"))
                    _maintain_grasp_close(claw, log, "recovery-ok")
                    break
            if not _claw_is_holding(claw, hold_latched):
                log("[GRASP] PINCE VIDE après recovery — PAS de pesée")
                claw.right_open()
                claw.wait_until_done(timeout=2.0)
                try:
                    _move_right_cartesian_to(
                        arm, right_pre, rq, log, "claw_empty_raise",
                        n_points=3, constraint_mode=RIGHT_PICK_IK_MODE,
                    )
                except Exception as exc:
                    log("[GRASP] raise après pince vide skip: %s", exc)
                    try:
                        _move_hand(
                            arm, "right", right_pre, rq, log, "claw_empty_raise_fb",
                            constraint_mode=RIGHT_PICK_IK_MODE, settle=0.4,
                        )
                    except Exception as exc2:
                        log("[GRASP] raise fb skip: %s", exc2)
                return False
        elif keep_if_holding and _claw_is_holding(claw, hold_latched):
            log("[GRASP] TIEN déjà (manner=%s excellent=%s) — garde FERMÉ",
                assess.get("manner"), excellent)
            _maintain_grasp_close(claw, log, "keep-hold")
    else:
        log("[GRASP] claw-hold skip (focus VISION)")

    if _claw_is_holding(claw, hold_latched):
        _maintain_grasp_close(claw, log, "pre-lift")

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
        # Latch > LiDAR : GRABBED puis faux EMPTY (dxy≈0.03) ne doit JAMAIS ouvrir
        if hold_latched or claw.right_holding() or int(
                getattr(claw, "_right_state", -1)) == 3:
            log("[VERIFY] grasp latched — skip perception EMPTY (no drop)")
        elif not _verify_parcel_lifted_perception(
            lidar, cam, tf_reader, name, (pick_x, pick_y), log
        ):
            log("[GRASP] %s EMPTY après close — PAS de pesée", name)
            return _safe_abort_grasp_open(
                claw, arm, right_pre, rq, log, "empty_raise_open", hold_latched)
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
            return _safe_abort_grasp_open(
                claw, arm, right_pre, rq, log, "lift_ik_fail", hold_latched)

    claw.wait_until_done(timeout=2.0)
    rs = int(getattr(claw, "_right_state", -1))
    log("[GRASP] %s claw R=%d latched=%s", name, rs, hold_latched)
    # Pas de 2e detect_parcels complet ici : CPU/ROS crash observé mid-mission
    if not skip_claw:
        if _claw_is_holding(claw, hold_latched):
            if not claw.right_holding() and rs != 3:
                log("[GRASP] %s transport hold flicker — continue (latched)", name)
        elif not _claw_is_holding(claw, hold_latched):
            log("[GRASP] %s EMPTY au transport (claw) — abandon", name)
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
    Vérifie FK près de WEIGH_RELEASE_IK avant d'ouvrir — sinon pas de jaune.
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

    tol_xy = float(globals().get("WEIGH_RELEASE_TOL_XY", 0.04) or 0.04)
    tol_z = float(globals().get("WEIGH_RELEASE_TOL_Z", 0.06) or 0.06)
    retries = int(globals().get("WEIGH_RELEASE_PLACE_RETRIES", 3) or 3)
    placed = False
    for attempt in range(1, retries + 1):
        if not _move_hand(
            arm, "right", weigh, rq, log, "right_weigh_down_a%d" % attempt,
            constraint_mode=RIGHT_PICK_IK_MODE, settle=PICK_GRASP_MOVE_SLEEP,
        ):
            continue
        cur = _current_right_xyz(arm)
        if cur is None:
            log("[WEIGH] FK indisponible après descente — retry")
            continue
        dx = abs(float(cur[0]) - float(weigh[0]))
        dy = abs(float(cur[1]) - float(weigh[1]))
        dz = abs(float(cur[2]) - float(weigh[2]))
        ok = dx <= tol_xy and dy <= tol_xy and dz <= tol_z
        log("[WEIGH] place check act=(%.3f,%.3f,%.3f) tgt=(%.3f,%.3f,%.3f) "
            "Δ=(%.0f,%.0f,%.0f)mm → %s",
            cur[0], cur[1], cur[2], weigh[0], weigh[1], weigh[2],
            dx * 1000, dy * 1000, dz * 1000, "OK" if ok else "FAIL")
        if ok:
            placed = True
            break
        log("[WEIGH] hors pad (pas de jaune si on lâche ici) — retry %d/%d",
            attempt, retries)
        # Remonter un peu puis retenter
        _move_hand(
            arm, "right", right_transit, carry_quat, log,
            "right_weigh_retry_up_a%d" % attempt,
            constraint_mode=RIGHT_PICK_IK_MODE, settle=0.5,
        )
    if not placed:
        log("[WEIGH] ABORT — main pas sur la balance, NE PAS ouvrir (évite chute)")
        return False

    if not claw.right_holding() and int(getattr(claw, "_right_state", -1)) != 3:
        log("[WEIGH] ABORT — pince vide avant pose: %s", claw.describe_right())
        return False

    rospy.sleep(WEIGH_RELEASE_SETTLE)
    claw.right_open()
    claw.wait_until_done(timeout=3.0)
    rospy.sleep(GRIPPER_SETTLE_TIME)
    rospy.sleep(WEIGH_DWELL)
    log("[WEIGH] 已释放，等待称重 (dwell=%.1fs — zone doit devenir jaune)",
        float(WEIGH_DWELL))
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


def _stabilize_before_handoff(robot, log):
    """Stance sans go_home — droite tient encore le colis."""
    if robot is None:
        return
    log("[HANDOFF] stabilisation stance (pas de marche)")
    try:
        robot.stop()
        robot.switch_to_stance()
    except Exception as exc:
        log("[HANDOFF] stance skip: %s", exc)
    rospy.sleep(float(globals().get("HANDOFF_STANCE_SETTLE", 0.8) or 0.8))


def handoff_to_left(arm, claw, log, cam=None, parcel_name=None, tf_reader=None,
                    robot=None):
    """
    Passation orga :
      droite raise → xy align → handoff IK (lock joints droite)
      gauche XZ puis Y avec droite figée → close L / open R → retract.
    """
    log("[HANDOFF] 右手交给左手 (orga lock-right)")
    _stabilize_before_handoff(robot, log)
    claw.left_open()
    rospy.sleep(0.2)
    try:
        claw.right_close()
        claw.wait_until_done(timeout=2.0)
    except Exception as exc:
        log("[HANDOFF] maintain R close skip: %s", exc)
    rq = RIGHT_HANDOFF_QUAT
    right_final = list(RIGHT_HANDOFF_IK)
    left_settle = float(globals().get("HANDOFF_LEFT_MOVE_SETTLE", 1.5) or 1.5)

    locked_right = None
    if bool(globals().get("HANDOFF_RIGHT_VERIFY", True)):
        ok_r, locked_right = _ensure_right_at_handoff(arm, claw, log)
        if not ok_r or locked_right is None:
            log("[HANDOFF] ABORT — D pas à pose handoff stable (G ne bouge pas)")
            return False
        log("[HANDOFF] droite verrouillée @ handoff OK — approche gauche")
    else:
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
            raise_pt = [cur[0], cur[1], float(tz)]
            if not _move_hand(
                arm, "right", raise_pt, rq, log, "right_handoff_raise_z%.2f" % tz,
                constraint_mode=IK_MODE_THREE_POINT_MIXED, settle=PICK_ALIGN_MOVE_SLEEP,
            ):
                continue
            align_pt = [right_final[0], right_final[1], float(tz)]
            if not _move_hand(
                arm, "right", align_pt, rq, log, "right_handoff_xy_z%.2f" % tz,
                constraint_mode=IK_MODE_THREE_POINT_MIXED, settle=PICK_ALIGN_MOVE_SLEEP,
            ):
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
        locked_right = list(arm._last_cmd_deg[7:14])
        log("[HANDOFF] droite verrouillée — approche gauche")

    use_mirror = bool(globals().get("HANDOFF_USE_WRIST_MIRROR", True))
    # Toujours FK réelle de D si dispo (= vraie symétrie sur la pose actuelle)
    # Preset = fallback seulement si FK rate.
    if use_mirror:
        r_xyz, r_quat = _current_right_pose(arm)
        if r_xyz is None or r_quat is None:
            log("[HANDOFF] FK D indisponible — miroir depuis preset RIGHT_HANDOFF")
            left_xz, left_recv, llq = _mirror_left_target_from_right(
                list(RIGHT_HANDOFF_IK), list(RIGHT_HANDOFF_QUAT), log=log)
        else:
            left_xz, left_recv, llq = _mirror_left_target_from_right(
                r_xyz, r_quat, log=log)
    else:
        left_xz = list(LEFT_HANDOFF_RECEIVE_XZ_READY_IK)
        left_recv = list(LEFT_HANDOFF_RECEIVE_IK)
        llq = LEFT_HANDOFF_RECEIVE_QUAT
    # Fallbacks si bras déjà déplacé (2e colis) — IK gauche hors workspace
    left_offsets = [
        (0.0, 0.0, 0.0),
        (0.0, 0.03, 0.03),
        (0.02, 0.05, 0.05),
        (-0.02, 0.04, 0.02),
        (0.0, 0.08, 0.08),
    ]
    left_latched = False
    grab_retries = int(globals().get("HANDOFF_LEFT_GRAB_RETRIES", 3) or 3)
    hold_timeout = float(globals().get("HANDOFF_LEFT_HOLD_TIMEOUT", 2.5) or 2.5)
    require_left = bool(globals().get("HANDOFF_REQUIRE_LEFT_HOLD", True))
    left_held = False

    for attempt in range(1, grab_retries + 1):
        # Remettre gauche ouverte entre essais
        try:
            claw.left_open()
            claw.wait_until_done(timeout=2.0)
        except Exception as exc:
            log("[HANDOFF] left open skip: %s", exc)

        left_ok = False
        for dx, dy, dz in left_offsets:
            xz = [left_xz[0] + dx, left_xz[1] + dy, left_xz[2] + dz]
            recv = [left_recv[0] + dx, left_recv[1] + dy, left_recv[2] + dz]
            if dx or dy or dz:
                log("[HANDOFF] left retry dxyz=(%.2f,%.2f,%.2f) attempt=%d",
                    dx, dy, dz, attempt)
            if not _maintain_right_close(claw, log, "pre-left-move-%d" % attempt):
                log("[HANDOFF] ABORT — droite a lâché avant approche gauche")
                return False
            if not _move_left_keep_right(
                arm, xz, llq, locked_right, log, "left_receive_xz",
                constraint_mode=IK_MODE_THREE_POINT_MIXED, settle=left_settle,
            ):
                continue
            y_mid_step = bool(globals().get("HANDOFF_LEFT_Y_MID_STEP", True))
            if y_mid_step and abs(float(recv[1]) - float(xz[1])) > 0.04:
                mid_y = [recv[0], (float(xz[1]) + float(recv[1])) * 0.5, recv[2]]
                if not _move_left_keep_right(
                    arm, mid_y, llq, locked_right, log, "left_receive_mid_y",
                    constraint_mode=IK_MODE_THREE_POINT_MIXED, settle=left_settle,
                ):
                    continue
            if not _move_left_keep_right(
                arm, recv, llq, locked_right, log, "left_receive_y",
                constraint_mode=IK_MODE_THREE_POINT_MIXED, settle=left_settle,
            ):
                continue
            left_ok = True
            break
        if not left_ok:
            log("[HANDOFF] approche gauche IK fail (attempt %d/%d)",
                attempt, grab_retries)
            continue

        settle = float(globals().get("HANDOFF_COINCIDENCE_SETTLE", 0.8) or 0.8)
        log("[HANDOFF] coïncidence settle=%.1fs — puis L-wrist (attempt %d/%d)",
            settle, attempt, grab_retries)
        rospy.sleep(settle)

        # Cam poignet G : correction légère seulement (miroir poignet D fait le gros du travail)
        vision_ok = True
        recv_now = list(recv) if left_ok else list(left_recv)
        light_servo = bool(globals().get("HANDOFF_LEFT_LIGHT_SERVO", True))
        if light_servo and cam is not None:
            vision_ok, recv_now = _left_handoff_light_correct(
                arm, locked_right, llq, cam, parcel_name, recv_now, log,
                max_iters=int(globals().get("HANDOFF_LEFT_LIGHT_SERVO_ITERS", 2) or 2),
                dpix_thresh=float(globals().get("HANDOFF_LEFT_CORRECT_DPIX", 70) or 70),
            )
        elif (bool(globals().get("HANDOFF_LEFT_WRIST_SERVO", False))
              and cam is not None):
            vision_ok, recv_now = _left_handoff_wrist_lock(
                arm, locked_right, llq, cam, tf_reader,
                parcel_name, recv_now, log,
            )
            if not vision_ok and bool(globals().get(
                    "HANDOFF_LEFT_REQUIRE_VISION", False)):
                log("[HANDOFF] L-wrist FAIL — pas centrée, PAS de close, retry")
                try:
                    claw.left_open()
                    claw.wait_until_done(timeout=2.0)
                except Exception:
                    pass
                continue
        elif (bool(globals().get("HANDOFF_LEFT_WRIST_CHECK", True))
                and cam is not None and parcel_name):
            tgt = {"name": parcel_name, "center": list(recv_now)}
            obs_l = observe_hand(cam, tgt, hand="left", log=log, settle=0.25)
            log("[HANDOFF] left-wrist seen=%s aim=%s zone=%s area=%d Δpx=%.0f",
                obs_l.get("seen"), obs_l.get("aim"), obs_l.get("grip_zone"),
                int(obs_l.get("area", 0)), float(obs_l.get("dpix", -1)))
            log_wrist_event(
                "handoff_left_observe", log=log, name=parcel_name,
                seen=bool(obs_l.get("seen")), aim=obs_l.get("aim"),
                grip_zone=obs_l.get("grip_zone"),
                area=int(obs_l.get("area", 0)),
                dpix=float(obs_l.get("dpix", -1)),
                attempt=attempt,
            )
            require_seen = bool(globals().get(
                "HANDOFF_LEFT_REQUIRE_SEEN_FOR_CLOSE", True))
            max_dpix = float(globals().get(
                "HANDOFF_LEFT_CLOSE_MAX_DPIX", 120) or 120)
            if require_seen:
                dpix = float(obs_l.get("dpix", 999))
                zone = str(obs_l.get("grip_zone", "") or "")
                vision_ok = (
                    bool(obs_l.get("seen"))
                    and dpix <= max_dpix
                    and zone not in ("outside", ""))
                if not vision_ok:
                    log("[HANDOFF] vision G insuffisante (seen=%s Δpx=%.0f "
                        "zone=%s) — PAS de close",
                        obs_l.get("seen"), dpix, zone)

        if not vision_ok:
            log("[HANDOFF] skip close gauche — vision/receive pas prête "
                "(attempt %d/%d)", attempt, grab_retries)
            try:
                claw.left_open()
                claw.wait_until_done(timeout=2.0)
            except Exception:
                pass
            continue

        if not claw.right_holding():
            log("[HANDOFF] ABORT — DROITE a lâché avant close gauche: %s",
                claw.describe_right())
            return False

        _maintain_right_close(claw, log, "pre-left-close-%d" % attempt)
        log("[HANDOFF] tip CLOSE gauche (attempt %d) — DROITE maintient",
            attempt)
        claw.left_close()
        if not claw.wait_until_done(timeout=3.0):
            log("[HANDOFF] left close timeout")
        rospy.sleep(GRIPPER_CLOSE_HOLD)
        _maintain_right_close(claw, log, "mid-left-close-%d" % attempt)
        claw.left_close()
        claw.wait_until_done(timeout=2.0)
        _maintain_right_close(claw, log, "post-left-close-%d" % attempt)
        for _ in range(25):
            if int(getattr(claw, "_left_state", -1)) != 1:
                break
            rospy.sleep(0.08)

        # held=True seulement si L=state3 confirmé N fois — latched seul = flicker, IGNORÉ
        held, latched = _await_left_grabbed_strict(
            claw, log, "post-close-L-%d" % attempt, timeout=hold_timeout)
        ls = int(getattr(claw, "_left_state", -1))
        pos_l = float(getattr(claw, "_left_pos", -1.0))
        if hasattr(claw, "describe_left"):
            left_desc = claw.describe_left()
        else:
            left_desc = "L=%d pos=%.0f" % (ls, pos_l)
        log("[HANDOFF] left after close %s held=%s latched=%s "
            "(open R seulement si held=True)",
            left_desc, held, latched)
        log_wrist_event(
            "handoff_left_close", log=log, name=parcel_name or "?",
            claw_state=ls, claw_pos=pos_l, held=bool(held),
            attempt=attempt, vision_ok=bool(vision_ok),
        )

        if held:
            if not claw.right_holding():
                log("[HANDOFF] ABORT — DROITE lâchée alors que G Grabbed")
                return False
            left_held = True
            left_latched = True
            log("[HANDOFF] GAUCHE L=GRABBED confirmé — prêt open DROITE")
            break

        log("[HANDOFF] gauche PAS Grabbed (held=False latched=%s) — "
            "NE PAS ouvrir droite, retry", latched)
        if not claw.right_holding():
            log("[HANDOFF] ABORT — DROITE a lâché pendant échec G: %s",
                claw.describe_right())
            return False
        try:
            claw.left_open()
            claw.wait_until_done(timeout=2.0)
        except Exception:
            pass

    if require_left and not left_held:
        log("[HANDOFF] ABORT — gauche n'a jamais Grabbed, DROITE reste FERMÉE")
        log_wrist_event(
            "handoff_abort_no_left_hold", log=log,
            name=parcel_name or "?", retries=grab_retries,
        )
        return False

    if not left_held:
        log("[HANDOFF] WARN require_left=False — open R sans Grabbed L (risqué)")

    both = float(globals().get("HANDOFF_BOTH_HOLD_BEFORE_OPEN_R", 1.0) or 1.0)
    if not _handoff_dual_squeeze(claw, log, both):
        log("[HANDOFF] ABORT — maintien double prise échoué, DROITE FERMÉE")
        return False

    # Re-vérifie L=3 juste avant open — pas de confiance au flag seul
    if not _safe_open_right_after_left(claw, log, left_held):
        log("[HANDOFF] ABORT — open R refusé (G pas Grabbed ou D perdue)")
        return False

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
        log("[HANDOFF] WARN recul droite IK fail — G tient, on passe au bac")

    log("[HANDOFF] 交接完成 — D ouverte, G porte colis → bac")
    return True


def place_in_box_right(arm, claw, parcel_name, log):
    """
    Dépose au bac avec la MAIN DROITE seule — skip handoff.
    Hover au-dessus du trou (pas de contact box), ouvre, laisse tomber.
    """
    base = list(globals().get("RIGHT_BOX_DROP_BASE_IK", BOX_DROP_BASE_IK) or BOX_DROP_BASE_IK)
    # Offset par colis si défini (même table que bac gauche)
    off = BOX_DROP_OFFSET_BY_PARCEL.get(parcel_name, [0.0, 0.0, 0.0])
    base = [base[0] + off[0], base[1] + off[1], base[2] + off[2]]
    hover_z = float(globals().get("RIGHT_BOX_DROP_HOVER_Z", BOX_DROP_HOVER_Z) or BOX_DROP_HOVER_Z)
    x_tries = list(globals().get("RIGHT_BOX_DROP_IK_X_TRIES", [0.0, -0.04, 0.04]) or [0.0])
    y_tries = list(globals().get("RIGHT_BOX_DROP_IK_Y_TRIES", [0.0, 0.04, -0.04]) or [0.0])
    rq = RIGHT_HANDOFF_QUAT
    log("[BOX] RIGHT drop-from-above %s (hover z=%.2f, skip handoff)",
        parcel_name, hover_z)

    cur = _current_right_xyz(arm)
    if cur is not None:
        # Lever d'abord (évite frottement table/balance)
        raise_z = max(float(cur[2]) + 0.08, 0.35, hover_z - 0.05)
        if not _move_hand(
            arm, "right", [cur[0], cur[1], raise_z], rq, log, "right_box_raise",
            constraint_mode=IK_MODE_THREE_POINT_MIXED, settle=PICK_ALIGN_MOVE_SLEEP,
        ):
            log("[BOX] RIGHT raise fail — on tente hover quand même")

    for dx in x_tries:
        for dy in y_tries:
            hover = [base[0] + dx, base[1] + dy, hover_z]
            log("[BOX] RIGHT hover try xyz=(%.3f, %.3f, %.3f)",
                hover[0], hover[1], hover[2])
            if not _move_hand(
                arm, "right", hover, rq, log, "right_box_hover",
                constraint_mode=IK_MODE_THREE_POINT_MIXED,
                settle=PICK_ALIGN_MOVE_SLEEP,
            ):
                continue
            claw.right_open()
            claw.wait_until_done(timeout=3.0)
            rospy.sleep(GRIPPER_SETTLE_TIME)
            rospy.sleep(PLACE_DWELL)
            log("[BOX] RIGHT 已释放 (hover @ %.2f,%.2f,%.2f)",
                hover[0], hover[1], hover[2])
            # Reculer un peu pour dégager
            claw.right_open()
            retreat = [hover[0] - 0.08, hover[1] - 0.12, hover_z]
            _move_hand(
                arm, "right", retreat, rq, log, "right_box_retreat",
                constraint_mode=IK_MODE_THREE_POINT_MIXED, settle=0.4,
            )
            return True

    log("[BOX] RIGHT hover IK all failed")
    return False


def place_in_box(arm, claw, parcel_name, log):
    """
    Après handoff : main gauche porte le colis → hover bac → ouvrir.
    NE PAS toucher la box (sinon elle bascule).
    """
    hover_z = float(BOX_DROP_HOVER_Z)
    transit_z = float(globals().get("LEFT_BOX_TRANSIT_Z", 0.42) or 0.42)
    log("[BOX] gauche → bac %s (transit z=%.2f, hover z=%.2f)",
        parcel_name, transit_z, hover_z)

    held, _ = _await_left_hold(claw, log, "pre-box", timeout=1.0)
    if not held:
        log("[BOX] ABORT — gauche ne tient pas avant dépôt bac")
        return False

    left_settle = float(globals().get("HANDOFF_LEFT_MOVE_SETTLE", 1.5) or 1.5)

    base = _box_drop_ik(parcel_name)
    x_tries = [0.0, -0.03, 0.03, -0.06]
    y_tries = [0.0, 0.03, -0.03, 0.06]
    llq = LEFT_BOX_DROP_QUAT

    locked_right = list(arm._last_cmd_deg[7:14]) if len(getattr(arm, "_last_cmd_deg", [])) >= 14 else None
    if locked_right is None:
        log("[BOX] pas de joints droite — fallback dual IK hover")
        return _place_in_box_dual(arm, claw, parcel_name, log)

    cur = _current_left_xyz(arm)
    if cur is not None:
        raise_z = max(float(cur[2]) + 0.08, transit_z, hover_z - 0.12)
        if not _move_left_keep_right(
            arm, [cur[0], cur[1], raise_z], llq, locked_right, log, "left_box_raise",
            constraint_mode=IK_MODE_THREE_POINT_MIXED, settle=left_settle,
        ):
            log("[BOX] WARN lever gauche fail — on tente trajet bac quand même")
        else:
            transit_z = raise_z

    for dx in x_tries:
        for dy in y_tries:
            bx = base[0] + dx
            by = base[1] + dy
            transit_ik = [bx, by, transit_z]
            hover_ik = [bx, by, hover_z]
            log("[BOX] try transit=(%.3f,%.3f,%.3f) hover=(%.3f,%.3f,%.3f)",
                transit_ik[0], transit_ik[1], transit_ik[2],
                hover_ik[0], hover_ik[1], hover_ik[2])
            if not _move_left_keep_right(
                arm, transit_ik, llq, locked_right, log, "left_box_transit",
                constraint_mode=IK_MODE_THREE_POINT_MIXED, settle=left_settle,
            ):
                continue
            if not _move_left_keep_right(
                arm, hover_ik, llq, locked_right, log, "left_box_hover",
                constraint_mode=IK_MODE_THREE_POINT_MIXED, settle=left_settle,
            ):
                continue
            claw.left_open()
            claw.wait_until_done(timeout=3.0)
            rospy.sleep(GRIPPER_SETTLE_TIME)
            rospy.sleep(PLACE_DWELL)
            log("[BOX] 已释放 (gauche hover @ %.2f,%.2f,%.2f — no box contact)",
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


def process_parcel_after_grasp(arm, claw, parcel, log, cam=None, tf_reader=None,
                               robot=None):
    """Après saisie : pesée → (handoff|droite) → bac."""
    name = parcel["name"]
    skip_handoff = bool(globals().get("SKIP_HANDOFF", False))

    def _to_box():
        if skip_handoff:
            log("[BOX] SKIP_HANDOFF — droite pose dans bac (%s)", name)
            return place_in_box_right(arm, claw, name, log)
        if not handoff_to_left(
                arm, claw, log, cam=cam, parcel_name=name, tf_reader=tf_reader,
                robot=robot):
            return False
        log("[PARCEL] %s handoff OK — gauche dépose dans le bac", name)
        return place_in_box(arm, claw, name, log)

    if SKIP_WEIGH or TRAIN_PICK_BOX:
        log("[TRAIN] skip weigh — %s → bac (%s)",
            "droite" if skip_handoff else "handoff", name)
        if not _to_box():
            return False
        log("[PARCEL] %s train pick→box OK", name)
        return True
    if not place_on_weighing_area(arm, claw, log):
        return False
    if not regrasp_from_weighing(arm, claw, log):
        return False
    if not _to_box():
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
    elif bool(globals().get("SKIP_HANDOFF", False)):
        log("场景一：pesée → bac DROITE (SKIP_HANDOFF, pas de passation)")
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
    _log_pose_snapshot(arm, log, "go_home")
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
            max_fail = int(globals().get("GRASP_PARCEL_MAX_FAILS", 5) or 5)
            if FORCE_PARCEL_NAME:
                max_fail = max(max_fail, 5)
            log("[LOOP] 抓取失败 %s (%d/%d)，跳过本轮",
                name, grasp_failures[name], max_fail)
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
        if process_parcel_after_grasp(
                arm, claw, target, log, cam=cam, tf_reader=tf_reader,
                robot=robot):
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
