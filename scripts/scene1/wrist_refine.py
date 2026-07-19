#!/usr/bin/env python3
"""
Peaufinage prise Scene1 via caméra MAIN droite (cam_r).

Flux :
  1) Bras déjà au-dessus du colis (appelant)
  2) capture_once(camera=right) via perception.py
  3) Si même colis vu près de la pose tête → met à jour pick_ik
  4) Sinon garde la pose tête (pas de saut)

Active par défaut. Désactiver : SCENE1_ENABLE_WRIST_REFINE=0
"""
from __future__ import print_function

import importlib.util
import math
import os
import sys


# Max correction XY (m) — servo doux, pas de saut
WRIST_MAX_DELTA_XY = 0.04
# Accepte une det du même colis dans ce rayon autour de la pose tête
WRIST_MATCH_XY_M = 0.12
# Look-down : rester à cette hauteur pendant la capture
WRIST_LOOK_Z = 0.32
WRIST_SETTLE_SEC = 0.40


def _scene1_dir():
    # .../scene1/wrist_refine.py → .../scene1
    return os.path.dirname(os.path.abspath(__file__))


def _load_scene1_perception():
    """Charge scene1.perception (RGB-D)."""
    try:
        from scene1 import perception as mod
        return mod
    except Exception:
        pass
    path = os.path.join(_scene1_dir(), "perception.py")
    if not os.path.isfile(path):
        raise RuntimeError("perception.py introuvable (wrist): %s" % path)
    spec = importlib.util.spec_from_file_location("scene1_perception_wrist", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _xy_dist(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _capture_wrist(output_dir="/tmp/scene1_wrist", target_frame="base_link"):
    """Lance la perception RGB-D sur cam_r (main droite)."""
    import argparse
    import rospy

    perception = _load_scene1_perception()
    args = argparse.Namespace(
        camera="right",
        target_frame=target_frame,
        output_dir=output_dir,
        timeout=8.0,
        tf_timeout=0.8,
        min_area=80.0,
        max_area=90000.0,
        min_depth=0.12,
        max_depth=0.90,
    )
    os.makedirs(output_dir, exist_ok=True)
    rospy.loginfo("[WRIST] capture cam_r (perception)…")
    return perception.capture_once(args)


def _match_parcel_detection(detections, parcel_name, ref_xyz, max_xy=WRIST_MATCH_XY_M):
    """Choisit la det du même colis la plus proche de ref_xyz."""
    best = None
    best_d = 1e9
    for det in detections or []:
        name = det.get("name") or det.get("class")
        if name != parcel_name:
            continue
        xyz = det.get("base_link_xyz_m")
        if xyz is None or len(xyz) < 2:
            continue
        d = _xy_dist(xyz, ref_xyz)
        if d < best_d and d <= float(max_xy):
            best_d = d
            best = det
    return best, best_d


def refine_job_pick_with_wrist(sc1, arm_pub, arm_hold, job, pick_quat,
                               output_dir="/tmp/scene1_wrist"):
    """
    Au-dessus du colis : regarde avec cam_r + perception, corrige pick_ik.

    Returns job (éventuellement modifié). Ne lève jamais — fallback silencieux.
    """
    import rospy

    if os.environ.get("SCENE1_ENABLE_WRIST_REFINE", "1") != "1":
        rospy.loginfo("[WRIST] désactivé (SCENE1_ENABLE_WRIST_REFINE=0)")
        return job

    name = job.get("object")
    old_ik = list(job["right_pick_ik"])
    ref_xy = list(old_ik[:2])

    # S'assurer d'être en hauteur regard (abri + vue main)
    look_z = max(
        float(WRIST_LOOK_Z),
        float(getattr(sc1, "RIGHT_PICK_TRANSIT_IK_Z", WRIST_LOOK_Z) or WRIST_LOOK_Z),
    )
    look = [float(old_ik[0]), float(old_ik[1]), look_z]
    q = list(pick_quat)
    try:
        sc1._move_right_cartesian_to(
            arm_pub, arm_hold, look, q, "%s_wrist_look" % name,
            n_points=10, seg_time=0.55,
        )
        rospy.sleep(float(WRIST_SETTLE_SEC))
    except Exception as exc:
        rospy.logwarn("[WRIST] %s look fail: %s — skip refine", name, exc)
        return job

    try:
        result = _capture_wrist(output_dir=output_dir)
    except Exception as exc:
        rospy.logwarn("[WRIST] %s capture fail: %s — garde pose tête", name, exc)
        return job

    det, dist = _match_parcel_detection(
        result.get("detections") or [], name, ref_xy,
    )
    if det is None:
        rospy.logwarn(
            "[WRIST] %s pas vu cam_r (near tête) — garde pick_ik=%s found=%s",
            name,
            [round(v, 3) for v in old_ik],
            result.get("found_parcels"),
        )
        return job

    bx, by, bz = [float(v) for v in det["base_link_xyz_m"]]
    # Offset tip orga (FK ≠ TCP) — même idée que jobs_from_detections
    off = list(sc1.WORLD_TO_IK_OFFSET)
    source_world = [bx - float(off[0]), by - float(off[1]), bz - float(off[2])]
    if job.get("source_world") is not None:
        source_world[2] = float(job["source_world"][2])
    offset = list(sc1._right_pick_offset_for_parcel(name, source_world))

    new_ik = [
        bx + float(offset[0]),
        by + float(offset[1]),
        float(sc1.RIGHT_PICK_IK_Z) + float(offset[2]),
    ]
    # Clamp Δxy — peaufinage, pas téléport
    dx = float(new_ik[0]) - float(old_ik[0])
    dy = float(new_ik[1]) - float(old_ik[1])
    dxy = math.hypot(dx, dy)
    if dxy > float(WRIST_MAX_DELTA_XY):
        scale = float(WRIST_MAX_DELTA_XY) / max(dxy, 1e-6)
        new_ik[0] = float(old_ik[0]) + dx * scale
        new_ik[1] = float(old_ik[1]) + dy * scale
        rospy.loginfo(
            "[WRIST] %s clamp Δxy %.1f→%.1f cm",
            name, dxy * 100.0, WRIST_MAX_DELTA_XY * 100.0,
        )

    out = dict(job)
    out["right_pick_ik"] = list(new_ik)
    out["wrist_refined"] = True
    out["wrist_det_xy"] = [bx, by]
    out["wrist_delta_xy"] = math.hypot(
        float(new_ik[0]) - float(old_ik[0]),
        float(new_ik[1]) - float(old_ik[1]),
    )
    perc = dict(out.get("perception") or {})
    perc["wrist_base_link_xyz_m"] = [bx, by, bz]
    perc["wrist_pixel"] = det.get("pixel")
    out["perception"] = perc

    rospy.loginfo(
        "[WRIST] %s peaufiné dist=%.1fcm Δ=%.1fcm pick_ik %s → %s",
        name, dist * 100.0, out["wrist_delta_xy"] * 100.0,
        [round(v, 3) for v in old_ik],
        [round(v, 3) for v in new_ik],
    )
    return out
