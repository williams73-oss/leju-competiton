#!/usr/bin/env python3
"""
Vision caméra MAIN (poignet RGB-D) — Scene 1.

Architecture :
  Tête/LiDAR  = zone + pose GROSSIÈRE (déjà faite avant d'appeler ici)
  Caméra main = peaufinage TEMPS RÉEL : couleur + position dans l'image

Priorité = vision parfaite (centrer le colis sous la pince).
État pince (Grabbed) = autre sujet, traité après.

Outils utilisés :
  - RGB  : masque couleur LAB/HSV du colis cible
  - Depth: filtre les pixels trop proches/loins (rejette décor)
  - Servo relatif : erreur pixel → petit Δxy (pas de saut 20 cm)
"""
from __future__ import print_function

import math
import os
import sys

import numpy as np
import rospy

_scripts = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

from scene1.config import (  # noqa: E402
    LAB_COLOR_DIST_BY_NAME,
    LAB_COLOR_DIST_MAX,
    PARCEL_HSV_FALLBACK,
    PARCEL_REF_COLORS,
    TABLE_LAB_DIST_MAX,
    TABLE_PARCEL_Z,
    TABLE_REF_BGR,
    WRIST_ACCEPT_PX,
    WRIST_ALLOW_RAY,
    WRIST_CENTER_BIAS,
    WRIST_CLOSE_MAX_PIXEL_FRAC,
    WRIST_DEPTH_Z_MAX,
    WRIST_DEPTH_Z_MIN,
    WRIST_LAB_BOOST,
    WRIST_MAX_BLOB_FRAC,
    WRIST_MAX_DELTA_XY,
    WRIST_MIN_PIXELS,
    WRIST_ROI_FRAC,
    WRIST_SERVO_GAIN,
    WRIST_SERVO_SIGN_X,
    WRIST_SERVO_SIGN_Y,
    WRIST_SETTLE,
    WRIST_YAW_ENABLE,
    WRIST_YAW_SNAP_SQUARE,
)

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def _log(log, msg, *args):
    if log is not None:
        log(msg, *args)


# ---------------------------------------------------------------------------
# Couleur / frames
# ---------------------------------------------------------------------------

def _ref_bgr_for_name(name):
    for n, _label, bgr in PARCEL_REF_COLORS:
        if n == name:
            return bgr
    return None


def _hsv_range_for_name(name):
    for n, _label, lo, hi in PARCEL_HSV_FALLBACK:
        if n == name:
            return lo, hi
    return None


def _bgr_to_lab(ref_bgr):
    swatch = np.uint8([[list(ref_bgr)]])
    return cv2.cvtColor(swatch, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def _lab_mask(bgr, ref_bgr, dist_max):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = _bgr_to_lab(ref_bgr)
    dist = np.linalg.norm(lab - ref.reshape(1, 1, 3), axis=2)
    return (dist < float(dist_max)).astype(np.uint8) * 255, dist


def _hsv_mask(bgr, lo, hi):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Poignet : S bas fréquent (ombres) → assouplir S min un peu
    lo2 = [int(lo[0]), max(0, int(lo[1]) - 10), max(0, int(lo[2]) - 20)]
    hi2 = [int(hi[0]), 255, 255]
    return cv2.inRange(hsv, np.array(lo2, dtype=np.uint8), np.array(hi2, dtype=np.uint8))


def _get_wrist_frames(cam, hand):
    if hand == "left":
        return cam.get_left_wrist_rgb(), cam.get_left_wrist_depth(), "left"
    return cam.get_right_wrist_rgb(), cam.get_right_wrist_depth(), "right"


def _color_mask_for_target(bgr, name, depth=None, log=None):
    """
    Masque couleur poignet :
      LAB (modéré) → soft seulement si vide → HSV si sparse → table →
      si saturé (>20% image) resserrer LAB → depth soft.
    """
    ref = _ref_bgr_for_name(name)
    if ref is None:
        return None

    h, w = bgr.shape[:2]
    img_n = float(max(h * w, 1))
    sat_frac = float(globals().get("WRIST_MASK_SAT_FRAC", 0.20) or 0.20)

    base = float(LAB_COLOR_DIST_BY_NAME.get(name, LAB_COLOR_DIST_MAX))
    boost = float(globals().get("WRIST_LAB_BOOST", 14.0) or 14.0)
    soft_extra = float(globals().get("WRIST_LAB_SOFT_EXTRA", 8.0) or 8.0)
    thr = base + boost

    mask, dist_map = _lab_mask(bgr, ref, thr)
    n_lab = int(np.count_nonzero(mask))
    dmin = float(dist_map.min()) if dist_map.size else 999.0

    # Soft seulement si presque RIEN (évite area=260k)
    if n_lab < int(WRIST_MIN_PIXELS):
        thr2 = thr + soft_extra
        mask2, _ = _lab_mask(bgr, ref, thr2)
        n2 = int(np.count_nonzero(mask2))
        if n2 > n_lab:
            _log(log, "[HAND] %s LAB soft thr=%.0f→%.0f px %d→%d (dmin=%.1f)",
                 name, thr, thr2, n_lab, n2, dmin)
            mask, n_lab, thr = mask2, n2, thr2

    # HSV : seulement pour sauver un masque trop maigre (pas pour le gorgé)
    hsv_only_sparse = bool(globals().get("WRIST_HSV_ONLY_IF_SPARSE", True))
    hsv_rng = _hsv_range_for_name(name)
    if hsv_rng is not None:
        if (not hsv_only_sparse) or n_lab < max(200, int(WRIST_MIN_PIXELS) * 3):
            hm = _hsv_mask(bgr, hsv_rng[0], hsv_rng[1])
            mask = cv2.bitwise_or(mask, hm)
            n_lab = int(np.count_nonzero(mask))

    # Exclude table — plus strict au poignet
    if bool(globals().get("WRIST_USE_TABLE_EXCLUDE", True)):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        table_lab = _bgr_to_lab(TABLE_REF_BGR)
        table_dist = np.linalg.norm(lab - table_lab.reshape(1, 1, 3), axis=2)
        scale = float(globals().get("WRIST_TABLE_EXCLUDE_SCALE", 1.15) or 1.15)
        thr_table = float(TABLE_LAB_DIST_MAX) * scale
        non_table = (table_dist > thr_table).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, non_table)

    n_rgb = int(np.count_nonzero(mask))

    # Saturation : resserrer LAB (log area≈260k)
    if n_rgb / img_n > sat_frac:
        thr_t = max(base + 4.0, thr - 12.0)
        mask_t, _ = _lab_mask(bgr, ref, thr_t)
        if bool(globals().get("WRIST_USE_TABLE_EXCLUDE", True)):
            mask_t = cv2.bitwise_and(mask_t, non_table)
        n_t = int(np.count_nonzero(mask_t))
        if n_t >= int(WRIST_MIN_PIXELS):
            _log(log, "[HAND] %s masque saturé %.0f%% → tighten thr=%.0f px %d→%d",
                 name, 100.0 * n_rgb / img_n, thr_t, n_rgb, n_t)
            mask, n_rgb, thr = mask_t, n_t, thr_t

    # Depth soft
    depth_soft = bool(globals().get("WRIST_DEPTH_SOFT", True))
    if depth is not None and depth.shape[:2] == mask.shape[:2]:
        d = depth.astype(np.float32)
        valid = np.isfinite(d) & (d >= float(WRIST_DEPTH_Z_MIN)) & (d <= float(WRIST_DEPTH_Z_MAX))
        if int(np.count_nonzero(valid)) > 200:
            masked = mask.copy()
            masked[~valid] = 0
            n_d = int(np.count_nonzero(masked))
            if n_d >= max(int(WRIST_MIN_PIXELS), int(0.25 * max(n_rgb, 1))):
                mask = masked
            elif depth_soft and n_rgb >= int(WRIST_MIN_PIXELS):
                _log(log, "[HAND] %s depth coupe %d→%d — garde RGB-only",
                     name, n_rgb, n_d)

    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)

    n_final = int(np.count_nonzero(mask))
    if n_final < int(WRIST_MIN_PIXELS):
        _log(log, "[HAND] %s masque faible px=%d (LAB dmin=%.1f thr=%.0f)",
             name, n_final, dmin, thr)
    elif n_final / img_n > sat_frac:
        _log(log, "[HAND] %s masque encore gros px=%d (%.0f%%)",
             name, n_final, 100.0 * n_final / img_n)
    return mask


def _image_aim(cam, cam_key, rgb_shape):
    """
    Point visé = projection tip pince (pas le centre optique brut).
    Biais calibré logs seed30 : gros blob avec Δpx~200–240 = tip à côté.
    """
    h, w = int(rgb_shape[0]), int(rgb_shape[1])
    info = cam.get_camera_info(cam_key) if cam is not None else None
    if info is not None:
        cx = float(info["cx"])
        cy = float(info["cy"])
    else:
        cx, cy = 0.5 * w, 0.5 * h
        info = None
    try:
        from scene1.config import WRIST_AIM_BIAS_U, WRIST_AIM_BIAS_V
        cx += float(WRIST_AIM_BIAS_U)
        cy += float(WRIST_AIM_BIAS_V)
    except Exception:
        pass
    cx = max(0.0, min(float(w - 1), cx))
    cy = max(0.0, min(float(h - 1), cy))
    return cx, cy, info


def _best_blob_near_center(mask, min_px, aim_xy=None):
    """
    Blob du colis pour la pince.
    Cherche d'abord près du tip (aim), sinon meilleure tache dans toute l'image
    (cas fréquent : bras arrêté À CÔTÉ du colis).
    """
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    h, w = mask.shape[:2]
    if aim_xy is not None:
        cx_i, cy_i = float(aim_xy[0]), float(aim_xy[1])
    else:
        cx_i, cy_i = 0.5 * w, 0.5 * h
    img_area = float(max(h * w, 1))
    max_area = img_area * float(WRIST_MAX_BLOB_FRAC)
    roi = 0.5 * float(WRIST_ROI_FRAC)
    best_i, best_score = None, -1.0

    def _score_blob(i, prefer_near):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_px or area > max_area:
            return -1.0
        ux, uy = float(centroids[i][0]), float(centroids[i][1])
        if prefer_near:
            if abs(ux - cx_i) > roi * w or abs(uy - cy_i) > roi * h:
                return -1.0
        dist_n = math.hypot((ux - cx_i) / w, (uy - cy_i) / h)
        # Préférer taille colis réelle (~1–8% image), pas le décor saturé
        ideal = 0.04 * img_area
        size_term = float(area) * math.exp(-abs(math.log((area + 1.0) / ideal)) * 0.35)
        size_term = min(size_term, 0.12 * img_area)
        return size_term * (1.0 - WRIST_CENTER_BIAS * min(1.0, dist_n * 2.5))

    for i in range(1, n):
        sc = _score_blob(i, prefer_near=True)
        if sc > best_score:
            best_score = sc
            best_i = i

    # Fallback : toute l'image, blob le plus près du tip (répare « à côté »)
    if best_i is None:
        for i in range(1, n):
            sc = _score_blob(i, prefer_near=False)
            if sc > best_score:
                best_score = sc
                best_i = i

    if best_i is None:
        return None
    blob = np.zeros_like(mask)
    blob[labels == best_i] = 255
    return (
        blob,
        float(centroids[best_i][0]),
        float(centroids[best_i][1]),
        int(stats[best_i, cv2.CC_STAT_AREA]),
    )


def _blob_principal_yaw_deg(blob):
    """Angle principal du blob (deg) — 0 = axe image X."""
    ys, xs = np.where(blob > 0)
    if len(xs) < 30:
        return 0.0
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    mean = pts.mean(axis=0)
    c = pts - mean
    cov = (c.T @ c) / float(len(pts))
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    ang = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    while ang > 90.0:
        ang -= 180.0
    while ang < -90.0:
        ang += 180.0
    return float(ang)


def snap_yaw_to_square_deg(yaw_deg):
    """
    Colis carrés : fermeture pince seulement // ou ⊥ aux côtés.
    Snap 0° / ±90° — jamais diagonale (~45°) qui fait tomber.
    """
    y = float(yaw_deg)
    while y > 90.0:
        y -= 180.0
    while y < -90.0:
        y += 180.0
    return float(min((0.0, 90.0, -90.0), key=lambda c: abs(y - c)))


def square_axis_from_yaw(yaw_deg):
    """0 ou 90 — choix du quat de prise (axe horizontal vs perpendiculaire)."""
    snapped = snap_yaw_to_square_deg(yaw_deg)
    if abs(snapped) >= 45.0:
        return 90
    return 0


# ---------------------------------------------------------------------------
# Observation temps réel (cœur vision)
# ---------------------------------------------------------------------------

def observe_hand(cam, target, hand="right", log=None, settle=None):
    """
    Un regard caméra main : couleur + position du colis dans l'image.

    Returns dict:
      seen, centered, u, v, area, dpix, frac, cx, cy, rgb, depth, cam_key, name
    """
    empty = {
        "seen": False, "centered": False, "u": 0.0, "v": 0.0,
        "area": 0, "dpix": 1e9, "frac": 1.0, "cx": 0.0, "cy": 0.0,
        "rgb": None, "depth": None, "cam_key": "right",
        "name": target.get("name", "?") if target else "?",
    }
    if cam is None or not _HAS_CV2 or target is None:
        return empty

    name = target.get("name", "?")
    empty["name"] = name
    if name not in ("parcel_1", "parcel_2", "parcel_3", "parcel_4"):
        return empty

    rgb_key = "right_rgb" if hand != "left" else "left_rgb"
    depth_key = "right_depth" if hand != "left" else "left_depth"
    cam.wait_for_frame(rgb_key, timeout=1.2)
    cam.wait_for_frame(depth_key, timeout=0.8)
    rospy.sleep(float(WRIST_SETTLE if settle is None else settle))

    rgb, depth, cam_key = _get_wrist_frames(cam, hand)
    if rgb is None:
        _log(log, "[HAND] %s — pas d'image", name)
        return empty

    cx_i, cy_i, _info = _image_aim(cam, cam_key, rgb.shape)
    mask = _color_mask_for_target(rgb, name, depth=depth, log=log)
    if mask is None:
        _log(log, "[HAND] %s — pas de masque couleur", name)
        return empty

    found = _best_blob_near_center(mask, WRIST_MIN_PIXELS, aim_xy=(cx_i, cy_i))
    if found is None:
        n_px = int(np.count_nonzero(mask))
        _log(log, "[HAND] %s — colis NON VU (couleur, mask_px=%d)", name, n_px)
        return empty

    _blob, u_px, v_px, area = found
    h, w = rgb.shape[:2]
    dpix = math.hypot(u_px - cx_i, v_px - cy_i)
    frac = dpix / max(math.hypot(w, h), 1.0)
    # Seed30 fail-mode: area=78 Δpx=302 frac=0.21 → faux « CENTRÉ » (bruit)
    from scene1.config import (
        WRIST_LOCK_MIN_AREA as _LOCK_AREA,
        WRIST_UNDER_HAND_AREA as _UH_AREA,
        WRIST_UNDER_HAND_FRAC as _UH_FRAC,
        WRIST_UNDER_HAND_MAX_DPIX as _UH_DPIX,
    )
    min_lock = float(_LOCK_AREA)
    soft = float(WRIST_CLOSE_MAX_PIXEL_FRAC)
    # Gate tip : pixels proches du aim — PAS soft-only (faux CENTRÉ loin du tip)
    near = dpix <= float(WRIST_ACCEPT_PX)
    # Gros blob + trop loin du tip → PAS lock (sinon close vide, log Δpx=239)
    under_hand = (
        area >= float(_UH_AREA)
        and frac <= float(_UH_FRAC)
        and dpix <= float(_UH_DPIX)
    )
    centered = bool((near and area >= min_lock and frac <= soft) or under_hand)

    yaw_raw = 0.0
    yaw_snap = 0.0
    square_axis = 0
    if WRIST_YAW_ENABLE:
        yaw_raw = _blob_principal_yaw_deg(_blob)
        yaw_snap = snap_yaw_to_square_deg(yaw_raw) if WRIST_YAW_SNAP_SQUARE else yaw_raw
        square_axis = square_axis_from_yaw(yaw_snap)

    _log(log, "[HAND] %s VU area=%d Δpx=%.0f frac=%.2f yaw=%+.0f→%+.0f° axis=%d → %s",
         name, area, dpix, frac, yaw_raw, yaw_snap, square_axis,
         "CENTRÉ" if centered else "décalé")

    return {
        "seen": True,
        "centered": centered,
        "u": u_px,
        "v": v_px,
        "area": area,
        "dpix": dpix,
        "frac": frac,
        "cx": cx_i,
        "cy": cy_i,
        "rgb": rgb,
        "depth": depth,
        "cam_key": cam_key,
        "name": name,
        "blob": _blob,
        "yaw_deg": yaw_snap,
        "yaw_raw_deg": yaw_raw,
        "square_axis": square_axis,
    }


def wrist_sees_centered(cam, tf_reader, target, hand="right", log=None):
    """Compat : True si le colis est vu et assez centré."""
    obs = observe_hand(cam, target, hand=hand, log=log)
    if not obs["seen"]:
        _log(log, "[HAND] see-check %s — AUCUN colis", obs["name"])
        return False
    ok = bool(obs["centered"])
    _log(log, "[HAND] see-check %s area=%d frac=%.2f → %s",
         obs["name"], obs["area"], obs["frac"], "OK" if ok else "TROP LOIN")
    return ok


def wrist_pixel_error(cam, target, hand="right", log=None):
    """Compat : (ok, frac, area)."""
    obs = observe_hand(cam, target, hand=hand, log=log, settle=0.12)
    if not obs["seen"]:
        return False, 1.0, 0
    return True, float(obs["frac"]), int(obs["area"])


# ---------------------------------------------------------------------------
# Servo relatif pixel → Δxy (petits pas seulement)
# ---------------------------------------------------------------------------

def _servo_delta_xy(cam, tf_reader, cam_key, u_px, v_px, rgb_shape, log):
    """Δxy base = ray(blob) − ray(centre). Fallback depth estimée."""
    h, w = int(rgb_shape[0]), int(rgb_shape[1])
    info = cam.get_camera_info(cam_key)
    if info is not None:
        fx, fy = float(info["fx"]), float(info["fy"])
        cx_i, cy_i = float(info["cx"]), float(info["cy"])
        frame_id = info["frame_id"]
    else:
        fx = fy = 0.5 * w
        cx_i, cy_i = 0.5 * w, 0.5 * h
        frame_id = None

    pt_blob = cam.pixel_ray_to_table_plane(
        tf_reader, cam_key, u_px, v_px, table_z=TABLE_PARCEL_Z)
    pt_aim = cam.pixel_ray_to_table_plane(
        tf_reader, cam_key, cx_i, cy_i, table_z=TABLE_PARCEL_Z)
    if pt_blob is not None and pt_aim is not None:
        dx = float(pt_blob[0] - pt_aim[0]) * float(WRIST_SERVO_GAIN) * float(WRIST_SERVO_SIGN_X)
        dy = float(pt_blob[1] - pt_aim[1]) * float(WRIST_SERVO_GAIN) * float(WRIST_SERVO_SIGN_Y)
        dxy = math.hypot(dx, dy)
        _log(log,
             "[HAND] servo-ray Δpx=(%+.0f,%+.0f) → Δxy=(%+.1f,%+.1f)cm |Δ|=%.1fcm",
             u_px - cx_i, v_px - cy_i, dx * 100.0, dy * 100.0, dxy * 100.0)
        return (dx, dy), dxy

    if frame_id is None or fx < 1.0:
        return None, 0.0
    pos, quat = tf_reader.lookup("base_link", frame_id)
    if pos is None or quat is None:
        return None, 0.0
    d_assumed = float(pos[2]) - float(TABLE_PARCEL_Z)
    if d_assumed < 0.06 or d_assumed > 0.55:
        d_assumed = 0.20
    du = (float(u_px) - cx_i) / fx * d_assumed
    dv = (float(v_px) - cy_i) / fy * d_assumed
    rx, ry, _rz = cam._quat_rotate(quat, (du, dv, 0.0))
    dx = float(rx) * float(WRIST_SERVO_GAIN) * float(WRIST_SERVO_SIGN_X)
    dy = float(ry) * float(WRIST_SERVO_GAIN) * float(WRIST_SERVO_SIGN_Y)
    dxy = math.hypot(dx, dy)
    _log(log,
         "[HAND] servo-depth≈%.2f Δpx=(%+.0f,%+.0f) → Δxy=(%+.1f,%+.1f)cm",
         d_assumed, u_px - cx_i, v_px - cy_i, dx * 100.0, dy * 100.0)
    return (dx, dy), dxy


def refine_target_with_wrist(cam, tf_reader, target, hand="right", log=None,
                             max_delta_xy=None):
    """
    Un pas de peaufinage : observe → si décalé, applique un petit Δxy sur la pose tête.
    """
    if cam is None or tf_reader is None or not _HAS_CV2:
        return target

    name = target.get("name", "?")
    if name not in ("parcel_1", "parcel_2", "parcel_3", "parcel_4"):
        return target

    clamp = float(WRIST_MAX_DELTA_XY if max_delta_xy is None else max_delta_xy)
    obs = observe_hand(cam, target, hand=hand, log=log)
    if not obs["seen"]:
        out = dict(target)
        out["wrist_refined"] = False
        out["wrist_seen"] = False
        return out

    # Pose courante (après grille/tête), PAS center_raw UV seule — sinon saute ailleurs
    raw = target.get("center") or target.get("center_raw")
    ox, oy, oz = float(raw[0]), float(raw[1]), float(raw[2])

    if obs["centered"]:
        out = dict(target)
        out["wrist_refined"] = True
        out["wrist_seen"] = True
        out["wrist_centered"] = True
        out["wrist_source"] = "hand-centered"
        out["wrist_delta_xy"] = 0.0
        out["wrist_frac"] = obs["frac"]
        out["wrist_area"] = obs["area"]
        return out

    servo, dxy_servo = _servo_delta_xy(
        cam, tf_reader, obs["cam_key"], obs["u"], obs["v"], obs["rgb"].shape, log)
    if servo is None:
        out = dict(target)
        out["wrist_refined"] = False
        out["wrist_seen"] = True
        return out

    dx, dy = servo
    if dxy_servo > clamp:
        scale = clamp / max(dxy_servo, 1e-6)
        dx *= scale
        dy *= scale
        dxy_servo = clamp
        _log(log, "[HAND] %s clamp |Δ|→%.1fcm", name, clamp * 100.0)

    nx, ny = ox + dx, oy + dy
    nz = float(TABLE_PARCEL_Z)
    _log(log, "[HAND] %s APPLY Δxy=%.1fcm area=%d → (%.3f,%.3f)",
         name, dxy_servo * 100.0, obs["area"], nx, ny)

    out = dict(target)
    out["center_raw"] = (nx, ny, nz)
    out["center"] = (nx, ny, nz)
    out["wrist_refined"] = True
    out["wrist_seen"] = True
    out["wrist_centered"] = False
    out["wrist_source"] = "hand-servo"
    out["wrist_delta_xy"] = dxy_servo
    out["wrist_frac"] = obs["frac"]
    out["wrist_area"] = obs["area"]
    return out


def hand_vision_loop_status(obs_list):
    """Résumé debug d'une boucle de regards."""
    if not obs_list:
        return "aucun regard"
    last = obs_list[-1]
    n_seen = sum(1 for o in obs_list if o.get("seen"))
    return "regards=%d vus=%d last_frac=%.2f %s" % (
        len(obs_list), n_seen, last.get("frac", 1.0),
        "CENTRÉ" if last.get("centered") else "décalé",
    )
