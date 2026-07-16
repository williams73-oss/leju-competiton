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

import json
import math
import os
import sys
import time

import numpy as np
import rospy

_scripts = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

from scene1.config import (  # noqa: E402
    GRASP_AIM_CENTER_DPIX,
    GRASP_AIM_EDGE_DPIX,
    GRASP_HOLD_POS_EMPTY_MIN,
    GRASP_HOLD_POS_GOOD_MAX,
    GRASP_HOLD_POS_GOOD_MIN,
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
    WRIST_LOG_ENABLED,
    WRIST_LOG_PATH,
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
# Logs JSONL + jugement milieu / bord / vide
# ---------------------------------------------------------------------------

def _seed_hint():
    for key in ("SCENE1_SEED", "KUAVO_SEED", "seed"):
        v = os.environ.get(key, "").strip()
        if v != "":
            try:
                return int(v)
            except Exception:
                return v
    try:
        if rospy.core.is_initialized():
            return int(rospy.get_param("/challenge_cup/seed", -1))
    except Exception:
        pass
    return None


def wrist_log_path():
    env = os.environ.get("SCENE1_WRIST_LOG", "").strip()
    if env in ("0", "false", "False", "off", "OFF"):
        return None
    if env:
        return env
    cfg = (WRIST_LOG_PATH or "").strip()
    if cfg:
        return cfg
    home = os.environ.get("HOME") or os.path.expanduser("~") or "/tmp"
    return os.path.join(home, "scene1_wrist.jsonl")


def log_wrist_event(event, log=None, **fields):
    """
    Append 1 ligne JSON (analyse offline).
    Désactiver : WRIST_LOG_ENABLED=False ou SCENE1_WRIST_LOG=0
    """
    if not WRIST_LOG_ENABLED:
        return
    path = wrist_log_path()
    if path is None:
        return
    row = {
        "ts": time.time(),
        "event": str(event),
        "seed": _seed_hint(),
    }
    for k, v in fields.items():
        if v is None:
            continue
        if isinstance(v, (bool, int, float, str)):
            row[k] = v
        elif isinstance(v, (list, tuple)) and len(v) <= 8:
            try:
                row[k] = [float(x) for x in v]
            except Exception:
                row[k] = str(v)
        else:
            try:
                row[k] = float(v)
            except Exception:
                row[k] = str(v)
    try:
        parent = os.path.dirname(path)
        if parent:
            try:
                os.makedirs(parent, exist_ok=True)
            except TypeError:
                if not os.path.isdir(parent):
                    os.makedirs(parent)
        with open(path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as exc:
        _log(log, "[HAND] log skip: %s", exc)


def classify_aim(obs):
    """
    Où on vise sur le colis (caméra poignet) :
      unseen | center | edge | off

    Priorité : tip dans le cœur du blob (face) > Δpx seul.
    """
    if not obs or not obs.get("seen"):
        return "unseen"
    zone = obs.get("grip_zone")
    if zone == "center":
        return "center"
    if zone == "edge":
        return "edge"
    if zone == "outside":
        return "off"
    dpix = float(obs.get("dpix", 1e9))
    if obs.get("centered") or dpix <= float(GRASP_AIM_CENTER_DPIX):
        return "center"
    if dpix <= float(GRASP_AIM_EDGE_DPIX):
        return "edge"
    return "off"


def classify_hold(claw_pos=None, claw_state=None, holding=None):
    """
    Comment la pince serre (ouverture %) :
      empty | good | thin_edge | wide_edge | unknown
    - empty     : fermé à fond (pas d'objet)
    - good      : épaisseur plausible boîte carrée
    - thin_edge : peu ouvert (coin / tip / mauvais contact)
    - wide_edge : assez ouvert mais hors fenêtre (bord oblique / 2 objets)
    """
    if holding is False and claw_state not in (3,):
        if claw_pos is not None and float(claw_pos) >= float(GRASP_HOLD_POS_EMPTY_MIN):
            return "empty"
    if claw_state == 3 or holding is True:
        if claw_pos is None:
            return "good"
        p = float(claw_pos)
        if p >= float(GRASP_HOLD_POS_EMPTY_MIN):
            return "empty"
        if float(GRASP_HOLD_POS_GOOD_MIN) <= p <= float(GRASP_HOLD_POS_GOOD_MAX):
            return "good"
        if p < float(GRASP_HOLD_POS_GOOD_MIN):
            return "thin_edge"
        return "wide_edge"
    if claw_pos is not None:
        p = float(claw_pos)
        if p >= float(GRASP_HOLD_POS_EMPTY_MIN):
            return "empty"
        if float(GRASP_HOLD_POS_GOOD_MIN) <= p <= float(GRASP_HOLD_POS_GOOD_MAX):
            return "good"
        if p < float(GRASP_HOLD_POS_GOOD_MIN):
            return "thin_edge"
        return "wide_edge"
    return "unknown"


def assess_grasp_manner(obs=None, claw_pos=None, claw_state=None, holding=None):
    """
    Combine aim (caméra) + hold (pince) → manner + ok_for_weigh.

    manners :
      center_good  — vise milieu + épaisseur OK
      edge_hold    — tient mais vise bord / épaisseur bizarre
      empty        — ne tient pas
      unseen_hold  — tient sans avoir vu (suspect)
      unknown
    """
    aim = classify_aim(obs)
    hold = classify_hold(claw_pos, claw_state, holding)
    if hold == "empty":
        manner = "empty"
        ok = False
    elif aim == "center" and hold == "good":
        manner = "center_good"
        ok = True
    elif hold in ("good", "thin_edge", "wide_edge") and aim in ("edge", "off"):
        manner = "edge_hold"
        ok = False  # recovery plutôt que pesée aveugle
    elif hold in ("good", "thin_edge", "wide_edge") and aim == "unseen":
        manner = "unseen_hold"
        ok = False
    elif hold == "good":
        manner = "center_good"
        ok = True
    else:
        manner = "unknown"
        ok = bool(holding) and aim == "center"
    return {
        "aim": aim,
        "hold": hold,
        "manner": manner,
        "ok_for_weigh": ok,
        "dpix": float(obs.get("dpix", -1)) if obs else -1.0,
        "frac": float(obs.get("frac", -1)) if obs else -1.0,
        "area": int(obs.get("area", 0)) if obs else 0,
        "claw_pos": None if claw_pos is None else float(claw_pos),
        "claw_state": claw_state,
        "holding": holding,
    }


def is_excellent_grasp(obs=None, assess=None, claw_state=None, claw_pos=None,
                       holding=None, vision_locked=False):
    """
    Très bonne prise (1er essai typique) :
      - vision OK avant close (centré / zone center)
      - pince Grabbed + épaisseur plausible (hold=good)
    → une fois True, ne jamais rouvrir pour recovery.
    """
    if assess is None:
        assess = {}
    if not (holding or int(claw_state or -1) == 3):
        return False
    if assess.get("hold") == "empty":
        return False
    if claw_pos is not None and float(claw_pos) >= float(GRASP_HOLD_POS_EMPTY_MIN):
        return False
    if assess.get("manner") == "center_good":
        return True
    if assess.get("hold") == "good":
        if vision_locked:
            return True
        if obs and (
            obs.get("centered")
            or obs.get("grip_zone") == "center"
            or obs.get("tip_in_core")
        ):
            return True
    return False


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
    boost_map = globals().get("WRIST_LAB_BOOST_BY_NAME") or {}
    if isinstance(boost_map, dict) and name in boost_map:
        boost = float(boost_map[name])
    else:
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


def _blob_grip_geometry(blob, aim_xy):
    """
    Tip vs face du colis (blob) :
      grip_zone = center | edge | outside
      tip_in_core, tip_in_blob, half_diag, tip_rel (0=centre, 1=bord)
    """
    h, w = blob.shape[:2]
    ax, ay = float(aim_xy[0]), float(aim_xy[1])
    ys, xs = np.where(blob > 0)
    empty = {
        "grip_zone": "outside",
        "tip_in_core": False,
        "tip_in_blob": False,
        "half_diag": 1.0,
        "tip_rel": 1.0,
        "aspect": 1.0,
        "bbox": (0, 0, 0, 0),
    }
    if len(xs) < 20:
        return empty

    u0, v0 = float(xs.mean()), float(ys.mean())
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bw = max(1, x_max - x_min + 1)
    bh = max(1, y_max - y_min + 1)
    aspect = float(max(bw, bh)) / float(min(bw, bh))
    half_diag = 0.5 * math.hypot(float(bw), float(bh))
    tip_dist = math.hypot(ax - u0, ay - v0)
    tip_rel = tip_dist / max(half_diag, 1.0)

    ix = int(round(ax))
    iy = int(round(ay))
    tip_in_blob = bool(0 <= ix < w and 0 <= iy < h and blob[iy, ix] > 0)

    erode_k = int(globals().get("WRIST_CORE_ERODE", 7) or 7)
    if erode_k % 2 == 0:
        erode_k += 1
    erode_k = max(3, erode_k)
    kernel = np.ones((erode_k, erode_k), np.uint8)
    core = cv2.erode(blob, kernel, iterations=1)
    if int(np.count_nonzero(core)) < 40:
        # blob petit → cœur = moitié intérieure via distance transform
        core = blob.copy()
        tip_in_core = tip_in_blob and tip_rel <= 0.45
    else:
        tip_in_core = bool(0 <= ix < w and 0 <= iy < h and core[iy, ix] > 0)

    if tip_in_core:
        zone = "center"
    elif tip_in_blob or tip_rel <= 0.85:
        zone = "edge"
    else:
        zone = "outside"

    return {
        "grip_zone": zone,
        "tip_in_core": bool(tip_in_core),
        "tip_in_blob": bool(tip_in_blob),
        "half_diag": float(half_diag),
        "tip_rel": float(tip_rel),
        "aspect": float(aspect),
        "bbox": (x_min, y_min, bw, bh),
        "centroid": (u0, v0),
    }


def _yaw_confident(area, aspect):
    if bool(globals().get("WRIST_YAW_ENABLE", False)):
        return True
    if not bool(globals().get("WRIST_YAW_AUTO", True)):
        return False
    min_a = float(globals().get("WRIST_YAW_MIN_AREA", 8000) or 8000)
    max_asp = float(globals().get("WRIST_YAW_MAX_ASPECT", 1.55) or 1.55)
    return area >= min_a and aspect <= max_asp


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
        log_wrist_event("observe", log=log, seen=False, aim="unseen", reason="no_cam")
        return empty

    name = target.get("name", "?")
    empty["name"] = name
    if name not in ("parcel_1", "parcel_2", "parcel_3", "parcel_4"):
        log_wrist_event("observe", log=log, name=name, seen=False, aim="unseen",
                        reason="bad_name")
        return empty

    rgb_key = "right_rgb" if hand != "left" else "left_rgb"
    depth_key = "right_depth" if hand != "left" else "left_depth"
    cam.wait_for_frame(rgb_key, timeout=1.2)
    cam.wait_for_frame(depth_key, timeout=0.8)
    rospy.sleep(float(WRIST_SETTLE if settle is None else settle))

    rgb, depth, cam_key = _get_wrist_frames(cam, hand)
    if rgb is None:
        _log(log, "[HAND] %s — pas d'image", name)
        log_wrist_event("observe", log=log, name=name, hand=hand, seen=False,
                        aim="unseen", reason="no_rgb")
        return empty

    cx_i, cy_i, _info = _image_aim(cam, cam_key, rgb.shape)
    mask = _color_mask_for_target(rgb, name, depth=depth, log=log)
    if mask is None:
        _log(log, "[HAND] %s — pas de masque couleur", name)
        log_wrist_event("observe", log=log, name=name, hand=hand, seen=False,
                        aim="unseen", reason="no_mask")
        return empty

    found = _best_blob_near_center(mask, WRIST_MIN_PIXELS, aim_xy=(cx_i, cy_i))
    if found is None:
        n_px = int(np.count_nonzero(mask))
        _log(log, "[HAND] %s — colis NON VU (couleur, mask_px=%d)", name, n_px)
        log_wrist_event("observe", log=log, name=name, hand=hand, seen=False,
                        aim="unseen", reason="no_blob", mask_px=int(n_px))
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
    geom = _blob_grip_geometry(_blob, (cx_i, cy_i))
    tip_core_req = bool(globals().get("WRIST_TIP_IN_CORE_REQUIRED", True))
    near_ok = bool((near and area >= min_lock and frac <= soft) or under_hand)
    if tip_core_req:
        centered = bool(near_ok and geom.get("tip_in_core"))
        # tip dans blob mais pas cœur → pas encore lock (servo continue)
        if near_ok and geom.get("tip_in_blob") and not geom.get("tip_in_core"):
            centered = False
    else:
        centered = near_ok

    yaw_raw = 0.0
    yaw_snap = 0.0
    square_axis = 0
    do_yaw = _yaw_confident(area, float(geom.get("aspect", 99.0)))
    if do_yaw and (WRIST_YAW_ENABLE or bool(globals().get("WRIST_YAW_AUTO", True))):
        yaw_raw = _blob_principal_yaw_deg(_blob)
        yaw_snap = snap_yaw_to_square_deg(yaw_raw) if WRIST_YAW_SNAP_SQUARE else yaw_raw
        square_axis = square_axis_from_yaw(yaw_snap)

    _log(log, "[HAND] %s VU area=%d Δpx=%.0f frac=%.2f zone=%s tip_rel=%.2f "
         "yaw=%+.0f→%+.0f° axis=%d → %s",
         name, area, dpix, frac, geom.get("grip_zone"), geom.get("tip_rel", 1.0),
         yaw_raw, yaw_snap, square_axis,
         "CENTRÉ" if centered else "décalé")

    out = {
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
        "grip_zone": geom.get("grip_zone"),
        "tip_in_core": geom.get("tip_in_core"),
        "tip_in_blob": geom.get("tip_in_blob"),
        "tip_rel": geom.get("tip_rel"),
        "aspect": geom.get("aspect"),
    }
    aim = classify_aim(out)
    out["aim"] = aim
    log_wrist_event(
        "observe",
        log=log,
        name=name,
        hand=hand,
        seen=True,
        centered=bool(centered),
        aim=aim,
        grip_zone=str(geom.get("grip_zone")),
        tip_in_core=bool(geom.get("tip_in_core")),
        tip_rel=float(geom.get("tip_rel", 1.0)),
        area=int(area),
        dpix=float(dpix),
        frac=float(frac),
        u=float(u_px),
        v=float(v_px),
        yaw_deg=float(yaw_snap),
    )
    return out


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
        log_wrist_event("refine", log=log, name=name, refined=False, seen=False,
                        aim="unseen")
        return out

    # Pose courante (après grille/tête), PAS center_raw UV seule — sinon saute ailleurs
    raw = target.get("center") or target.get("center_raw")
    ox, oy, oz = float(raw[0]), float(raw[1]), float(raw[2])

    # Depth 3D : centroid blob → base_link (plus précis que ray table seul)
    depth3d = None
    depth_m = None
    blob = obs.get("blob")
    depth = obs.get("depth")
    if (bool(globals().get("WRIST_USE_DEPTH_3D", True))
            and blob is not None and depth is not None
            and hasattr(cam, "median_depth_in_mask")
            and hasattr(cam, "pixel_to_base_link")):
        zmin = float(globals().get("WRIST_DEPTH_Z_GRASP_MIN", 0.06) or 0.06)
        zmax = float(globals().get("WRIST_DEPTH_Z_GRASP_MAX", 0.45) or 0.45)
        depth_m = cam.median_depth_in_mask(depth, blob, z_min=zmin, z_max=zmax)
        if depth_m is not None:
            depth3d = cam.pixel_to_base_link(
                tf_reader, obs["cam_key"], obs["u"], obs["v"], depth_m)
            if depth3d is not None:
                _log(log, "[HAND] %s depth3d z_cam=%.3f → (%.3f,%.3f,%.3f)",
                     name, depth_m, depth3d[0], depth3d[1], depth3d[2])

    if obs["centered"]:
        out = dict(target)
        if depth3d is not None:
            max_d = float(globals().get("WRIST_DEPTH_3D_MAX_DELTA", 0.04) or 0.04)
            dx = float(depth3d[0]) - ox
            dy = float(depth3d[1]) - oy
            if math.hypot(dx, dy) <= max_d:
                out["center"] = (float(depth3d[0]), float(depth3d[1]), float(oz))
                out["center_raw"] = out["center"]
                out["wrist_source"] = "hand-centered-depth3d"
            else:
                out["wrist_source"] = "hand-centered"
        else:
            out["wrist_source"] = "hand-centered"
        out["wrist_refined"] = True
        out["wrist_seen"] = True
        out["wrist_centered"] = True
        out["wrist_delta_xy"] = 0.0
        out["wrist_frac"] = obs["frac"]
        out["wrist_area"] = obs["area"]
        out["aim"] = obs.get("aim", classify_aim(obs))
        out["grip_zone"] = obs.get("grip_zone")
        log_wrist_event(
            "refine", log=log, name=name, refined=True, seen=True,
            centered=True, aim=out["aim"], grip_zone=obs.get("grip_zone"),
            delta_xy=0.0, depth_m=depth_m,
            area=int(obs["area"]), dpix=float(obs["dpix"]), frac=float(obs["frac"]),
        )
        return out

    # Prefer depth3d delta when available; else ray servo
    dx = dy = 0.0
    dxy_servo = 0.0
    source = "hand-servo"
    if depth3d is not None:
        dx = float(depth3d[0]) - ox
        dy = float(depth3d[1]) - oy
        dxy_servo = math.hypot(dx, dy)
        source = "hand-depth3d"
        _log(log, "[HAND] %s depth3d Δxy=(%+.1f,%+.1f)cm",
             name, dx * 100.0, dy * 100.0)
    else:
        servo, dxy_servo = _servo_delta_xy(
            cam, tf_reader, obs["cam_key"], obs["u"], obs["v"], obs["rgb"].shape, log)
        if servo is None:
            out = dict(target)
            out["wrist_refined"] = False
            out["wrist_seen"] = True
            log_wrist_event("refine", log=log, name=name, refined=False, seen=True,
                            aim=obs.get("aim", classify_aim(obs)), reason="no_servo")
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
    _log(log, "[HAND] %s APPLY Δxy=%.1fcm area=%d zone=%s → (%.3f,%.3f) [%s]",
         name, dxy_servo * 100.0, obs["area"], obs.get("grip_zone"), nx, ny, source)

    out = dict(target)
    out["center_raw"] = (nx, ny, nz)
    out["center"] = (nx, ny, nz)
    out["wrist_refined"] = True
    out["wrist_seen"] = True
    out["wrist_centered"] = False
    out["wrist_source"] = source
    out["wrist_delta_xy"] = dxy_servo
    out["wrist_frac"] = obs["frac"]
    out["wrist_area"] = obs["area"]
    out["aim"] = obs.get("aim", classify_aim(obs))
    out["grip_zone"] = obs.get("grip_zone")
    log_wrist_event(
        "refine", log=log, name=name, refined=True, seen=True, centered=False,
        aim=out["aim"], grip_zone=obs.get("grip_zone"),
        delta_xy=float(dxy_servo), depth_m=depth_m, source=source,
        area=int(obs["area"]), dpix=float(obs["dpix"]), frac=float(obs["frac"]),
        center=(nx, ny, nz),
    )
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
