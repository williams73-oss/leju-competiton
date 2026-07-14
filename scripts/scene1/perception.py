#!/usr/bin/env python3
"""
Perception Scene 1 — tête RGB + LiDAR Mid360.

Rôle : détecter colis (qui/où), balance, bac.
Pas de mouvement bras ici (tête pitch=20 via le runner).
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
_pkg = os.path.dirname(_scripts)
sys.path.insert(0, os.path.join(_pkg, "src"))
from perception_api import CameraReader, LidarReader, TFReader

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    import open3d as o3d
    _HAS_O3D = True
except ImportError:
    _HAS_O3D = False

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from scene1.config import *  # noqa: F401,F403

def _cluster_points_xy(points, eps, min_points):
    """
    Regroupe les points LiDAR en amas (algorithme type DBSCAN simplifié en 2D).
    Deux points à moins de `eps` mètres appartiennent au même amas.
    """
    if points is None or len(points) == 0:
        return []

    xy = points[:, :2]  # on ignore z pour le regroupement horizontal
    n = len(xy)
    visited = np.zeros(n, dtype=bool)
    clusters = []
    eps2 = eps * eps

    for i in range(n):
        if visited[i]:
            continue
        # Parcours en largeur (BFS) : partir du point i et étendre l'amas
        queue = [i]
        visited[i] = True
        members = [i]
        while queue:
            j = queue.pop()
            dist2 = np.sum((xy - xy[j]) ** 2, axis=1)
            neighbors = np.where((dist2 <= eps2) & (~visited))[0]
            for k in neighbors:
                visited[k] = True
                queue.append(int(k))
                members.append(int(k))
        if len(members) >= min_points:
            clusters.append(members)
    return clusters


def _summarize_cluster(points, indices, seed_xy=None):
    """Calcule le centre et la taille d'un amas de points LiDAR."""
    pts = points[indices]
    if seed_xy is not None and len(pts) > 4:
        dist = np.hypot(pts[:, 0] - seed_xy[0], pts[:, 1] - seed_xy[1])
        keep_n = max(3, int(len(pts) * 0.65))
        pts = pts[np.argsort(dist)[:keep_n]]
    # Centre robuste (percentiles) + biais LiDAR calibré (dx≈+5.5 cm sim).
    center = np.array([
        0.5 * (float(np.percentile(pts[:, 0], 25))
               + float(np.percentile(pts[:, 0], 75))),
        0.5 * (float(np.percentile(pts[:, 1], 25))
               + float(np.percentile(pts[:, 1], 75))),
        float(np.median(pts[:, 2])),
    ], dtype=np.float64)
    corr = np.asarray(LIDAR_IK_CORR, dtype=np.float64)
    center = center + corr
    return {
        "center": tuple(float(v) for v in center),
        "size_xy": (float(pts[:, 0].ptp()), float(pts[:, 1].ptp())),
        "n_points": len(indices),
    }


def _sort_parcels(clusters):
    """Trie les colis de gauche à droite puis de bas en haut (y puis x)."""
    return sorted(clusters, key=lambda c: (c["center"][1], c["center"][0]))


def _relative_grid_mids(centers_xy):
    """Médianes X/Y des centres détectés — grille 2×2 relative (tous seeds)."""
    xs = [c[0] for c in centers_xy]
    ys = [c[1] for c in centers_xy]
    return float(np.median(xs)), float(np.median(ys))


def _slot_name_from_xy(x, y, x_mid, y_mid):
    """
    Convention scène 1 (indépendante du seed) :
      x petit = proche robot → parcel_1/2
      x grand = loin        → parcel_3/4
      y petit (plus négatif) → parcel_1/3
      y grand                → parcel_2/4
    """
    near = x < x_mid
    low_y = y < y_mid
    if near and low_y:
        return "parcel_1"
    if near and not low_y:
        return "parcel_2"
    if (not near) and low_y:
        return "parcel_3"
    return "parcel_4"


def _assign_names_relative_grid(clusters, log, source_tag="lidar-spatial"):
    """Nomme les amas par grille 2×2 relative aux médianes (pas seed0)."""
    if not clusters:
        return []
    centers = [c["center"][:2] for c in clusters]
    x_mid, y_mid = _relative_grid_mids(centers)
    log("[FUSE] grille relative x_mid=%.3f y_mid=%.3f (anti-seed0)", x_mid, y_mid)

    # Si collision de noms (2 amas même quadrant), départager par distance au coin
    used = set()
    out = []
    # Traiter les plus isolés d'abord : tri par |x-xmid|+|y-ymid| décroissant
    ordered = sorted(
        clusters,
        key=lambda c: -(abs(c["center"][0] - x_mid) + abs(c["center"][1] - y_mid)))
    for lc in ordered:
        cx, cy, cz = lc["center"]
        name = _slot_name_from_xy(cx, cy, x_mid, y_mid)
        if name in used:
            # quadrant déjà pris → choisir le nom libre le plus cohérent
            free = [n for n in PARCEL_NAMES if n not in used]
            if not free:
                break
            best, best_d = free[0], float("inf")
            for n in free:
                # coin nominal relatif du slot
                sx = x_mid - 0.07 if n in ("parcel_1", "parcel_2") else x_mid + 0.07
                sy = y_mid - 0.10 if n in ("parcel_1", "parcel_3") else y_mid + 0.10
                d = math.hypot(cx - sx, cy - sy)
                if d < best_d:
                    best, best_d = n, d
            name = best
        used.add(name)
        p = dict(lc)
        p["name"] = name
        p["color"] = _color_label_for_parcel(name)
        p["source"] = source_tag
        out.append(p)
        log("[FUSE] %s (%s) ← %s relatif (%.3f, %.3f, %.3f)",
            name, p["color"], source_tag, cx, cy, cz)
    return _sort_parcels(out)


def _geometry_quality_relative(parcels):
    """
    Qualité géométrie multi-seed : cohérence interne grille 2×2.
    Retourne (named_count, struct_err_xy).
    Ne compare PAS à PARCEL_WORLD_POS (seed0).
    """
    by_name = {p["name"]: p for p in parcels if p.get("name") in PARCEL_NAMES}
    named = len(by_name)
    if named == 0:
        return 0, 1.0

    # Hors zone table → erreur forte
    for p in by_name.values():
        x, y, z = p["center"]
        if not (TABLE_X_RANGE[0] - 0.05 <= x <= TABLE_X_RANGE[1] + 0.05
                and TABLE_Y_RANGE[0] - 0.05 <= y <= TABLE_Y_RANGE[1] + 0.05):
            return named, 0.25

    if named < 4:
        return named, 0.15 + 0.05 * (4 - named)

    c1, c2 = by_name["parcel_1"]["center"], by_name["parcel_2"]["center"]
    c3, c4 = by_name["parcel_3"]["center"], by_name["parcel_4"]["center"]
    x_near = 0.5 * (c1[0] + c2[0])
    x_far = 0.5 * (c3[0] + c4[0])
    y_lo = 0.5 * (c1[1] + c3[1])
    y_hi = 0.5 * (c2[1] + c4[1])
    _, row_dy = _grid_spacing()
    # Collapse rangée (seed0: gap~9 cm au lieu de ~21) → rejeter l'attempt.
    # Ancien seuil 0.06 laissait passer gap=0.096 avec err_struct~1.5 cm.
    min_row_gap = 0.55 * abs(row_dy)  # ~0.115 m
    if x_far - x_near < 0.06 or y_hi - y_lo < min_row_gap:
        return named, 0.12

    # Colonnes hors plage table (ex. jaune x=0.17) → attempt pourri.
    for name, lo, hi in (
            ("parcel_1", 0.24, 0.42), ("parcel_2", 0.24, 0.42),
            ("parcel_3", 0.40, 0.58), ("parcel_4", 0.40, 0.58)):
        x = by_name[name]["center"][0]
        if not (lo <= x <= hi):
            return named, 0.12

    targets = {
        "parcel_1": (x_near, y_lo),
        "parcel_2": (x_near, y_hi),
        "parcel_3": (x_far, y_lo),
        "parcel_4": (x_far, y_hi),
    }
    max_err = 0.0
    for name, p in by_name.items():
        tx, ty = targets[name]
        max_err = max(max_err, math.hypot(p["center"][0] - tx, p["center"][1] - ty))
    return named, max_err


def _lidar_clusters_seeded(pts, log):
    """Fallback : 4 graines autour des médianes du nuage (anti-seed0)."""
    if pts is None or len(pts) < 4:
        return []
    x_mid = float(np.median(pts[:, 0]))
    y_mid = float(np.median(pts[:, 1]))
    # Offsets typiques grille 2×2 sur table (~15 cm)
    seeds = [
        ("parcel_1", x_mid - 0.07, y_mid - 0.10),
        ("parcel_2", x_mid - 0.07, y_mid + 0.10),
        ("parcel_3", x_mid + 0.07, y_mid - 0.10),
        ("parcel_4", x_mid + 0.07, y_mid + 0.10),
    ]
    clusters = []
    for name, ex, ey in seeds:
        dist = np.hypot(pts[:, 0] - ex, pts[:, 1] - ey)
        idx = np.where(dist <= LIDAR_SEED_RADIUS)[0]
        if len(idx) < LIDAR_SEED_MIN_POINTS:
            continue
        summary = _summarize_cluster(pts, idx, seed_xy=(ex, ey))
        summary["source"] = "lidar-seed"
        summary["seed_name"] = name
        clusters.append(summary)
        cx, cy, cz = summary["center"]
        log("[DETECT]   seed_%s: center=(%.3f, %.3f, %.3f) n=%d",
            name, cx, cy, cz, summary["n_points"])
    return _sort_parcels(clusters)[:4]


def _lidar_clusters_grid(pts, log):
    """Fallback : 4 quadrants relatifs aux médianes du nuage (anti-seed0)."""
    if pts is None or len(pts) < 4:
        return []
    x_mid = float(np.median(pts[:, 0]))
    y_mid = float(np.median(pts[:, 1]))
    log("[DETECT] grille pts relative x_mid=%.3f y_mid=%.3f", x_mid, y_mid)
    masks = [
        (pts[:, 0] < x_mid) & (pts[:, 1] < y_mid),
        (pts[:, 0] < x_mid) & (pts[:, 1] >= y_mid),
        (pts[:, 0] >= x_mid) & (pts[:, 1] < y_mid),
        (pts[:, 0] >= x_mid) & (pts[:, 1] >= y_mid),
    ]
    clusters = []
    for i, mask in enumerate(masks):
        idx = np.where(mask)[0]
        if len(idx) < 2:
            continue
        summary = _summarize_cluster(pts, idx)
        summary["source"] = "lidar-grid"
        clusters.append(summary)
        cx, cy, cz = summary["center"]
        log("[DETECT]   grid_%d: center=(%.3f, %.3f, %.3f) n=%d",
            i + 1, cx, cy, cz, summary["n_points"])
    return _sort_parcels(clusters)[:4]


def _lidar_clusters_open3d(pts, log):
    """Niveau 2 : RANSAC plan table + DBSCAN 3D sur points au-dessus."""
    if not _HAS_O3D or pts is None or len(pts) < 20:
        return []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    try:
        # Open3D 0.13 (Py3.8) : segment_plane(dist, ransac_n, num_iter) — pas de probability
        result = pcd.segment_plane(LIDAR_O3D_PLANE_DIST, 3, 1000)
        plane, inliers = result[0], result[1]
        if hasattr(plane, "flatten"):
            plane = plane.flatten()
    except Exception as exc:
        log("[DETECT] LiDAR Open3D: RANSAC échec (%s)", exc)
        return []

    a, b, c, d = float(plane[0]), float(plane[1]), float(plane[2]), float(plane[3])
    norm = math.sqrt(a * a + b * b + c * c) + 1e-9
    dists = np.abs(pts[:, 0] * a + pts[:, 1] * b + pts[:, 2] * c + d) / norm
    above = dists > LIDAR_O3D_PLANE_DIST
    below_max = dists < LIDAR_O3D_ABOVE_MAX
    parcel_pts = pts[above & below_max]
    if len(parcel_pts) < LIDAR_O3D_MIN_POINTS * 2:
        log("[DETECT] LiDAR Open3D: %d pts au-dessus plan (insuffisant)",
            len(parcel_pts))
        return []

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(parcel_pts.astype(np.float64))
    labels = np.array(pcd2.cluster_dbscan(
        eps=LIDAR_O3D_EPS, min_points=LIDAR_O3D_MIN_POINTS, print_progress=False))

    candidates = []
    for label in sorted(set(labels)):
        if label < 0:
            continue
        idx = np.where(labels == label)[0]
        if len(idx) < LIDAR_O3D_MIN_POINTS:
            continue
        summary = _summarize_cluster(parcel_pts, idx)
        sx, sy = summary["size_xy"]
        if sx < MIN_PARCEL_SIZE_XY or sy < MIN_PARCEL_SIZE_XY:
            continue
        if sx > MAX_PARCEL_SIZE_XY or sy > MAX_PARCEL_SIZE_XY:
            continue
        summary["source"] = "lidar-o3d"
        candidates.append(summary)

    log("[DETECT] LiDAR Open3D: plan table + %d amas 3D", len(candidates))
    for i, c in enumerate(candidates):
        cx, cy, cz = c["center"]
        log("[DETECT]   o3d_%d: center=(%.3f, %.3f, %.3f) n=%d",
            i + 1, cx, cy, cz, c["n_points"])
    return _sort_parcels(candidates)[:4]


def _lidar_clusters_legacy(pts, log):
    candidates = []
    raw = _cluster_points_xy(pts, CLUSTER_EPS_XY, MIN_CLUSTER_POINTS)
    log("[DETECT] LiDAR: DBSCAN brut %d amas", len(raw))
    for indices in raw:
        summary = _summarize_cluster(pts, indices)
        sx, sy = summary["size_xy"]
        if sx < MIN_PARCEL_SIZE_XY or sy < MIN_PARCEL_SIZE_XY:
            continue
        if sx > MAX_PARCEL_SIZE_XY or sy > MAX_PARCEL_SIZE_XY:
            continue
        summary["source"] = "lidar"
        candidates.append(summary)

    if not candidates and len(pts) >= 12:
        log("[DETECT] LiDAR: DBSCAN 无有效簇 (region %d pts)，使用 seed 位置聚类", len(pts))
        candidates = _lidar_clusters_seeded(pts, log)
        if len(candidates) < 3:
            log("[DETECT] LiDAR: seed 仅 %d 簇，补充 2×2 网格", len(candidates))
            grid = _lidar_clusters_grid(pts, log)
            seen = {tuple(round(c["center"][i], 2) for i in range(3)) for c in candidates}
            for g in grid:
                key = tuple(round(g["center"][i], 2) for i in range(3))
                if key not in seen:
                    candidates.append(g)
                    seen.add(key)

    clusters = _sort_parcels(candidates)[:4]
    log("[DETECT] LiDAR: 找到 %d 个几何簇", len(clusters))
    for i, c in enumerate(clusters):
        cx, cy, cz = c["center"]
        log("[DETECT]   cluster_%d: center=(%.3f, %.3f, %.3f) size=%.3f×%.3f",
            i + 1, cx, cy, cz, c["size_xy"][0], c["size_xy"][1])
    return clusters


def _lidar_clusters_raw(lidar, log):
    """LiDAR : Open3D (N2) si dispo, sinon DBSCAN 2D + seed/grid."""
    pts = lidar.get_points_in_region(
        x_range=TABLE_X_RANGE,
        y_range=TABLE_Y_RANGE,
        z_range=PARCEL_Z_RANGE,
    )
    all_pts = lidar.get_points()
    if all_pts is not None and len(all_pts) > 0:
        log("[DETECT] LiDAR: total=%d x=[%.3f, %.3f] y=[%.3f, %.3f] z=[%.3f, %.3f]",
            len(all_pts),
            float(all_pts[:, 0].min()), float(all_pts[:, 0].max()),
            float(all_pts[:, 1].min()), float(all_pts[:, 1].max()),
            float(all_pts[:, 2].min()), float(all_pts[:, 2].max()))
    if pts is None or len(pts) == 0:
        log("[DETECT] LiDAR: 未获取到点云 (region 内 0 点, x=%s y=%s z=%s)",
            TABLE_X_RANGE, TABLE_Y_RANGE, PARCEL_Z_RANGE)
        return []

    log("[DETECT] LiDAR: region %d pts x=[%.3f, %.3f] y=[%.3f, %.3f] z=[%.3f, %.3f]",
        len(pts),
        float(pts[:, 0].min()), float(pts[:, 0].max()),
        float(pts[:, 1].min()), float(pts[:, 1].max()),
        float(pts[:, 2].min()), float(pts[:, 2].max()))

    if _HAS_O3D:
        o3d_clusters = _lidar_clusters_open3d(pts, log)
        if len(o3d_clusters) >= 2:
            return o3d_clusters
        log("[DETECT] LiDAR Open3D: <2 amas → fallback DBSCAN 2D")
    else:
        log("[DETECT] LiDAR: open3d absent → DBSCAN 2D")

    return _lidar_clusters_legacy(pts, log)


# =============================================================================
# PERCEPTION — ÉTAPE 2 : couleur RGB (QUEL colis ?)
# =============================================================================

def _is_valid_base_point(x, y, z):
    return (BASE_VALID_X[0] <= x <= BASE_VALID_X[1]
            and BASE_VALID_Y[0] <= y <= BASE_VALID_Y[1]
            and BASE_VALID_Z[0] <= z <= BASE_VALID_Z[1])


def _is_valid_parcel_table_point(x, y, z, rgb_backed=False):
    """Zone table colis (IK). rgb_backed : z serré autour de la table."""
    z_lo = PARCEL_RGB_Z_RANGE[0] if rgb_backed else PARCEL_Z_RANGE[0]
    z_hi = PARCEL_RGB_Z_RANGE[1] if rgb_backed else PARCEL_Z_RANGE[1]
    return (TABLE_X_RANGE[0] - 0.02 <= x <= TABLE_X_RANGE[1] + 0.02
            and TABLE_Y_RANGE[0] - 0.02 <= y <= TABLE_Y_RANGE[1] + 0.02
            and z_lo <= z <= z_hi)


def _validates_rgb_point(name, x, y, z):
    if name in PARCEL_NAMES:
        return _is_valid_parcel_table_point(x, y, z, rgb_backed=True)
    return _is_valid_base_point(x, y, z)


def _rgb_depth_table_point(cam, tf_reader, u_px, v_px, depth_m, log, name, via):
    """
    Niveau 2 : depth + ray→plan table. Depth seul rejette souvent (z≈+0.2).
    Priorité depth si xy cohérent avec ray ; sinon ray-plane.
    """
    pt_ray = None
    if cam is not None and tf_reader is not None:
        pt_ray = cam.pixel_ray_to_table_plane(
            tf_reader, "head", u_px, v_px, TABLE_PARCEL_Z)

    pt_depth = None
    if depth_m is not None and cam is not None and tf_reader is not None:
        pt_depth = cam.pixel_to_base_link(
            tf_reader, "head", u_px, v_px, depth_m)

    if pt_depth is not None and _validates_rgb_point(name, *pt_depth):
        if pt_ray is not None and _validates_rgb_point(name, *pt_ray):
            err_xy = math.hypot(pt_depth[0] - pt_ray[0], pt_depth[1] - pt_ray[1])
            if err_xy > RGB_DEPTH_XY_TOL:
                log("[COLOR] %s (%s): depth/ray Δxy=%.3f → ray-plane", name, via, err_xy)
                return pt_ray, "rgb-ray"
        if abs(pt_depth[2] - TABLE_PARCEL_Z) > 0.06:
            if pt_ray is not None and _validates_rgb_point(name, *pt_ray):
                log("[COLOR] %s (%s): depth z=%.3f aberrant → ray-plane",
                    name, via, pt_depth[2])
                return pt_ray, "rgb-ray"
        return pt_depth, "rgb-depth"

    if pt_depth is not None:
        x, y, z = pt_depth
        if (TABLE_X_RANGE[0] - 0.02 <= x <= TABLE_X_RANGE[1] + 0.02
                and TABLE_Y_RANGE[0] - 0.02 <= y <= TABLE_Y_RANGE[1] + 0.02):
            if pt_ray is not None and _validates_rgb_point(name, *pt_ray):
                err_xy = math.hypot(x - pt_ray[0], y - pt_ray[1])
                if err_xy > RGB_DEPTH_XY_TOL:
                    log("[COLOR] %s (%s): zsnap xy/ray Δxy=%.3f → ray-plane",
                        name, via, err_xy)
                    return pt_ray, "rgb-ray"
            log("[COLOR] %s (%s): z=%.3f → plan table (%.3f)", name, via, z, TABLE_PARCEL_Z)
            snapped = (x, y, TABLE_PARCEL_Z)
            if _validates_rgb_point(name, *snapped):
                return snapped, "rgb-depth+zsnap"

    if pt_ray is not None and _validates_rgb_point(name, *pt_ray):
        return pt_ray, "rgb-ray"

    if pt_ray is not None:
        log("[COLOR] %s (%s): rgb-ray hors table (%.3f, %.3f, %.3f), ignoré",
            name, via, pt_ray[0], pt_ray[1], pt_ray[2])
    return None, None


def _hungarian_pairs(cost_matrix):
    """Assignation optimale (scipy ou brute force n≤6)."""
    cost = np.asarray(cost_matrix, dtype=np.float64)
    n, m = cost.shape
    size = max(n, m)
    padded = np.full((size, size), HUNGARIAN_BIG_COST)
    padded[:n, :m] = cost

    if _HAS_SCIPY:
        row_ind, col_ind = linear_sum_assignment(padded)
        return [(int(r), int(c)) for r, c in zip(row_ind, col_ind)
                if r < n and c < m]

    best_perm = None
    best_val = float("inf")
    for perm in __import__("itertools").permutations(range(size)):
        val = sum(padded[i, perm[i]] for i in range(size))
        if val < best_val:
            best_val = val
            best_perm = perm
    return [(i, best_perm[i]) for i in range(n) if best_perm[i] < m]


def _source_priority(source):
    order = (
        "lidar-spatial", "lidar+color", "lidar-o3d", "lidar", "lidar-seed", "lidar-grid",
        "rgb-depth", "rgb-ray", "rgb-depth+zsnap",
        "couleur+row-x", "row-infer", "couleur",
    )
    src = source or ""
    for i, tag in enumerate(order):
        if tag in src:
            return i
    return len(order)


def _bgr_to_lab(ref_bgr):
    """Tuple BGR → vecteur LAB (float32)."""
    swatch = np.uint8([[[ref_bgr[0], ref_bgr[1], ref_bgr[2]]]])
    return cv2.cvtColor(swatch, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def _color_mask_hsv(roi, lo, hi, remaining):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.bitwise_and(mask, remaining)


def _apply_color_u_roi(mask, name):
    """Restreint le masque à une bande u (colonne image) si définie pour ce colis."""
    u_range = COLOR_U_RANGE_BY_NAME.get(name)
    if u_range is None or mask is None or mask.size == 0:
        return mask
    rh, rw = mask.shape[:2]
    u0 = int(rw * u_range[0])
    u1 = int(rw * u_range[1])
    out = np.zeros_like(mask)
    out[:, u0:u1] = mask[:, u0:u1]
    return out


def _best_color_blob(mask, name, min_px, max_area):
    """
    Parmi les composantes connexes, garde un blob dans [min_px, max_area].
    Pour le bleu : préfère le blob le plus bas (table), pas le ciel.
    """
    if mask is None:
        return None, 0
    area_raw = int(mask.sum()) // 255
    if area_raw < min_px:
        return None, area_raw

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    prefer_low_v = name in COLOR_PREFER_LOW_V
    best_label = None
    best_score = None
    for lab_id in range(1, n_labels):
        a = int(stats[lab_id, cv2.CC_STAT_AREA])
        if a < min_px or a > max_area:
            continue
        if prefer_low_v:
            # centroid y plus grand = plus bas dans l'image = mieux
            score = (float(centroids[lab_id][1]), a)
        else:
            score = (float(a),)
        if best_score is None or score > best_score:
            best_score = score
            best_label = lab_id
    if best_label is None:
        # secours : erode pour casser le ciel, puis retente
        eroded = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=2)
        if int(eroded.sum()) // 255 >= min_px:
            return _best_color_blob(eroded, name, min_px, max_area)
        return None, area_raw
    blob = np.zeros_like(mask)
    blob[labels == best_label] = 255
    return blob, int(stats[best_label, cv2.CC_STAT_AREA])


def _depth_gate_roi(depth, h, w, v0):
    """Masque ROI : pixels avec profondeur table (coupe ciel / infini)."""
    rh = h - v0
    if depth is None:
        return np.full((rh, w), 255, dtype=np.uint8)
    if depth.shape[0] != h or depth.shape[1] != w:
        # resize nearest si résolution différente
        depth_r = cv2.resize(depth.astype(np.float32), (w, h),
                             interpolation=cv2.INTER_NEAREST)
    else:
        depth_r = depth
    roi_d = depth_r[v0:, :]
    valid = np.isfinite(roi_d) & (roi_d >= COLOR_DEPTH_Z_MIN) & (
        roi_d <= COLOR_DEPTH_Z_MAX)
    return (valid.astype(np.uint8) * 255)


def _append_color_hit(hits, remaining, name, label, mask, h, w, v0,
                      depth, cam, tf_reader, log, via="LAB"):
    max_area = int(h * w * MAX_COLOR_AREA_RATIO)
    min_px = MIN_COLOR_PIXELS_BY_NAME.get(name, MIN_COLOR_PIXELS)
    max_area_name = COLOR_MAX_AREA_BY_NAME.get(name)
    if max_area_name is not None:
        max_area = min(max_area, max_area_name)
    if name == "parcel_4":
        max_area = min(max_area, MAX_BLUE_MASK_AREA)

    mask = _apply_color_u_roi(mask, name)
    blob, area_info = _best_color_blob(mask, name, min_px, max_area)
    if blob is None:
        if area_info < min_px:
            log("[COLOR] %s (%s): 有效像素不足 (%s, need>=%d, raw=%d)",
                name, label, via, min_px, area_info)
        else:
            log("[COLOR] %s (%s): aucun blob valide (%s, raw=%d, max=%d)",
                name, label, via, area_info, max_area)
        return remaining

    area = int(blob.sum()) // 255
    ys, xs = np.where(blob > 0)
    u_px = float(xs.mean())
    v_px = float(ys.mean() + v0)
    v_norm = v_px / h
    v_max = COLOR_V_NORM_MAX_BY_NAME.get(name, COLOR_V_NORM_MAX)
    if v_norm > v_max:
        log("[COLOR] %s (%s): v=%.2f trop bas (bord image)，跳过", name, label, v_norm)
        return remaining
    # bleu ciel : v trop haut dans l'image (proche du haut)
    if name == "parcel_4" and v_norm < 0.45:
        log("[COLOR] %s (%s): v=%.2f trop haut (ciel)，跳过", name, label, v_norm)
        return remaining

    remaining = cv2.bitwise_and(remaining, cv2.bitwise_not(blob))
    mask_full = np.zeros((h, w), dtype=np.uint8)
    mask_full[v0:, :] = blob

    center_base = None
    depth_m = None
    rgb_src = None
    if cam is not None and tf_reader is not None:
        if depth is not None:
            depth_m = cam.median_depth_in_mask(
                depth, mask_full, z_min=DEPTH_Z_MIN, z_max=DEPTH_Z_MAX)
        center_base, rgb_src = _rgb_depth_table_point(
            cam, tf_reader, u_px, v_px, depth_m, log, name, via)

    hits.append({
        "name": name,
        "color": label,
        "u_norm": u_px / w,
        "v_norm": v_norm,
        "u_px": u_px,
        "v_px": v_px,
        "area": area,
        "depth_m": depth_m,
        "center_base": center_base,
        "rgb_source": rgb_src,
    })
    if center_base is not None:
        log("[COLOR] %s (%s): u=%.2f v=%.2f depth=%s → base (%.3f, %.3f, %.3f) area=%d [%s]",
            name, label, hits[-1]["u_norm"], v_norm,
            "%.3f" % depth_m if depth_m is not None else "—",
            center_base[0], center_base[1], center_base[2], area,
            rgb_src or via)
    else:
        log("[COLOR] %s (%s): u=%.2f v=%.2f area=%d [%s]",
            name, label, hits[-1]["u_norm"], v_norm, area, via)
    return remaining


def _detect_color_parcels(rgb, depth, cam, tf_reader, log):
    """
    Segmentation couleur LAB (matériaux scene1.yaml), secours HSV.
    Cherche dans la bande table ; si refs trop loin, élargit ROI / seuils.
    (Les colis ont souvent depth NaN → ne pas gate la recherche sur depth.)
    """
    if not _HAS_CV2:
        log("[COLOR] opencv-python 未安装，跳过颜色识别")
        return []

    h, w = rgb.shape[:2]
    v0 = int(h * COLOR_ROI_V_START)
    roi = rgb[v0:, :, :]
    rh, rw = roi.shape[:2]
    lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    table_lab = _bgr_to_lab(TABLE_REF_BGR)
    table_dist = np.sqrt(np.sum((lab_roi - table_lab) ** 2, axis=2))
    non_table = ((table_dist > TABLE_LAB_DIST_MAX).astype(np.uint8)) * 255

    # Bande table (tête pitch=20) — ignore un peu le haut du ROI (ciel)
    table_band = np.zeros((rh, rw), dtype=np.uint8)
    band_v0 = int(rh * COLOR_TABLE_BAND_V0)
    table_band[band_v0:, :] = 255
    remaining = cv2.bitwise_and(non_table, table_band)
    depth_gate = _depth_gate_roi(depth, h, w, v0)
    thr_boost = 0.0
    log("[COLOR] ROI gate: search=%d non_table=%d depth_ok=%d band_v0=%.2f",
        int(remaining.sum()) // 255,
        int(non_table.sum()) // 255,
        int(depth_gate.sum()) // 255,
        COLOR_TABLE_BAND_V0)

    def _dmin_any(mask_bool):
        if int(mask_bool.sum()) < 100:
            return 999.0
        return min(
            float(np.sqrt(np.sum(
                (lab_roi - _bgr_to_lab(ref_bgr)) ** 2, axis=2)[mask_bool].min()))
            for _, _, ref_bgr in PARCEL_REF_COLORS)

    search_probe = remaining > 0
    dmin_any = _dmin_any(search_probe)
    if dmin_any > COLOR_SOFT_DMIN:
        soft_non_table = ((table_dist > COLOR_SOFT_TABLE_THR).astype(np.uint8)) * 255
        remaining = cv2.bitwise_and(soft_non_table, table_band)
        thr_boost = COLOR_SOFT_THR_BOOST
        dmin_any = _dmin_any(remaining > 0)
        log("[COLOR] dmin_any=%.1f → soft non_table (thr=%.0f) boost=+%.0f",
            dmin_any, COLOR_SOFT_TABLE_THR, thr_boost)
        try:
            dbg = _labo_image_path()
            cv2.imwrite(dbg, rgb)
            log("[COLOR] debug RGB sauvé: %s", dbg)
        except Exception:
            pass

    # Dernier recours : tout le ROI (colis hors bande basse)
    if dmin_any > COLOR_FULLROI_DMIN:
        remaining = np.ones((rh, rw), dtype=np.uint8) * 255
        thr_boost = max(thr_boost, COLOR_SOFT_THR_BOOST + 8.0)
        dmin_any = _dmin_any(remaining > 0)
        log("[COLOR] dmin_any=%.1f → FULL ROI search boost=+%.0f",
            dmin_any, thr_boost)

    search = remaining > 0
    if int(search.sum()) > 100:
        for name, label, ref_bgr in PARCEL_REF_COLORS:
            ref_lab = _bgr_to_lab(ref_bgr)
            dist = np.sqrt(np.sum((lab_roi - ref_lab) ** 2, axis=2))
            dmin = float(dist[search].min())
            thr = LAB_COLOR_DIST_BY_NAME.get(name, LAB_COLOR_DIST_MAX) + thr_boost
            log("[COLOR] diag %s (%s): LAB dmin=%.1f thr=%.1f",
                name, label, dmin, thr)
        mean_bgr = roi[search].mean(axis=0)
        log("[COLOR] diag mean BGR search: (%.0f, %.0f, %.0f)",
            mean_bgr[0], mean_bgr[1], mean_bgr[2])

    hits = []
    detected = set()

    # Assignation pixel → couleur la plus proche (évite que jaune mange bleu)
    ref_labs = []
    ref_meta = []
    thr_list = []
    for name, label, ref_bgr in PARCEL_REF_COLORS:
        ref_labs.append(_bgr_to_lab(ref_bgr))
        ref_meta.append((name, label, ref_bgr))
        thr_list.append(
            LAB_COLOR_DIST_BY_NAME.get(name, LAB_COLOR_DIST_MAX) + thr_boost)
    ref_labs = np.stack(ref_labs, axis=0)  # 4×3
    diff = lab_roi[:, :, None, :] - ref_labs[None, None, :, :]
    dist_all = np.sqrt(np.sum(diff * diff, axis=3))
    best_k = np.argmin(dist_all, axis=2)
    best_d = np.min(dist_all, axis=2)
    search_m = remaining > 0

    for k, (name, label, _) in enumerate(ref_meta):
        thr = thr_list[k]
        mask = ((best_k == k) & (best_d < thr) & search_m).astype(np.uint8) * 255
        kk = 3 if name != "parcel_4" else 5
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((kk, kk), np.uint8))
        n_before = len(hits)
        remaining = _append_color_hit(
            hits, remaining, name, label, mask, h, w, v0,
            depth, cam, tf_reader, log, via="LAB")
        if len(hits) > n_before:
            detected.add(name)

    for name, label, lo, hi in PARCEL_HSV_FALLBACK:
        if name in detected:
            continue
        mask = cv2.inRange(
            hsv_roi, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.bitwise_and(mask, remaining)
        n_before = len(hits)
        remaining = _append_color_hit(
            hits, remaining, name, label, mask, h, w, v0,
            depth, cam, tf_reader, log, via="HSV")
        if len(hits) > n_before:
            detected.add(name)

    return hits


# =============================================================================
# PERCEPTION — ÉTAPE 3 : fusion LiDAR + couleur
# =============================================================================

def _lidar_norm_uv(cluster, xs, ys):
    """
    Projette un centre LiDAR 3D → (u, v) image approx.
    Calibré tête orga pitch=20 (pas l'ancien look_down).

    Avec ≥2 amas : rang relatif (robuste multi-seed) —
      x↑ (rangée proche) → v↑ ; y↑ (droite) → u↑
    Sinon : affine absolu HEAD20_* (secours 1 amas).
    """
    x, y, _ = cluster["center"]
    if xs is not None and ys is not None and len(xs) >= 2:
        x_min, x_max = float(min(xs)), float(max(xs))
        y_min, y_max = float(min(ys)), float(max(ys))
        tx = 0.5 if (x_max - x_min) < 1e-6 else (x - x_min) / (x_max - x_min)
        ty = 0.5 if (y_max - y_min) < 1e-6 else (y - y_min) / (y_max - y_min)
        u = HEAD20_UV_U_LEFT + HEAD20_UV_U_SPAN * ty
        v = HEAD20_UV_V_FAR + HEAD20_UV_V_SPAN * tx
    else:
        # absolu pitch=20 : inverse approx de _uv_to_table_xy
        ax, bx = HEAD20_X_FROM_V
        ay, by = HEAD20_Y_FROM_U
        v = 0.5 if abs(bx) < 1e-9 else (x - ax) / bx
        u = 0.5 if abs(by) < 1e-9 else (y - ay) / by
    u = max(0.05, min(0.95, float(u)))
    v = max(0.40, min(0.99, float(v)))
    return u, v


def _world_to_ik(x, y, z):
    ox, oy, oz = WORLD_TO_IK_OFFSET
    return x + ox, y + oy, z + oz


def _uv_to_table_xy(u_norm, v_norm):
    """
    u,v image → x,y table (base_link IK).
    Secours seulement (rgb-ray fail) — calibré pitch=20.
    Préférer rgb-depth / pixel_ray quand dispo.
    """
    ax, bx = HEAD20_X_FROM_V
    ay, by = HEAD20_Y_FROM_U
    x = ax + bx * float(v_norm)
    y = ay + by * float(u_norm)
    x = max(TABLE_X_RANGE[0], min(TABLE_X_RANGE[1], x))
    y = max(TABLE_Y_RANGE[0], min(TABLE_Y_RANGE[1], y))
    return x, y


def _table_z_from_lidar(lidar_clusters):
    if lidar_clusters:
        return float(np.median([c["center"][2] for c in lidar_clusters]))
    return TABLE_PARCEL_Z


def _parcel_from_color_hit(ch, z, log, source="couleur"):
    """Position 3D seulement si center_base valide ; sinon None (ne pas inventer UV)."""
    if ch.get("center_base") is not None:
        x, y, z = ch["center_base"]
        source = ch.get("rgb_source") or "rgb-depth"
        if not _is_valid_parcel_table_point(x, y, z, rgb_backed=True):
            log("[FUSE] %s (%s): center_base hors table, ignoré",
                ch["name"], ch["color"])
            return None
    else:
        # UV→XY trop fragile (logs: x≈0.16 hors LiDAR) → ne pas inventer une pose
        log("[FUSE] %s (%s): pas de 3D caméra fiable — nom seulement",
            ch["name"], ch["color"])
        return None
    log("[FUSE] %s (%s) ← %s (%.3f, %.3f, %.3f)",
        ch["name"], ch["color"], source, x, y, z)
    return {
        "name": ch["name"],
        "color": ch["color"],
        "center": (x, y, z),
        "size_xy": (0.06, 0.05),
        "n_points": 0,
        "source": source,
    }


def _fuse_color_names_to_lidar(color_hits, lidar_clusters, log):
    """
    Chaque hit couleur a un nom. Coller au LiDAR le plus proche en UV image
    (pas aux poses seed0).
    """
    if not color_hits or not lidar_clusters:
        return [], set()

    xs = [c["center"][0] for c in lidar_clusters]
    ys = [c["center"][1] for c in lidar_clusters]
    remaining = list(range(len(lidar_clusters)))
    fused = []
    used_lidar = set()
    matched_names = set()

    ordered = sorted(
        color_hits,
        key=lambda ch: 0 if ch.get("center_base") is not None else 1)

    x_mid = float(globals().get("LIDAR_GRID_X_MID", 0.381))
    y_mid = float(globals().get("LIDAR_GRID_Y_MID", -0.214))

    def _fuse_cost(ch, li):
        u, v = _lidar_norm_uv(lidar_clusters[li], xs, ys)
        d = math.hypot(ch["u_norm"] - u, ch["v_norm"] - v)
        cx, cy, _ = lidar_clusters[li]["center"]
        cb = ch.get("center_base")
        if cb is not None:
            # XY caméra pèse plus que UV relatif (head20: UV plantait
            # jaune/bleu sur la rangée basse → Δxy ~19 cm).
            d = 0.30 * d + 0.70 * math.hypot(cx - cb[0], cy - cb[1]) / 0.25
        # Colonne : parcel_1/2 = gauche, parcel_3/4 = droite.
        name = ch["name"]
        if name in ("parcel_1", "parcel_2") and cx > x_mid + 0.02:
            d += 0.85
        elif name in ("parcel_3", "parcel_4") and cx < x_mid - 0.02:
            d += 0.85
        # Rangée : parcel_1/3 = y bas (loin), parcel_2/4 = y haut (proche).
        # Seed4 : orange↔bleu swapaient sans ça.
        if name in ("parcel_1", "parcel_3") and cy > y_mid + 0.02:
            d += 0.85
        elif name in ("parcel_2", "parcel_4") and cy < y_mid - 0.02:
            d += 0.85
        return d

    for ch in ordered:
        name = ch["name"]
        if name not in PARCEL_NAMES or name in matched_names:
            continue
        # Préférer bonne colonne ET bonne rangée
        prefer = []
        for li in remaining:
            cx, cy = lidar_clusters[li]["center"][0], lidar_clusters[li]["center"][1]
            col_ok = True
            row_ok = True
            if name in ("parcel_1", "parcel_2"):
                col_ok = cx <= x_mid + 0.02
            elif name in ("parcel_3", "parcel_4"):
                col_ok = cx >= x_mid - 0.02
            if name in ("parcel_1", "parcel_3"):
                row_ok = cy <= y_mid + 0.02
            elif name in ("parcel_2", "parcel_4"):
                row_ok = cy >= y_mid - 0.02
            if col_ok and row_ok:
                prefer.append(li)
        if not prefer:
            for li in remaining:
                cx = lidar_clusters[li]["center"][0]
                if name in ("parcel_1", "parcel_2") and cx <= x_mid + 0.02:
                    prefer.append(li)
                elif name in ("parcel_3", "parcel_4") and cx >= x_mid - 0.02:
                    prefer.append(li)
        candidates = prefer if prefer else remaining
        best_li, best_d = None, float("inf")
        for li in candidates:
            d = _fuse_cost(ch, li)
            if d < best_d:
                best_d, best_li = d, li
        if best_li is None or best_d > MAX_FUSE_UV_DIST + 0.15:
            log("[FUSE] %s (%s): aucun LiDAR UV proche (best=%.3f) — skip",
                name, ch["color"], best_d if best_li is not None else -1)
            continue
        remaining.remove(best_li)
        used_lidar.add(best_li)
        matched_names.add(name)
        parcel = dict(lidar_clusters[best_li])
        parcel["name"] = name
        parcel["color"] = ch["color"]
        parcel["source"] = "lidar+color"
        fused.append(parcel)
        cx, cy, cz = parcel["center"]
        log("[FUSE] %s (%s) ← LiDAR←couleur UV (%.3f, %.3f, %.3f) d_uv=%.3f",
            name, ch["color"], cx, cy, cz, best_d)

    return fused, used_lidar


def _fix_row_structure(parcels, log):
    """
    1) Si orange est sur la rangée haute et bleu en bas → swap centres (seed4).
    2) Aligne Y droite sur voisins gauche (1↔3, 2↔4).
    3) Si rangée haute trop collée à la basse (|y2-y1|<0.14) → écarte de row_dy.
    """
    by_name = {p["name"]: p for p in parcels if p["name"] in PARCEL_NAMES}
    p1 = by_name.get("parcel_1")
    p2 = by_name.get("parcel_2")
    p3 = by_name.get("parcel_3")
    p4 = by_name.get("parcel_4")
    if p1 is None or p2 is None:
        return parcels

    y1, y2 = p1["center"][1], p2["center"][1]
    # s'assurer brown plus bas (plus négatif) que yellow
    if y1 > y2 + 0.05:
        log("[FUSE] row-fix: swap parcel_1↔parcel_2 (y inversés)")
        c1, c2 = p1["center"], p2["center"]
        p1["center"], p2["center"] = (c1[0], c2[1], c1[2]), (c2[0], c1[1], c2[2])
        y1, y2 = p1["center"][1], p2["center"][1]

    if p3 is not None and p4 is not None:
        y3, y4 = p3["center"][1], p4["center"][1]
        d3_lo = abs(y3 - y1)
        d3_hi = abs(y3 - y2)
        # Orange trop proche de la rangée jaune → swap avec bleu
        if d3_hi + 0.04 < d3_lo:
            log("[FUSE] row-fix: swap parcel_3↔parcel_4 "
                "(orange y=%.3f sur rangée haute)", y3)
            c3, c4 = p3["center"], p4["center"]
            p3["center"], p4["center"] = c4, c3
            p3["source"] = (p3.get("source") or "?") + "+row-swap"
            p4["source"] = (p4.get("source") or "?") + "+row-swap"
            y3, y4 = p3["center"][1], p4["center"][1]

        # Snap Y voisins horizontaux (max 12 cm)
        for left_p, right_p, tag in ((p1, p3, "brown←orange"), (p2, p4, "yellow←blue")):
            if left_p is None or right_p is None:
                continue
            ly, ry = left_p["center"][1], right_p["center"][1]
            if 0.02 < abs(ly - ry) <= 0.12:
                rx, _, rz = right_p["center"]
                if "center_raw" not in right_p:
                    right_p["center_raw"] = right_p["center"]
                right_p["center"] = (rx, ly, rz)
                right_p["source"] = (right_p.get("source") or "?") + "+row-y"
                log("[FUSE] row-fix %s: y %.3f → %.3f", tag, ry, ly)

    # Écarter rangée haute si collapse (jaune trop proche marron)
    _, row_dy = _grid_spacing()
    y1, y2 = p1["center"][1], p2["center"][1]
    gap = y2 - y1  # doit être ~+0.21
    if gap < 0.16:
        target_y2 = y1 + row_dy
        max_d = float(globals().get("FUSE_MAX_RESHAPE_XY", 0.18) or 0.18)
        if abs(target_y2 - y2) <= max_d + 0.05:
            for name, target_y in (("parcel_2", target_y2),):
                p = by_name.get(name)
                if p is None:
                    continue
                cx, cy, cz = p["center"]
                if "center_raw" not in p:
                    p["center_raw"] = (cx, cy, cz)
                p["center"] = (cx, target_y, cz)
                p["source"] = (p.get("source") or "?") + "+row-gap"
                log("[FUSE] row-fix %s: y %.3f → %.3f (gap bas→haut)",
                    name, cy, target_y)
            p4 = by_name.get("parcel_4")
            p2 = by_name.get("parcel_2")
            if p4 is not None and p2 is not None:
                cx, cy, cz = p4["center"]
                ty = p2["center"][1]
                if abs(cy - ty) > 0.02:
                    if "center_raw" not in p4:
                        p4["center_raw"] = (cx, cy, cz)
                    p4["center"] = (cx, ty, cz)
                    p4["source"] = (p4.get("source") or "?") + "+row-gap"
                    log("[FUSE] row-fix parcel_4: y %.3f → %.3f (suit jaune)",
                        cy, ty)

    return _sort_parcels(list(by_name.values()))


def _inject_rgb_parcels(parcels, color_hits, log):
    """Colis sans LiDAR fiable : injecter couleur seulement si 3D caméra valide."""
    by_name = {p["name"]: p for p in parcels if p["name"] in PARCEL_NAMES}
    hits_by_name = {ch["name"]: ch for ch in color_hits if ch["name"] in PARCEL_NAMES}

    for name in PARCEL_NAMES:
        ch = hits_by_name.get(name)
        if ch is None:
            continue
        cur = by_name.get(name)
        cb = ch.get("center_base")
        # LiDAR sur mauvaise rangée vs RGB : corriger (jaune/bleu → rangée haute)
        if (cur is not None and cb is not None
                and name in ("parcel_2", "parcel_4")
                and _is_valid_parcel_table_point(
                    cb[0], cb[1], cb[2], rgb_backed=True)):
            cy = cur["center"][1]
            if cy < cb[1] - 0.08:
                cx, _, cz = cur["center"]
                cur = dict(cur)
                if _is_lidar_backed(cur.get("source", "")):
                    cur["center"] = (cx, cb[1], cz)
                    cur["source"] = (cur.get("source") or "lidar") + "+rgb-y"
                else:
                    cur["center"] = (cb[0], cb[1], cb[2])
                    cur["source"] = ch.get("rgb_source") or "rgb-depth"
                cur["color"] = ch["color"]
                by_name[name] = cur
                log("[FUSE] %s: LiDAR y=%.3f trop bas vs RGB y=%.3f → corrigé",
                    name, cy, cb[1])
                continue
        # garder LiDAR fiable ; remplacer grid / row-infer / absent
        if cur is not None and _is_reliable_lidar_parcel(cur):
            # juste enrichir le flag couleur si déjà bon LiDAR
            if "color" not in (cur.get("source") or ""):
                cur = dict(cur)
                cur["source"] = (cur.get("source") or "lidar") + "+color"
                cur["color"] = ch["color"]
                by_name[name] = cur
            continue
        if cur is not None and _is_lidar_backed(cur.get("source", "")) and (
                "grid" not in (cur.get("source") or "")
                and "row-infer" not in (cur.get("source") or "")):
            cur = dict(cur)
            cur["source"] = (cur.get("source") or "lidar") + "+color"
            cur["color"] = ch["color"]
            by_name[name] = cur
            continue
        p = _parcel_from_color_hit(ch, TABLE_PARCEL_Z, log, ch.get("rgb_source") or "couleur")
        if p is None:
            continue
        by_name[name] = p
        log("[FUSE] %s (%s) ← injecté %s (vraie détection caméra)",
            name, p["color"], p["source"])

    return _sort_parcels(list(by_name.values()))


def _fix_collapsed_high_row(parcels, color_hits, log):
    """
    Si jaune/bleu n'ont pas clairement la rangée haute, reprendre Y RGB
    ou +row_dy. Ne lifter QUE si la rangée basse est crédible (anti-chute).
    """
    if not bool(globals().get("FUSE_ENABLE_ROW_LIFT", True)):
        log("[FUSE] row-lift OFF (contrat zone stable)")
        return parcels
    by_name = {p["name"]: dict(p) for p in parcels if p["name"] in PARCEL_NAMES}
    if len(by_name) < 2:
        return parcels
    hits = {ch["name"]: ch for ch in (color_hits or [])
            if ch.get("name") in PARCEL_NAMES}
    # Rangée basse = y clairement bas. Médiane avec brown à -0.05 → lift à +0.04
    # (hors table) → bras tire de travers → chute.
    low_ys = [
        by_name[n]["center"][1]
        for n in ("parcel_1", "parcel_3") if n in by_name
        if by_name[n]["center"][1] <= -0.18
    ]
    if not low_ys:
        log("[FUSE] skip row-lift: pas de rangée basse crédible")
        return parcels
    y_low = float(np.median(low_ys))
    _, row_dy = _grid_spacing()
    high_ok = y_low + 0.65 * abs(row_dy)
    y_hi_max = TABLE_Y_RANGE[1] - 0.01  # ne jamais sortir de la table

    for name in ("parcel_2", "parcel_4"):
        p = by_name.get(name)
        if p is None:
            continue
        cx, cy, cz = p["center"]
        if cy > high_ok:
            continue
        ch = hits.get(name)
        fixed = False
        if ch is not None and ch.get("center_base") is not None:
            bx, by, bz = ch["center_base"]
            if (by > y_low + 0.10 and _is_valid_parcel_table_point(
                    bx, by, bz, rgb_backed=True)):
                if _is_lidar_backed(p.get("source", "")):
                    p["center"] = (cx, by, cz)
                    p["source"] = (p.get("source") or "?") + "+rgb-y"
                else:
                    p["center"] = (bx, by, bz)
                    p["source"] = ch.get("rgb_source") or "rgb-depth"
                p["color"] = ch["color"]
                fixed = True
                log("[FUSE] %s: collapse rangée → y %.3f→%.3f (RGB)",
                    name, cy, p["center"][1])
        if not fixed:
            new_y = min(y_low + abs(row_dy), y_hi_max)
            if new_y <= cy + 0.02:
                continue
            if not _is_valid_parcel_table_point(cx, new_y, cz):
                log("[FUSE] skip row-lift %s: y=%.3f hors table", name, new_y)
                continue
            p["center"] = (cx, new_y, cz)
            p["source"] = (p.get("source") or "?") + "+row-lift"
            log("[FUSE] %s: collapse rangée → y %.3f→%.3f (row-lift)",
                name, cy, new_y)
        by_name[name] = p

    return _sort_parcels(list(by_name.values()))


def _fill_missing_parcels(parcels, color_hits, log):
    """Complète colis manquants : rgb-ray d'abord, row-infer relatif (anti-seed0)."""
    by_name = {p["name"]: p for p in parcels if p["name"] in PARCEL_NAMES}
    hits_by_name = {ch["name"]: ch for ch in color_hits if ch["name"] in PARCEL_NAMES}
    _, row_dy = _grid_spacing()
    if len(by_name) >= len(PARCEL_NAMES):
        return _sort_parcels(list(by_name.values()))

    for name in PARCEL_NAMES:
        if name in by_name:
            continue
        ch = hits_by_name.get(name)
        if ch is not None and ch.get("center_base") is not None:
            src = ch.get("rgb_source") or "rgb-ray"
            if _is_rgb_backed(src):
                p = _parcel_from_color_hit(ch, TABLE_PARCEL_Z, log, src)
                if p is not None:
                    by_name[name] = p
                    continue

    # row-infer : même x que le voisin, y décalé d'un pas de grille relatif
    for left, right in (("parcel_1", "parcel_2"), ("parcel_3", "parcel_4")):
        for missing, partner, sign in (
                (right, left, +1.0), (left, right, -1.0)):
            if missing in by_name or partner not in by_name:
                continue
            px, py, pz = by_name[partner]["center"]
            infer_y = py + sign * abs(row_dy)
            p = {
                "name": missing,
                "color": _color_label_for_parcel(missing),
                "center": (px, infer_y, pz),
                "size_xy": (0.06, 0.05),
                "n_points": 0,
                "source": "row-infer",
            }
            by_name[missing] = p
            log("[FUSE] %s (%s) ← row-infer relatif (%.3f, %.3f, %.3f) depuis %s",
                missing, p["color"], px, infer_y, pz, partner)

    return _sort_parcels(list(by_name.values()))


def _parcels_from_scene_layout(log):
    """Secours : positions nominales scene1 seed0 (world→IK), si capteurs échouent."""
    parcels = []
    for name in PARCEL_NAMES:
        wx, wy, wz = PARCEL_WORLD_POS[name]
        cx, cy, cz = _world_to_ik(wx, wy, wz)
        parcels.append({
            "name": name,
            "color": "layout",
            "center": (cx, cy, cz),
            "size_xy": (0.06, 0.05),
            "n_points": 0,
        })
        log("[FUSE] %s ← layout scène IK (%.3f, %.3f, %.3f)", name, cx, cy, cz)
    return _sort_parcels(parcels)


def _parcels_from_color_only(color_hits, log, lidar_clusters=None):
    """Secours : couleurs OK mais LiDAR vide → positions estimées sur la table."""
    z = _table_z_from_lidar(lidar_clusters or [])
    parcels = []
    for ch in color_hits:
        p = _parcel_from_color_hit(ch, z, log, "couleur seule")
        if p is not None:
            parcels.append(p)
        else:
            # dernier recours : UV recalibré (mieux que rien si 0 LiDAR)
            x, y = _uv_to_table_xy(ch["u_norm"], ch["v_norm"])
            parcels.append({
                "name": ch["name"],
                "color": ch["color"],
                "center": (x, y, TABLE_PARCEL_Z),
                "size_xy": (0.06, 0.05),
                "n_points": 0,
                "source": "couleur-uv",
            })
            log("[FUSE] %s (%s) ← couleur-uv secours (%.3f, %.3f, %.3f)",
                ch["name"], ch["color"], x, y, TABLE_PARCEL_Z)
    return _sort_parcels(parcels)


def _expected_parcel_positions():
    return {name: _world_to_ik(*PARCEL_WORLD_POS[name]) for name in PARCEL_NAMES}


def _color_label_for_parcel(name):
    for n, label, _ in PARCEL_REF_COLORS:
        if n == name:
            return label
    return "unknown"


def _is_rgb_backed(source):
    s = source or ""
    return (
        s in ("rgb-ray", "rgb-depth", "rgb-depth+zsnap", "couleur")
        or s.startswith("rgb-")
        or "couleur" in s
    )


def _is_lidar_backed(source):
    s = source or ""
    return "lidar" in s and "couleur" not in s


def _snap_row_x_from_lidar_neighbors(parcels, log):
    """Colis couleur-only sur même colonne : reprendre x du voisin LiDAR."""
    by_name = {p["name"]: p for p in parcels if p["name"] in PARCEL_NAMES}
    for left, right in (("parcel_1", "parcel_2"), ("parcel_3", "parcel_4")):
        for src_name, dst_name in ((left, right), (right, left)):
            sp = by_name.get(src_name)
            dp = by_name.get(dst_name)
            if sp is None or dp is None:
                continue
            if not _is_lidar_backed(sp.get("source", "")):
                continue
            if _is_lidar_backed(dp.get("source", "")):
                continue
            if _is_rgb_backed(dp.get("source", "")):
                continue
            cx, cy, cz = dp["center"]
            sx = sp["center"][0]
            dp["center"] = (sx, cy, cz)
            dp["source"] = "couleur+row-x"
            log("[FUSE] %s: x %.3f → %.3f (même colonne que %s)",
                dst_name, cx, sx, src_name)
    return parcels


def _grid_spacing():
    """Espacement grille 2×2 typique (mètres IK) — indépendant du seed."""
    return 0.15, 0.21  # col_dx (proche→loin), row_dy (y bas→haut)


def _is_reliable_lidar_parcel(p):
    if p is None:
        return False
    return (_is_lidar_backed(p.get("source", ""))
            and p.get("n_points", 0) >= LIDAR_SEED_MIN_POINTS)


def _snap_grid_geometry(parcels, log):
    """
    Contrainte 2×2 scene1 : colonne droite LiDAR fiable → corrige X colonne gauche.
    Même logique pour Y entre voisins de rangée (parcel_1↔3, parcel_2↔4).
    """
    if not bool(globals().get("FUSE_ENABLE_GRID_SNAP", True)):
        log("[FUSE] grid-snap OFF (contrat zone stable — LiDAR←couleur)")
        return parcels
    by_name = {p["name"]: p for p in parcels if p["name"] in PARCEL_NAMES}
    if len(by_name) < 2:
        return parcels

    col_dx, row_dy = _grid_spacing()
    p1, p2 = by_name.get("parcel_1"), by_name.get("parcel_2")
    p3, p4 = by_name.get("parcel_3"), by_name.get("parcel_4")

    right_x = None
    if _is_reliable_lidar_parcel(p3):
        right_x = p3["center"][0]
    elif _is_reliable_lidar_parcel(p4):
        right_x = p4["center"][0]
    elif p3 is not None and _is_lidar_backed(p3.get("source", "")):
        right_x = p3["center"][0]
    elif p4 is not None and _is_lidar_backed(p4.get("source", "")):
        right_x = p4["center"][0]

    if right_x is not None:
        # Colonne « loin » doit être crédible. Si right_x≈0.18 (colonne proche
        # mal nommée), left=right-0.15→0.03 hors table (mission seed0).
        if right_x < 0.30:
            log("[FUSE] skip grid-x: right_x=%.3f trop proche (pas colonne loin)",
                right_x)
            right_x = None

    if right_x is not None:
        left_x = right_x - col_dx
        left_x = max(float(TABLE_X_RANGE[0]), min(float(TABLE_X_RANGE[1]), left_x))
        # Si RGB gauche fiable, ancrer aussi la colonne droite (anti biais résiduel).
        rgb_left = []
        for name in ("parcel_1", "parcel_2"):
            p = by_name.get(name)
            if p is not None and _is_rgb_backed(p.get("source", "")):
                rgb_left.append(p["center"][0])
        if rgb_left:
            left_rgb = float(np.median(rgb_left))
            # RGB gauche hors table / absurde → ne pas blender
            if left_rgb < TABLE_X_RANGE[0] - 0.05 or left_rgb > TABLE_X_RANGE[1]:
                log("[FUSE] skip RGB-left blend: left_rgb=%.3f hors plage", left_rgb)
            else:
                expect_r = left_rgb + col_dx
                if abs(right_x - expect_r) > 0.025:
                    blend_r = 0.55 * right_x + 0.45 * expect_r
                    log("[FUSE] colonne droite x %.3f → %.3f (blend RGB gauche=%.3f)",
                        right_x, blend_r, left_rgb)
                    for name in ("parcel_3", "parcel_4"):
                        rp = by_name.get(name)
                        if rp is None:
                            continue
                        rx, ry, rz = rp["center"]
                        if "center_raw" not in rp:
                            rp["center_raw"] = (rx, ry, rz)
                        rp["center"] = (blend_r, ry, rz)
                        if "grid-x" not in (rp.get("source") or ""):
                            rp["source"] = (rp.get("source") or "?") + "+grid-x"
                    right_x = blend_r
                    left_x = max(float(TABLE_X_RANGE[0]), right_x - col_dx)

        for name in ("parcel_1", "parcel_2"):
            p = by_name.get(name)
            if p is None:
                continue
            cx, cy, cz = p["center"]
            if abs(cx - left_x) < 0.015 and "grid" in (p.get("source") or ""):
                continue
            # Ne pas tirer hors table
            if left_x < TABLE_X_RANGE[0] - 0.01:
                log("[FUSE] skip grid-x %s: left_x=%.3f hors table", name, left_x)
                continue
            max_d = float(globals().get("FUSE_MAX_RESHAPE_XY", 0.04) or 0.04)
            if abs(cx - left_x) > max_d:
                log("[FUSE] skip grid-x %s: Δx=%.3f > %.3f (garde mesure)",
                    name, abs(cx - left_x), max_d)
                continue
            # Garder la pose LiDAR réelle pour la saisie/touch (évite viser le vide)
            if "center_raw" not in p:
                p["center_raw"] = (cx, cy, cz)
            p["center"] = (left_x, cy, cz)
            p["source"] = (p.get("source") or "?") + "+grid-x"
            log("[FUSE] %s: x %.3f → %.3f (grille 2×2, colonne droite x=%.3f)",
                name, cx, left_x, right_x)

    # Y : ne JAMAIS écraser un LiDAR fiable (logs head20 : jaune collé
    # sur y du bleu quand bleu est sur la mauvaise rangée → Δxy ~19 cm).
    # grid-y = correction douce seulement si Δy déjà petit, ou si gauche faible.
    row_pairs = (("parcel_1", "parcel_3"), ("parcel_2", "parcel_4"))
    for left_name, right_name in row_pairs:
        lp, rp = by_name.get(left_name), by_name.get(right_name)
        if lp is None or rp is None:
            continue
        if not _is_reliable_lidar_parcel(rp):
            continue
        _, ry, rz = rp["center"]
        cx, cy, cz = lp["center"]
        if abs(cy - ry) < 0.015:
            continue

        # Ancre droite sur la mauvaise rangée ? (ex. bleu ≈ y marron/orange)
        other_left = "parcel_1" if left_name == "parcel_2" else "parcel_2"
        other_right = "parcel_3" if right_name == "parcel_4" else "parcel_4"
        wrong_row = False
        for oname in (other_left, other_right):
            op = by_name.get(oname)
            if op is None:
                continue
            if abs(ry - op["center"][1]) < 0.08:
                wrong_row = True
                log("[FUSE] skip grid-y %s←%s: %s y=%.3f trop proche de %s "
                    "(mauvaise rangée)",
                    left_name, right_name, right_name, ry, oname)
                break
        if wrong_row:
            continue

        # LiDAR+couleur à gauche : garder le y réel (saisie)
        if _is_reliable_lidar_parcel(lp):
            log("[FUSE] %s: conserve y=%.3f (LiDAR fiable, pas de grid-y "
                "vers %s=%.3f)", left_name, cy, right_name, ry)
            continue

        # Δy trop grand → association suspecte, ne pas forcer
        if abs(cy - ry) > 0.10:
            log("[FUSE] skip grid-y %s←%s: Δy=%.3f trop grand",
                left_name, right_name, abs(cy - ry))
            continue

        lp["center"] = (cx, ry, cz if abs(cz - TABLE_PARCEL_Z) < 0.08 else rz)
        lp["source"] = (lp.get("source") or "?") + "+grid-y"
        log("[FUSE] %s: y %.3f → %.3f (même rangée que %s, gauche faible)",
            left_name, cy, ry, right_name)

    if p2 is not None and _is_reliable_lidar_parcel(p4):
        if _is_rgb_backed(p2.get("source", "")) or _is_reliable_lidar_parcel(p2):
            log("[FUSE] parcel_2: conserve source=%s (pas de grille)",
                p2.get("source"))
        elif "row-infer" in (p2.get("source") or "") or not _is_lidar_backed(
                p2.get("source", "")):
            _, py4, _ = p4["center"]
            # Ne pas coller le jaune si bleu est déjà sur la rangée basse
            row_conflict = False
            for oname in ("parcel_1", "parcel_3"):
                op = by_name.get(oname)
                if op is not None and abs(py4 - op["center"][1]) < 0.08:
                    row_conflict = True
                    log("[FUSE] skip grid-2x2 yellow: parcel_4 y=%.3f ≈ %s",
                        py4, oname)
                    break
            if not row_conflict:
                left_x = (by_name["parcel_1"]["center"][0]
                          if p1 else p2["center"][0])
                p2["center"] = (left_x, py4, TABLE_PARCEL_Z)
                p2["source"] = "grid-2x2"
                log("[FUSE] %s (yellow) ← grille 2×2 (%.3f, %.3f, %.3f) "
                    "depuis %s",
                    p2["name"], left_x, py4, TABLE_PARCEL_Z, "parcel_4")

    return _fix_row_structure(_sort_parcels(list(by_name.values())), log)


def _name_clusters_spatial(lidar_clusters, log):
    """Associe chaque cluster LiDAR à un colis via grille 2×2 relative (anti-seed0)."""
    return _assign_names_relative_grid(
        lidar_clusters, log, source_tag="lidar-spatial")


def _fuse_color_lidar_hungarian(color_hits, lidar_clusters, table_z, log):
    """Niveau 2 : assignation optimale couleur ↔ cluster LiDAR (Hongrois)."""
    nc, nl = len(color_hits), len(lidar_clusters)
    if nc == 0 or nl == 0:
        return [], set()

    xs = [c["center"][0] for c in lidar_clusters]
    ys = [c["center"][1] for c in lidar_clusters]
    cost = np.full((nc, nl), HUNGARIAN_BIG_COST)
    for i, ch in enumerate(color_hits):
        for j, lc in enumerate(lidar_clusters):
            u, v = _lidar_norm_uv(lc, xs, ys)
            du = ch["u_norm"] - u
            dv = ch["v_norm"] - v
            d_uv = du * du + dv * dv
            cb = ch.get("center_base")
            if cb is not None:
                cx, cy, _ = lc["center"]
                d_xy = (math.hypot(cx - cb[0], cy - cb[1]) / 0.25) ** 2
                cost[i, j] = 0.30 * d_uv + 0.70 * d_xy
            else:
                cost[i, j] = d_uv

    pairs = _hungarian_pairs(cost)
    fuse_dist_sq_max = MAX_FUSE_UV_DIST * MAX_FUSE_UV_DIST
    used_lidar = set()
    matched_ci = set()
    fused = []

    for ci, li in pairs:
        if ci >= nc:
            continue
        ch = color_hits[ci]
        d2 = float(cost[ci, li]) if li < nl else HUNGARIAN_BIG_COST
        if li < nl and d2 <= fuse_dist_sq_max:
            used_lidar.add(li)
            matched_ci.add(ci)
            parcel = dict(lidar_clusters[li])
            parcel["name"] = ch["name"]
            parcel["color"] = ch["color"]
            parcel["source"] = "lidar+color"
            fused.append(parcel)
            cx, cy, cz = parcel["center"]
            log("[FUSE] %s (%s) ← LiDAR Hungarian (%.3f, %.3f, %.3f) dist=%.3f",
                parcel["name"], parcel["color"], cx, cy, cz, math.sqrt(d2))

    # Ne plus inventer de pose UV pour les non-matchés (cassait la fusion)
    for ci, ch in enumerate(color_hits):
        if ci in matched_ci:
            continue
        log("[FUSE] %s (%s): Hungarian non matché — nom gardé pour collage spatial",
            ch["name"], ch["color"])

    return fused, used_lidar


def _fuse_lidar_and_color(lidar_clusters, color_hits, log):
    """
    Fusion anti-seed0 : LiDAR = position, RGB = nom.
    1) couleur → LiDAR via UV
    2) Hungarian UV
    3) grille 2×2 relative
    """
    if not lidar_clusters:
        if color_hits:
            log("[FUSE] LiDAR 无簇，使用颜色估计位置")
            return _parcels_from_color_only(color_hits, log, lidar_clusters)
        return []

    if len(lidar_clusters) >= 2 and len(color_hits) < 2:
        log("[FUSE] 颜色仅 %d/4 → 使用 LiDAR 空间命名 (%d 几何簇)",
            len(color_hits), len(lidar_clusters))
        named = _name_clusters_spatial(lidar_clusters, log)
        named = _inject_rgb_parcels(named, color_hits, log)
        return _snap_row_x_from_lidar_neighbors(named, log)

    if not color_hits:
        log("[FUSE] 无颜色信息，grille relative (anti-seed0)")
        return _name_clusters_spatial(lidar_clusters, log)

    # Chemin principal : couleur → LiDAR via UV (pas seed0)
    fused, used = _fuse_color_names_to_lidar(color_hits, lidar_clusters, log)

    # Compléter avec Hungarian sur les clusters / hits restants
    remain_hits = [ch for ch in color_hits
                   if ch["name"] not in {p["name"] for p in fused}]
    remain_lidar = [lc for i, lc in enumerate(lidar_clusters) if i not in used]
    if remain_hits and remain_lidar:
        table_z = _table_z_from_lidar(lidar_clusters)
        fused2, used2 = _fuse_color_lidar_hungarian(
            remain_hits, remain_lidar, table_z, log)
        # remap used2 indices into original lidar list
        remain_idx = [i for i in range(len(lidar_clusters)) if i not in used]
        for local_i in used2:
            if local_i < len(remain_idx):
                used.add(remain_idx[local_i])
        fused.extend(fused2)

    for i, lc in enumerate(lidar_clusters):
        if i in used:
            continue
        parcel = dict(lc)
        parcel["name"] = "parcel_unknown_%d" % (i + 1)
        parcel["color"] = "unknown"
        parcel["source"] = "lidar"
        fused.append(parcel)
        cx, cy, cz = parcel["center"]
        log("[FUSE] 未匹配颜色的簇 → parcel_unknown_%d @ (%.3f, %.3f, %.3f)",
            i + 1, cx, cy, cz)

    # Renommer les unknown restants : grille fixe (pas mid d'1 seul amas).
    # Si le slot est déjà pris par un match médiocre → swap (seed0 mission :
    # amas brun 0.40/-0.34 jeté car spatial montrait blue alors UV avait déjà blue).
    named_ok = [p for p in fused if p["name"] in PARCEL_NAMES]
    unknowns = [p for p in fused if p["name"] not in PARCEL_NAMES]
    if unknowns:
        x_mid = float(globals().get("LIDAR_GRID_X_MID", 0.381))
        y_mid = float(globals().get("LIDAR_GRID_Y_MID", -0.214))

        def _slot_cost(name, cx, cy):
            sx = x_mid - 0.07 if name in ("parcel_1", "parcel_2") else x_mid + 0.07
            sy = y_mid - 0.10 if name in ("parcel_1", "parcel_3") else y_mid + 0.10
            return math.hypot(cx - sx, cy - sy)

        by_name = {p["name"]: p for p in named_ok}
        for lc in unknowns:
            cx, cy, cz = lc["center"]
            preferred = _slot_name_from_xy(cx, cy, x_mid, y_mid)
            p = dict(lc)
            p["color"] = _color_label_for_parcel(preferred)
            p["source"] = "lidar-spatial+reassign"
            if preferred not in by_name:
                p["name"] = preferred
                named_ok.append(p)
                by_name[preferred] = p
                log("[FUSE] unknown → %s @ (%.3f, %.3f, %.3f)",
                    preferred, cx, cy, cz)
                continue
            old = by_name[preferred]
            ox, oy = old["center"][0], old["center"][1]
            if _slot_cost(preferred, cx, cy) + 0.04 < _slot_cost(preferred, ox, oy):
                free = [n for n in PARCEL_NAMES if n not in by_name]
                # libérer preferred : déplacer l'ancien vers le meilleur free
                displaced = dict(old)
                if free:
                    best_f = min(free, key=lambda n: _slot_cost(n, ox, oy))
                    displaced["name"] = best_f
                    displaced["color"] = _color_label_for_parcel(best_f)
                    displaced["source"] = (displaced.get("source") or "?") + "+displaced"
                    by_name[best_f] = displaced
                    log("[FUSE] displace %s → %s (%.3f, %.3f) au profit unknown",
                        preferred, best_f, ox, oy)
                named_ok = [q for q in named_ok if q["name"] != preferred]
                if free:
                    named_ok.append(displaced)
                p["name"] = preferred
                named_ok.append(p)
                by_name[preferred] = p
                log("[FUSE] unknown remplace %s @ (%.3f, %.3f) ← (%.3f, %.3f)",
                    preferred, cx, cy, ox, oy)
            else:
                free = [n for n in PARCEL_NAMES if n not in by_name]
                if not free:
                    continue
                best_f = min(free, key=lambda n: _slot_cost(n, cx, cy))
                p["name"] = best_f
                p["color"] = _color_label_for_parcel(best_f)
                named_ok.append(p)
                by_name[best_f] = p
                log("[FUSE] unknown slot %s pris → %s @ (%.3f, %.3f)",
                    preferred, best_f, cx, cy)
        fused = _sort_parcels(named_ok)

    fused = _inject_rgb_parcels(fused, color_hits, log)
    fused = _snap_row_x_from_lidar_neighbors(fused, log)
    return _snap_grid_geometry(fused, log)


def detect_parcels(lidar, cam, tf_reader, log):
    """
    Point d'entrée perception : enchaîne LiDAR → RGB+depth → fusion.
    Retourne une liste de dicts : {name, color, center, size_xy, n_points}.

    Si PERCEPTION_BACKEND=graphics : pipeline ych adaptée (src/scene3_task.py).
    """
    backend = str(globals().get("PERCEPTION_BACKEND", "lidar") or "lidar").lower()
    if backend in ("graphics", "ych", "scene3"):
        log("[DETECT] backend=graphics (ych depth→cluster→couleur)")
        repo = _repo_root()
        src_dir = os.path.join(repo, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        try:
            import scene3_task as ych_geo
            parcels = ych_geo.detect_parcels_graphics(cam, tf_reader, log)
        except Exception as exc:
            log("[DETECT] graphics échoué: %s — fallback lidar", exc)
            parcels = []
        if parcels:
            for p in parcels:
                cx, cy, cz = p["center"]
                log("[DETECT] 最终 %s (%s) [%s]: center=(%.3f, %.3f, %.3f)",
                    p["name"], p.get("color", "?"), p.get("source", "?"),
                    cx, cy, cz)
            log("[DETECT] graphics terminé，共 %d 个快递", len(parcels))
            return parcels
        log("[DETECT] graphics vide → fallback lidar+couleur")

    lidar.wait_for_points(timeout=3.0, min_points=50)
    cam.wait_for_frame("head_rgb", timeout=2.0)
    cam.wait_for_frame("head_depth", timeout=2.0)

    log("[DETECT] N2 perception: open3d=%s scipy=%s",
        _HAS_O3D, _HAS_SCIPY)
    lidar_clusters = _lidar_clusters_raw(lidar, log)
    rgb = cam.get_head_rgb()
    depth = cam.get_head_depth()
    n_depth = cam.depth_valid_count(depth, DEPTH_Z_MIN, DEPTH_Z_MAX) if depth is not None else 0
    log("[DEPTH] pixels valides (%.2f–%.2f m): %d",
        DEPTH_Z_MIN, DEPTH_Z_MAX, n_depth)
    color_hits = (
        _detect_color_parcels(rgb, depth, cam, tf_reader, log)
        if rgb is not None else []
    )
    # Frame trop tôt (tête pas encore baissée) → 2e capture
    if rgb is not None and len(color_hits) < 2:
        log("[COLOR] seulement %d/4 → nouvelle frame après stabilisation",
            len(color_hits))
        rospy.sleep(1.0)
        cam.wait_for_frame("head_rgb", timeout=2.0)
        cam.wait_for_frame("head_depth", timeout=2.0)
        rgb2 = cam.get_head_rgb()
        depth2 = cam.get_head_depth()
        if rgb2 is not None:
            hits2 = _detect_color_parcels(rgb2, depth2, cam, tf_reader, log)
            if len(hits2) > len(color_hits):
                color_hits = hits2
                rgb, depth = rgb2, depth2
    if rgb is None:
        log("[COLOR] 未获取到 RGB，仅使用 LiDAR 几何簇")

    parcels = _fuse_lidar_and_color(lidar_clusters, color_hits, log)
    parcels = _fill_missing_parcels(parcels, color_hits, log)
    parcels = _fix_collapsed_high_row(parcels, color_hits, log)
    parcels = _snap_grid_geometry(parcels, log)
    # Re-lift après grid-y (ancre droite parfois encore sur mauvaise rangée).
    parcels = _fix_collapsed_high_row(parcels, color_hits, log)
    if not parcels and not PERCEPTION_ONLY:
        log("[DETECT] 感知全失败，使用 scene1 布局估计 (seed0 固定点位)")
        parcels = _parcels_from_scene_layout(log)
    for p in parcels:
        cx, cy, cz = p["center"]
        src = p.get("source", "?")
        log("[DETECT] 最终 %s (%s) [%s]: center=(%.3f, %.3f, %.3f)",
            p["name"], p.get("color", "?"), src, cx, cy, cz)
    log("[DETECT] 融合完成，共 %d 个快递", len(parcels))
    return parcels


def _target_dict(name, kind, color, center, source, hand):
    return {
        "name": name,
        "kind": kind,
        "color": color,
        "center": center,
        "source": source,
        "hand": hand,
    }


def _parcel_to_target(p):
    t = _target_dict(
        p["name"], "parcel", p.get("color", "?"),
        p["center"], p.get("source", "?"), "right",
    )
    # Pose LiDAR avant snap grille — pour viser le carton, pas le vide
    if "center_raw" in p:
        t["center_raw"] = p["center_raw"]
    return t


def _is_valid_landmark_point(name, x, y, z):
    if name == "weighing_area":
        return (0.22 <= x <= 0.58 and -0.72 <= y <= -0.28
                and -0.18 <= z <= 0.22)
    if name == "sorting_box":
        return (0.38 <= x <= 0.92 and -0.05 <= y <= 0.52
                and -0.18 <= z <= 0.55)
    return _is_valid_base_point(x, y, z)


def _landmark_from_mask(name, label, mask, h, w, v0, depth, cam, tf_reader,
                        log, hand, via, z_fallback=None):
    """Centre masque couleur → cible touch (depth/TF si valide)."""
    area = int(mask.sum()) // 255
    if area < MIN_LANDMARK_COLOR_PIXELS:
        return None
    ys, xs = np.where(mask > 0)
    u_px = float(xs.mean())
    v_px = float(ys.mean() + v0)
    mask_full = np.zeros((h, w), dtype=np.uint8)
    mask_full[v0:, :] = mask
    center = None
    source = via
    z_table = TABLE_PARCEL_Z if z_fallback is None else z_fallback
    if depth is not None and cam is not None and tf_reader is not None:
        depth_m = cam.median_depth_in_mask(
            depth, mask_full, z_min=DEPTH_Z_MIN, z_max=DEPTH_Z_MAX)
        pt, rgb_src = _rgb_depth_table_point(
            cam, tf_reader, u_px, v_px, depth_m, log, name, via)
        if pt is not None and _is_valid_landmark_point(name, *pt):
            center = (pt[0], pt[1], z_table if name == "weighing_area" else pt[2])
            source = rgb_src or "rgb-depth"
    if center is None and cam is not None and tf_reader is not None:
        pt_ray = cam.pixel_ray_to_table_plane(
            tf_reader, "head", u_px, v_px, z_table)
        if pt_ray is not None and _is_valid_landmark_point(name, *pt_ray):
            center = pt_ray
            source = "rgb-ray"
    if center is None:
        x, y = _uv_to_table_xy(u_px / w, v_px / h)
        z = TABLE_PARCEL_Z if z_fallback is None else z_fallback
        candidate = (x, y, z)
        if not _is_valid_landmark_point(name, *candidate):
            log("[LANDMARK] %s: couleur hors zone valide (%.3f, %.3f, %.3f), rejeté",
                name, x, y, z)
            return None
        center = candidate
        source = "couleur"
    log("[LANDMARK] %s (%s) [%s]: (%.3f, %.3f, %.3f) area=%d",
        name, label, source, center[0], center[1], center[2], area)
    return _target_dict(name, "landmark", label, center, source, hand)


def _lidar_sorting_box_center(pts, ref_xy, log):
    """Centre bac : points sol + percentile x (LiDAR voit le bord proche du robot)."""
    ref_x, ref_y = ref_xy
    zlo, zhi = BOX_FLOOR_LIDAR_Z_RANGE
    floor = pts[(pts[:, 2] >= zlo) & (pts[:, 2] <= zhi)]
    use = floor if len(floor) >= MIN_BOX_LIDAR_POINTS else pts
    if len(use) < MIN_BOX_LIDAR_POINTS:
        return None
    if len(floor) < MIN_BOX_LIDAR_POINTS:
        log("[LANDMARK] sorting_box: peu de points sol (%d), fallback tous points",
            len(floor))
    dist = np.hypot(use[:, 0] - ref_x, use[:, 1] - ref_y)
    keep_n = max(MIN_BOX_LIDAR_POINTS, int(len(use) * 0.6))
    near = use[np.argsort(dist)[:keep_n]]
    cx = float(np.percentile(near[:, 0], BOX_LIDAR_X_PERCENTILE))
    cy = float(np.percentile(near[:, 1], BOX_LIDAR_Y_PERCENTILE))
    cz = float(np.median(near[:, 2]))
    if not _is_valid_landmark_point("sorting_box", cx, cy, cz):
        return None
    return cx, cy, cz


def _lidar_landmark_center(pts, ref_xy, name):
    """Centre robuste landmark : trim vers ref + médiane (évite bords du bac)."""
    ref_x, ref_y = ref_xy
    dist = np.hypot(pts[:, 0] - ref_x, pts[:, 1] - ref_y)
    keep_n = max(MIN_BOX_LIDAR_POINTS, int(len(pts) * 0.55))
    near = pts[np.argsort(dist)[:keep_n]]
    cx = float(np.median(near[:, 0]))
    cy = float(np.median(near[:, 1]))
    cz = float(np.median(near[:, 2]))
    if not _is_valid_landmark_point(name, cx, cy, cz):
        return None
    return cx, cy, cz


def _detect_weighing_area(lidar, rgb, depth, cam, tf_reader, log):
    """Zone pesée : LiDAR (hors table colis) puis marqueur vert LAB dans ROI dédiée."""
    refs = _landmark_ref_positions()
    ref_x, ref_y, ref_z = refs["weighing_area"]
    center = None
    source = None

    pts = lidar.get_points_in_region(
        WEIGH_LIDAR_X_RANGE, WEIGH_LIDAR_Y_RANGE, WEIGH_LIDAR_Z_RANGE)
    if pts is not None and len(pts) >= MIN_WEIGH_LIDAR_POINTS:
        ref_xy = (ref_x, ref_y)
        trimmed = _lidar_landmark_center(pts, ref_xy, "weighing_area")
        if trimmed is not None:
            cx, cy, cz = trimmed
            center = (cx, cy, cz)
            source = "lidar"
            log("[LANDMARK] weighing_area [lidar]: (%.3f, %.3f, %.3f) n=%d",
                cx, cy, cz, len(pts))

    color_tgt = None
    if _HAS_CV2 and rgb is not None:
        h, w = rgb.shape[:2]
        v0 = int(h * COLOR_ROI_V_START)
        roi = rgb[v0:, :, :]
        rh, rw = roi.shape[:2]
        roi_gate = np.zeros((rh, rw), dtype=np.uint8)
        ru0 = int(rw * WEIGH_COLOR_U_RANGE[0])
        ru1 = int(rw * WEIGH_COLOR_U_RANGE[1])
        rv0 = int(rh * WEIGH_COLOR_V_RANGE[0])
        rv1 = int(rh * WEIGH_COLOR_V_RANGE[1])
        roi_gate[rv0:rv1, ru0:ru1] = 255

        lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = _bgr_to_lab(WEIGH_MARKER_REF_BGR)
        dist = np.sqrt(np.sum((lab_roi - ref_lab) ** 2, axis=2))
        mask = (dist < WEIGH_LAB_DIST_MAX).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, roi_gate)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        if int(mask.sum()) // 255 >= MIN_LANDMARK_COLOR_PIXELS:
            color_tgt = _landmark_from_mask(
                "weighing_area", "balance", mask, h, w, v0, depth, cam, tf_reader,
                log, hand="right", via="LAB-weigh",
            )
        else:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hsv_mask = cv2.inRange(
                hsv, np.array(WEIGH_MARKER_HSV[0], np.uint8),
                np.array(WEIGH_MARKER_HSV[1], np.uint8))
            hsv_mask = cv2.bitwise_and(hsv_mask, roi_gate)
            hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            area = int(hsv_mask.sum()) // 255
            if MIN_LANDMARK_COLOR_PIXELS <= area <= MAX_WEIGH_MARKER_AREA:
                color_tgt = _landmark_from_mask(
                    "weighing_area", "balance", hsv_mask, h, w, v0, depth, cam, tf_reader,
                    log, hand="right", via="HSV-weigh",
                )
            else:
                log("[LANDMARK] weighing_area: couleur insuffisante (area=%d)", area)

    if color_tgt is not None:
        cx, cy, cz = color_tgt["center"]
        err = math.hypot(cx - ref_x, cy - ref_y)
        if err <= WEIGH_COLOR_MAX_ERR_XY:
            if center is None or err < math.hypot(center[0] - ref_x, center[1] - ref_y):
                center = color_tgt["center"]
                source = color_tgt["source"]
        else:
            log("[LANDMARK] weighing_area: couleur rejetée err_xy=%.3f m (>%.2f)",
                err, WEIGH_COLOR_MAX_ERR_XY)

    if center is None:
        log("[LANDMARK] weighing_area: non détecté")
        return None
    return _target_dict("weighing_area", "landmark", "balance", center, source, "right")


def _detect_sorting_box(lidar, rgb, depth, cam, tf_reader, log):
    """Bac de tri : LiDAR sol+percentile, secours/fusion couleur vert bac."""
    ref_x, ref_y, ref_z = BOX_DROP_BASE_IK
    ref_xy = (ref_x, ref_y)
    candidates = []

    pts = lidar.get_points_in_region(
        BOX_LIDAR_X_RANGE, BOX_LIDAR_Y_RANGE, BOX_LIDAR_Z_RANGE)
    if pts is not None and len(pts) >= MIN_BOX_LIDAR_POINTS:
        trimmed = _lidar_sorting_box_center(pts, ref_xy, log)
        if trimmed is not None:
            cx, cy, cz = trimmed
            candidates.append(("lidar-floor", (cx, cy, cz), len(pts)))
            log("[LANDMARK] sorting_box [lidar-floor]: (%.3f, %.3f, %.3f) n=%d",
                cx, cy, cz, len(pts))

    if _HAS_CV2 and rgb is not None:
        h, w = rgb.shape[:2]
        v0 = int(h * COLOR_ROI_V_START)
        roi = rgb[v0:, :, :]
        rh, rw = roi.shape[:2]
        roi_gate = np.zeros((rh, rw), dtype=np.uint8)
        ru0 = int(rw * BOX_COLOR_U_RANGE[0])
        ru1 = int(rw * BOX_COLOR_U_RANGE[1])
        rv0 = int(rh * BOX_COLOR_V_RANGE[0])
        rv1 = int(rh * BOX_COLOR_V_RANGE[1])
        roi_gate[rv0:rv1, ru0:ru1] = 255
        lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = _bgr_to_lab(BOX_MARKER_REF_BGR)
        dist = np.sqrt(np.sum((lab_roi - ref_lab) ** 2, axis=2))
        mask = (dist < LAB_COLOR_DIST_MAX).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, roi_gate)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        if int(mask.sum()) // 255 >= MIN_LANDMARK_COLOR_PIXELS:
            tgt = _landmark_from_mask(
                "sorting_box", "bac", mask, h, w, v0, depth, cam, tf_reader,
                log, hand="left", via="LAB-box",
                z_fallback=TABLE_PARCEL_Z,
            )
            if tgt is not None:
                cx, cy, cz = tgt["center"]
                err = math.hypot(cx - ref_x, cy - ref_y)
                if err <= 0.18:
                    candidates.append((tgt["source"], tgt["center"], int(mask.sum()) // 255))
        elif not candidates:
            log("[LANDMARK] sorting_box: couleur insuffisante dans ROI bac")

    if not candidates:
        log("[LANDMARK] sorting_box: non détecté")
        return None

    source, center, _ = min(
        candidates,
        key=lambda c: math.hypot(c[1][0] - ref_x, c[1][1] - ref_y),
    )
    if len(candidates) > 1:
        log("[LANDMARK] sorting_box: choix [%s] err_xy=%.3f m (candidats=%d)",
            source, math.hypot(center[0] - ref_x, center[1] - ref_y),
            len(candidates))
    return _target_dict("sorting_box", "landmark", "bac", center, source, "right")


def detect_all_touch_targets(lidar, cam, tf_reader, log):
    """Colis + balance + bac — uniquement ce que la perception voit."""
    parcels = detect_parcels(lidar, cam, tf_reader, log)
    targets = [_parcel_to_target(p) for p in parcels if p.get("name") in PARCEL_NAMES]
    weigh = _detect_weighing_area(
        lidar, cam.get_head_rgb(), cam.get_head_depth(), cam, tf_reader, log)
    if weigh is not None:
        targets.append(weigh)
    box = _detect_sorting_box(lidar, cam.get_head_rgb(), cam.get_head_depth(),
                              cam, tf_reader, log)
    if box is not None:
        targets.append(box)
    log("[DETECT] cibles touch total: %d (colis + landmarks)", len(targets))
    return targets


def _landmark_ref_positions():
    """Refs IK opérationnelles (balance release, point dépose bac)."""
    wa = WEIGH_RELEASE_IK
    sb = BOX_DROP_BASE_IK
    return {
        "weighing_area": (wa[0], wa[1], TABLE_PARCEL_Z),
        "sorting_box": (sb[0], sb[1], sb[2]),
    }


def log_touch_targets_report(targets, log, include_parcels=True):
    """Rapport perception pour colis + landmarks."""
    if include_parcels:
        log_perception_report([t for t in targets if t["kind"] == "parcel"], log)
    refs = _landmark_ref_positions()
    log("[REPORT] ========== Landmarks (base_link IK) ==========")
    by_name = {t["name"]: t for t in targets if t["kind"] == "landmark"}
    max_err = 0.0
    for name in ("weighing_area", "sorting_box"):
        ex, ey, ez = refs[name]
        if name not in by_name:
            log("[REPORT] %s: MANQUANT  ref=(%.3f, %.3f, %.3f)", name, ex, ey, ez)
            continue
        cx, cy, cz = by_name[name]["center"]
        err_xy = math.hypot(cx - ex, cy - ey)
        max_err = max(max_err, err_xy)
        src = by_name[name].get("source", "?")
        log("[REPORT] %s [%s]: détecté=(%.3f, %.3f, %.3f)  ref=(%.3f, %.3f, %.3f)  err_xy=%.3f m",
            name, src, cx, cy, cz, ex, ey, ez, err_xy)
    log("[REPORT] landmarks détectés %d/2  max err_xy %.3f m",
        len(by_name), max_err)
    log("[REPORT] ============================================")
    return max_err, len(by_name)


def _is_safe_touch_target(target, log):
    """Refuse les cibles qui ont fait tomber le robot (y trop latéral, x trop loin)."""
    name = target.get("name", "?")
    cx, cy, cz = target["center"]
    if target.get("kind") == "parcel" or name in PARCEL_NAMES:
        if not (TOUCH_MIN_X <= cx <= TOUCH_MAX_X and -0.38 <= cy <= -0.04):
            log("[TOUCH] SKIP %s: hors zone safe (%.3f, %.3f, %.3f)", name, cx, cy, cz)
            return False
        if cz < TOUCH_TABLE_Z_MIN or cz > TOUCH_TABLE_Z_MAX + 0.05:
            log("[TOUCH] SKIP %s: z aberrant %.3f", name, cz)
            return False
        src = target.get("source") or ""
        if "row-infer" in src or src == "layout":
            log("[TOUCH] SKIP %s: source trop faible (%s)", name, src)
            return False
    if name == "weighing_area":
        if not TOUCH_LANDMARKS:
            log("[TOUCH] SKIP weighing_area: landmarks désactivés")
            return False
        if not (0.28 <= cx <= 0.55 and -0.70 <= cy <= -0.35):
            log("[TOUCH] SKIP weighing_area: hors zone (%.3f, %.3f)", cx, cy)
            return False
    if name == "sorting_box":
        if not TOUCH_LANDMARKS:
            log("[TOUCH] SKIP sorting_box: landmarks désactivés")
            return False
        if not (0.40 <= cx <= 0.75 and 0.05 <= cy <= 0.45):
            log("[TOUCH] SKIP sorting_box: hors zone (%.3f, %.3f)", cx, cy)
            return False
    return True


def _targets_for_touch(targets):
    by_name = {t["name"]: t for t in targets}
    ordered = []
    n_parcels = 0
    for name in TOUCH_ORDER:
        if name not in by_name:
            continue
        if name in PARCEL_NAMES:
            if n_parcels >= TOUCH_MAX_PARCELS:
                continue
            n_parcels += 1
        elif not TOUCH_LANDMARKS:
            continue
        ordered.append(by_name[name])
    return ordered


def log_perception_report(parcels, log):
    """
    Rapport multi-seed : coords détectées + erreur de structure 2×2 relative.
    Ne compare plus à PARCEL_WORLD_POS (biais seed0).
    """
    detected = {p["name"]: p for p in parcels if p["name"] in PARCEL_NAMES}
    named, max_err = _geometry_quality_relative(parcels)
    log("[REPORT] ========== 感知坐标 (base_link IK, anti-seed0) ==========")
    for name in PARCEL_NAMES:
        if name not in detected:
            log("[REPORT] %s: MANQUANT", name)
            continue
        cx, cy, cz = detected[name]["center"]
        src = detected[name].get("source", "?")
        log("[REPORT] %s [%s]: détecté=(%.3f, %.3f, %.3f)  color=%s",
            name, src, cx, cy, cz, detected[name].get("color", "?"))
    log("[REPORT] 识别 %d/4  err_structure_2x2 %.3f m (pas ref seed0)",
        named, max_err)
    log("[REPORT] ============================================")
    return named, max_err


def _perception_score(parcels, named, max_err):
    """
    Score N2 : priorité 4/4, err_xy basse, évite row-infer / layout.
    Plus bas = meilleur.
    """
    infer = sum(
        1 for p in parcels
        if p.get("name") in PARCEL_NAMES
        and ("row-infer" in (p.get("source") or "")
             or p.get("source") == "layout")
    )
    bad_rgb = sum(
        1 for p in parcels
        if p.get("name") in PARCEL_NAMES
        and "rgb-depth+zsnap" in (p.get("source") or "")
    )
    # Bonus léger si row-lift / rgb-y a corrigé la rangée haute.
    row_fixed = sum(
        1 for p in parcels
        if p.get("name") in ("parcel_2", "parcel_4")
        and any(t in (p.get("source") or "") for t in ("row-lift", "rgb-y"))
    )
    return ((4 - named) * 10.0 + max_err + infer * 0.03 + bad_rgb * 0.05
            - 0.01 * row_fixed)


def run_scene1_perception_only(arm, head, log):
    """Mode test : voir les colis, afficher les coordonnées, pas de saisie."""
    log("=" * 50)
    log("场景一：仅感知测试 — 识别坐标 (PERCEPTION_ONLY)")
    log("=" * 50)

    log("[STEP 1] 切换手臂到外部控制模式")
    arm.switch_to_external_control()
    rospy.sleep(0.5)
    arm.go_home()
    rospy.sleep(0.5)
    log_scene_landmarks(log)

    lidar = LidarReader()
    cam = CameraReader()
    tf_reader = TFReader()
    rospy.sleep(0.5)

    best_parcels = []
    best_named = 0
    best_err = float("inf")
    best_score = float("inf")
    best_colorish = -1

    # Comme l'orga : baisser la tête UNE fois, puis laisser (pas de monte/descend)
    log("[PERCEPT] tête orga yaw=%.1f pitch=%.1f (fixe pendant détection)",
        HEAD_LOOK_YAW, HEAD_LOOK_PITCH)
    head.look_at(HEAD_LOOK_YAW, HEAD_LOOK_PITCH)
    rospy.sleep(HEAD_SETTLE_SEC)

    for attempt in range(1, PERCEPTION_ATTEMPTS + 1):
        log("[PERCEPT] --- 尝试 %d/%d ---", attempt, PERCEPTION_ATTEMPTS)
        # republier la même pose (hold), sans look_forward
        head.look_at(HEAD_LOOK_YAW, HEAD_LOOK_PITCH)
        rospy.sleep(0.3)

        parcels = detect_parcels(lidar, cam, tf_reader, log)
        if len(parcels) < 4:
            inspect_table_depth(cam, log)

        rgb = cam.get_head_rgb()
        depth = cam.get_head_depth()
        landmarks = []
        weigh = _detect_weighing_area(lidar, rgb, depth, cam, tf_reader, log)
        if weigh is not None:
            landmarks.append(weigh)
        box = _detect_sorting_box(lidar, rgb, depth, cam, tf_reader, log)
        if box is not None:
            landmarks.append(box)

        named, max_err = log_perception_report(parcels, log)
        score = _perception_score(parcels, named, max_err)
        lm_err, lm_count = log_touch_targets_report(
            [_parcel_to_target(p) for p in parcels] + landmarks, log,
            include_parcels=False)
        n_colorish = sum(
            1 for p in parcels
            if _is_rgb_backed(p.get("source", ""))
            or "color" in (p.get("source") or "")
            or "lidar+color" in (p.get("source") or ""))
        log("[PERCEPT] score N2=%.4f (err=%.3f, row-infer pénalité incluse)",
            score, max_err)
        log("[PERCEPT] N1b landmarks %d/2  err_max=%.3f m  colorish=%d",
            lm_count, lm_err, n_colorish)

        # Préférer géométrie bonne + plus de couleur (évite garder un attempt grid-only)
        # Ne pas verrouiller un best avec err≥8 cm (collapse rangée head20).
        take = False
        if score < best_score - 1e-6 and max_err < 0.08:
            take = True
        elif (n_colorish > best_colorish and max_err < 0.08
              and named >= 4):
            take = True
        elif (best_err >= 0.08 and max_err < best_err and named >= 4
              and score <= best_score + 0.02):
            take = True
        elif best_named < 4 and named >= best_named and score <= best_score:
            take = True
        if take:
            best_parcels = parcels
            best_named = named
            best_err = max_err
            best_score = score
            best_colorish = n_colorish

        geometry_ok = (named >= 4 and max_err < PERCEPTION_ERR_TARGET
                       and lm_count >= 2 and lm_err < LANDMARK_ERR_TARGET)
        log("[PERCEPT] STAT attempt=%d/%d named=%d/4 err_xy=%.3f colorish=%d "
            "lm=%d/2 lm_err=%.3f geo_ok=%d",
            attempt, PERCEPTION_ATTEMPTS, named, max_err, n_colorish,
            lm_count, lm_err, 1 if geometry_ok else 0)

        if not PERCEPTION_FORCE_ALL_ATTEMPTS:
            if geometry_ok and (not PERCEPTION_ONLY or n_colorish >= 2
                                or attempt >= PERCEPTION_ATTEMPTS):
                log("[PERCEPT] N2 4/4 + N1b landmarks OK (< %.0f cm)，停止重试",
                    PERCEPTION_ERR_TARGET * 100)
                break
            if geometry_ok and PERCEPTION_ONLY and n_colorish < 2:
                log("[PERCEPT] géométrie OK mais couleur faible (%d) → retry",
                    n_colorish)
        rospy.sleep(0.5)

    log("[DONE] 感知测试完成：最佳结果 %d/4  err_max=%.3f m  score=%.4f colorish=%d",
        best_named, best_err, best_score, best_colorish)
    for p in best_parcels:
        cx, cy, cz = p["center"]
        log("[DONE]   %s (%s) [%s]: (%.3f, %.3f, %.3f)",
            p["name"], p.get("color", "?"), p.get("source", "?"), cx, cy, cz)
    if GT_COMPARE:
        log_gt_compare_debug(best_parcels, log)
    log_study_det_csv(best_parcels, log)
    log("场景一：感知测试结束")


def _current_seed():
    """Seed du run (env SEED du script docker, sinon 0). Debug only."""
    try:
        return int(os.environ.get("SEED", "0"))
    except (TypeError, ValueError):
        return 0


def _repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "..", ".."))


def _repo_csv_path(filename):
    """Chemin sous le repo ; crée le dossier parent (labo/scene1/...)."""
    path = os.path.join(_repo_root(), filename)
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        try:
            os.makedirs(parent)
        except OSError:
            pass
    return path


def _labo_image_path(filename=None):
    name = filename or globals().get(
        "COLOR_DEBUG_JPG", "labo/scene1/images/scene1_color_debug.jpg")
    return _repo_csv_path(name)


def log_study_det_csv(parcels, log):
    """
    Tableau étude : coords détectées (IK) seulement — lecture humaine.
    Ne pilote rien. Append par seed.
    """
    name = globals().get("STUDY_DET_CSV", "scene1_study_det_algo.csv")
    path = _repo_csv_path(name)
    seed = _current_seed()
    detected = {p["name"]: p for p in parcels if p.get("name") in PARCEL_NAMES}
    write_header = not os.path.isfile(path)
    try:
        import csv
        with open(path, "a") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "seed", "name", "color", "source",
                    "det_ik_x", "det_ik_y", "det_ik_z",
                ])
            for pname in PARCEL_NAMES:
                p = detected.get(pname)
                if p is None:
                    w.writerow([seed, pname, "?", "missing", "", "", ""])
                    continue
                cx, cy, cz = p["center"]
                w.writerow([
                    seed, pname, p.get("color", "?"), p.get("source", "?"),
                    cx, cy, cz,
                ])
        log("[STUDY] det CSV (algo only): %s", path)
    except Exception as exc:
        log("[STUDY] écriture det CSV échouée: %s", exc)


def _load_layout_world_poses(seed, log):
    """
    Positions monde MuJoCo via challenge_secret (labo).
    Fallback seed0 : PARCEL_WORLD_POS. Jamais utilisé pour piloter le bras.
    """
    try:
        import rospkg
        lib_dir = os.path.join(
            rospkg.RosPack().get_path("challenge_cup_simulator"), "lib")
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)
        import challenge_secret
        layout = challenge_secret.get_object_layout("scene1", int(seed))
        out = {}
        for name in PARCEL_NAMES:
            if name not in layout or "pos" not in layout[name]:
                continue
            pos = layout[name]["pos"]
            out[name] = (float(pos[0]), float(pos[1]), float(pos[2]))
        if out:
            log("[GT_CMP] layout secret seed=%d → %d colis (world)", seed, len(out))
            return out, "challenge_secret"
    except Exception as exc:
        log("[GT_CMP] challenge_secret indisponible: %s", exc)

    if int(seed) == 0:
        log("[GT_CMP] fallback PARCEL_WORLD_POS (seed0 nominal)")
        return {n: tuple(PARCEL_WORLD_POS[n]) for n in PARCEL_NAMES}, "parcel_world_pos_seed0"
    log("[GT_CMP] pas de GT pour seed=%d — skip compare", seed)
    return {}, "none"


def _gt_compare_csv_path():
    name = GT_COMPARE_CSV if "GT_COMPARE_CSV" in globals() else "scene1_gt_compare_debug.csv"
    return _repo_csv_path(name)


def log_gt_compare_debug(parcels, log):
    """
    DEBUG ONLY : détecté (IK) vs GT monde→IK. Écrit CSV baseline.
    Ne modifie pas `parcels`, ne pilote rien.
    """
    seed = _current_seed()
    gt_world, gt_src = _load_layout_world_poses(seed, log)
    if not gt_world:
        return

    detected = {p["name"]: p for p in parcels if p.get("name") in PARCEL_NAMES}
    csv_path = _gt_compare_csv_path()
    write_header = not os.path.isfile(csv_path)
    log("[GT_CMP] ========== compare debug (ne pilote PAS le bras) ==========")
    log("[GT_CMP] seed=%d  gt_src=%s  lien: IK = world + WORLD_TO_IK_OFFSET",
        seed, gt_src)
    log("[GT_CMP] offset=(%.6f, %.6f, %.6f)", *WORLD_TO_IK_OFFSET)

    rows = []
    for name in PARCEL_NAMES:
        wx, wy, wz = gt_world[name]
        gx, gy, gz = _world_to_ik(wx, wy, wz)
        p = detected.get(name)
        if p is None:
            log("[GT_CMP] %s: MANQUANT | GT_ik=(%.3f, %.3f, %.3f)", name, gx, gy, gz)
            rows.append([
                seed, name, "?", "missing",
                wx, wy, wz, gx, gy, gz,
                "", "", "", "", "", "", "", "",
            ])
            continue
        cx, cy, cz = p["center"]
        ex, ey, ez = cx - gx, cy - gy, cz - gz
        err_xy = math.hypot(ex, ey)
        err_xyz = math.sqrt(ex * ex + ey * ey + ez * ez)
        color = p.get("color", "?")
        src = p.get("source", "?")
        log("[GT_CMP] %s (%s) [%s]: det_ik=(%.3f, %.3f, %.3f)  "
            "GT_ik=(%.3f, %.3f, %.3f)  Δxy=%.1f cm  Δxyz=%.1f cm",
            name, color, src, cx, cy, cz, gx, gy, gz,
            err_xy * 100.0, err_xyz * 100.0)
        if name == "parcel_2":
            log("[GT_CMP] ★ JAUNE parcel_2: Δxy=%.1f cm (det vs GT)", err_xy * 100.0)
        rows.append([
            seed, name, color, src,
            wx, wy, wz, gx, gy, gz,
            cx, cy, cz, ex, ey, ez, err_xy, err_xyz,
        ])

    try:
        import csv
        with open(csv_path, "a") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "seed", "name", "color", "source",
                    "gt_world_x", "gt_world_y", "gt_world_z",
                    "gt_ik_x", "gt_ik_y", "gt_ik_z",
                    "det_ik_x", "det_ik_y", "det_ik_z",
                    "err_x", "err_y", "err_z", "err_xy_m", "err_xyz_m",
                ])
            for row in rows:
                w.writerow(row)
        log("[GT_CMP] CSV baseline (lecture humaine only): %s", csv_path)
    except Exception as exc:
        log("[GT_CMP] écriture CSV échouée: %s", exc)
    log("[GT_CMP] ========================================================")


def log_scene_landmarks(log):
    """Affiche dans les logs les positions fixes balance / bac (debug)."""
    w = SCENE_LANDMARKS["weighing_area"]
    b = SCENE_LANDMARKS["sorting_box"]
    log("[LANDMARK] 称重区 center=(%.2f, %.2f, %.2f) release_z=%.2f",
        w["center"][0], w["center"][1], w["center"][2], w["release_z"])
    log("[LANDMARK] 分拣箱 center=(%.2f, %.2f, %.2f) drop_z=%.2f",
        b["center"][0], b["center"][1], b["center"][2], b["drop_z"])


def inspect_table_depth(cam, log):
    """
    Secours debug : analyse la carte de profondeur si LiDAR ne trouve pas assez de colis.
    """
    depth = cam.get_head_depth()
    if depth is None:
        log("[DETECT] 深度图: 未获取到数据")
        return None

    count = cam.depth_valid_count(depth, DEPTH_Z_MIN, DEPTH_Z_MAX)
    if count < 500:
        log("[DETECT] 深度图: 有效像素太少 (%d)", count)
        return None

    valid = np.isfinite(depth) & (depth > DEPTH_Z_MIN) & (depth < DEPTH_Z_MAX)
    avg_dist = float(depth[valid].mean())
    h, w = depth.shape
    ys, xs = np.where(valid)
    offset_x = (xs.mean() - w / 2.0) / w
    log("[DETECT] 深度图: 平均距离 %.2f m, 水平偏移 %.2f (正=偏右), pixels=%d",
        avg_dist, offset_x, count)
    return {"avg_dist": avg_dist, "offset_x": float(offset_x), "valid_pixels": count}
