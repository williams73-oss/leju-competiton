#!/usr/bin/env python3
"""
Scène 1 — Pesée et tri des colis (快递称重与摆放).

Objets dans la scène :
  - Colis : parcel_1 … parcel_4 (couleurs fixes : marron, jaune, orange, bleu)
  - Balance : weighing_area_0p2m_square
  - Bac de tri : sorting_box_0p4_0p3_0p3
  - Table : challenge_table

Flux de la mission (répété 4 fois) :
  1. Détecter les colis sur la table (LiDAR + caméra)
  2. Choisir le plus proche, s'en approcher, le saisir (main droite)
  3. Le poser sur la balance → reprendre → passer à la main gauche
  4. Le déposer dans le bac
"""

import math
import os
import sys

import numpy as np
import rospy

# Permet d'importer perception_api depuis le dossier src/ du package
_pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_pkg, "src"))
from perception_api import CameraReader, LidarReader

# OpenCV est optionnel : sans lui, on ne peut pas reconnaître les couleurs
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# =============================================================================
# CONSTANTES — PERCEPTION (LiDAR + couleur)
# =============================================================================

PARCEL_NAMES = ["parcel_1", "parcel_2", "parcel_3", "parcel_4"]

# Zone de filtrage LiDAR dans le repère base_link du robot :
#   x = devant le robot, y = gauche/droite, z = hauteur
TABLE_X_RANGE = (0.0, 1.1)       # de 0 m à 1,1 m devant le robot
TABLE_Y_RANGE = (-0.85, 0.25)    # bande latérale couvrant la table
PARCEL_Z_RANGE = (0.84, 0.96)    # hauteur des colis posés sur la table (~0,9 m)

# Clustering : regrouper les points LiDAR proches en "amas" (1 amas ≈ 1 colis)
CLUSTER_EPS_XY = 0.10            # distance max entre 2 points du même amas (m)
MIN_CLUSTER_POINTS = 12            # minimum de points pour valider un amas
MIN_PARCEL_SIZE_XY = 0.02          # taille min d'un colis (évite le bruit)
MAX_PARCEL_SIZE_XY = 0.18          # taille max (évite la table entière)
MIN_COLOR_PIXELS = 200             # pixels HSV min pour accepter une couleur

# Couleurs HSV OpenCV — ordre important : bleu/jaune d'abord, marron en dernier
# pour éviter que marron/orange/jaune se chevauchent dans l'image
PARCEL_COLOR_SPECS = [
    ("parcel_4", "blue", (90, 50, 50), (130, 255, 255)),
    ("parcel_2", "yellow", (20, 80, 80), (35, 255, 255)),
    ("parcel_3", "orange", (8, 100, 80), (18, 255, 255)),
    ("parcel_1", "brown", (5, 40, 40), (20, 180, 180)),
]


# =============================================================================
# CONSTANTES — POINTS FIXES DE LA SCÈNE (ne changent PAS avec le seed)
# =============================================================================
# Coordonnées IK [x, y, z] en mètres dans le repère local du robot.
# Calibrées d'après le script organisateur collect_scene1_handoff_dataset.py.

WEIGH_TRANSIT_Z = 0.326217          # hauteur de transport vers la balance
WEIGH_RELEASE_IK = [0.396, -0.574, 0.146217]   # point de pose sur la balance
WEIGH_REGRASP_IK = [0.396, -0.574, -0.04]      # reprise après pesée
LEFT_PRESET_2_IK = [0.313, 0.239, 0.282]       # main gauche en attente
RIGHT_HANDOFF_IK = [0.246, -0.044645, 0.3016983]  # point de passation
RIGHT_HANDOFF_TRANSIT_Z = 0.40     # hauteur intermédiaire avant passation
LEFT_HANDOFF_RECEIVE_IK = [0.266, 0.04645, 0.2216983]  # main gauche reçoit
RIGHT_HANDOFF_RETRACT_Y = -0.30    # recul de la main droite après passation
BOX_DROP_BASE_IK = [0.605966, 0.226114, 0.556217]  # point de dépose dans le bac

# Chaque colis a un petit décalage dans le bac pour ne pas se superposer (grille 2×2)
BOX_DROP_OFFSET_BY_PARCEL = {
    "parcel_1": [0.0, -0.01, 0.0],
    "parcel_2": [0.0, 0.04, 0.0],
    "parcel_3": [0.04, -0.03, 0.0],
    "parcel_4": [0.04, 0.03, 0.0],
}

# Repères pour les logs (balance et bac) — utile pour le debug
SCENE_LANDMARKS = {
    "weighing_area": {
        "center": tuple(WEIGH_RELEASE_IK[:2]) + (0.88,),
        "release_z": WEIGH_RELEASE_IK[2],
    },
    "sorting_box": {
        "center": tuple(BOX_DROP_BASE_IK),
        "drop_z": BOX_DROP_BASE_IK[2],
    },
}


# =============================================================================
# CONSTANTES — MOUVEMENT DU BRAS ET DES PINCES
# =============================================================================

GRASP_QUAT = [0.0, 0.0, 0.0, 1.0]  # orientation neutre (quaternion x,y,z,w) pour la saisie
LEFT_WAIT_Y_OFFSET = 0.15           # main gauche décalée sur y pendant saisie droite
RIGHT_GRASP_Y_OFFSET = -0.02        # fine correction y pour la pince droite
APPROACH_Z_OFFSET = 0.10            # approche : 10 cm au-dessus du colis
GRASP_Z_OFFSET = -0.01              # descente légèrement sous le centre détecté
LIFT_Z_OFFSET = 0.12                # élévation après saisie
PLACE_APPROACH_Z = 0.06             # approche au-dessus du bac avant dépose
ARM_SETTLE_TIME = 1.2               # pause après mouvement bras (s)
GRIPPER_SETTLE_TIME = 0.4           # pause après ouverture/fermeture pince
WEIGH_RELEASE_SETTLE = 1.5          # stabilisation avant ouverture sur balance
WEIGH_DWELL = 1.0                   # attente après pose (simulation pesée)
PLACE_DWELL = 0.8                   # attente après dépose dans le bac
MAX_PARCELS = 4                     # nombre total de colis à traiter


# =============================================================================
# MATH — conversion angles → quaternion (pour orienter les mains au bon angle)
# =============================================================================

def _matmul3(a, b):
    """Multiplication de deux matrices 3×3."""
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def _rotation_matrix_to_quat_xyzw(matrix):
    """Convertit une matrice de rotation 3×3 en quaternion [x, y, z, w]."""
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        w = math.sqrt(trace + 1.0) / 2.0
        x = (matrix[2][1] - matrix[1][2]) / (4.0 * w)
        y = (matrix[0][2] - matrix[2][0]) / (4.0 * w)
        z = (matrix[1][0] - matrix[0][1]) / (4.0 * w)
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        x = math.sqrt(max(0.0, 1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2])) / 2.0
        y = (matrix[0][1] + matrix[1][0]) / (4.0 * x)
        z = (matrix[0][2] + matrix[2][0]) / (4.0 * x)
        w = (matrix[2][1] - matrix[1][2]) / (4.0 * x)
    elif matrix[1][1] > matrix[2][2]:
        y = math.sqrt(max(0.0, 1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2])) / 2.0
        x = (matrix[0][1] + matrix[1][0]) / (4.0 * y)
        z = (matrix[1][2] + matrix[2][1]) / (4.0 * y)
        w = (matrix[0][2] - matrix[2][0]) / (4.0 * y)
    else:
        z = math.sqrt(max(0.0, 1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1])) / 2.0
        x = (matrix[0][2] + matrix[2][0]) / (4.0 * z)
        y = (matrix[1][2] + matrix[2][1]) / (4.0 * z)
        w = (matrix[1][0] - matrix[0][1]) / (4.0 * z)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        return [0.0, 0.0, 0.0, 1.0]
    return [x / norm, y / norm, z / norm, w / norm]


def _quat_from_ypr_deg(first_ypr_deg, second_ypr_deg=None):
    """
    Construit un quaternion à partir d'angles yaw/pitch/roll en degrés.
    Le robot Kuavo utilise deux jeux d'angles (comme dans arm_control.py).
    """
    yaw, pitch, _roll = [math.radians(float(v)) for v in first_ypr_deg]
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    matrix = [
        [cy * cp, -sy, cy * sp],
        [sy * cp, cy, sy * sp],
        [-sp, 0.0, cp],
    ]
    if second_ypr_deg is not None:
        manual_yaw, manual_pitch, manual_roll = [math.radians(float(v)) for v in second_ypr_deg]
        manual = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        if abs(manual_yaw) > 0.01:
            c, s = math.cos(manual_yaw), math.sin(manual_yaw)
            manual = _matmul3([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], manual)
        if abs(manual_pitch) > 0.01:
            c, s = math.cos(manual_pitch), math.sin(manual_pitch)
            manual = _matmul3([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], manual)
        if abs(manual_roll) > 0.01:
            c, s = math.cos(manual_roll), math.sin(manual_roll)
            manual = _matmul3([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], manual)
        matrix = _matmul3(matrix, manual)
    return _rotation_matrix_to_quat_xyzw(matrix)


# Quaternions pré-calculés pour chaque phase du mouvement (évite de recalculer à chaque fois)
LEFT_PRESET_2_QUAT = _quat_from_ypr_deg([-146.440, 4.966, 0.0], [0.0, 0.0, 96.580])
RIGHT_WEIGH_RELEASE_QUAT = _quat_from_ypr_deg([0, -100, 0.0], [90.0, 0.0, 0.0])
RIGHT_WEIGH_REGRASP_QUAT = _quat_from_ypr_deg([0, -60, 0.0], [90.0, 0.0, 0.0])
RIGHT_HANDOFF_QUAT = _quat_from_ypr_deg([-0.839, -100.0, 0.0], [90.0, -20.0, 90.0])
LEFT_HANDOFF_RECEIVE_QUAT = _quat_from_ypr_deg([-0.839, -100.0, 0.0], [-90.0, -0.0, -90.0])
LEFT_BOX_DROP_QUAT = _quat_from_ypr_deg([-0.328, -100.935, 0.0], [-90.0, 0.0, 0.369])


# =============================================================================
# PERCEPTION — ÉTAPE 1 : clustering LiDAR (OÙ sont les objets ?)
# =============================================================================

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


def _summarize_cluster(points, indices):
    """Calcule le centre et la taille d'un amas de points LiDAR."""
    pts = points[indices]
    center = pts.mean(axis=0)
    return {
        "center": tuple(float(v) for v in center),  # (x, y, z) en mètres
        "size_xy": (float(pts[:, 0].ptp()), float(pts[:, 1].ptp())),  # largeur × profondeur
        "n_points": len(indices),
    }


def _sort_parcels(clusters):
    """Trie les colis de gauche à droite puis de bas en haut (y puis x)."""
    return sorted(clusters, key=lambda c: (c["center"][1], c["center"][0]))


def _lidar_clusters_raw(lidar, log):
    """
    Détection LiDAR pure : retourne jusqu'à 4 amas géométriques SANS nom de colis.
    Chaque amas = dict avec 'center', 'size_xy', 'n_points'.
    """
    pts = lidar.get_points_in_region(
        x_range=TABLE_X_RANGE,
        y_range=TABLE_Y_RANGE,
        z_range=PARCEL_Z_RANGE,
    )
    if pts is None or len(pts) == 0:
        log("[DETECT] LiDAR: 未获取到点云")
        return []

    log("[DETECT] LiDAR: 桌面高度带内 %d 个点", len(pts))

    candidates = []
    for indices in _cluster_points_xy(pts, CLUSTER_EPS_XY, MIN_CLUSTER_POINTS):
        summary = _summarize_cluster(pts, indices)
        sx, sy = summary["size_xy"]
        # Filtrer les amas trop petits (bruit) ou trop grands (pas un colis)
        if sx < MIN_PARCEL_SIZE_XY or sy < MIN_PARCEL_SIZE_XY:
            continue
        if sx > MAX_PARCEL_SIZE_XY or sy > MAX_PARCEL_SIZE_XY:
            continue
        candidates.append(summary)

    clusters = _sort_parcels(candidates)[:4]
    log("[DETECT] LiDAR: 找到 %d 个几何簇", len(clusters))
    for i, c in enumerate(clusters):
        cx, cy, cz = c["center"]
        log("[DETECT]   cluster_%d: center=(%.3f, %.3f, %.3f) size=%.3f×%.3f",
            i + 1, cx, cy, cz, c["size_xy"][0], c["size_xy"][1])
    return clusters


# =============================================================================
# PERCEPTION — ÉTAPE 2 : couleur RGB (QUEL colis ?)
# =============================================================================

def _detect_color_parcels(rgb, log):
    """
    Analyse l'image caméra : masque HSV par couleur → centre de chaque zone colorée.
    Retourne une liste avec name, color, u_norm, v_norm (position normalisée 0–1 dans l'image).
    """
    if not _HAS_CV2:
        log("[COLOR] opencv-python 未安装，跳过颜色识别")
        return []

    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    # Masque "remaining" : une fois qu'une couleur est extraite, on la retire pour éviter les doublons
    remaining = np.full((h, w), 255, dtype=np.uint8)
    hits = []

    for name, label, lo, hi in PARCEL_COLOR_SPECS:
        mask = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
        mask = cv2.bitwise_and(mask, remaining)
        if int(mask.sum()) // 255 < MIN_COLOR_PIXELS:
            log("[COLOR] %s (%s): 有效像素不足", name, label)
            continue
        remaining = cv2.bitwise_and(remaining, cv2.bitwise_not(mask))
        ys, xs = np.where(mask > 0)
        hits.append({
            "name": name,
            "color": label,
            "u_norm": float(xs.mean() / w),   # 0 = gauche image, 1 = droite
            "v_norm": float(ys.mean() / h),   # 0 = haut image, 1 = bas
            "area": len(xs),
        })
        log("[COLOR] %s (%s): u=%.2f v=%.2f area=%d",
            name, label, hits[-1]["u_norm"], hits[-1]["v_norm"], hits[-1]["area"])
    return hits


# =============================================================================
# PERCEPTION — ÉTAPE 3 : fusion LiDAR + couleur
# =============================================================================

def _lidar_norm_uv(cluster, xs, ys):
    """
    Projette approximativement un centre LiDAR 3D en coordonnées image (u, v).
    Nécessaire pour associer un amas LiDAR à une tache de couleur dans l'image.
    """
    x, y, _ = cluster["center"]
    x_n = (x - min(xs)) / max(max(xs) - min(xs), 1e-3)
    y_n = (y - min(ys)) / max(max(ys) - min(ys), 1e-3)
    u = 0.5 - 0.45 * (y_n - 0.5)
    v = 0.25 + 0.50 * x_n
    return u, v


def _fuse_lidar_and_color(lidar_clusters, color_hits, log):
    """
    Fusion : LiDAR donne la position 3D, RGB donne le nom (parcel_1 … parcel_4).
    On fait correspondre chaque couleur au cluster LiDAR le plus proche en (u, v).
    """
    if not lidar_clusters:
        return []

    if not color_hits:
        # Secours : pas de caméra → on nomme parcel_1, parcel_2… par ordre spatial
        log("[FUSE] 无颜色信息，按空间顺序命名")
        named = []
        for i, c in enumerate(lidar_clusters):
            p = dict(c)
            p["name"] = PARCEL_NAMES[i] if i < len(PARCEL_NAMES) else "parcel_%d" % (i + 1)
            p["color"] = "unknown"
            named.append(p)
        return named

    xs = [c["center"][0] for c in lidar_clusters]
    ys = [c["center"][1] for c in lidar_clusters]
    used = set()
    fused = []

    for ch in color_hits:
        best_i = None
        best_d = float("inf")
        # Chercher le cluster LiDAR dont la projection image est la plus proche de la couleur
        for i, lc in enumerate(lidar_clusters):
            if i in used:
                continue
            u, v = _lidar_norm_uv(lc, xs, ys)
            du = ch["u_norm"] - u
            dv = ch["v_norm"] - v
            d = du * du + dv * dv  # distance au carré (évite sqrt)
            if d < best_d:
                best_d = d
                best_i = i
        if best_i is None:
            continue
        used.add(best_i)
        parcel = dict(lidar_clusters[best_i])
        parcel["name"] = ch["name"]
        parcel["color"] = ch["color"]
        fused.append(parcel)
        cx, cy, cz = parcel["center"]
        log("[FUSE] %s (%s) ← LiDAR (%.3f, %.3f, %.3f) match_dist=%.3f",
            parcel["name"], parcel["color"], cx, cy, cz, math.sqrt(best_d))

    # Clusters LiDAR restants sans couleur associée
    for i, lc in enumerate(lidar_clusters):
        if i in used:
            continue
        parcel = dict(lc)
        parcel["name"] = "parcel_unknown_%d" % (i + 1)
        parcel["color"] = "unknown"
        fused.append(parcel)
        log("[FUSE] 未匹配颜色的簇 → %s @ (%.3f, %.3f, %.3f)",
            parcel["name"], *parcel["center"])

    return _sort_parcels(fused)


def detect_parcels(lidar, cam, log):
    """
    Point d'entrée perception : enchaîne LiDAR → RGB → fusion.
    Retourne une liste de dicts : {name, color, center, size_xy, n_points}.
    """
    lidar_clusters = _lidar_clusters_raw(lidar, log)
    rgb = cam.get_head_rgb()
    color_hits = _detect_color_parcels(rgb, log) if rgb is not None else []
    if rgb is None:
        log("[COLOR] 未获取到 RGB，仅使用 LiDAR 几何簇")

    parcels = _fuse_lidar_and_color(lidar_clusters, color_hits, log)
    for p in parcels:
        cx, cy, cz = p["center"]
        log("[DETECT] 最终 %s (%s): center=(%.3f, %.3f, %.3f)",
            p["name"], p.get("color", "?"), cx, cy, cz)
    log("[DETECT] 融合完成，共 %d 个快递", len(parcels))
    return parcels


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
    Indique distance moyenne et décalage horizontal de la table.
    """
    depth = cam.get_head_depth()
    if depth is None:
        log("[DETECT] 深度图: 未获取到数据")
        return None

    valid = np.isfinite(depth) & (depth > 0.35) & (depth < 1.2)
    count = int(valid.sum())
    if count < 500:
        log("[DETECT] 深度图: 有效像素太少 (%d)", count)
        return None

    avg_dist = float(depth[valid].mean())
    h, w = depth.shape
    ys, xs = np.where(valid)
    offset_x = (xs.mean() - w / 2.0) / w
    log("[DETECT] 深度图: 平均距离 %.2f m, 水平偏移 %.2f (正=偏右)", avg_dist, offset_x)
    return {"avg_dist": avg_dist, "offset_x": float(offset_x)}


# =============================================================================
# DÉCISION — choisir quel colis saisir en premier
# =============================================================================

def _parcel_xy_distance(parcel):
    """Distance horizontale du colis à l'origine du robot (approximation : le plus proche)."""
    cx, cy, _ = parcel["center"]
    return math.hypot(cx, cy)


def select_nearest_parcel(parcels, log):
    """Retourne le colis le plus proche du robot (stratégie simple pour commencer)."""
    if not parcels:
        log("[SELECT] 无可用快递")
        return None
    target = min(parcels, key=_parcel_xy_distance)
    dist = _parcel_xy_distance(target)
    cx, cy, cz = target["center"]
    log("[SELECT] 目标 %s (%s) center=(%.3f, %.3f, %.3f) dist=%.3f m",
        target["name"], target.get("color", "?"), cx, cy, cz, dist)
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


def _solve_and_move(arm, left_xyz, right_xyz, log, label,
                    left_quat=None, right_quat=None):
    """
    Étape standard de mouvement :
      1. Appeler le service IK avec positions + orientations des deux mains
      2. Si succès → envoyer les 14 angles de articulations au bras
    """
    lq = GRASP_QUAT if left_quat is None else left_quat
    rq = GRASP_QUAT if right_quat is None else right_quat
    ok, joints = arm.solve_ik(left_xyz, lq, right_xyz, rq)
    if not ok:
        log("[MOVE] %s IK 求解失败", label)
        return False
    arm.go_to_joints(joints)
    rospy.sleep(ARM_SETTLE_TIME)
    return True


# =============================================================================
# ACTION — saisie main droite sur la table
# =============================================================================

def grasp_parcel_right(arm, claw, parcel, log):
    """
    Séquence de saisie avec la main droite :
      approche haute → ouvrir pince → descendre → fermer → lever.
    `parcel["center"]` vient de la perception LiDAR.
    """
    cx, cy, cz = parcel["center"]
    log("[GRASP] 开始抓取 %s (%s)", parcel["name"], parcel.get("color", "?"))

    left_wait = [cx, cy + LEFT_WAIT_Y_OFFSET, cz + APPROACH_Z_OFFSET + 0.05]
    right_above = [cx, cy + RIGHT_GRASP_Y_OFFSET, cz + APPROACH_Z_OFFSET]
    if not _solve_and_move(arm, left_wait, right_above, log, "预抓取"):
        return False

    claw.open()
    claw.right_open()
    rospy.sleep(GRIPPER_SETTLE_TIME)

    right_grasp = [cx, cy + RIGHT_GRASP_Y_OFFSET, cz + GRASP_Z_OFFSET]
    if not _solve_and_move(arm, left_wait, right_grasp, log, "下降抓取"):
        return False

    claw.right_close()
    if not claw.wait_until_done(timeout=3.0):
        log("[GRASP] 夹爪动作超时")
    rospy.sleep(GRIPPER_SETTLE_TIME)

    right_lift = [cx, cy + RIGHT_GRASP_Y_OFFSET, cz + LIFT_Z_OFFSET]
    if not _solve_and_move(arm, left_wait, right_lift, log, "抬起"):
        return False

    if claw.is_grabbed():
        log("[GRASP] %s 抓取成功", parcel["name"])
        return True

    log("[GRASP] %s 抓取失败，张开重试", parcel["name"])
    claw.right_open()
    rospy.sleep(GRIPPER_SETTLE_TIME)
    return False


# =============================================================================
# ACTION — balance, reprise, passation, bac
# =============================================================================

def place_on_weighing_area(arm, claw, log):
    """
    Phase pesée :
      1. Déplacer le colis en hauteur vers la balance
      2. Descendre au point de release
      3. Ouvrir la pince droite et attendre (simulation = zone devient jaune)
    """
    log("[WEIGH] 搬运到称重区")
    left_preset = list(LEFT_PRESET_2_IK)
    right_transit = _with_ik_z(WEIGH_RELEASE_IK, WEIGH_TRANSIT_Z)
    if not _solve_and_move(
        arm, left_preset, right_transit, log, "称重高位横移",
        left_quat=LEFT_PRESET_2_QUAT, right_quat=RIGHT_WEIGH_RELEASE_QUAT,
    ):
        return False

    if not _solve_and_move(
        arm, left_preset, list(WEIGH_RELEASE_IK), log, "称重下降释放",
        left_quat=LEFT_PRESET_2_QUAT, right_quat=RIGHT_WEIGH_RELEASE_QUAT,
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
    Reprendre le colis sur la balance (main droite) après la pesée.
    Même x/y que la pose, z plus bas pour attraper le colis posé.
    """
    log("[WEIGH] 二次抓取")
    left_preset = list(LEFT_PRESET_2_IK)
    regrasp_pre = _with_ik_z(WEIGH_REGRASP_IK, WEIGH_RELEASE_IK[2])
    if not _solve_and_move(
        arm, left_preset, regrasp_pre, log, "二次抓取对齐",
        left_quat=LEFT_PRESET_2_QUAT, right_quat=RIGHT_WEIGH_REGRASP_QUAT,
    ):
        return False

    if not _solve_and_move(
        arm, left_preset, list(WEIGH_REGRASP_IK), log, "二次抓取下降",
        left_quat=LEFT_PRESET_2_QUAT, right_quat=RIGHT_WEIGH_REGRASP_QUAT,
    ):
        return False

    claw.right_close()
    claw.wait_until_done(timeout=3.0)
    rospy.sleep(GRIPPER_SETTLE_TIME)
    if not claw.is_grabbed():
        log("[WEIGH] 二次抓取失败")
        return False

    right_lift = _with_ik_z(WEIGH_REGRASP_IK, WEIGH_TRANSIT_Z)
    if not _solve_and_move(
        arm, left_preset, right_lift, log, "二次抓取抬起",
        left_quat=LEFT_PRESET_2_QUAT, right_quat=RIGHT_WEIGH_REGRASP_QUAT,
    ):
        return False

    log("[WEIGH] 二次抓取成功")
    return True


def handoff_to_left(arm, claw, log):
    """
    Passation droite → gauche :
      1. Main droite se place au point de handoff
      2. Main gauche s'approche et ferme
      3. Main droite ouvre et recule sur y (évite collision)
    """
    log("[HANDOFF] 右手交给左手")
    left_preset = list(LEFT_PRESET_2_IK)
    right_transit = _with_ik_z(RIGHT_HANDOFF_IK, RIGHT_HANDOFF_TRANSIT_Z)
    if not _solve_and_move(
        arm, left_preset, right_transit, log, "交接高位对齐",
        left_quat=LEFT_PRESET_2_QUAT, right_quat=RIGHT_HANDOFF_QUAT,
    ):
        return False

    if not _solve_and_move(
        arm, left_preset, list(RIGHT_HANDOFF_IK), log, "交接下降",
        left_quat=LEFT_PRESET_2_QUAT, right_quat=RIGHT_HANDOFF_QUAT,
    ):
        return False

    if not _solve_and_move(
        arm, list(LEFT_HANDOFF_RECEIVE_IK), list(RIGHT_HANDOFF_IK), log, "左手接收",
        left_quat=LEFT_HANDOFF_RECEIVE_QUAT, right_quat=RIGHT_HANDOFF_QUAT,
    ):
        return False

    claw.left_close()
    claw.wait_until_done(timeout=3.0)
    rospy.sleep(GRIPPER_SETTLE_TIME)
    if not claw.is_grabbed():
        log("[HANDOFF] 左手未夹住")
        return False

    claw.right_open()
    claw.wait_until_done(timeout=3.0)
    rospy.sleep(GRIPPER_SETTLE_TIME)

    right_retract = [
        RIGHT_HANDOFF_IK[0],
        RIGHT_HANDOFF_IK[1] + RIGHT_HANDOFF_RETRACT_Y,
        RIGHT_HANDOFF_IK[2],
    ]
    if not _solve_and_move(
        arm, list(LEFT_HANDOFF_RECEIVE_IK), right_retract, log, "右手退让",
        left_quat=LEFT_HANDOFF_RECEIVE_QUAT, right_quat=RIGHT_HANDOFF_QUAT,
    ):
        return False

    log("[HANDOFF] 交接完成")
    return True


def place_in_box(arm, claw, parcel_name, log):
    """
    Dépose dans le bac avec la main gauche.
    `parcel_name` sert à choisir le petit décalage dans BOX_DROP_OFFSET_BY_PARCEL.
    """
    log("[BOX] 左手放入分拣箱 %s", parcel_name)
    drop_ik = _box_drop_ik(parcel_name)
    approach_ik = _with_ik_z(drop_ik, drop_ik[2] + PLACE_APPROACH_Z)

    if not _solve_and_move(
        arm, approach_ik, list(RIGHT_HANDOFF_IK), log, "箱上方预备",
        left_quat=LEFT_BOX_DROP_QUAT, right_quat=RIGHT_HANDOFF_QUAT,
    ):
        return False

    if not _solve_and_move(
        arm, drop_ik, list(RIGHT_HANDOFF_IK), log, "箱内下降",
        left_quat=LEFT_BOX_DROP_QUAT, right_quat=RIGHT_HANDOFF_QUAT,
    ):
        return False

    claw.left_open()
    claw.wait_until_done(timeout=3.0)
    rospy.sleep(GRIPPER_SETTLE_TIME)
    rospy.sleep(PLACE_DWELL)
    log("[BOX] 已释放")
    return True


def process_parcel_after_grasp(arm, claw, parcel, log):
    """Enchaîne les 4 phases après une saisie réussie sur la table."""
    name = parcel["name"]
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


def approach_and_grasp(robot, arm, claw, target, log):
    """
    Approche du colis :
      - posture ready
      - avancer le robot si le colis est trop loin (cx > 0.55 m)
      - saisie main droite
    """
    arm.go_ready()
    rospy.sleep(0.8)

    cx, _, _ = target["center"]
    if cx > 0.55:
        # Plus le colis est loin, plus on avance longtemps (max 2 s)
        forward_duration = min(2.0, max(0.0, (cx - 0.45) / 0.05))
        if forward_duration > 0.1:
            log("[APPROACH] 前进 %.1f s 靠近目标", forward_duration)
            robot.move_forward(0.05, duration=forward_duration)
            robot.stop()
            rospy.sleep(0.5)

    return grasp_parcel_right(arm, claw, target, log)


# =============================================================================
# MISSION PRINCIPALE — boucle sur les 4 colis
# =============================================================================

def run_scene1(robot, arm, claw, head, log):
    """
    Point d'entrée appelé par challenge_task.py.
    Paramètres injectés :
      robot — RobotMover (déplacements)
      arm   — ArmController (IK + joints)
      claw  — ClawController (pinces)
      head  — HeadController (caméra / LiDAR orientation)
      log   — fonction de log (comme rospy.loginfo)
    """
    log("=" * 50)
    log("场景一：快递称重与摆放 — 任务开始")
    log("=" * 50)

    # --- Initialisation ---
    log("[STEP 1] 切换手臂到外部控制模式")
    arm.switch_to_external_control()  # obligatoire avant d'envoyer des commandes bras
    rospy.sleep(0.5)
    arm.go_home()
    rospy.sleep(0.5)

    log_scene_landmarks(log)

    lidar = LidarReader()
    cam = CameraReader()
    rospy.sleep(0.5)

    completed = 0  # colis traités avec succès
    failed = 0     # tentatives ratées

    # --- Boucle : un colis à la fois jusqu'à 4 réussites ---
    while completed < MAX_PARCELS:
        log("[LOOP] 第 %d/%d 个快递 (失败 %d)", completed + 1, MAX_PARCELS, failed)

        log("[STEP 2] 低头观察桌面")
        head.look_down(20)  # incliner la tête pour voir la table
        rospy.sleep(0.8)

        log("[STEP 3] LiDAR + RGB 融合检测")
        parcels = detect_parcels(lidar, cam, log)
        if len(parcels) < 4 - completed:
            log("[STEP 3] 检测数量偏少，补充深度图检查")
            inspect_table_depth(cam, log)

        head.look_forward()
        rospy.sleep(0.3)

        if not parcels:
            log("[LOOP] 未检测到剩余快递，提前结束")
            break

        target = select_nearest_parcel(parcels, log)
        log("[STEP 4] 目标: %s", target["name"])

        log("[STEP 5] 抓取")
        grasped = approach_and_grasp(robot, arm, claw, target, log)
        if not grasped:
            failed += 1
            log("[LOOP] 抓取失败，跳过本轮")
            arm.go_home()
            rospy.sleep(0.5)
            continue  # passer au tour suivant sans incrémenter completed

        log("[STEP 6] 称重 → 二次抓取 → 交接 → 入箱")
        if process_parcel_after_grasp(arm, claw, target, log):
            completed += 1
            log("[LOOP] 成功 %d/%d", completed, MAX_PARCELS)
        else:
            failed += 1
            log("[LOOP] 后处理失败")

        arm.go_home()
        rospy.sleep(0.5)

    log("[DONE] 完成 %d/%d 个快递，失败 %d 次", completed, MAX_PARCELS, failed)
    log("场景一：任务结束")
