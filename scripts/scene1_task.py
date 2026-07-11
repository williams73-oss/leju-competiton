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
from perception_api import CameraReader, LidarReader, TFReader

# OpenCV est optionnel : sans lui, on ne peut pas reconnaître les couleurs
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


# =============================================================================
# CONSTANTES — PERCEPTION (LiDAR + couleur)
# =============================================================================

# =============================================================================
# MODE — choisir UNE seule option True
#   PERCEPTION_ONLY : voir coordonnées, pas de mouvement bras
#   TOUCH_TEST      : perception puis toucher chaque colis (valider visuellement)
#   (les deux False) : mission complète saisie / balance / bac
# =============================================================================
PERCEPTION_ONLY = False
TOUCH_TEST = False

# Toucher le dessus du colis (pince ouverte, pas de fermeture)
TOUCH_Z_ABOVE_CENTER = 0.02       # +2 cm au-dessus du centre détecté ≈ dessus colis
TOUCH_Z_ABOVE_BOX = 0.06          # bac : bord un peu plus haut
TOUCH_DWELL = 1.0                 # pause contact pour observer dans la simu
TOUCH_USE_FORWARD = False         # pas de marche — risque chute en external control
TOUCH_ARM_SETTLE = 2.0            # plus long que ARM_SETTLE_TIME (stabilité MPC)
TOUCH_TABLE_Z_MIN = -0.12
TOUCH_TABLE_Z_MAX = 0.02          # table IK z ≈ -0.04 ; rejette rgb-depth z>0
TOUCH_MIN_X = 0.22                # cibles trop proches = coords aberrantes

# Landmarks scène — couleur (scene1.yaml) + zones LiDAR IK
WEIGH_MARKER_REF_BGR = (66, 158, 26)       # weighing_area 0.10, 0.62, 0.26
BOX_MARKER_REF_BGR = (87, 158, 26)         # drop_box 0.10, 0.62, 0.34
WEIGH_MARKER_HSV = ((40, 100, 100), (85, 255, 255))
WEIGH_LIDAR_X_RANGE = (0.28, 0.52)
WEIGH_LIDAR_Y_RANGE = (-0.72, -0.42)
WEIGH_LIDAR_Z_RANGE = (-0.20, 0.25)
MIN_WEIGH_LIDAR_POINTS = 4
WEIGH_LAB_DIST_MAX = 42.0
WEIGH_COLOR_U_RANGE = (0.02, 0.42)
WEIGH_COLOR_V_RANGE = (0.35, 0.92)
WEIGH_COLOR_MAX_ERR_XY = 0.18
BOX_LIDAR_X_RANGE = (0.42, 0.88)
BOX_LIDAR_Y_RANGE = (0.02, 0.45)
BOX_LIDAR_Z_RANGE = (-0.15, 0.45)
BOX_FLOOR_LIDAR_Z_RANGE = (-0.11, 0.03)   # sol vert du bac (évite murs z≈0.09)
BOX_LIDAR_X_PERCENTILE = 82               # bord proche → intérieur dépose
BOX_LIDAR_Y_PERCENTILE = 72               # sol visible biaisé latéralement
BOX_COLOR_U_RANGE = (0.48, 0.98)
BOX_COLOR_V_RANGE = (0.35, 0.95)
MIN_LANDMARK_COLOR_PIXELS = 80
MAX_WEIGH_MARKER_AREA = 12000
MIN_BOX_LIDAR_POINTS = 6
LANDMARK_WORLD_POS = {
    "weighing_area": (-0.17, -0.56, 0.880),
    "sorting_box": (0.10, 0.29, 0.880),
}
TOUCH_ORDER = [
    "weighing_area",
    "parcel_1", "parcel_2", "parcel_3", "parcel_4",
    "sorting_box",
]

PARCEL_NAMES = ["parcel_1", "parcel_2", "parcel_3", "parcel_4"]

# Zone de filtrage LiDAR dans le repère base_link (IK) du robot :
#   x = devant, y = latéral, z = hauteur (table ~ z=-0.04, pas z=0.88 monde)
# Conversion world→IK : collect_scene1_handoff_dataset.py WORLD_TO_IK_OFFSET
WORLD_TO_IK_OFFSET = (0.565966, -0.013886, -0.923783)
PARCEL_WORLD_POS = {
    "parcel_1": (-0.26, -0.31, 0.880),
    "parcel_2": (-0.26, -0.09, 0.880),
    "parcel_3": (-0.11, -0.31, 0.880),
    "parcel_4": (-0.11, -0.09, 0.880),
}
TABLE_X_RANGE = (0.20, 0.55)     # colis IK x ≈ 0.31 … 0.46
TABLE_Y_RANGE = (-0.45, -0.05)   # colis IK y ≈ -0.32 … -0.10
PARCEL_Z_RANGE = (-0.20, 0.25)   # centre colis IK z ≈ -0.04 (world 0.88 − offset)
PARCEL_RGB_Z_RANGE = (-0.12, 0.05)  # rgb-ray/depth : table z≈-0.04 ; rejette z≈+0.2
TABLE_PARCEL_Z = -0.04           # hauteur saisie (RIGHT_PICK_IK_Z ≈ -0.03)
MAX_COLOR_AREA_RATIO = 0.25      # rejette masque HSV > 25 % image (faux bleu ciel)
COLOR_ROI_V_START = 0.15         # ignorer le ciel : ROI = bande basse de l'image

# Clustering : regrouper les points LiDAR proches en "amas" (1 amas ≈ 1 colis)
CLUSTER_EPS_XY = 0.08            # points clairsemés sur la table
MIN_CLUSTER_POINTS = 3             # peu de retours LiDAR par colis
MIN_PARCEL_SIZE_XY = 0.008         # filtre bruit très petit
MAX_PARCEL_SIZE_XY = 0.14          # au-dessus → découpe grille 2×2
LIDAR_GRID_X_MID = 0.381           # séparation parcel_1/2 vs 3/4 (IK x)
LIDAR_GRID_Y_MID = -0.214          # séparation parcel_1/3 vs 2/4 (IK y)
LIDAR_SEED_RADIUS = 0.13           # regroupement LiDAR autour positions seed0 IK
LIDAR_SEED_MIN_POINTS = 2          # grid_2 (jaune) souvent < 3 pts
MIN_COLOR_PIXELS = 120             # pixels HSV min (colis lointains / petits)
MIN_COLOR_PIXELS_BY_NAME = {"parcel_2": 70}
LAB_COLOR_DIST_BY_NAME = {"parcel_2": 58.0}
MAX_FUSE_UV_DIST = 0.12            # au-delà : couleur seule, pas le mauvais cluster LiDAR
PERCEPTION_ATTEMPTS = 3            # tentatives avant rapport final
PERCEPTION_ERR_TARGET = 0.05       # N2 : arrêt si 4/4 colis err_xy < 5 cm
LANDMARK_ERR_TARGET = 0.05         # N1b : balance + bac err_xy < 5 cm
SPATIAL_NAME_MAX_ERR = 0.12        # nommage LiDAR → parcel le plus proche (m)
MAX_BLUE_MASK_AREA = 80000         # rejette faux bleu (ciel/table)
DEPTH_Z_MIN = 0.35                 # profondeur valide table (m)
DEPTH_Z_MAX = 1.2
RGB_DEPTH_XY_TOL = 0.10            # depth vs ray-plane : max Δxy pour accepter depth
LIDAR_O3D_PLANE_DIST = 0.015       # RANSAC distance au plan table (m)
LIDAR_O3D_ABOVE_MAX = 0.07         # points colis au-dessus du plan (m)
LIDAR_O3D_EPS = 0.065              # DBSCAN 3D (m)
LIDAR_O3D_MIN_POINTS = 3
HUNGARIAN_BIG_COST = 1e4

# Couleurs nominales scene1.yaml (RGBA → BGR OpenCV) — distance LAB
TABLE_REF_BGR = (26, 77, 26)       # table_top 0.10, 0.30, 0.10
LAB_COLOR_DIST_MAX = 50.0          # seuil ΔE approx en espace LAB
TABLE_LAB_DIST_MAX = 32.0          # exclusion table verte
COLOR_V_NORM_MAX = 0.96            # table visible en bas avec look_down
COLOR_V_NORM_MAX_BY_NAME = {"parcel_2": 0.93}  # jaune : rejette énormes masques bord
COLOR_MAX_AREA_BY_NAME = {"parcel_2": 12000}   # jaune : masque >12k px = faux positif
BASE_VALID_X = (0.18, 0.58)        # base_link IK valide pour rgb-depth
BASE_VALID_Y = (-0.48, -0.02)
BASE_VALID_Z = (-0.18, 0.08)
# Secours HSV si LAB insuffisant (scene1 colis pâles)
PARCEL_HSV_FALLBACK = [
    ("parcel_2", "yellow", (12, 40, 80), (48, 255, 255)),
    ("parcel_1", "brown", (10, 30, 120), (30, 160, 240)),
    ("parcel_3", "orange", (5, 60, 100), (25, 255, 255)),
    ("parcel_4", "blue", (95, 25, 160), (125, 90, 255)),
]
PARCEL_REF_COLORS = [
    ("parcel_2", "yellow", (97, 199, 235)),   # jaune en premier (évite masque bleu/table)
    ("parcel_1", "brown", (186, 214, 224)),
    ("parcel_3", "orange", (71, 140, 230)),
    ("parcel_4", "blue", (235, 214, 199)),
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


def _summarize_cluster(points, indices, seed_xy=None):
    """Calcule le centre et la taille d'un amas de points LiDAR."""
    pts = points[indices]
    if seed_xy is not None and len(pts) > 4:
        dist = np.hypot(pts[:, 0] - seed_xy[0], pts[:, 1] - seed_xy[1])
        keep_n = max(3, int(len(pts) * 0.65))
        pts = pts[np.argsort(dist)[:keep_n]]
    center = pts.mean(axis=0)
    return {
        "center": tuple(float(v) for v in center),
        "size_xy": (float(pts[:, 0].ptp()), float(pts[:, 1].ptp())),
        "n_points": len(indices),
    }


def _sort_parcels(clusters):
    """Trie les colis de gauche à droite puis de bas en haut (y puis x)."""
    return sorted(clusters, key=lambda c: (c["center"][1], c["center"][0]))


def _lidar_clusters_seeded(pts, log):
    """Fallback : amas par proximité aux positions nominales seed0 (IK)."""
    expected = _expected_parcel_positions()
    clusters = []
    for name in PARCEL_NAMES:
        ex, ey, _ = expected[name]
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
    """Fallback : 4 quadrants fixes (scene1 seed0, 2×2 sur la table)."""
    masks = [
        (pts[:, 0] < LIDAR_GRID_X_MID) & (pts[:, 1] < LIDAR_GRID_Y_MID),
        (pts[:, 0] < LIDAR_GRID_X_MID) & (pts[:, 1] >= LIDAR_GRID_Y_MID),
        (pts[:, 0] >= LIDAR_GRID_X_MID) & (pts[:, 1] < LIDAR_GRID_Y_MID),
        (pts[:, 0] >= LIDAR_GRID_X_MID) & (pts[:, 1] >= LIDAR_GRID_Y_MID),
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


def _append_color_hit(hits, remaining, name, label, mask, h, w, v0,
                      depth, cam, tf_reader, log, via="LAB"):
    area = int(mask.sum()) // 255
    max_area = int(h * w * MAX_COLOR_AREA_RATIO)
    min_px = MIN_COLOR_PIXELS_BY_NAME.get(name, MIN_COLOR_PIXELS)
    if area < min_px:
        log("[COLOR] %s (%s): 有效像素不足 (%s, need>=%d)", name, label, via, min_px)
        return remaining
    if area > max_area:
        log("[COLOR] %s (%s): 面积过大 (%d)，跳过", name, label, area)
        return remaining
    if name == "parcel_4" and area > MAX_BLUE_MASK_AREA:
        log("[COLOR] %s (%s): masque bleu trop large (%d)，跳过", name, label, area)
        return remaining
    max_area_name = COLOR_MAX_AREA_BY_NAME.get(name)
    if max_area_name is not None and area > max_area_name:
        log("[COLOR] %s (%s): masque trop large (%d > %d)，跳过",
            name, label, area, max_area_name)
        return remaining
    ys, xs = np.where(mask > 0)
    u_px = float(xs.mean())
    v_px = float(ys.mean() + v0)
    v_norm = v_px / h
    v_max = COLOR_V_NORM_MAX_BY_NAME.get(name, COLOR_V_NORM_MAX)
    if v_norm > v_max:
        log("[COLOR] %s (%s): v=%.2f trop bas (bord image)，跳过", name, label, v_norm)
        return remaining

    remaining = cv2.bitwise_and(remaining, cv2.bitwise_not(mask))
    mask_full = np.zeros((h, w), dtype=np.uint8)
    mask_full[v0:, :] = mask

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
    Segmentation couleur par distance LAB aux matériaux scene1.yaml.
    Si depth + TF disponibles → centre 3D base_link (rgb-depth).
    """
    if not _HAS_CV2:
        log("[COLOR] opencv-python 未安装，跳过颜色识别")
        return []

    h, w = rgb.shape[:2]
    v0 = int(h * COLOR_ROI_V_START)
    roi = rgb[v0:, :, :]
    rh, rw = roi.shape[:2]
    lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    table_lab = _bgr_to_lab(TABLE_REF_BGR)
    table_dist = np.sqrt(np.sum((lab_roi - table_lab) ** 2, axis=2))
    remaining = ((table_dist > TABLE_LAB_DIST_MAX).astype(np.uint8)) * 255
    hits = []
    detected = set()

    for name, label, ref_bgr in PARCEL_REF_COLORS:
        ref_lab = _bgr_to_lab(ref_bgr)
        dist_max = LAB_COLOR_DIST_BY_NAME.get(name, LAB_COLOR_DIST_MAX)
        dist = np.sqrt(np.sum((lab_roi - ref_lab) ** 2, axis=2))
        mask = (dist < dist_max).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.bitwise_and(mask, remaining)
        area_before = int(mask.sum()) // 255
        min_px = MIN_COLOR_PIXELS_BY_NAME.get(name, MIN_COLOR_PIXELS)
        if area_before >= min_px:
            remaining = _append_color_hit(
                hits, remaining, name, label, mask, h, w, v0,
                depth, cam, tf_reader, log, via="LAB")
            detected.add(name)
        elif area_before > 0:
            log("[COLOR] %s (%s): LAB pixels=%d < %d", name, label, area_before, min_px)

    for name, label, lo, hi in PARCEL_HSV_FALLBACK:
        if name in detected:
            continue
        mask = _color_mask_hsv(roi, lo, hi, remaining)
        remaining = _append_color_hit(
            hits, remaining, name, label, mask, h, w, v0,
            depth, cam, tf_reader, log, via="HSV")

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


def _world_to_ik(x, y, z):
    ox, oy, oz = WORLD_TO_IK_OFFSET
    return x + ox, y + oy, z + oz


def _uv_to_table_xy(u_norm, v_norm):
    """u,v image → x,y table (base_link IK), calibré scene1 seed0."""
    x = 0.91 - 0.83 * v_norm
    y = -0.55 + 0.75 * u_norm
    return x, y


def _table_z_from_lidar(lidar_clusters):
    if lidar_clusters:
        return float(np.median([c["center"][2] for c in lidar_clusters]))
    return TABLE_PARCEL_Z


def _parcel_from_color_hit(ch, z, log, source="couleur"):
    if ch.get("center_base") is not None:
        x, y, z = ch["center_base"]
        source = ch.get("rgb_source") or "rgb-depth"
    else:
        x, y = _uv_to_table_xy(ch["u_norm"], ch["v_norm"])
        z = TABLE_PARCEL_Z
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


def _inject_rgb_parcels(parcels, color_hits, log):
    """Colis sans LiDAR : injecter rgb-ray / rgb-depth depuis les hits couleur."""
    by_name = {p["name"]: p for p in parcels if p["name"] in PARCEL_NAMES}
    hits_by_name = {ch["name"]: ch for ch in color_hits if ch["name"] in PARCEL_NAMES}

    for name in PARCEL_NAMES:
        ch = hits_by_name.get(name)
        if ch is None or ch.get("center_base") is None:
            continue
        src = ch.get("rgb_source") or "rgb-ray"
        if not _is_rgb_backed(src):
            continue
        cur = by_name.get(name)
        if cur is not None and _source_priority(cur.get("source", "")) <= _source_priority(src):
            continue
        p = _parcel_from_color_hit(ch, TABLE_PARCEL_Z, log, src)
        by_name[name] = p
        log("[FUSE] %s (%s) ← injecté %s (vraie détection caméra)",
            name, p["color"], src)

    return _sort_parcels(list(by_name.values()))


def _fill_missing_parcels(parcels, color_hits, log):
    """Complète colis manquants : rgb-ray d'abord, row-infer seulement en secours."""
    by_name = {p["name"]: p for p in parcels if p["name"] in PARCEL_NAMES}
    hits_by_name = {ch["name"]: ch for ch in color_hits if ch["name"] in PARCEL_NAMES}
    expected = _expected_parcel_positions()
    if len(by_name) >= len(PARCEL_NAMES):
        return _sort_parcels(list(by_name.values()))

    for name in PARCEL_NAMES:
        if name in by_name:
            continue
        ch = hits_by_name.get(name)
        if ch is not None and ch.get("center_base") is not None:
            src = ch.get("rgb_source") or "rgb-ray"
            if _is_rgb_backed(src):
                by_name[name] = _parcel_from_color_hit(ch, TABLE_PARCEL_Z, log, src)
                continue

    for left, right in (("parcel_1", "parcel_2"), ("parcel_3", "parcel_4")):
        for missing, partner in ((right, left), (left, right)):
            if missing in by_name or partner not in by_name:
                continue
            px, py, pz = by_name[partner]["center"]
            ex_m, ey_m, _ = expected[missing]
            ex_p, ey_p, _ = expected[partner]
            infer_y = py + (ey_m - ey_p)
            p = {
                "name": missing,
                "color": _color_label_for_parcel(missing),
                "center": (px, infer_y, pz),
                "size_xy": (0.06, 0.05),
                "n_points": 0,
                "source": "row-infer",
            }
            by_name[missing] = p
            log("[FUSE] %s (%s) ← row-infer secours (%.3f, %.3f, %.3f) depuis %s",
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
    parcels = [_parcel_from_color_hit(ch, z, log, "couleur seule") for ch in color_hits]
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
    return s in ("rgb-ray", "rgb-depth", "rgb-depth+zsnap") or s.startswith("rgb-")


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
    """Espacement nominal grille 2×2 (scene1 seed0, IK)."""
    e = _expected_parcel_positions()
    col_dx = e["parcel_3"][0] - e["parcel_1"][0]
    row_dy = e["parcel_2"][1] - e["parcel_1"][1]
    return col_dx, row_dy


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
        left_x = right_x - col_dx
        for name in ("parcel_1", "parcel_2"):
            p = by_name.get(name)
            if p is None:
                continue
            cx, cy, cz = p["center"]
            if abs(cx - left_x) < 0.015 and "grid" in (p.get("source") or ""):
                continue
            p["center"] = (left_x, cy, cz)
            p["source"] = (p.get("source") or "?") + "+grid-x"
            log("[FUSE] %s: x %.3f → %.3f (grille 2×2, colonne droite x=%.3f)",
                name, cx, left_x, right_x)

    row_pairs = (("parcel_1", "parcel_3"), ("parcel_2", "parcel_4"))
    for left_name, right_name in row_pairs:
        lp, rp = by_name.get(left_name), by_name.get(right_name)
        if lp is None or rp is None:
            continue
        if not _is_reliable_lidar_parcel(rp):
            continue
        _, ry, rz = rp["center"]
        cx, cy, cz = lp["center"]
        if abs(cy - ry) < 0.015 and "grid" in (lp.get("source") or ""):
            continue
        lp["center"] = (cx, ry, cz if abs(cz - TABLE_PARCEL_Z) < 0.08 else rz)
        lp["source"] = (lp.get("source") or "?") + "+grid-y"
        log("[FUSE] %s: y %.3f → %.3f (même rangée que %s)",
            left_name, cy, ry, right_name)

    if p2 is not None and _is_reliable_lidar_parcel(p4):
        cx, cy, cz = p2["center"]
        _, py4, pz4 = p4["center"]
        left_x = by_name["parcel_1"]["center"][0] if p1 else cx
        if "row-infer" in (p2.get("source") or "") or not _is_lidar_backed(p2.get("source", "")):
            p2["center"] = (left_x, py4, TABLE_PARCEL_Z)
            p2["source"] = "grid-2x2"
            log("[FUSE] %s (yellow) ← grille 2×2 (%.3f, %.3f, %.3f) depuis %s",
                p2["name"], left_x, py4, TABLE_PARCEL_Z, "parcel_4")

    return _sort_parcels(list(by_name.values()))


def _name_clusters_spatial(lidar_clusters, log):
    """Associe chaque cluster LiDAR au colis le plus proche (positions seed0 IK)."""
    expected = _expected_parcel_positions()
    remaining = list(lidar_clusters)
    used_names = set()
    out = []

    while remaining and len(used_names) < len(PARCEL_NAMES):
        best_ci = best_name = None
        best_d = float("inf")
        for ci, lc in enumerate(remaining):
            cx, cy, _ = lc["center"]
            for name in PARCEL_NAMES:
                if name in used_names:
                    continue
                ex, ey, _ = expected[name]
                d = math.hypot(cx - ex, cy - ey)
                if d < best_d:
                    best_d, best_ci, best_name = d, ci, name
        if best_name is None or best_d > SPATIAL_NAME_MAX_ERR:
            break
        lc = remaining.pop(best_ci)
        p = dict(lc)
        p["name"] = best_name
        p["color"] = _color_label_for_parcel(best_name)
        p["source"] = "lidar-spatial"
        used_names.add(best_name)
        out.append(p)
        cx, cy, cz = p["center"]
        log("[FUSE] %s (%s) ← LiDAR spatial (%.3f, %.3f, %.3f) err_xy=%.3f m",
            best_name, p["color"], cx, cy, cz, best_d)

    return _sort_parcels(out)


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
            cost[i, j] = du * du + dv * dv

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

    for ci, ch in enumerate(color_hits):
        if ci in matched_ci:
            continue
        fused.append(_parcel_from_color_hit(
            ch, table_z, log, "couleur (Hungarian 未匹配)"))

    return fused, used_lidar


def _fuse_lidar_and_color(lidar_clusters, color_hits, log):
    """
    Fusion : LiDAR donne la position 3D, RGB donne le nom — assignation Hungarian (N2).
    """
    if not lidar_clusters:
        if color_hits:
            log("[FUSE] LiDAR 无簇，使用颜色估计位置")
            return _parcels_from_color_only(color_hits, log, lidar_clusters)
        return []

    if len(lidar_clusters) >= 2 and len(color_hits) < 3:
        log("[FUSE] 颜色仅 %d/4 → 使用 LiDAR 空间命名 (%d 几何簇)",
            len(color_hits), len(lidar_clusters))
        named = _name_clusters_spatial(lidar_clusters, log)
        named = _inject_rgb_parcels(named, color_hits, log)
        return _snap_row_x_from_lidar_neighbors(named, log)

    if not color_hits:
        log("[FUSE] 无颜色信息，按空间顺序命名")
        named = []
        for i, c in enumerate(lidar_clusters):
            p = dict(c)
            p["name"] = PARCEL_NAMES[i] if i < len(PARCEL_NAMES) else "parcel_%d" % (i + 1)
            p["color"] = "unknown"
            p["source"] = "lidar"
            named.append(p)
        return named

    table_z = _table_z_from_lidar(lidar_clusters)

    fused, used = _fuse_color_lidar_hungarian(
        color_hits, lidar_clusters, table_z, log)

    for i, lc in enumerate(lidar_clusters):
        if i in used:
            continue
        parcel = dict(lc)
        parcel["name"] = "parcel_unknown_%d" % (i + 1)
        parcel["color"] = "unknown"
        parcel["source"] = "lidar"
        fused.append(parcel)
        log("[FUSE] 未匹配颜色的簇 → %s @ (%.3f, %.3f, %.3f)",
            parcel["name"], *parcel["center"])

    named = [p for p in fused if p["name"] in PARCEL_NAMES]
    if lidar_clusters:
        spatial = _name_clusters_spatial(lidar_clusters, log)
        by_name = {p["name"]: p for p in named}
        for p in spatial:
            cur = by_name.get(p["name"])
            if cur is None or _source_priority(p.get("source", "")) <= _source_priority(cur.get("source", "")):
                by_name[p["name"]] = p
        merged = _sort_parcels(list(by_name.values()))
        merged = _inject_rgb_parcels(merged, color_hits, log)
        merged = _snap_row_x_from_lidar_neighbors(merged, log)
        return _snap_grid_geometry(merged, log)

    fused = _inject_rgb_parcels(fused, color_hits, log)
    fused = _snap_row_x_from_lidar_neighbors(fused, log)
    return _snap_grid_geometry(fused, log)


def detect_parcels(lidar, cam, tf_reader, log):
    """
    Point d'entrée perception : enchaîne LiDAR → RGB+depth → fusion.
    Retourne une liste de dicts : {name, color, center, size_xy, n_points}.
    """
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
    if rgb is None:
        log("[COLOR] 未获取到 RGB，仅使用 LiDAR 几何簇")

    parcels = _fuse_lidar_and_color(lidar_clusters, color_hits, log)
    parcels = _fill_missing_parcels(parcels, color_hits, log)
    parcels = _snap_grid_geometry(parcels, log)
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
    return _target_dict(
        p["name"], "parcel", p.get("color", "?"),
        p["center"], p.get("source", "?"), "right",
    )


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
    return _target_dict("sorting_box", "landmark", "bac", center, source, "left")


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


def _targets_for_touch(targets):
    by_name = {t["name"]: t for t in targets}
    return [by_name[n] for n in TOUCH_ORDER if n in by_name]


def log_perception_report(parcels, log):
    """Compare détections vs positions nominales scene1 (base_link IK)."""
    expected = _expected_parcel_positions()
    detected = {p["name"]: p for p in parcels if p["name"] in PARCEL_NAMES}
    log("[REPORT] ========== 感知坐标 (base_link IK) ==========")
    max_err = 0.0
    for name in PARCEL_NAMES:
        ex, ey, ez = expected[name]
        if name not in detected:
            log("[REPORT] %s: MANQUANT  参考=(%.3f, %.3f, %.3f)", name, ex, ey, ez)
            continue
        cx, cy, cz = detected[name]["center"]
        err_xy = math.hypot(cx - ex, cy - ey)
        err_z = abs(cz - ez)
        max_err = max(max_err, err_xy)
        src = detected[name].get("source", "?")
        log("[REPORT] %s [%s]: détecté=(%.3f, %.3f, %.3f)  ref=(%.3f, %.3f, %.3f)  err_xy=%.3f m  err_z=%.3f m",
            name, src, cx, cy, cz, ex, ey, ez, err_xy, err_z)
    named = len(detected)
    log("[REPORT] 识别 %d/4  最大横向误差 %.3f m", named, max_err)
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
    return (4 - named) * 10.0 + max_err + infer * 0.03 + bad_rgb * 0.05


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

    for attempt in range(1, PERCEPTION_ATTEMPTS + 1):
        log("[PERCEPT] --- 尝试 %d/%d ---", attempt, PERCEPTION_ATTEMPTS)
        head.look_down(20)
        rospy.sleep(1.5)

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

        head.look_forward()
        rospy.sleep(0.3)

        named, max_err = log_perception_report(parcels, log)
        score = _perception_score(parcels, named, max_err)
        lm_err, lm_count = log_touch_targets_report(
            [_parcel_to_target(p) for p in parcels] + landmarks, log,
            include_parcels=False)
        log("[PERCEPT] score N2=%.4f (err=%.3f, row-infer pénalité incluse)",
            score, max_err)
        log("[PERCEPT] N1b landmarks %d/2  err_max=%.3f m", lm_count, lm_err)
        if score < best_score:
            best_parcels = parcels
            best_named = named
            best_err = max_err
            best_score = score

        if (named >= 4 and max_err < PERCEPTION_ERR_TARGET
                and lm_count >= 2 and lm_err < LANDMARK_ERR_TARGET):
            log("[PERCEPT] N2 4/4 + N1b landmarks OK (< %.0f cm)，停止重试",
                PERCEPTION_ERR_TARGET * 100)
            break
        rospy.sleep(0.5)

    log("[DONE] 感知测试完成：最佳结果 %d/4  err_max=%.3f m  score=%.4f",
        best_named, best_err, best_score)
    for p in best_parcels:
        cx, cy, cz = p["center"]
        log("[DONE]   %s (%s): (%.3f, %.3f, %.3f)",
            p["name"], p.get("color", "?"), cx, cy, cz)
    log("场景一：感知测试结束")


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


def _solve_and_move_touch(arm, left_xyz, right_xyz, log, label,
                          left_quat=None, right_quat=None):
    ok = _solve_and_move(arm, left_xyz, right_xyz, log, label,
                         left_quat=left_quat, right_quat=right_quat)
    if ok:
        rospy.sleep(TOUCH_ARM_SETTLE)
    return ok


def touch_with_right_hand(arm, claw, target, log):
    """
    Approche et contact léger : pince droite ouverte, pas de saisie.
    Valide visuellement si la position perçue est correcte.
    """
    target = _sanitize_touch_target(target, log)
    cx, cy, cz = target["center"]
    touch_z = cz + TOUCH_Z_ABOVE_CENTER
    if target.get("kind") == "landmark" and target["name"] == "weighing_area":
        touch_z = cz + TOUCH_Z_ABOVE_CENTER + 0.01
    name = target["name"]
    log("[TOUCH] %s (%s/%s) cible contact (%.3f, %.3f, %.3f)",
        name, target.get("color", "?"), target.get("kind", "?"), cx, cy, touch_z)

    left_wait = [cx, cy + LEFT_WAIT_Y_OFFSET, touch_z + APPROACH_Z_OFFSET + 0.05]
    right_above = [cx, cy + RIGHT_GRASP_Y_OFFSET, touch_z + APPROACH_Z_OFFSET]
    if not _solve_and_move_touch(arm, left_wait, right_above, log, "TOUCH 预接近"):
        log("[TOUCH] %s IK 失败 (approche)", name)
        return False

    claw.open()
    claw.right_open()
    rospy.sleep(GRIPPER_SETTLE_TIME)

    right_touch = [cx, cy + RIGHT_GRASP_Y_OFFSET, touch_z]
    if not _solve_and_move_touch(arm, left_wait, right_touch, log, "TOUCH 下降接触"):
        log("[TOUCH] %s IK 失败 (contact)", name)
        return False

    log("[TOUCH] %s 接触 — 观察 simu (%.1f s)", name, TOUCH_DWELL)
    rospy.sleep(TOUCH_DWELL)

    if not _solve_and_move_touch(arm, left_wait, right_above, log, "TOUCH 抬起"):
        log("[TOUCH] %s IK 失败 (retrait)", name)
        return False

    log("[TOUCH] %s terminé OK", name)
    return True


def touch_with_left_hand(arm, claw, target, log):
    """Contact léger main gauche (bac) — pose neutre pour stabilité."""
    target = _sanitize_touch_target(target, log)
    cx, cy, cz = target["center"]
    touch_z = cz + TOUCH_Z_ABOVE_BOX
    name = target["name"]
    log("[TOUCH] %s (%s) main gauche → (%.3f, %.3f, %.3f)",
        name, target.get("color", "?"), cx, cy, touch_z)

    right_wait = [0.24, -0.16, 0.30]
    left_above = [cx, cy, touch_z + APPROACH_Z_OFFSET]
    if not _solve_and_move_touch(
        arm, left_above, right_wait, log, "TOUCH 左臂预接近",
        left_quat=GRASP_QUAT, right_quat=GRASP_QUAT,
    ):
        log("[TOUCH] %s IK 失败 (approche gauche)", name)
        return False

    claw.open()
    claw.left_open()
    rospy.sleep(GRIPPER_SETTLE_TIME)

    left_touch = [cx, cy, touch_z]
    if not _solve_and_move_touch(
        arm, left_touch, right_wait, log, "TOUCH 左臂接触",
        left_quat=GRASP_QUAT, right_quat=GRASP_QUAT,
    ):
        log("[TOUCH] %s IK 失败 (contact gauche)", name)
        return False

    log("[TOUCH] %s 接触 — 观察 simu (%.1f s)", name, TOUCH_DWELL)
    rospy.sleep(TOUCH_DWELL)

    if not _solve_and_move_touch(
        arm, left_above, right_wait, log, "TOUCH 左臂抬起",
        left_quat=GRASP_QUAT, right_quat=GRASP_QUAT,
    ):
        return False

    log("[TOUCH] %s terminé OK (gauche)", name)
    return True


def touch_target(robot, arm, claw, target, log):
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
        return touch_with_left_hand(arm, claw, target, log)
    return touch_with_right_hand(arm, claw, target, log)


def approach_and_touch(robot, arm, claw, target, log):
    """Toucher une cible depuis posture stable (home)."""
    return touch_target(robot, arm, claw, target, log)


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

    log("[STEP 2] 低头感知 — colis, balance, bac")
    head.look_down(20)
    rospy.sleep(1.5)
    targets = detect_all_touch_targets(lidar, cam, tf_reader, log)
    head.look_forward()
    rospy.sleep(0.3)

    log_touch_targets_report(targets, log)
    touch_list = _targets_for_touch(targets)
    log("[TOUCH] %d cibles à toucher (balance → colis → bac)", len(touch_list))
    for t in touch_list:
        cx, cy, cz = t["center"]
        log("[TOUCH]   planifié %s (%s) main=%s → (%.3f, %.3f, %.3f)",
            t["name"], t.get("color", "?"), t.get("hand", "?"), cx, cy, cz)

    if not touch_list:
        log("[TOUCH] rien détecté — 中止")
        return

    log("[STEP 3] contact — bras depuis home (stable)")
    _stable_between_touches(arm, robot, log)

    touched = 0
    failed = 0
    for i, target in enumerate(touch_list, 1):
        log("[TOUCH] --- %d/%d : %s ---", i, len(touch_list), target["name"])
        if approach_and_touch(robot, arm, claw, target, log):
            touched += 1
        else:
            failed += 1
        _stable_between_touches(arm, robot, log)

    log("[DONE] 触摸测试完成：成功 %d/%d  失败 %d",
        touched, len(touch_list), failed)
    log("场景一：触摸测试结束 — simu : pince sur chaque élément détecté ?")


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


# =============================================================================
# DÉCISION — choisir quel colis saisir en premier
# =============================================================================

def _parcel_xy_distance(parcel):
    """Distance horizontale du colis à l'origine du robot (approximation : le plus proche)."""
    cx, cy, _ = parcel["center"]
    return math.hypot(cx, cy)


def _parcel_select_score(parcel):
    """Score bas = meilleur candidat. Pénalise sources peu fiables."""
    score = _parcel_xy_distance(parcel)
    src = parcel.get("source") or ""
    if "row-infer" in src or src == "layout":
        score += 0.25
    elif "grid-2x2" in src or "grid-x" in src:
        score += 0.12
    elif not _is_lidar_backed(src) and not _is_rgb_backed(src):
        score += 0.18
    return score


def select_nearest_parcel(parcels, log, exclude_names=None, skip_failures=None):
    """Choisit le colis le plus accessible (distance + fiabilité perception)."""
    exclude = set(exclude_names or ())
    failures = skip_failures or {}
    candidates = [
        p for p in parcels
        if p.get("name") not in exclude and failures.get(p.get("name"), 0) < 2
    ]
    if not candidates:
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


def _solve_and_move(arm, left_xyz, right_xyz, log, label,
                    left_quat=None, right_quat=None):
    """
    Étape standard de mouvement :
      1. Appeler le service IK avec positions + orientations des deux mains
      2. Si succès → envoyer les 14 angles de articulations au bras
    """
    lq = GRASP_QUAT if left_quat is None else left_quat
    rq = GRASP_QUAT if right_quat is None else right_quat
    try:
        ok, joints = arm.solve_ik(left_xyz, lq, right_xyz, rq)
    except rospy.exceptions.ROSException as exc:
        log("[MOVE] %s IK 服务不可用: %s", label, exc)
        return False
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
    if abs(cz - TABLE_PARCEL_Z) > 0.03:
        log("[GRASP] %s z %.3f → %.3f (table)", parcel["name"], cz, TABLE_PARCEL_Z)
        cz = TABLE_PARCEL_Z
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


def approach_and_grasp(robot, arm, claw, target, head, lidar, cam, tf_reader, log):
    """
    Approche du colis :
      - posture ready
      - avancer le robot si le colis est trop loin (cx > 0.55 m)
      - re-détection après marche
      - saisie main droite
    """
    arm.go_ready()
    rospy.sleep(0.8)

    cx, _, _ = target["center"]
    moved = False
    if cx > 0.55:
        forward_duration = min(2.0, max(0.0, (cx - 0.45) / 0.05))
        if forward_duration > 0.1:
            log("[APPROACH] 前进 %.1f s 靠近目标", forward_duration)
            robot.move_forward(0.05, duration=forward_duration)
            robot.stop()
            rospy.sleep(0.5)
            moved = True

    if moved and head is not None and lidar is not None and cam is not None:
        log("[APPROACH] 前进后重新检测 %s", target["name"])
        head.look_down(20)
        rospy.sleep(1.2)
        fresh = detect_parcels(lidar, cam, tf_reader, log)
        head.look_forward()
        rospy.sleep(0.3)
        for p in fresh:
            if p.get("name") == target.get("name"):
                target = p
                cx, cy, cz = target["center"]
                log("[APPROACH] %s 更新坐标 (%.3f, %.3f, %.3f)",
                    target["name"], cx, cy, cz)
                break

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
    if PERCEPTION_ONLY:
        run_scene1_perception_only(arm, head, log)
        return
    if TOUCH_TEST:
        run_scene1_touch_test(robot, arm, claw, head, log)
        return

    log("=" * 50)
    log("场景一：快递称重与摆放 — 任务开始")
    log("=" * 50)

    # --- Initialisation ---
    log("[STEP 1] 切换手臂到外部控制模式")
    arm.switch_to_external_control()  # obligatoire avant d'envoyer des commandes bras
    rospy.sleep(0.5)
    log("[STEP 1b] 等待 IK 服务就绪")
    try:
        rospy.wait_for_service("/ik/two_arm_hand_pose_cmd_srv", timeout=60.0)
        log("[STEP 1b] IK 服务就绪")
    except rospy.exceptions.ROSException:
        log("[STEP 1b] IK 服务超时 — 任务中止")
        return
    arm.go_home()
    rospy.sleep(0.5)

    log_scene_landmarks(log)

    lidar = LidarReader()
    cam = CameraReader()
    tf_reader = TFReader()
    rospy.sleep(0.5)

    completed = 0  # colis traités avec succès
    failed = 0     # tentatives ratées
    completed_names = set()
    grasp_failures = {}  # name → nb échecs saisie

    # --- Boucle : un colis à la fois jusqu'à 4 réussites ---
    while completed < MAX_PARCELS:
        log("[LOOP] 第 %d/%d 个快递 (失败 %d)", completed + 1, MAX_PARCELS, failed)

        log("[STEP 2] 低头观察桌面")
        head.look_down(20)  # incliner la tête pour voir la table
        rospy.sleep(1.5)

        log("[STEP 3] LiDAR + RGB 融合检测")
        parcels = detect_parcels(lidar, cam, tf_reader, log)
        parcels = [p for p in parcels if p.get("name") not in completed_names]
        if len(parcels) < MAX_PARCELS - completed:
            log("[STEP 3] 检测数量偏少，补充深度图检查")
            inspect_table_depth(cam, log)

        head.look_forward()
        rospy.sleep(0.3)

        if not parcels:
            log("[LOOP] 未检测到剩余快递，提前结束")
            break

        target = select_nearest_parcel(
            parcels, log, exclude_names=completed_names, skip_failures=grasp_failures)
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
            arm.go_home()
            rospy.sleep(0.5)
            continue  # passer au tour suivant sans incrémenter completed

        log("[STEP 6] 称重 → 二次抓取 → 交接 → 入箱")
        if process_parcel_after_grasp(arm, claw, target, log):
            completed += 1
            completed_names.add(target["name"])
            log("[LOOP] 成功 %d/%d", completed, MAX_PARCELS)
        else:
            failed += 1
            log("[LOOP] 后处理失败")

        arm.go_home()
        rospy.sleep(0.5)

    log("[DONE] 完成 %d/%d 个快递，失败 %d 次", completed, MAX_PARCELS, failed)
    log("场景一：任务结束")
