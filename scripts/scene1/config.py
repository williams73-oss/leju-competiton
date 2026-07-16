#!/usr/bin/env python3
"""Constantes Scene 1 — modes, perception, landmarks, bras."""
from __future__ import print_function
import math

# =============================================================================
# CONSTANTES — PERCEPTION (LiDAR + couleur)
# =============================================================================

# =============================================================================
# MODE — choisir UNE seule option True
#   PERCEPTION_ONLY : voir coordonnées, pas de mouvement bras (équipe perception)
#   TOUCH_TEST      : perception puis toucher (équipe contrôle)
#   (les deux False) : mission complète saisie / balance / bac
# =============================================================================
PERCEPTION_ONLY = False  # False = mission saisie (seed0 test)
TOUCH_TEST = False
# Backend détection colis :
#   "lidar"     = pipeline équipe (LiDAR + couleur + fuse) — défaut
#   "graphics"  = essai ych adapté (src/scene3_task.py) — abandonné seed0 (2/4, err 0.25)
PERCEPTION_BACKEND = "lidar"
DEBUG_STOP_IK_LABEL = ""  # ex. "right_x_to_pick_pre" = debug arrêt IK
DEBUG_STOP_AFTER_FIRST_IK = False
# Mission orga-flow : déjà câblé dans actions.py (grasp→weigh→handoff→box).
# Anti-triche : XY colis = detect_parcels ; points fixes scène = constantes orga.

# Tête — aligné script orga collect_scene1_handoff_dataset.py
# Orga: HEAD_TARGET = [0.0, 20.0], HEAD_SETTLE_TIME = 0.8
# Une fois baissée, on NE remonte PAS (pas de look_forward entre essais)
HEAD_LOOK_YAW = 0.0
HEAD_LOOK_PITCH = 20.0
HEAD_SETTLE_SEC = 0.8       # comme orga

# Debug labo UNIQUEMENT : comparer détection vs layout MuJoCo (world→IK).
# N'alimente JAMAIS le bras / la mission — log + CSV baseline seulement.
# Artefacts → labo/scene1/ (pas la racine repo / pas le code constructeur)
LABO_SCENE1_REL = "labo/scene1"
GT_COMPARE = False  # False en mission
GT_COMPARE_CSV = "labo/scene1/csv/scene1_gt_compare_fix_xbias.csv"
STUDY_GT_CSV = "labo/scene1/csv/scene1_study_gt_mujoco.csv"
STUDY_DET_CSV = "labo/scene1/csv/scene1_study_det_fix_xbias.csv"
COLOR_DEBUG_JPG = "labo/scene1/images/scene1_color_debug.jpg"

# Toucher le dessus du colis (pince ouverte, pas de fermeture)
TOUCH_Z_ABOVE_CENTER = 0.03       # juste au-dessus du centre (~dessus carton)
TOUCH_Z_ABOVE_BOX = 0.06
TOUCH_DWELL = 1.0
TOUCH_USE_FORWARD = False
TOUCH_ARM_SETTLE = 1.5
TOUCH_APPROACH_Z = 0.14           # approche encore haute, descente ensuite
TOUCH_TABLE_Z_MIN = -0.12
TOUCH_TABLE_Z_MAX = 0.02
TOUCH_MIN_X = 0.22
TOUCH_MAX_X = 0.52
TOUCH_LANDMARKS = False
TOUCH_MAX_PARCELS = 1             # 1er run : parcel_1 seulement
TOUCH_Y_OFFSET = 0.0              # pas de biais y (avant RIGHT_GRASP_Y_OFFSET=-0.02)

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
# Ordre demandé : colis 1→4, puis bac, puis zone de pesée (truck)
TOUCH_ORDER = [
    "parcel_1", "parcel_2", "parcel_3", "parcel_4",
    "sorting_box",
    "weighing_area",
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
TABLE_X_RANGE = (0.18, 0.58)     # seed30 : x≈0.25–0.48 (IK fail sous 0.18)
TABLE_Y_RANGE = (-0.42, 0.00)     # zone table seed30
PARCEL_Z_RANGE = (-0.15, 0.12)   # rejette amas flottants z≈0.22
PARCEL_RGB_Z_RANGE = (-0.12, 0.05)
TABLE_PARCEL_Z = -0.04
MIN_PICK_IK_X = 0.22               # sous ça : IK / workspace KO

# Contrat tête = ZONE stable (LiDAR←couleur UV).
# Labo seeds 0–9 : grid OFF → err_structure bloqué à 0.12 m (jaune sur colonne droite).
# grid-x ON + Δ max 18 cm : corrige association gauche ; row-lift reste OFF (bras/touch).
FUSE_ENABLE_GRID_SNAP = True
FUSE_ENABLE_ROW_LIFT = False
FUSE_MAX_RESHAPE_XY = 0.18         # jaune mal collé ~15 cm → 4 cm skippait toujours le snap
MAX_COLOR_AREA_RATIO = 0.18      # rejette masque > 18 % ROI (faux bleu ciel)
COLOR_ROI_V_START = 0.08         # pitch=20 : colis dès ~v=0.60 (plus haut qu'avec look_down)
COLOR_TABLE_BAND_V0 = 0.10       # fraction haute du ROI ignorée
COLOR_SOFT_DMIN = 65.0           # si refs trop loin → élargir recherche
COLOR_SOFT_TABLE_THR = 18.0      # seuil table assoupli en mode soft
COLOR_FULLROI_DMIN = 72.0        # si encore trop loin → chercher tout le ROI
COLOR_SOFT_THR_BOOST = 22.0      # +tolérance LAB en mode soft / full-ROI

# Clustering : regrouper les points LiDAR proches en "amas" (1 amas ≈ 1 colis)
CLUSTER_EPS_XY = 0.055           # plus serré → moins fusionner 2 colis voisins
MIN_CLUSTER_POINTS = 3             # peu de retours LiDAR par colis
MIN_PARCEL_SIZE_XY = 0.008         # filtre bruit très petit
MAX_PARCEL_SIZE_XY = 0.14          # au-dessus → découpe grille 2×2
LIDAR_GRID_X_MID = 0.381           # séparation parcel_1/2 vs 3/4 (IK x)
LIDAR_GRID_Y_MID = -0.214          # séparation parcel_1/3 vs 2/4 (IK y)
LIDAR_SEED_RADIUS = 0.14           # un peu plus large pour parcel_2 (jaune, peu de pts)
LIDAR_SEED_MIN_POINTS = 2          # grid_2 (jaune) souvent < 3 pts
MIN_COLOR_PIXELS = 80              # colis pâles / lointains (avant 120 trop strict)
MIN_COLOR_PIXELS_BY_NAME = {
    "parcel_2": 45,                # jaune : souvent peu de pixels LAB/HSV
    "parcel_1": 60,
    "parcel_3": 40,                # FOCUS orange : plus facile à voir
    "parcel_4": 90,
}
LAB_COLOR_DIST_BY_NAME = {
    "parcel_2": 78.0,              # jaune pâle (sim souvent saturée basse)
    "parcel_1": 72.0,
    "parcel_3": 85.0,              # FOCUS orange : plus tolérant
    "parcel_4": 66.0,              # bleu : 58 trop strict → colorish=3 (seeds 1/4/7)
}
MAX_FUSE_UV_DIST = 0.28            # pitch=20 : proj. relative + marge

# UV ↔ table IK — recalibré tête orga pitch=20 (COLOR u/v logs ↔ GT_ik seed2)
# Convention image : v↑ = rangée proche (x↑) ; u↑ = côté droit (y↑)
HEAD20_UV_U_LEFT = 0.28            # u min (gauche) sur plage relative
HEAD20_UV_U_SPAN = 0.50            # largeur u relative
HEAD20_UV_V_FAR = 0.58             # v min (rangée loin, x petit)
HEAD20_UV_V_SPAN = 0.40            # largeur v relative
# Secours absolu (1 amas / pas de peers) : x=a+b*v , y=c+d*u
HEAD20_X_FROM_V = (-0.01, 0.48)
HEAD20_Y_FROM_U = (-0.52, 0.55)
PERCEPTION_ATTEMPTS = 8            # validation multi-seed (assez, pas 20)
PERCEPTION_FORCE_ALL_ATTEMPTS = False  # stop tôt si géométrie+couleur OK
PERCEPTION_ERR_TARGET = 0.05       # N2 : err_structure_2x2 < 5 cm (anti-seed0)
LANDMARK_ERR_TARGET = 0.05         # N1b : balance + bac err_xy < 5 cm
SPATIAL_NAME_MAX_ERR = 0.12        # nommage LiDAR → parcel le plus proche (m)
MAX_BLUE_MASK_AREA = 25000         # blob bleu max après filtre profondeur
DEPTH_Z_MIN = 0.35                 # profondeur valide table (m)
DEPTH_Z_MAX = 1.2
# Profondeur typique des colis sur table (tête pitch=20) — coupe le ciel
COLOR_DEPTH_Z_MIN = 0.40
COLOR_DEPTH_Z_MAX = 0.95
RGB_DEPTH_XY_TOL = 0.10            # depth vs ray-plane : max Δxy pour accepter depth
LIDAR_O3D_PLANE_DIST = 0.015       # RANSAC distance au plan table (m)
LIDAR_O3D_ABOVE_MAX = 0.07         # points colis au-dessus du plan (m)
LIDAR_O3D_EPS = 0.045              # DBSCAN 3D plus serré (sépare colis ~15 cm)
LIDAR_O3D_MIN_POINTS = 3
# Biais systématique LiDAR→base_link (fix_rowy seeds 0/1/3/6 : dx≈+5.5 cm).
# Corrige l'ancre colonne droite avant grid-x (sinon toute la grille décale).
LIDAR_IK_CORR = (-0.050, 0.0, 0.0)
HUNGARIAN_BIG_COST = 1e4

# Couleurs nominales scene1.yaml (RGBA → BGR OpenCV) — distance LAB
TABLE_REF_BGR = (26, 77, 26)       # table_top 0.10, 0.30, 0.10
LAB_COLOR_DIST_MAX = 55.0          # un peu plus tolérant (éclairage sim)
TABLE_LAB_DIST_MAX = 36.0          # exclusion table (ne pas manger les colis pâles)
# pitch=20 : table + colis vers le bas/milieu (v≈0.60–0.99)
COLOR_V_NORM_MAX = 0.995
COLOR_V_NORM_MAX_BY_NAME = {}
COLOR_MAX_AREA_BY_NAME = {
    "parcel_2": 20000,
    "parcel_4": 25000,
}
# ROI image normalisée (u dans le ROI couleur)
COLOR_U_RANGE_BY_NAME = {
    "parcel_4": (0.30, 0.95),
    "parcel_2": (0.05, 0.75),
}
# Prefere blobs bas dans l'image (table) pour le bleu (évite ciel)
COLOR_PREFER_LOW_V = {"parcel_4"}
BASE_VALID_X = (0.18, 0.58)        # base_link IK valide pour rgb-depth
BASE_VALID_Y = (-0.48, -0.02)
BASE_VALID_Z = (-0.18, 0.08)
# Secours HSV si LAB insuffisant (scene1 colis pâles — S bas)
PARCEL_HSV_FALLBACK = [
    ("parcel_2", "yellow", (8, 20, 60), (55, 255, 255)),
    ("parcel_1", "brown", (4, 15, 70), (40, 200, 255)),
    ("parcel_3", "orange", (2, 30, 70), (30, 255, 255)),
    ("parcel_4", "blue", (90, 10, 120), (135, 130, 255)),
]
PARCEL_REF_COLORS = [
    ("parcel_2", "yellow", (97, 199, 235)),
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
RIGHT_HANDOFF_TRANSIT_FALLBACK_ZS = [0.37, 0.35]  # orga: si 0.40 IK fail
# Gauche : vraie symétrie de RIGHT_HANDOFF (y_L = -y_R) + offsets ox/oz
# RIGHT y=-0.044645 → LEFT y=+0.044645 ; xz ready = +0.10 en Y avant serrage
LEFT_HANDOFF_RECEIVE_XZ_READY_IK = [0.266, 0.145, 0.2816983]  # pré-approche (y_L + 0.10)
LEFT_HANDOFF_RECEIVE_IK = [0.266, 0.044645, 0.2816983]  # miroir de RIGHT (y = -y_R)
RIGHT_HANDOFF_RETRACT_Y = -0.30    # recul de la main droite après passation
BOX_DROP_BASE_IK = [0.58, 0.24, 0.556217]   # xy plus centré sur ouverture bac
BOX_DROP_HOVER_Z = 0.66            # au-dessus, assez bas pour viser le trou (sans toucher)
LEFT_BOX_TRANSIT_Z = 0.42          # lever gauche après handoff avant trajet bac
BOX_DROP_IK_X_FALLBACK_DELTAS = [-0.04, 0.0, 0.04, -0.08]

# Chaque colis a un petit décalage dans le bac pour ne pas se superposer (grille 2×2)
BOX_DROP_OFFSET_BY_PARCEL = {
    "parcel_1": [0.0, 0.0, 0.0],
    "parcel_2": [0.0, 0.02, 0.0],
    "parcel_3": [0.0, 0.0, 0.0],   # FOCUS: centre bac (pas décalé hors trou)
    "parcel_4": [0.02, 0.02, 0.0],
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
LIFT_Z_OFFSET = 0.30                # lever haut après saisie (orga) — évite la table
PLACE_APPROACH_Z = 0.06            # pesée seulement — bac = hover drop
ARM_SETTLE_TIME = 1.5
GRIPPER_SETTLE_TIME = 0.4           # pause après ouverture/fermeture pince
WEIGH_RELEASE_SETTLE = 1.5          # stabilisation avant ouverture sur balance
WEIGH_DWELL = 2.5                   # laisser le temps à la zone de devenir jaune
WEIGH_RELEASE_TOL_XY = 0.04         # m — sinon colis hors pad → pas de jaune
WEIGH_RELEASE_TOL_Z = 0.06
WEIGH_RELEASE_PLACE_RETRIES = 3
PLACE_DWELL = 1.0                   # laisser le colis tomber dans le bac
MAX_PARCELS = 4
MAX_MISSION_FAILURES = 8
FORCE_PARCEL_NAME = None            # None = les 4 colis (brun→jaune→orange→bleu selon score)

# Pipeline : prise → pesée → handoff → bac (mode validé)
TRAIN_PICK_BOX = False
SKIP_WEIGH = False
# False = orga : droite pèse → passation → gauche dépose bac.
SKIP_HANDOFF = False
# Cibles lab (unused si SKIP_HANDOFF=False)
RIGHT_BOX_DROP_BASE_IK = [0.52, 0.12, 0.556217]
RIGHT_BOX_DROP_HOVER_Z = 0.62
RIGHT_BOX_DROP_TRANSIT_Z = 0.40
RIGHT_BOX_DROP_IK_Y_TRIES = [0.0, 0.04, -0.04, 0.08, 0.12]
RIGHT_BOX_DROP_IK_X_TRIES = [0.0, -0.04, 0.04, -0.08, 0.06]
# Pause coïncidence : gauches fermée + angles alignés, avant ouverture droite
HANDOFF_COINCIDENCE_SETTLE = 0.8
# Double hold SEULEMENT après Grabbed gauche confirmé — sinon chute
HANDOFF_STANCE_SETTLE = 0.8          # stance avant approche gauche (évite chute)
HANDOFF_LEFT_MOVE_SETTLE = 1.5       # plus lent que pick — bras G bouge, D tient colis
HANDOFF_LEFT_Y_MID_STEP = True       # approche Y en 2 temps (xz → mid-y → receive)
HANDOFF_BOTH_HOLD_BEFORE_OPEN_R = 1.0
HANDOFF_LEFT_HOLD_TIMEOUT = 2.5     # attendre L=Grabbed après close
HANDOFF_LEFT_GRAB_RETRIES = 3       # retries approche si gauche rate
# Ne JAMAIS ouvrir droite si gauche n'a pas Grabbed
HANDOFF_REQUIRE_LEFT_HOLD = True

# Handoff = vraie symétrie sagittale D→G : y_L = -y_R, x/z ≈ D (+ ox/oz), quat miroir
# Caméra poignet G = correction légère seulement après le miroir
HANDOFF_USE_WRIST_MIRROR = True
HANDOFF_MIRROR_X_OFFSET = 0.02       # left_x = right_x + offset
HANDOFF_MIRROR_Z_OFFSET = -0.02      # left_z = right_z + offset
HANDOFF_MIRROR_Y_PRE_APPROACH = 0.10 # XZ ready : +Y avant receive (même x/z)
HANDOFF_LEFT_LIGHT_SERVO = True      # cam poignet G = petite correction seulement
HANDOFF_LEFT_LIGHT_SERVO_ITERS = 2
HANDOFF_LEFT_CORRECT_DPIX = 70.0     # servo seulement si Δpx > seuil
# False = close G après miroir même si pas parfait (cam corrige si besoin)
HANDOFF_LEFT_REQUIRE_VISION = False
HANDOFF_LEFT_REQUIRE_SEEN_FOR_CLOSE = True   # pas de close G si colis non vu
HANDOFF_LEFT_CLOSE_MAX_DPIX = 120.0          # Δpx max avant close gauche
HANDOFF_LEFT_GRABBED_HITS = 3                # lectures L=GRABBED (state=3) consécutives
# D DOIT être à la pose handoff habituelle avant que G bouge (évite chute)
HANDOFF_RIGHT_VERIFY = True
HANDOFF_RIGHT_TOL_XY = 0.035         # m
HANDOFF_RIGHT_TOL_Z = 0.05           # m
HANDOFF_RIGHT_PLACE_RETRIES = 3
# False = miroir depuis FK réelle de D (vraie symétrie sur pose actuelle)
HANDOFF_MIRROR_FROM_PRESET = False

# Vérif saisie (LiDAR/RGB après lift) — Phase 1B : abort si encore sur table.
# /mujoco/qpos est INTERDIT (anti-triche).
GRASP_VERIFY_ENABLED = True
GRASP_VERIFY_ABORT_ON_EMPTY = True
GRASP_VERIFY_STILL_ON_TABLE_XY = 0.12   # encore près du pick = vide (strict)
GRASP_VERIFY_TABLE_Z_MAX = 0.12


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
RIGHT_PICK_QUAT = _quat_from_ypr_deg([0, -90, 0.0], [90.0, 0.0, 0.0])  # prise // axes table
# Grippage perpendiculaire (colis carré tourné ~90°) — yaw verrouillé avant plongée
RIGHT_PICK_QUAT_AXIS90 = _quat_from_ypr_deg([0, -90, 0.0], [90.0, 0.0, 90.0])
RIGHT_WEIGH_RELEASE_QUAT = _quat_from_ypr_deg([0, -100, 0.0], [90.0, 0.0, 0.0])
RIGHT_WEIGH_REGRASP_QUAT = _quat_from_ypr_deg([0, -60, 0.0], [90.0, 0.0, 0.0])
RIGHT_HANDOFF_QUAT = _quat_from_ypr_deg([-0.839, -100.0, 0.0], [90.0, -20.0, 90.0])
# Miroir de la droite : same first YPR, second = [-yaw, pitch, -roll]
LEFT_HANDOFF_RECEIVE_QUAT = _quat_from_ypr_deg([-0.839, -100.0, 0.0], [-90.0, -20.0, -90.0])
LEFT_BOX_DROP_QUAT = _quat_from_ypr_deg([-0.328, -100.935, 0.0], [-90.0, 0.0, 0.369])

# Offset tip — Z plus bas (souvent pince juste au-dessus → vide pos≈89%)
# Orga: pick_z=-0.03 / near -0.05 ; on vise un cran plus bas pour mordre le carton.
RIGHT_CLAW_TIP_OFFSET = [0.02, 0.01, -0.010]
RIGHT_PICK_IK_Z = -0.04
RIGHT_PICK_TRANSIT_IK_Z = 0.22
RIGHT_PICK_NEAR_FAR_Y_THRESHOLD = -0.20
RIGHT_PICK_OFFSET_FAR_ROW = [-0.03, 0.0, 0.0]       # orga far
RIGHT_PICK_OFFSET_NEAR_ROW = [-0.03, 0.02, -0.02]   # orga near (z plus bas)
RIGHT_PICK_OFFSET_BY_PARCEL = {}
# Si 1er close = vide (REACHED ≥85%) → recovery plonge encore plus bas
GRASP_EMPTY_DEEPER_Z = 0.025
GRASP_PICK_Z_MIN = -0.065          # plancher sécurité (ne pas rentrer dans table)
RIGHT_PICK_YZ_ALIGN_SAFE_IK_X = 0.184
# Modes IK orga (pas de triche — paramètres de solve)
IK_MODE_POS_HARD_ORI_SOFT = 0x02
IK_MODE_POS_HARD_ORI_HARD = 0x03
IK_MODE_THREE_POINT_MIXED = 0x06
RIGHT_PICK_IK_MODE = IK_MODE_THREE_POINT_MIXED       # 0x06 approche
RIGHT_GRASP_FINAL_IK_MODE = IK_MODE_POS_HARD_ORI_HARD  # 0x03 descente
IK_MAJOR_ITERATIONS = 500
PICK_ALIGN_MOVE_SLEEP = 1.2   # orga PICK_ALIGN_MOVE_TIME
PICK_GRASP_MOVE_SLEEP = 1.4   # orga PICK_GRASP_MOVE_TIME
CARTESIAN_LIFT_POINTS = 4
CARTESIAN_LIFT_SEG_SLEEP = 0.35
GRIPPER_CLOSE_HOLD = 0.7      # orga GRIPPER_CLOSE_HOLD_TIME

# Levée "croix" orga PRESET_POINTS_DEG (5 waypoints) — évite de raser la table
ARM_RAISE_PRESET_DEG = [
    [20, 0, 0, -30, 0, 0, 0, 20, 0, 0, -30, 0, 0, 0],
    [20, 90, 0, -55, 0, 0, 0, 20, -90, 0, -55, 0, 0, 0],
    [20, 60, 0, -75, 0, 0, 0, 20, -60, 0, -75, 0, 0, 0],
    [29.89, 30.67, 29.889, -139.1, -59.33, 0, 0,
     29.89, -30.67, -29.889, -139.1, 59.33, 0, 0],
    [29.89, 10.67, 9.889, -139.1, -59.33, 0, 0,
     29.89, -10.67, -9.889, -139.1, 59.33, 0, 0],
]
ARM_RAISE_STEP_SLEEP = 1.2
ARM_CLEAR_TABLE_Z = 0.28   # hauteur mini avant tout déplacement horizontal

# =============================================================================
# VISION MAIN (priorité) — tête = zone ; main = peaufinage temps réel
# Pince (Grabbed) = après, une fois la vision stable
# =============================================================================
WRIST_ALLOW_RAY = True
WRIST_DEPTH_Z_MIN = 0.08
WRIST_DEPTH_Z_MAX = 0.55
WRIST_MIN_PIXELS = 50
# Log saturé area≈260k → faux centroid ; viser blobs colis (~15–80k)
WRIST_MAX_BLOB_FRAC = 0.16
WRIST_MASK_SAT_FRAC = 0.20         # si masque > 20% image → resserrer LAB
WRIST_ROI_FRAC = 0.85
WRIST_MAX_DELTA_XY = 0.02
WRIST_SERVO_GAIN = 0.45
WRIST_SERVO_ITERS = 4
WRIST_SERVO_SIGN_X = 1.0
WRIST_SERVO_SIGN_Y = -1.0
WRIST_CENTER_BIAS = 0.85
WRIST_ACCEPT_PX = 90.0
# Sweet spot run CENTRÉ Δpx=49 area≈40k
WRIST_LAB_BOOST = 8.0
WRIST_LAB_SOFT_EXTRA = 8.0
WRIST_USE_TABLE_EXCLUDE = True
WRIST_TABLE_EXCLUDE_SCALE = 1.15   # plus strict que tête (évite table « brun »)
WRIST_DEPTH_SOFT = True
WRIST_HSV_ONLY_IF_SPARSE = True    # HSV n'élargit pas un masque déjà plein
WRIST_SETTLE = 0.35
WRIST_YAW_ENABLE = False            # OFF par défaut (instable mid-grasp)
WRIST_YAW_AUTO = True               # active yaw seulement si blob confiant
WRIST_YAW_MIN_AREA = 8000
WRIST_YAW_MAX_ASPECT = 1.55         # trop allongé → pas un carré net
WRIST_YAW_MAX_DEG = 90.0
WRIST_YAW_SNAP_SQUARE = True
WRIST_REQUIRE_SEE_BEFORE_CLOSE = True
WRIST_CLOSE_MAX_PIXEL_FRAC = 0.10
WRIST_LOCK_MIN_AREA = 3000
WRIST_UNDER_HAND_AREA = 12000
WRIST_UNDER_HAND_FRAC = 0.12
WRIST_UNDER_HAND_MAX_DPIX = 120
WRIST_POST_PLUNGE_MAX_DPIX = 170   # post-plongée Δpx monte (parallaxe tip)
WRIST_AIM_BIAS_U = 40.0
WRIST_AIM_BIAS_V = 70.0
WRIST_APPROACH_NUDGE = True
WRIST_APPROACH_NUDGE_MAX = 0.02
WRIST_SHALLOW_PLUNGE = 0.05
WRIST_CLOSE_EVEN_IF_IK_FAIL = True
WRIST_MID_DESCEND = False
WRIST_VISION_ONLY_GATE = True
WRIST_SKIP_CLAW_HOLD_CHECK = False

# LAB boost par colis (poignet plus proche → seuils dédiés, bleu/jaune sensibles)
WRIST_LAB_BOOST_BY_NAME = {
    "parcel_1": 8.0,   # brun
    "parcel_2": 12.0,  # jaune
    "parcel_3": 10.0,  # orange
    "parcel_4": 14.0,  # bleu
}

# Tip vs blob (milieu / bord) — erosion = « cœur » de la face
WRIST_CORE_ERODE = 7                # pixels (impair)
WRIST_TIP_IN_CORE_REQUIRED = True   # CENTRÉ seulement si tip dans le cœur
WRIST_USE_DEPTH_3D = True           # refine XYZ via depth médian du blob
WRIST_DEPTH_3D_MAX_DELTA = 0.04     # clamp m vs pose tête
WRIST_DEPTH_Z_GRASP_MIN = 0.06
WRIST_DEPTH_Z_GRASP_MAX = 0.45
HANDOFF_LEFT_WRIST_CHECK = True     # log observe avant close
# False = miroir poignet D + light correct ; True = ancien servo long
HANDOFF_LEFT_WRIST_SERVO = False
HANDOFF_LEFT_SERVO_ITERS = 4
HANDOFF_LEFT_MAX_DELTA_XY = 0.02
HANDOFF_LEFT_SERVO_SIGN_X = 1.0
HANDOFF_LEFT_SERVO_SIGN_Y = -1.0   # calibrer si servo inverse

# --- Logs JSONL (option A) ---
# Désactiver : SCENE1_WRIST_LOG=0  |  chemin : SCENE1_WRIST_LOG=/tmp/foo.jsonl
WRIST_LOG_ENABLED = True
WRIST_LOG_PATH = ""  # vide = ~/scene1_wrist.jsonl (ou $SCENE1_WRIST_LOG)

# --- Qualité de prise (milieu vs bord vs vide) ---
# pos% après close : vide≈85–95 ; bonne épaisseur colis≈35–75 ; bord/corner souvent hors
GRASP_HOLD_POS_EMPTY_MIN = 85.0     # ≥ → air / fermé à fond
GRASP_HOLD_POS_GOOD_MIN = 30.0
GRASP_HOLD_POS_GOOD_MAX = 78.0
GRASP_AIM_EDGE_DPIX = 110.0         # tip loin du centroid → vise probablement le bord
GRASP_AIM_CENTER_DPIX = 90.0        # aligné WRIST_ACCEPT_PX

# --- Recovery close + persistance colis ---
GRASP_RECOVERY_MAX = 2              # retries sur place (ouvrir↑wrist↓close)
GRASP_PARCEL_MAX_FAILS = 5          # fails mission avant de changer de colis
# Si Grabbed confirmé → NE PAS rouvrir pour recovery (edge_hold / manner)
GRASP_KEEP_IF_HOLDING = True
GRASP_MAINTAIN_CLOSE = True         # re-close après détection prise (maintient serrage)
# Sonde pression : close → pause → re-serre → lire state/pos/effort (sait s'il a pris)
GRASP_SQUEEZE_PROBE = True
GRASP_SQUEEZE_PULSES = 2            # nb de re-serrages après 1er close
GRASP_SQUEEZE_PAUSE = 0.35          # s entre pulses (laisse l'effort / Grabbed se publier)
GRASP_SQUEEZE_EFFORT_MIN = 0.5      # |effort| min pour confirmer contact (sim souvent ~5)
# Si 1ère prise EXCELLENTE (vision lock + Grabbed + épaisseur good) → lock définitif
GRASP_LOCK_EXCELLENT_FIRST = True
# 1ère prise excellente (vision lock + Grabbed + good) → jamais rouvrir
GRASP_EXCELLENT_LATCH = True

