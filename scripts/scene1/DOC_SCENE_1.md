# DOC SCENE 1 — Vision & saisie (équipe)

**Auteur vision :** Williams  
**Dernière MAJ :** 15 juillet 2026  
**Rôle :** détection tête + refine caméra main ; grasp / weigh / handoff sur structure orga.

**中文版：** [`DOC_SCENE_1_zh.md`](./DOC_SCENE_1_zh.md)  
**Dépôt :** https://github.com/williams73-oss/leju-competiton

---

## 0. Quelqu’un peut-il cloner et lancer ?

**Oui sous conditions.**  
Il faut l’env simu officiel + installer ce dépôt comme `challenge_cup_task_template` (§1).  
Ce dépôt **n’est pas** toute la simu.

- **Ce qui marche :** après §1, `rosrun … --scene scene1` lance la mission (detect → grasp → weigh → handoff → bac). Code + `robot_api` étendu présents.
- **Pas garanti :** 4/4 sur tous les seeds. Meilleur cas documenté : seed **30**. Il faut OpenCV + services simu (pince / IK / FK).
- **Ne marche pas seul :** clone sans Docker / `kuavo_ws` orga.

---

## 1. De zéro (clone → install → run)

### 1.1 Prérequis

- Image / workspace officiel `kuavo_challenge_cup_2026` (comme un participant)
- Savoir lancer la simu + `challenge_task.py` officiel
- GPU / display comme l’orga
- **Dépendances Python (vision) :** dans le conteneur / env simu :
  ```bash
  pip3 install opencv-python open3d
  # ou le script monorepo docker/install_perception_deps.sh (pas dans ce dépôt GitHub)
  ```
  Sans `opencv-python` → couleur / wrist faibles ou skippés → mission peu crédible.  
  Sans `open3d` → fallback LiDAR 2D (tourne, un peu moins bien).

### 1.2 Cloner

```bash
cd /tmp
git clone https://github.com/williams73-oss/leju-competiton.git
```

Contenu utile :

| Chemin | Rôle |
|--------|------|
| `scripts/challenge_task.py` | Entrée 3 scènes |
| `scripts/scene1_task.py` | Entrée Scene1 |
| `scripts/scene1/` | perception / actions / config |
| `src/robot_api.py` | bras / pince (+ FK, IK 1 main, hold…) |
| `src/perception_api.py` | cams + LiDAR + TF |

### 1.3 Installer dans le workspace

**A — Remplacer le paquet (recommandé)**

```bash
WS=~/leju-kuavo-challenge-cup-2026   # ou /root/kuavo_ws dans le conteneur
PKG=$WS/src/challenge_cup_task_template

mv "$PKG" "${PKG}.bak_$(date +%Y%m%d)"
cp -a /tmp/leju-competiton "$PKG"
```

**B — Overlay Scene1 seulement**

```bash
PKG=~/leju-kuavo-challenge-cup-2026/src/challenge_cup_task_template
SRC=/tmp/leju-competiton

cp -a "$SRC/scripts/scene1" "$PKG/scripts/"
cp -a "$SRC/scripts/scene1_task.py" "$PKG/scripts/"
cp -a "$SRC/src/robot_api.py" "$PKG/src/"
cp -a "$SRC/src/perception_api.py" "$PKG/src/"
```

Rebuilder le paquet si besoin (`catkin build challenge_cup_task_template`).

### 1.4 Modes (`scripts/scene1/config.py`)

| Réglage | Effet |
|---------|--------|
| `PERCEPTION_ONLY = True` | Detect only, pas de bras |
| `TOUCH_TEST = True` | Detect puis toucher |
| **les deux `False`** | **Mission complète** (défaut équipe) |

Défaut actuel : mission (`False` / `False`), `FORCE_PARCEL_NAME = None` (4 colis).

### 1.5 Lancer

Dans l’env sourcé (conteneur ou host) :

```bash
source /root/kuavo_ws/devel/setup.zsh   # adapter le chemin
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 30
```

Changer `--seed` selon besoin (`0`, `30`, `400`…).

### 1.6 Logs utiles

```text
DETECT | COLOR | FUSE | WRIST | VISION | Grabbed | DONE | claw
```

Succès type : `VISION OK`, `claw R=3` Grabbed, puis pesée / handoff.

### 1.7 Optionnel — scripts Docker monorepo

Si tu as le gros repo `leju-kuavo-challenge-cup-2026` **avec** `docker/run_scene1_*.sh` :

```bash
cd ~/leju-kuavo-challenge-cup-2026
bash docker/stop_scene1.sh
bash docker/run_scene1_mission.sh 30 900
```

Ces scripts **ne sont pas** dans `leju-competiton` sur GitHub. Sans eux → **`rosrun`** (§1.5).

`Ctrl+C` n’arrête pas forcément Docker → `stop_scene1.sh` si dispo.

**Ne pas pusher :** `labo/scene1/**` (csv, logs, images), gros CSV runtime.

---

## 2. Objectif

Pipeline orga : 4 colis → droite → peser → regrasp → gauche → bac.

**Anti-triche :** jamais piloter le bras avec `/mujoco/qpos` / GT.  
`GT_COMPARE = False` en mission.

---

## 3. Fichiers

| Fichier | Rôle |
|---------|------|
| `perception.py` | Tête : LiDAR + RGB LAB/HSV + depth → 4 colis |
| `wrist_vision.py` | Main : blob + petit Δxy |
| `config.py` | Modes, seuils, WRIST_*, FORCE_PARCEL |
| `actions.py` | Grasp / weigh / handoff |
| `../scene1_task.py` | Entrée |
| `../../src/perception_api.py` | Cams + LiDAR + TF |
| `../../src/robot_api.py` | Bras / pince (+ API étendue) |

---

## 4. Architecture

```
HEAD detect → approche → HAND refine → GRASP close
```

---

## 5. Validé

| Seed / colis | Résultat |
|--------------|----------|
| Seed **30**, `parcel_1` | VISION OK + Grabbed, pesée + handoff OK |
| Seed **0** tête | 4/4, `err_structure_2x2 ≈ 0.009` m |
| Seed 0, bleu / jaune | Approche OK ; grip / tip parfois à peaufiner |

---

## 6. Pièges & fixes

| Problème | Fix |
|----------|-----|
| Plongée trop profonde → IK fail | Tip shallow (`RIGHT_PICK_IK_Z`, tip offset) |
| Faux « holding » en MOVING | `right_holding()` ignore MOVING |
| Servo wrist trop fort | `WRIST_MAX_DELTA_XY = 0.02`, 4 iters |
| Yaw mid-grasp | `WRIST_YAW_ENABLE = False` |
| Ancien `robot_api` | Remplacer par celui de ce dépôt |

---

## 7. Config (aperçu)

```python
PERCEPTION_ONLY = False
TOUCH_TEST = False
GT_COMPARE = False
FORCE_PARCEL_NAME = None

RIGHT_CLAW_TIP_OFFSET = [0.02, 0.01, -0.005]
RIGHT_PICK_IK_Z = -0.005
WRIST_SERVO_ITERS = 4
WRIST_MAX_DELTA_XY = 0.02
WRIST_YAW_ENABLE = False
WRIST_REQUIRE_SEE_BEFORE_CLOSE = True
WRIST_VISION_ONLY_GATE = True
```

| Name | Couleur |
|------|---------|
| `parcel_1` | brun / gris |
| `parcel_2` | jaune |
| `parcel_3` | orange |
| `parcel_4` | bleu |

---

## 8. `robot_api` — APIs requises par `actions.py`

Déjà dans `src/robot_api.py` de ce dépôt :

| Méthode / import | Rôle |
|------------------|------|
| `_read_arm_joints_rad()` | 14 joints bras (capteurs) |
| `call_fk()` | FK |
| `solve_ik_one_hand()` | IK une main |
| `_last_cmd_deg` | Dernière commande articulaire |
| `right_holding()` / `describe_right()` | État pince droite |
| `fkSrv`, `sensorsData` | Types ROS |

Sans ce fichier → crash au grasp / handoff.

---

## 9. Couches perception

1. LiDAR — OÙ  
2. RGB tête LAB/HSV — QUI  
3. Depth tête  
4. Fusion + grille 2×2  
5. Caméra poignet — refine avant close  

Critères labo tête : `named == 4/4`, `err_structure_2x2 < 0.05` m, `colorish == 4`.

---

## 10. Liens

- Dépôt équipe : https://github.com/williams73-oss/leju-competiton  
- Réf. orga : `collect_scene1_handoff_dataset.py`
