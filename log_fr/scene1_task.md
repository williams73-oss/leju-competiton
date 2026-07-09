# scene1_task.py — traduction FR des tests

| Log chinois | Français | Résultat attendu |
|-------------|----------|------------------|
| `场景一：快递称重与摆放 — 任务开始` | Début tests scène 1 | — |
| `[STEP 1] 切换手臂到外部控制模式` | Bras en mode externe (obligatoire) | OK si pas d'erreur timeout bras |
| `[TEST 1/5] 头部左右看` | Test tête gauche / droite / avant | Robot regarde autour |
| `[TEST 2/5] 读取头部 RGB 图像` | Test caméra RGB tête | |
| `头部 RGB 尺寸: W × H` | Résolution image | **OK** si chiffres affichés |
| `未获取到 RGB，请检查 opencv-python 是否安装` | **ÉCHEC** — installer opencv dans Docker | |
| `[TEST 3/5] 读取激光雷达点云` | Test lidar | |
| `点云点数: N，范围 x:[...] y:[...]` | N points lidar | **OK** — principal pour trouver colis |
| `未获取到点云` | **ÉCHEC** lidar | |
| `[TEST 4/5] 读取关节角度` | Test capteurs articulations | |
| `双臂关节角度(度): 左臂 [...]` | Angles bras gauche | **OK** |
| `未获取到传感器数据` | **ÉCHEC** capteurs | |
| `[TEST 5/5] 查询场景物体位置` | Test TF sur colis | |
| `parcel_X: 查询失败（物体可能不在 TF 树中）` | **Normal en compétition** — colis pas dans TF | Utiliser lidar/caméra |
| `场景一：任务结束` | Fin tests | Succès si TEST 2–4 OK |

## Checklist rapide

- [ ] TEST 2 : dimensions RGB
- [ ] TEST 3 : > 0 points lidar
- [ ] TEST 4 : angles bras
- [ ] TEST 5 : échec TF = **normal**
