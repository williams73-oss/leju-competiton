# Simulateur — bruit terminal (hors `>>>`)

Messages que tu ne contrôles pas. Cherche surtout les lignes **`>>>`**.

| Message | Français | Action |
|---------|----------|--------|
| `HumanoidInterfaceDrake INFO` | Init modèle robot Drake | Ignorer (bruit) |
| `MPC node is ready` | Contrôleur marche prêt | Bon signe |
| `Start spinning now` | MPC tourne | Bon signe |
| `[mpc] arm control mode changed, but target not updated` | Mode bras externe sans nouvelle consigne | Warning normal après STEP 1 |
| `libdrake.so: cannot open shared object` | **Erreur** bibliothèque Drake | Relancer conteneur / vérifier image Docker |
| `humanoid_sqp_mpc has died` | **Crash** contrôleur | Sim morte — relancer propre |
| `terminate called without an active exception` | Node C++ arrêté brutalement | Souvent bénin si sim continue |
| `MuJoCo version 3.0.1` | MuJoCo chargé | OK |
| `Left gripper actuator found` | Pinces sim détectées dans MuJoCo | OK |
| `sim_leju_claw_interface: ... ready` | **Service pince prêt** | Doit apparaître avant `夹爪服务已连接` |
| `等待仿真就绪超时 (120s)` | Sim lente mais continue parfois | Attendre ou relancer |
| `场景初始化服务不可用` | **FATAL** — colis pas initialisés | Sim pas saine — relancer Docker |
| `new node registered with same name` | **Double lancement** | Ctrl+C + `killall roslaunch` |

## Ordre normal d'un bon lancement

1. Beaucoup de logs Drake / MPC (1–2 min)
2. `初始化 parcel_1` … `parcel_4 完成`
3. `反作弊监控已启动` + timer
4. `>>>` lignes de **challenge_task**
5. `>>>` lignes **TEST 1/5** … **5/5**
6. `场景一：任务结束`
