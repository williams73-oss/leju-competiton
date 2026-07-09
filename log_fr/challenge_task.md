# challenge_task.py — traduction FR des logs `>>>`

Lignes dans l'ordre d'apparition.

| Log chinois (terminal) | Français |
|------------------------|----------|
| `=== 场景一：包裹称重与摆放任务启动 ===` | Démarrage mission scène 1 (colis, balance, tri) |
| `等待 ROS 节点就绪（10 秒）…` | Pause 10 s — laisser démarrer tous les nodes sim (dont pince) |
| `初始化 RobotMover...` | Création contrôleur marche |
| `初始化 ArmController...` | Création contrôleur bras |
| `等待夹爪服务 /control_robot_leju_claw（最多 120 秒）...` | Attente service pince (max 2 min) |
| `仍在等待夹爪服务… X 秒` | Toujours en attente… X secondes |
| `夹爪服务已连接。` | Pince OK |
| `夹爪服务超时。诊断: rosnode list \| grep sim_leju` | Pince absente — voir diagnostic |
| `请 Ctrl+C 后 killall roslaunch，再单独启动一次 rosrun` | Relance propre : un seul rosrun |
| `警告：场景一传感器测试继续（无夹爪）` | Scène 1 continue sans pince (tests capteurs OK) |
| `警告：夹爪不可用，open() 已跳过` | Stub pince — pas de vraie pince |
| `初始化 HeadController...` | Création contrôleur tête |
| `场景实例已初始化，控制器就绪。` | Tout prêt — tests scène 1 commencent |
| `场景一：… 任务执行完毕。` | Fin du script (puis Ctrl+C pour quitter) |

## Erreurs fréquentes (pas dans `>>>`)

| Message simulateur | Français |
|--------------------|----------|
| `timeout ... /control_robot_leju_claw` | Service pince jamais apparu |
| `new node registered with same name` | Double lancement — tuer l'ancien roslaunch |
| `challenge_sim_launcher: 初始化 parcel_X 完成` | Colis X placé (OK) |
| `反作弊监控已启动` | Anti-triche actif (OK) |
| `比赛计时器已启动` | Chronomètre démarré (OK) |
