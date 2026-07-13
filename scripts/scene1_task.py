#!/usr/bin/env python3
"""
Scène 1 — Pesée et tri des colis (快递称重与摆放).

Orchestration seule. Logique découpée :
  - scene1.config       : constantes / modes
  - scene1.perception   : tête + LiDAR + fusion
  - scene1.wrist_vision : refine cam_r (équipe contrôle)
  - scene1.actions      : bras / saisie (équipe contrôle)

Compte rendu : docs/scene1/STATUS.md
"""
from __future__ import print_function
import os
import sys

# scripts/ sur le path (rosrun) + package parent pour `import scene1`
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
_pkg = os.path.dirname(_scripts_dir)
sys.path.insert(0, os.path.join(_pkg, "src"))

from scene1.config import PERCEPTION_ONLY, TOUCH_TEST
from scene1.perception import run_scene1_perception_only
from scene1.actions import run_scene1_touch_test, run_scene1_mission


def run_scene1(robot, arm, claw, head, log):
    """
    Point d'entrée appelé par challenge_task.py.
      robot — RobotMover
      arm   — ArmController
      claw  — ClawController
      head  — HeadController
      log   — logger
    """
    if PERCEPTION_ONLY:
        run_scene1_perception_only(arm, head, log)
        return
    if TOUCH_TEST:
        run_scene1_touch_test(robot, arm, claw, head, log)
        return
    run_scene1_mission(robot, arm, claw, head, log)
