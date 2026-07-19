#!/usr/bin/env python3
"""
挑战杯三场景统一任务入口。

推荐运行方式：
  rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3
  rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 3
  rosrun challenge_cup_task_template challenge_task.py --scene scene3 --seed 3
"""

import argparse
import os
import sys


SCENE_CONFIGS = {
    "scene1": {
        "node_name": "challenge_task_scene1",
        "title": "场景一：包裹称重与摆放",
    },
    "scene2": {
        "node_name": "challenge_task_scene2",
        "title": "场景二：分拣归档",
    },
    "scene3": {
        "node_name": "challenge_task_scene3",
        "title": "场景三：SMT 料盘出库",
    },
}


class _NullClaw(object):
    """夹爪服务不可用时的占位（仅场景一传感器测试）。"""

    def open(self, position=None):
        import rospy
        rospy.logwarn(">>> 警告：夹爪不可用，open() 已跳过")

    def close(self, position=None):
        import rospy
        rospy.logwarn(">>> 警告：夹爪不可用，close() 已跳过")

    def set_position(self, left_percent, right_percent):
        import rospy
        rospy.logwarn(">>> 警告：夹爪不可用，set_position() 已跳过")

    def left_open(self):
        self.open()

    def left_close(self):
        self.close()

    def right_open(self):
        self.open()

    def right_close(self):
        self.close()

    def is_grabbed(self):
        return False

    def is_moving(self):
        return False

    def wait_until_done(self, timeout=5.0):
        return True


def _wait_for_claw_service(log, timeout=120.0):
    import rospy
    import time

    log("等待夹爪服务 /control_robot_leju_claw（最多 %.0f 秒）...", timeout)
    start = time.time()
    last_progress = -15
    while time.time() - start < timeout:
        try:
            rospy.wait_for_service("/control_robot_leju_claw", timeout=5.0)
            log("夹爪服务已连接。")
            return True
        except rospy.ROSException:
            elapsed = int(time.time() - start)
            if elapsed - last_progress >= 15:
                last_progress = elapsed
                log("仍在等待夹爪服务… %d 秒", elapsed)
    return False


def _create_claw(log, scene, ClawController):
    import rospy

    if not _wait_for_claw_service(log, timeout=120.0):
        log("夹爪服务超时。诊断: rosnode list | grep sim_leju ; rosservice list | grep claw")
        log("请 Ctrl+C 后 killall roslaunch，再单独启动一次 rosrun")
        if scene == "scene1":
            log("警告：场景一传感器测试继续（无夹爪）")
            return _NullClaw()
        raise rospy.ROSException(
            "timeout exceeded while waiting for service /control_robot_leju_claw"
        )
    return ClawController()


def _load_launcher():
    # 公共启动器位于受保护包 challenge_cup_simulator/utils/（选手不可改动），
    # 从那里导入，确保完整性校验无法被绕过。
    try:
        import rospkg
        sim_utils = os.path.join(rospkg.RosPack().get_path("challenge_cup_simulator"), "utils")
    except Exception:
        sim_utils = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..", "challenge_cup_simulator", "utils")
    sys.path.insert(0, sim_utils)
    from challenge_sim_launcher import ChallengeSimLauncher
    return ChallengeSimLauncher


def run_scene(scene, seed, node_name=None, timeout=120,
              time_limit=None, timer_gui=True):
    if scene not in SCENE_CONFIGS:
        raise ValueError("unknown scene: {}".format(scene))

    config = SCENE_CONFIGS[scene]
    ChallengeSimLauncher = _load_launcher()

    launcher = ChallengeSimLauncher(
        scene=scene,
        seed=seed,
        match_time_limit=time_limit,
        timer_gui=timer_gui,
    )
    launcher.start(node_name=node_name or config["node_name"], timeout=timeout)

    import rospy

    def log(msg, *args):
        if args:
            msg = msg % args
        rospy.loginfo(">>> " + msg)

    _scripts_dir = os.path.dirname(os.path.abspath(__file__))
    _pkg_dir = os.path.dirname(_scripts_dir)
    sys.path.insert(0, os.path.join(_pkg_dir, "src"))
    sys.path.insert(0, _scripts_dir)

    from robot_api import RobotMover, ArmController, ClawController, HeadController

    log("=== %s任务启动 ===", config["title"])

    log("等待 ROS 节点就绪（10 秒）…")
    rospy.sleep(10.0)

    log("初始化 RobotMover...")
    robot = RobotMover()
    log("初始化 ArmController...")
    arm = ArmController()
    claw = _create_claw(log, scene, ClawController)
    log("初始化 HeadController...")
    head = HeadController()

    rospy.sleep(1.0)
    log("场景实例已初始化，控制器就绪。")

    if scene == "scene1":
        from scene1_task import run_scene1
        run_scene1(robot, arm, claw, head, log, seed=seed)
    elif scene == "scene2":
        from scene2_task import run_scene2
        run_scene2(robot, arm, claw, head, log)
    elif scene == "scene3":
        from scene3_task import run_scene3
        run_scene3(robot, arm, claw, head, log)

    log("%s 任务执行完毕。", config["title"])
    rospy.spin()


def main():
    parser = argparse.ArgumentParser(description="挑战杯三场景统一任务入口")
    parser.add_argument("--scene", choices=sorted(SCENE_CONFIGS), default="scene1",
                        help="要启动的比赛场景")
    parser.add_argument("--seed", type=int, default=0,
                        help="场景种子；正式评测 seed 由组委会指定")
    parser.add_argument("--node-name", default=None,
                        help="ROS 节点名；默认按 scene 自动设置")
    parser.add_argument("--timeout", type=int, default=120,
                        help="等待仿真就绪的超时时间，单位秒")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="比赛时长，单位秒；默认读取 CHALLENGE_TIME_LIMIT，未设置则不限时")
    parser.add_argument("--no-timer-gui", action="store_true",
                        help="不弹出计时器窗口，仅保留后台计时日志")
    args = parser.parse_args()

    run_scene(
        scene=args.scene,
        seed=args.seed,
        node_name=args.node_name,
        timeout=args.timeout,
        time_limit=args.time_limit,
        timer_gui=not args.no_timer_gui,
    )


if __name__ == "__main__":
    main()
