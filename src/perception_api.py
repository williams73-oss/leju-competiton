#!/usr/bin/env python3
"""
感知接口封装模块。

封装相机(RGB/深度)、激光雷达、传感器数据和 TF 坐标系的读取，
让任务逻辑中可以用简单的方法获取感知数据。

用法示例:
    from perception_api import CameraReader, LidarReader, SensorReader

    cam = CameraReader()
    rgb = cam.get_head_rgb()          # 获取头部 RGB 图像 (numpy array)
    depth = cam.get_head_depth()      # 获取头部深度图

    lidar = LidarReader()
    points = lidar.get_points()       # 获取点云 (numpy array N×3)

    sensor = SensorReader()
    joint_q = sensor.get_joint_q()    # 获取 28 个关节角度
"""

import rospy
import numpy as np
import struct

# ---- ROS 消息类型 ----
from sensor_msgs.msg import CompressedImage, PointCloud2, JointState, Imu, CameraInfo
from sensor_msgs import point_cloud2 as pc2
from kuavo_msgs.msg import sensorsData
from geometry_msgs.msg import TransformStamped
import tf2_ros

# ---- 图像解码 ----
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# ============================================================
#  CameraReader —— 相机读取
# ============================================================
class CameraReader:
    """
    读取机器人头部和腕部相机的 RGB / 深度图像。

    用法:
        cam = CameraReader()
        rgb = cam.get_head_rgb()       # 头部 RGB，numpy (H, W, 3)
        depth = cam.get_head_depth()   # 头部深度，numpy (H, W) 单位 米
        left_rgb = cam.get_left_wrist_rgb()   # 左腕 RGB
        right_depth = cam.get_right_wrist_depth()  # 右腕深度
    """

    # 默认话题名
    TOPICS = {
        "head_rgb":   "/cam_h/color/image_raw/compressed",
        "head_depth": "/cam_h/depth/image_raw/compressedDepth",
        "left_rgb":   "/cam_l/color/image_raw/compressed",
        "left_depth": "/cam_l/depth/image_rect_raw/compressedDepth",
        "right_rgb":  "/cam_r/color/image_raw/compressed",
        "right_depth":"/cam_r/depth/image_rect_raw/compressedDepth",
    }
    INFO_TOPICS = {
        "head": "/cam_h/color/camera_info",
        "left": "/cam_l/color/camera_info",
        "right": "/cam_r/color/camera_info",
    }
    DEPTH_FRAME = {
        "head": "Head Camera View",
        "left": "Left Wrist Camera View",
        "right": "Right wrist Camera View",
    }

    def __init__(self):
        if not _HAS_CV2:
            rospy.logwarn("CameraReader: 未安装 opencv-python，图像解码不可用。"
                          "请在容器内执行: pip install opencv-python")

        self._cache = {}  # key → CompressedImage
        self._cam_info = {}  # head|left|right → dict fx,fy,cx,cy,frame_id

        # 订阅所有图像话题
        self._subs = {}
        for key, topic in self.TOPICS.items():
            self._subs[key] = rospy.Subscriber(
                topic, CompressedImage,
                lambda msg, k=key: self._callback(k, msg),
                queue_size=1
            )
        for key, topic in self.INFO_TOPICS.items():
            rospy.Subscriber(
                topic, CameraInfo,
                lambda msg, k=key: self._info_callback(k, msg),
                queue_size=1
            )

        rospy.sleep(0.3)  # 等第一帧到来

    def _info_callback(self, key, msg):
        self._cam_info[key] = {
            "fx": float(msg.K[0]),
            "fy": float(msg.K[4]),
            "cx": float(msg.K[2]),
            "cy": float(msg.K[5]),
            "width": int(msg.width),
            "height": int(msg.height),
            "frame_id": msg.header.frame_id or self.DEPTH_FRAME.get(key, key),
        }

    def wait_for_frame(self, key="head_rgb", timeout=2.0, poll=0.05):
        """等待指定相机至少收到一帧。"""
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        while rospy.Time.now() < deadline and not rospy.is_shutdown():
            if self._cache.get(key) is not None:
                return True
            rospy.sleep(poll)
        return self._cache.get(key) is not None

    def _callback(self, key, msg):
        self._cache[key] = msg

    # ---- RGB 获取 ----

    def get_head_rgb(self):
        """返回头部 RGB 图像 (BGR numpy H×W×3)，无数据返回 None。"""
        return self._get_rgb("head_rgb")

    def get_left_wrist_rgb(self):
        """返回左腕 RGB 图像。"""
        return self._get_rgb("left_rgb")

    def get_right_wrist_rgb(self):
        """返回右腕 RGB 图像。"""
        return self._get_rgb("right_rgb")

    # ---- 深度图获取 ----

    def get_head_depth(self):
        """返回头部深度图 (numpy H×W 单位 米)，无数据返回 None。"""
        return self._get_depth("head_depth")

    def get_left_wrist_depth(self):
        """返回左腕深度图。"""
        return self._get_depth("left_depth")

    def get_right_wrist_depth(self):
        """返回右腕深度图。"""
        return self._get_depth("right_depth")

    def get_camera_info(self, cam_key="head"):
        """返回相机内参 dict，无数据返回 None。"""
        return self._cam_info.get(cam_key)

    def depth_valid_count(self, depth, z_min=0.35, z_max=1.2):
        """统计有效深度像素数（米）。"""
        if depth is None:
            return 0
        valid = np.isfinite(depth) & (depth > z_min) & (depth < z_max)
        return int(valid.sum())

    @staticmethod
    def _quat_rotate(quat, point):
        """四元数 (x,y,z,w) 旋转向量。"""
        x, y, z, w = quat
        vx, vy, vz = point
        ix = w * vx + y * vz - z * vy
        iy = w * vy + z * vx - x * vz
        iz = w * vz + x * vy - y * vx
        iw = -x * vx - y * vy - z * vz
        return (
            ix * w + iw * -x + iy * -z - iz * -y,
            iy * w + iw * -y + iz * -x - ix * -z,
            iz * w + iw * -z + ix * -y - iy * -x,
        )

    def pixel_to_base_link(self, tf_reader, cam_key, u_px, v_px, depth_m,
                           target_frame="base_link"):
        """
        像素 + 深度 → base_link 3D 点（相机光学系：Z 前，X 右，Y 下）。
        失败返回 None。
        """
        info = self.get_camera_info(cam_key)
        if info is None or depth_m is None or depth_m <= 0:
            return None
        fx, fy, cx, cy = info["fx"], info["fy"], info["cx"], info["cy"]
        frame_id = info["frame_id"]
        x_c = (float(u_px) - cx) * depth_m / fx
        y_c = (float(v_px) - cy) * depth_m / fy
        z_c = float(depth_m)
        pos, quat = tf_reader.lookup(target_frame, frame_id)
        if pos is None:
            return None
        rx, ry, rz = self._quat_rotate(quat, (x_c, y_c, z_c))
        return (pos[0] + rx, pos[1] + ry, pos[2] + rz)

    def pixel_ray_to_table_plane(self, tf_reader, cam_key, u_px, v_px,
                                 table_z=-0.04, target_frame="base_link"):
        """
        像素射线与 base_link 水平面 z=table_z 求交（不依赖深度值）。
        适用于 colis sur table — depth 的 z 经常偏正。
        """
        info = self.get_camera_info(cam_key)
        if info is None:
            return None
        fx, fy, cx, cy = info["fx"], info["fy"], info["cx"], info["cy"]
        frame_id = info["frame_id"]
        dir_c = ((float(u_px) - cx) / fx,
                 (float(v_px) - cy) / fy,
                 1.0)
        pos, quat = tf_reader.lookup(target_frame, frame_id)
        if pos is None:
            return None
        dx, dy, dz = self._quat_rotate(quat, dir_c)
        if abs(dz) < 1e-6:
            return None
        t = (float(table_z) - pos[2]) / dz
        if t <= 0:
            return None
        return (
            pos[0] + t * dx,
            pos[1] + t * dy,
            float(table_z),
        )

    def median_depth_in_mask(self, depth, mask, z_min=0.35, z_max=1.2):
        """在 mask 区域取深度中值（米），无效返回 None。"""
        if depth is None or mask is None:
            return None
        if depth.shape[:2] != mask.shape[:2]:
            return None
        vals = depth[mask > 0]
        vals = vals[np.isfinite(vals) & (vals > z_min) & (vals < z_max)]
        if len(vals) < 5:
            return None
        return float(np.median(vals))

    # ---- 检查是否有新数据 ----

    def has_new(self, key="head_rgb"):
        """检查指定相机是否有新数据到达。"""
        return key in self._cache and self._cache[key] is not None

    # ---- 内部方法 ----

    def _get_rgb(self, key):
        """解码 JPEG 压缩的 RGB 图像。"""
        if not _HAS_CV2:
            return None
        msg = self._cache.get(key)
        if msg is None:
            return None
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR 格式

    def _get_depth(self, key):
        """
        解码 compressedDepth 图像。

        compressedDepth 使用 PNG 压缩，16-bit 灰度，
        需要把像素值拆成高 8 位和低 8 位再合成毫米值。
        """
        if not _HAS_CV2:
            return None
        msg = self._cache.get(key)
        if msg is None:
            return None

        data = bytes(msg.data)
        depth_quant_a = depth_quant_b = None
        if len(data) >= 12 and data[0] != 0x89:
            depth_quant_a = struct.unpack("f", data[0:4])[0]
            depth_quant_b = struct.unpack("f", data[4:8])[0]
            png_data = data[12:]
        else:
            png_data = data

        buf = np.frombuffer(png_data, dtype=np.uint8)
        raw = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

        if raw is None:
            return None

        if len(raw.shape) == 3:
            raw = raw[:, :, 0]

        use_quant = (
            depth_quant_a is not None
            and abs(depth_quant_a) > 1e-6
            and depth_quant_b is not None
        )
        if use_quant:
            depth_m = depth_quant_a / (raw.astype(np.float32) - depth_quant_b)
            depth_m[raw == 0] = np.nan
            valid = np.isfinite(depth_m) & (depth_m > 0.01)
            if int(valid.sum()) >= 10:
                return depth_m

        # Sim MuJoCo : quant a=b=0 → PNG 16-bit millimètres
        depth_m = raw.astype(np.float32) / 1000.0
        depth_m[raw == 0] = np.nan
        return depth_m


# ============================================================
#  LidarReader —— 激光雷达点云
# ============================================================
class LidarReader:
    """
    读取 Mid360 激光雷达点云。

    用法:
        lidar = LidarReader()
        points = lidar.get_points()     # N×3 numpy array (x, y, z) 单位 米
        points_xy = lidar.get_points_2d()  # N×2 只看 xy 平面
    """

    def __init__(self):
        self._points = None  # N×3
        self._header = None

        rospy.Subscriber("/lidar/points", PointCloud2,
                         self._callback, queue_size=1)
        # LiDAR 初始化需要时间，等久一点
        rospy.sleep(2.0)

    def _callback(self, msg):
        """解析 PointCloud2 → numpy N×3（支持 mixed fields / 24 bytes/point）。"""
        try:
            pts = list(pc2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True))
            if not pts:
                return
            self._points = np.asarray(pts, dtype=np.float32)
            self._header = msg.header
        except Exception as e:
            rospy.logwarn_throttle(10, "点云解析失败: %s, fields=%s",
                                   e, [f.name for f in msg.fields])

    def wait_for_points(self, timeout=3.0, min_points=50, poll=0.1):
        """等待至少 min_points 个点云到达。"""
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        while rospy.Time.now() < deadline and not rospy.is_shutdown():
            if self._points is not None and len(self._points) >= min_points:
                return True
            rospy.sleep(poll)
        return self._points is not None and len(self._points) >= min_points

    def get_points(self):
        """返回点云 (N×3 numpy array)，单位 米。无数据返回 None。"""
        return self._points

    def get_points_2d(self):
        """返回 xy 平面上的点 (N×2)。"""
        if self._points is None:
            return None
        return self._points[:, :2]

    def get_points_in_region(self, x_range, y_range, z_range=None):
        """
        获取指定区域内的点云。

        参数:
            x_range — (x_min, x_max) 单位 米
            y_range — (y_min, y_max) 单位 米
            z_range — (z_min, z_max) 可选，单位 米
        返回:
            N×3 numpy array
        """
        if self._points is None:
            return None
        pts = self._points
        mask = (pts[:, 0] >= x_range[0]) & (pts[:, 0] <= x_range[1]) & \
               (pts[:, 1] >= y_range[0]) & (pts[:, 1] <= y_range[1])
        if z_range is not None:
            mask &= (pts[:, 2] >= z_range[0]) & (pts[:, 2] <= z_range[1])
        return pts[mask]


# ============================================================
#  SensorReader —— 传感器数据
# ============================================================
class SensorReader:
    """
    读取 /sensors_data_raw 中的关节角度、IMU、末端信息。

    关节顺序: 左腿 6 → 右腿 6 → 左臂 7 → 右臂 7 → 头 2 = 28 个

    用法:
        sensor = SensorReader()
        q = sensor.get_joint_q()       # 28 个关节位置 (rad)
        quat = sensor.get_imu_quat()   # IMU 四元数姿态
        acc = sensor.get_imu_acc()     # 加速度 (m/s²)
    """

    def __init__(self):
        self._data = None
        rospy.Subscriber("/sensors_data_raw", sensorsData,
                         self._callback, queue_size=1)
        rospy.sleep(0.3)

    def _callback(self, msg):
        self._data = msg

    def get_joint_q(self):
        """返回 28 个关节位置，单位 弧度。"""
        if self._data is None:
            return None
        return list(self._data.joint_data.joint_q)

    def get_joint_v(self):
        """返回 28 个关节速度，单位 rad/s。"""
        if self._data is None:
            return None
        return list(self._data.joint_data.joint_v)

    def get_joint_degrees(self):
        """返回 28 个关节位置，单位 度。"""
        q_rad = self.get_joint_q()
        if q_rad is None:
            return None
        return [np.degrees(q) for q in q_rad]

    def get_arm_joint_degrees(self):
        """
        返回双臂 14 个关节角度，单位 度。
        顺序: 左臂 7 + 右臂 7。
        """
        q_deg = self.get_joint_degrees()
        if q_deg is None:
            return None
        return q_deg[12:26]  # 索引 12~25 是手臂

    def get_imu_quat(self):
        """返回 IMU 姿态四元数 (x, y, z, w)。"""
        if self._data is None:
            return None
        q = self._data.imu_data.quat
        return (q.x, q.y, q.z, q.w)

    def get_imu_acc(self):
        """返回加速度 (x, y, z)，单位 m/s²。"""
        if self._data is None:
            return None
        a = self._data.imu_data.acc
        return (a.x, a.y, a.z)

    def get_imu_gyro(self):
        """返回角速度 (x, y, z)，单位 rad/s。"""
        if self._data is None:
            return None
        g = self._data.imu_data.gyro
        return (g.x, g.y, g.z)

    def get_claw_position(self):
        """返回夹爪位置 [左%, 右%]，0=张开 100=闭合。"""
        if self._data is None:
            return None
        return list(self._data.end_effector_data.position)


# ============================================================
#  TFReader —— 坐标系查询
# ============================================================
class TFReader:
    """
    查询 TF 变换，获取物体/机器人部件在坐标系中的位置。

    用法:
        tf_reader = TFReader()
        pos, quat = tf_reader.lookup("base_link", "parcel_1")
        if pos is not None:
            print(f"快递位置: x={pos[0]:.2f} y={pos[1]:.2f} z={pos[2]:.2f}")
    """

    def __init__(self):
        self._buffer = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._buffer)
        rospy.sleep(0.3)

    def lookup(self, from_frame, to_frame, timeout=1.0):
        """
        查询 to_frame 在 from_frame 坐标系下的位姿。

        参数:
            from_frame — 参考坐标系，如 "base_link"
            to_frame   — 目标坐标系，如 "parcel_1"
            timeout    — 超时秒数
        返回:
            (position, quaternion) 或 (None, None)
            position — (x, y, z) 单位 米
            quaternion — (x, y, z, w)
        """
        try:
            trans = self._buffer.lookup_transform(
                from_frame, to_frame,
                rospy.Time(0),  # 最新
                rospy.Duration(timeout)
            )
            t = trans.transform.translation
            r = trans.transform.rotation
            return (t.x, t.y, t.z), (r.x, r.y, r.z, r.w)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5, "TF 查询失败 %s→%s: %s",
                                   from_frame, to_frame, e)
            return None, None

    def get_distance(self, from_frame, to_frame):
        """返回两个坐标系之间的直线距离（米）。查询失败返回 None。"""
        pos, _ = self.lookup(from_frame, to_frame)
        if pos is None:
            return None
        return np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
