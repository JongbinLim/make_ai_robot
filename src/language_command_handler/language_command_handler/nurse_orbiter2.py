#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from geometry_msgs.msg import PoseStamped, PointStamped
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String

# -------------------------------
# Mission constants
# -------------------------------
TARGET_ROOM_POSE = (-7.5, -25.0, 3.13)  # (x, y, yaw)
TARGET_LABEL = "nurse"

CONTROL_PERIOD = 0.3  # 너무 길면 orbit이 멈춘 것처럼 보임
WAYPOINT_TOLERANCE = 0.25  # 너무 크면 "도착"으로 판정되어 안 움직일 수 있음

# SEARCH
SEARCH_YAW_STEP = math.radians(30)
SEARCH_YAW_TOL  = math.radians(5)
SEARCH_SETTLE_TICKS = 1
PERCEPTION_FRESH_SEC = 1.5
DETECT_CONFIRM_N = 1

# CENTERING
CENTER_TOL_PX = 25
MAX_YAW_STEP = math.radians(15)
CENTERING_STABLE_TICKS = 2
CENTERING_FRESH_STRICT_SEC = 0.3

# Camera fallback
IMG_WIDTH_FALLBACK = 720.0
IMG_CENTER_U_FALLBACK = IMG_WIDTH_FALLBACK * 0.5

# ORBIT
ORBIT_RADIUS = 1.0
ORBIT_POINTS = 8
ORBIT_CCW = True        # True=반시계(보통 오른쪽으로 90도 꺾이며 시작), False=시계
ORBIT_REPEAT = True     # 계속 빙빙 돌기


class NurseOrbiter(Node):
    def __init__(self):
        super().__init__("nurse_orbiter")

        self.state = "GO_TO_ROOM"
        self._state_entered = True

        # pose
        self.current_pose: PoseStamped | None = None

        # perception
        self.detected_labels: list[str] = []
        self.last_labels_time = None

        self.nurse_u = -1.0
        self.nurse_depth = -1.0
        self.last_nurse_center_time = None

        self.camera_info: CameraInfo | None = None

        # searching
        self.search_yaw = 0.0
        self.search_steps_done = 0
        self.max_search_steps = int(2 * math.pi / SEARCH_YAW_STEP)
        self.search_phase = "ROTATE"
        self.search_settle_counter = 0
        self.detect_confirm_count = 0

        # centering
        self.center_stable_cnt = 0

        # frozen nurse map
        self.nurse_map_x: float | None = None
        self.nurse_map_y: float | None = None

        # orbit
        self.orbit_waypoints: list[tuple[float, float, float]] = []
        self.current_waypoint_idx = 0

        qos = QoSProfile(depth=10)
        self.pose_sub = self.create_subscription(PoseStamped, "/go1_pose", self.pose_callback, qos)
        self.label_sub = self.create_subscription(String, "/detections/labels", self.label_callback, qos)
        self.nurse_center_sub = self.create_subscription(PointStamped, "/detections/nurse_center", self.nurse_center_callback, qos)
        self.caminfo_sub = self.create_subscription(CameraInfo, "/camera_top/camera_info", self.camera_info_callback, qos)

        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.timer = self.create_timer(CONTROL_PERIOD, self.control_loop)

        self.get_logger().info("NurseOrbiter (simple freeze+orbit) started.")

    # ---------------- callbacks ----------------
    def pose_callback(self, msg: PoseStamped):
        self.current_pose = msg

    def label_callback(self, msg: String):
        self.detected_labels = [l.strip() for l in msg.data.split(",") if l.strip()]
        self.last_labels_time = self.get_clock().now()

    def nurse_center_callback(self, msg: PointStamped):
        self.nurse_u = float(msg.point.x)
        self.nurse_depth = float(msg.point.z)
        self.last_nurse_center_time = self.get_clock().now()

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info = msg

    # ---------------- utils ----------------
    @staticmethod
    def wrap_to_pi(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    def time_since(self, t) -> float:
        if t is None:
            return float("inf")
        return (self.get_clock().now() - t).nanoseconds * 1e-9

    def labels_fresh(self) -> bool:
        return self.time_since(self.last_labels_time) <= PERCEPTION_FRESH_SEC

    def nurse_center_fresh(self) -> bool:
        return self.time_since(self.last_nurse_center_time) <= PERCEPTION_FRESH_SEC and self.nurse_u >= 0.0

    def get_yaw_from_pose(self, pose: PoseStamped) -> float:
        q = pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def publish_goal(self, x: float, y: float, yaw: float):
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "map"
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)

        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        goal.pose.orientation.w = cy
        goal.pose.orientation.z = sy
        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        self.goal_pub.publish(goal)

    @staticmethod
    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def get_image_center_u(self) -> float:
        if self.camera_info is not None and len(self.camera_info.k) >= 9:
            return float(self.camera_info.k[2])
        if self.camera_info is not None and self.camera_info.width > 0:
            return float(self.camera_info.width) * 0.5
        return IMG_CENTER_U_FALLBACK

    def pixel_to_bearing(self, u_px: float) -> float:
        if self.camera_info is not None and len(self.camera_info.k) >= 9:
            fx = float(self.camera_info.k[0])
            cx = float(self.camera_info.k[2])
            if fx > 1e-6:
                return math.atan2((u_px - cx), fx)

        # fallback HFOV 90deg
        w = IMG_WIDTH_FALLBACK
        hfov = math.radians(90.0)
        x_norm = (u_px - (w * 0.5)) / (w * 0.5)
        return x_norm * (hfov * 0.5)

    def reset_search(self, curr_yaw: float):
        self.search_yaw = curr_yaw
        self.search_steps_done = 0
        self.search_phase = "ROTATE"
        self.search_settle_counter = 0
        self.detect_confirm_count = 0

    def set_state(self, s: str):
        if self.state != s:
            self.state = s
            self._state_entered = True
            if s != "CENTERING":
                self.center_stable_cnt = 0

    # ---- freeze nurse in map ----
    def freeze_nurse_map(self, curr_x: float, curr_y: float, curr_yaw: float) -> bool:
        if not self.nurse_center_fresh():
            return False
        if self.nurse_depth is None or self.nurse_depth <= 0.0:
            return False

        bearing = self.pixel_to_bearing(self.nurse_u)
        abs_yaw = curr_yaw - bearing  # robot->nurse 방향
        nx = curr_x + self.nurse_depth * math.cos(abs_yaw)
        ny = curr_y + self.nurse_depth * math.sin(abs_yaw)

        self.nurse_map_x = nx
        self.nurse_map_y = ny
        return True

    def build_orbit_waypoints_with_entry(self, cx: float, cy: float, curr_x: float, curr_y: float) -> list[tuple[float, float, float]]:
        """
        0번: entry(원 위의 '현재 방사선 방향' 점) - yaw는 entry로 가는 heading (먼저 이동하게)
        1번~: 원 궤도 - yaw는 접선(tangent)
        """
        r = ORBIT_RADIUS
        direction = 1.0 if ORBIT_CCW else -1.0
        delta = 2.0 * math.pi / ORBIT_POINTS

        # center->robot 각도 (원 위 entry 점)
        start_angle = math.atan2(curr_y - cy, curr_x - cx)

        entry_x = cx + r * math.cos(start_angle)
        entry_y = cy + r * math.sin(start_angle)
        entry_yaw = math.atan2(entry_y - curr_y, entry_x - curr_x)  # ✅ 먼저 이동(앞으로/경로)하게

        wps = [(entry_x, entry_y, self.wrap_to_pi(entry_yaw))]

        # orbit points (entry 각도는 이미 썼으니 i+1로 "앞쪽"부터)
        for i in range(ORBIT_POINTS):
            theta = start_angle + direction * delta * (i + 1)
            wx = cx + r * math.cos(theta)
            wy = cy + r * math.sin(theta)
            wyaw = self.wrap_to_pi(theta + direction * (math.pi / 2.0))  # ✅ 접선
            wps.append((wx, wy, wyaw))

        return wps

    # ---------------- main loop ----------------
    def control_loop(self):
        if self.current_pose is None:
            return

        curr_x = self.current_pose.pose.position.x
        curr_y = self.current_pose.pose.position.y
        curr_yaw = self.get_yaw_from_pose(self.current_pose)

        # 1) GO_TO_ROOM
        if self.state == "GO_TO_ROOM":
            if self._state_entered:
                self.get_logger().info("State=GO_TO_ROOM")
                self._state_entered = False
                self.nurse_map_x = None
                self.nurse_map_y = None
                self.orbit_waypoints = []
                self.current_waypoint_idx = 0

            tx, ty, tyaw = TARGET_ROOM_POSE
            dist = math.hypot(tx - curr_x, ty - curr_y)
            if dist > 1.0:
                self.publish_goal(tx, ty, tyaw)
            else:
                self.get_logger().info("Arrived -> SEARCHING")
                self.reset_search(curr_yaw)
                self.set_state("SEARCHING")
            return

        # 2) SEARCHING
        if self.state == "SEARCHING":
            if self._state_entered:
                self.get_logger().info("State=SEARCHING")
                self._state_entered = False

            self.publish_goal(curr_x, curr_y, self.search_yaw)
            yaw_err = abs(self.wrap_to_pi(self.search_yaw - curr_yaw))

            if self.search_phase == "ROTATE":
                if yaw_err <= SEARCH_YAW_TOL:
                    self.search_phase = "SETTLE"
                    self.search_settle_counter = 0
                return

            if self.search_phase == "SETTLE":
                self.search_settle_counter += 1
                if self.search_settle_counter >= SEARCH_SETTLE_TICKS:
                    self.search_phase = "CHECK"
                return

            detected = (self.labels_fresh() and (TARGET_LABEL in self.detected_labels)) or self.nurse_center_fresh()
            self.detect_confirm_count = (self.detect_confirm_count + 1) if detected else 0

            if self.detect_confirm_count >= DETECT_CONFIRM_N:
                self.detect_confirm_count = 0
                self.get_logger().info("Detected -> CENTERING")
                self.set_state("CENTERING")
                return

            self.search_yaw = self.wrap_to_pi(self.search_yaw + SEARCH_YAW_STEP)
            self.search_steps_done += 1
            self.search_phase = "ROTATE"
            self.search_settle_counter = 0

            if self.search_steps_done >= self.max_search_steps:
                self.get_logger().warn("Not found after full rotation. Restart scan.")
                self.search_steps_done = 0
            return

        # 3) CENTERING
        if self.state == "CENTERING":
            if self._state_entered:
                self.get_logger().info("State=CENTERING")
                self._state_entered = False
                self.center_stable_cnt = 0

            if not self.nurse_center_fresh():
                self.get_logger().warn("Lost nurse_center -> SEARCHING")
                self.reset_search(curr_yaw)
                self.set_state("SEARCHING")
                return

            if self.time_since(self.last_nurse_center_time) > CENTERING_FRESH_STRICT_SEC:
                self.publish_goal(curr_x, curr_y, curr_yaw)
                return

            center_u = self.get_image_center_u()
            du = self.nurse_u - center_u

            bearing = self.pixel_to_bearing(self.nurse_u)
            yaw_step = self.clamp(-bearing, -MAX_YAW_STEP, MAX_YAW_STEP)
            target_yaw = self.wrap_to_pi(curr_yaw + yaw_step)

            # yaw only
            self.publish_goal(curr_x, curr_y, target_yaw)

            if abs(du) <= CENTER_TOL_PX:
                self.center_stable_cnt += 1
                if self.center_stable_cnt >= CENTERING_STABLE_TICKS:
                    ok = self.freeze_nurse_map(curr_x, curr_y, curr_yaw)
                    if not ok or self.nurse_map_x is None or self.nurse_map_y is None:
                        self.get_logger().warn("Freeze failed -> SEARCHING")
                        self.reset_search(curr_yaw)
                        self.set_state("SEARCHING")
                        return

                    cx, cy = self.nurse_map_x, self.nurse_map_y
                    self.orbit_waypoints = self.build_orbit_waypoints_with_entry(cx, cy, curr_x, curr_y)
                    self.current_waypoint_idx = 0
                    self.get_logger().info(f"Frozen nurse=({cx:.2f},{cy:.2f}), orbit R={ORBIT_RADIUS:.2f}")
                    self.set_state("ORBITING")
                return
            else:
                self.center_stable_cnt = 0
                return

        # 4) ORBITING
        if self.state == "ORBITING":
            if self._state_entered:
                self.get_logger().info("State=ORBITING")
                self._state_entered = False

            if not self.orbit_waypoints:
                self.get_logger().warn("No orbit waypoints -> FINISHED")
                self.set_state("FINISHED")
                return

            if self.current_waypoint_idx >= len(self.orbit_waypoints):
                if ORBIT_REPEAT:
                    # entry(0)은 이미 했으니 1부터 다시 계속 돌기
                    self.current_waypoint_idx = 1
                else:
                    self.get_logger().info("Orbit completed -> FINISHED")
                    self.set_state("FINISHED")
                    return

            wx, wy, wyaw = self.orbit_waypoints[self.current_waypoint_idx]
            dist = math.hypot(wx - curr_x, wy - curr_y)

            if dist < WAYPOINT_TOLERANCE:
                self.current_waypoint_idx += 1
                # 멈칫 방지: 다음 점 바로 publish
                if self.current_waypoint_idx < len(self.orbit_waypoints):
                    wx, wy, wyaw = self.orbit_waypoints[self.current_waypoint_idx]
                    self.publish_goal(wx, wy, wyaw)
            else:
                self.publish_goal(wx, wy, wyaw)
            return

        # 5) FINISHED
        if self.state == "FINISHED":
            if self._state_entered:
                self.get_logger().info("State=FINISHED")
                self._state_entered = False
            return


def main(args=None):
    rclpy.init(args=args)
    node = NurseOrbiter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
