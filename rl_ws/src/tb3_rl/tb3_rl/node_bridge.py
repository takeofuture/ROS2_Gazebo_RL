# tb3_rl/node_bridge.py
import math
import time
import subprocess
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    ok: bool = False


class TB3Bridge(Node):
    """
    ROS2側：
      - publish : /cmd_vel (geometry_msgs/Twist)
      - subscribe: /odom (nav_msgs/Odometry)

    Gazebo側：
      - reset: gz service /world/<world>/set_pose
      - world pose: gz topic /world/<world>/pose/info

    重要:
      /odom は world そのものではないので、reset直後に 1回だけ
      odom->world の 2D剛体変換 (R,t) を推定して固定する。
    """
    def __init__(self, world_name="arena", model_name="tb3"):
        super().__init__("tb3_bridge")
        self.world_name = world_name
        self.model_name = model_name

        self._odom_pose = Pose2D()

        # odom -> world transform: world = R(dyaw)*odom + (dx,dy)
        self._has_calib = False
        self._dx = 0.0
        self._dy = 0.0
        self._dyaw = 0.0

        self.sub_odom = self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._odom_pose = Pose2D(float(p.x), float(p.y), float(quat_to_yaw(q)), True)

    def spin_for(self, sec: float = 0.2):
        t_end = time.time() + sec
        while time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_odom_ok(self, timeout_sec: float = 2.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._odom_pose.ok:
                return True
        return False

    def _apply_odom_to_world(self, od: Pose2D) -> Pose2D:
        if not od.ok:
            return od
        if not self._has_calib:
            return od

        c = math.cos(self._dyaw)
        s = math.sin(self._dyaw)
        xw = c * od.x - s * od.y + self._dx
        yw = s * od.x + c * od.y + self._dy
        yaww = wrap_pi(od.yaw + self._dyaw)
        return Pose2D(xw, yw, yaww, True)

    def get_pose(self) -> Pose2D:
        return self._apply_odom_to_world(self._odom_pose)

    def send_cmd(self, linear_x: float, angular_z: float, duration: float):
        tw = Twist()
        tw.linear.x = float(linear_x)
        tw.angular.z = float(angular_z)

        self.pub_cmd.publish(tw)
        self.spin_for(duration)

        tw.linear.x = 0.0
        tw.angular.z = 0.0
        self.pub_cmd.publish(tw)
        self.spin_for(0.05)

    # ---------------------------
    # Gazebo world pose fetch
    # ---------------------------
    def _gz_world_xyyaw(self, timeout_sec=1.5) -> Optional[Tuple[float, float, float]]:
        topic = f"/world/{self.world_name}/pose/info"
        cmd = f"gz topic -e -t {topic} -n 1"
        try:
            r = subprocess.run(
                ["bash", "-lc", cmd],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None

        if r.returncode != 0 or not r.stdout:
            return None

        txt = r.stdout
        key = f'name: "{self.model_name}"'
        i = txt.find(key)
        if i < 0:
            return None

        cut = txt[i:i + 1200]

        mx = re.search(r"position\s*{[^}]*x:\s*([-\d.eE]+)", cut, re.DOTALL)
        my = re.search(r"position\s*{[^}]*y:\s*([-\d.eE]+)", cut, re.DOTALL)
        mw = re.search(r"orientation\s*{[^}]*w:\s*([-\d.eE]+)", cut, re.DOTALL)
        mz = re.search(r"orientation\s*{[^}]*z:\s*([-\d.eE]+)", cut, re.DOTALL)
        if not (mx and my and mw and mz):
            return None

        wx = float(mx.group(1))
        wy = float(my.group(1))
        qw = float(mw.group(1))
        qz = float(mz.group(1))
        wyaw = math.atan2(2 * qw * qz, 1 - 2 * (qz * qz))
        return wx, wy, wyaw

    def calibrate_odom_to_world(self, world_x: float, world_y: float, world_yaw: float):
        """
        world = R(dyaw)*odom + (dx,dy)
        dyaw = world_yaw - odom_yaw
        dx,dy は上式から解く
        """
        od = self._odom_pose
        if not od.ok:
            self.get_logger().warn("[CALIB] no odom; skip")
            return

        self._dyaw = wrap_pi(world_yaw - od.yaw)

        c = math.cos(self._dyaw)
        s = math.sin(self._dyaw)
        rx = c * od.x - s * od.y
        ry = s * od.x + c * od.y
        self._dx = world_x - rx
        self._dy = world_y - ry
        self._has_calib = True

        self.get_logger().info(
            f"Calib odom->world: dx={self._dx:.3f} dy={self._dy:.3f} dyaw={self._dyaw:.3f} "
            f"(odom=({od.x:.3f},{od.y:.3f},{od.yaw:.3f}) world=({world_x:.3f},{world_y:.3f},{world_yaw:.3f}))"
        )

    # ---------------------------
    # Reset
    # ---------------------------
    def reset_pose(self, x=-2.0, y=-2.0, z=0.02, yaw=0.0, timeout_sec=8.0, retries=3):
        self._has_calib = False  # resetごとにキャリブやり直す

        half = yaw * 0.5
        qz = math.sin(half)
        qw = math.cos(half)

        def call_set_pose(zv):
            cmd = (
                f"gz service -s /world/{self.world_name}/set_pose "
                f"--reqtype gz.msgs.Pose --reptype gz.msgs.Boolean "
                f"--req 'name:\"{self.model_name}\" "
                f"position:{{x:{x},y:{y},z:{zv}}} "
                f"orientation:{{w:{qw}, z:{qz}}}'"
            )
            return subprocess.run(
                ["bash", "-lc", cmd],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )

        last = ""
        ok = False
        for i in range(1, retries + 1):
            try:
                # 2段階ワープ（床に埋まって詰まるのを避ける）
                call_set_pose(0.30)
                r = call_set_pose(z)

                out = (r.stdout or "").strip().lower()
                err = (r.stderr or "").strip()
                last = err or out

                if r.returncode == 0 and ("true" in out or out == ""):
                    ok = True
                    break

                self.get_logger().warn(f"[RESET] attempt {i}/{retries} failed rc={r.returncode} last={last}")
            except subprocess.TimeoutExpired:
                last = f"timeout>{timeout_sec}s"
                self.get_logger().warn(f"[RESET] attempt {i}/{retries} timed out")
            time.sleep(0.2)

        if not ok:
            self.get_logger().warn(f"[RESET] FAILED after {retries} attempts (last={last})")

        # odom更新を待つ
        self.spin_for(0.4)
        self.wait_odom_ok(timeout_sec=2.0)

        # world pose を取れるならそれを真値に（取れなければ reset target を真値に）
        w = None
        for _ in range(6):
            w = self._gz_world_xyyaw(timeout_sec=1.5)
            if w is not None:
                break
            time.sleep(0.1)

        od = self._odom_pose
        if w is not None:
            wx, wy, wyaw = w
            self.get_logger().info(f"[RESET] world pose from gz: ({wx:.3f},{wy:.3f},{wyaw:.3f})")
            self.get_logger().info(f"[RESET] raw odom pose     : ({od.x:.3f},{od.y:.3f},{od.yaw:.3f})")
            self.calibrate_odom_to_world(wx, wy, yaw)  # yawは「目標yaw」を使う（揺れを減らす）
        else:
            self.get_logger().warn("[RESET] gz world pose unavailable; using reset target as world truth")
            self.calibrate_odom_to_world(x, y, yaw)

        p = self.get_pose()
        self.get_logger().info(f"Pose(after calib): {p}")


def main():
    rclpy.init()
    node = TB3Bridge(world_name="arena", model_name="tb3")

    node.get_logger().info("Reset to (-2,-2)")
    node.reset_pose(-2.0, -2.0, 0.02, 0.0)

    node.get_logger().info("Forward")
    node.send_cmd(0.50, 0.0, 0.70)
    node.get_logger().info(f"Pose: {node.get_pose()}")

    node.get_logger().info("Left")
    node.send_cmd(0.0, 0.6, 0.30)
    node.get_logger().info(f"Pose: {node.get_pose()}")

    node.get_logger().info("Forward")
    node.send_cmd(0.50, 0.0, 0.70)
    node.get_logger().info(f"Pose: {node.get_pose()}")

    node.get_logger().info("Done")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
