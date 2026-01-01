# tb3_rl/env.py
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy

from .node_bridge import TB3Bridge


def wrap_pi(a):
    while a > math.pi:
        a -= 2*math.pi
    while a < -math.pi:
        a += 2*math.pi
    return a


class TB3ArenaEnv(gym.Env):
    """
    離散3行動:
      0: forward
      1: turn_left
      2: turn_right
    観測:
      [x, y, yaw, dx, dy, dist, heading_to_goal]
    done:
      goal到達 OR 壁/中央障害物に近づきすぎ OR step_limit
    """
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # arena: [-4,4]、壁厚は0.2なので余裕を見て閾値
        self.arena_min = -4.0
        self.arena_max = 4.0
        self.wall_margin = 0.25  # 壁に近い＝失敗
        self.center = (0.0, 0.0)
        self.center_box_half = 0.6  # center_box size 1.2 -> half 0.6
        self.center_margin = 0.20   # 箱に近い＝失敗

        self.start = (-2.0, -2.0, 0.0)  # 右下固定
        self.goal = (3.5, 3.5)          # 左上固定
        self.goal_radius = 0.35

        # action: 3
        self.action_space = spaces.Discrete(3)

        # observation: 7次元
        high = np.array([10,10, math.pi, 10,10, 20, math.pi], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # ROS
        if not rclpy.ok():
            rclpy.init()
        self.bridge = TB3Bridge(world_name="arena", model_name="tb3")
        #self.bridge = TB3Bridge(model_name="tb3")

        self.max_steps = 250
        self.step_count = 0
        self.prev_dist = None

    def _spin_some(self, n=5):
        # まとめて受信する
        for _ in range(n):
            rclpy.spin_once(self.bridge, timeout_sec=0.05)

    def _obs(self, pose):
        x, y, yaw = pose.x, pose.y, pose.yaw
        dx = self.goal[0] - x
        dy = self.goal[1] - y
        dist = math.hypot(dx, dy)
        goal_yaw = math.atan2(dy, dx)
        heading = wrap_pi(goal_yaw - yaw)
        return np.array([x, y, yaw, dx, dy, dist, heading], dtype=np.float32)

    def _is_collision(self, x, y):
        # 壁に近い
        if (x < self.arena_min + self.wall_margin or
            x > self.arena_max - self.wall_margin or
            y < self.arena_min + self.wall_margin or
            y > self.arena_max - self.wall_margin):
            return True

        # 中央箱に近い（AABBで判定）
        if (abs(x - self.center[0]) < (self.center_box_half + self.center_margin) and
            abs(y - self.center[1]) < (self.center_box_half + self.center_margin)):
            return True

        return False

    def _is_goal(self, x, y):
        return math.hypot(self.goal[0]-x, self.goal[1]-y) < self.goal_radius

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        # ★必ず開始位置へ戻す
        self.bridge.reset_pose(x=self.start[0], y=self.start[1], z=0.02, yaw=self.start[2])
        #self.bridge.reset(x=self.start[0], y=self.start[1], z=0.02, yaw=self.start[2], prefer_respawn=False)
        # ここは「あなたが成功した set_pose」サービスで戻す
        # すでに動いてるので、ここは後で実装（下のメモ参照）
        # まずは pose が取れるまで待つだけでもOK
        for _ in range(10):
            self._spin_some()
            #p = self.bridge.get_pose()
            #if p.ok:
            #    break
        p = self.bridge.get_pose()
        obs = self._obs(p)
        self.prev_dist = float(obs[5])
        return obs, {}

    def step(self, action):
        self.step_count += 1
        # 実際にコマンドを送る（前進/左/右）
        if action == 0:
            self.bridge.send_cmd(0.50, 0.0, duration=0.70)
        elif action == 1:
            self.bridge.send_cmd(0.0, 0.6, duration=0.30)
        else:
            self.bridge.send_cmd(0.0, -0.6, duration=0.30)

        self._spin_some()
        p = self.bridge.get_pose()
        obs = self._obs(p)
        x, y = float(obs[0]), float(obs[1])
        dist = float(obs[5])

        terminated = False
        success = False
        collision = False

        if self._is_goal(x, y):
            terminated = True
            success = True
        elif self._is_collision(x, y):
            terminated = True
            collision = True
            reason = self._collision_reason(x, y)
            self.bridge.get_logger().warn(f"[COLLISION] reason={reason} x={x:.3f} y={y:.3f}")
        truncated = self.step_count >= self.max_steps

        # 報酬：距離が縮んだら＋、遠ざかったら−（簡単で効く）
        # + ゴール到達は大きく、衝突は大きくマイナス
        progress = self.prev_dist - dist
        reward = 1.0 * progress
        if success:
            reward += 5.0
        if collision:
            reward -= 5.0
        if truncated and not success:
            reward -= 1.0

        self.prev_dist = dist

        info = {"success": success, "collision": collision, "dist": dist}

        if terminated:
            self.bridge.get_logger().info(
                f"[DONE] x={x:.3f} y={y:.3f} yaw={float(obs[2]):.3f} "
                f"goal={success} collision={collision} dist={dist:.3f}"
            )

        return obs, reward, terminated, truncated, info

    def _collision_reason(self, x, y):
        if x < self.arena_min + self.wall_margin: return "wall_xmin"
        if x > self.arena_max - self.wall_margin: return "wall_xmax"
        if y < self.arena_min + self.wall_margin: return "wall_ymin"
        if y > self.arena_max - self.wall_margin: return "wall_ymax"
        if (abs(x) < (self.center_box_half + self.center_margin) and
            abs(y) < (self.center_box_half + self.center_margin)):
            return "center_box"
        return None
