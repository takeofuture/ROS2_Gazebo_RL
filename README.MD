# ROS2 JAZZY + GAZEBO Sim on Ubuntu24.04　で簡単な強化学習 
強化学習の内容は所定のスタート位置（このア場合は-2.0, -2.0）からゴール(-3.5,-3.5)に向かって障害物を避けて進むというものです。

## 事前準備とインストール
Ubuntu24.04の動いている環境でROS2を導入ください
以下のブログはRASPBERRY PI5で紹介してますが基本的な手順はPCやサーバーでも同じですので参考にしてください。
いずれROS2から直接、実機を動かすことを見越して、RASPBERRYにインストールしてます。
個々の内容はCPUだけのPCでも十分動きます。

* Ubuntu24.04のインストール

https://zenn.dev/takeofuture/articles/e58703a74dd6ad

* ROS2のインストール

https://zenn.dev/takeofuture/articles/e99cc33d4a7bc3
* Gazebo Simのインストール（これはPCで実施してます。PCはGAZEBO　SIMでRASPBERRYは実機を想定）

https://zenn.dev/takeofuture/articles/40d83d88ff6138

* 手動でシミュレーション内のロボットを動かす

https://zenn.dev/takeofuture/articles/a40714ecf8917b

## 開発環境準備
```
# apt install python3 python3-pip python3-venv
# python3 -m venv ~/pyenv_rl_demo
# source ~/pyenv_rl_demo/bin/activate
(pyenv_rl_demo)$ pip install -r requirement.txt
```

## 実験環境の構築
** ROS2の開発場所を作る
```
(pyenv_rl_demo)$ mkdir -p ~/rl_demo/rl_ws/src
(pyenv_rl_demo)$ cd ~/rl_demo/rl_ws/src
```
** Pythonパッケージ雛形を作成（ament_python）
```
ros2 pkg create tb3_rl --build-type ament_python --dependencies rclpy geometry_msgs ros_gz_interfaces
```
** 上記のPYTHONをsrc/tb3_rl/tb3_rl内に配置してください。作業スペースでGIT　CLONEしても多分大丈夫だとは思います。
```
(pyenv_rl_demo) $ tree
.
├── checkpoints
│   └── ppo_tb3_arena_9000_steps.zip
└── src
    └── tb3_rl
        ├── package.xml
        ├── resource
        │   └── tb3_rl
        ├── setup.cfg
        ├── setup.py
        ├── tb3_rl
        │   ├── env.py
        │   ├── infer.py
        │   ├── node_bridge.py
        │   ├── resume.py
        │   └── train.py
        └── test
            ├── test_copyright.py
            ├── test_flake8.py
            └── test_pep257.py
```

## 実行準備
背景や実行結果の録画などはに関しては上記の記事および以下の記事で簡単に触れてます。

https://zenn.dev/takeofuture/articles/7da22e8751886e

それでは手順にそって進めます。
* 定義した世界（環境）を立ち上げる

https://zenn.dev/takeofuture/articles/40d83d88ff6138

```
(pyenv_rl_demo) $ gz sim -r /root/rl_demo/worlds/arena.sdf &
```
そのあと、ロボットを(-2.0, -2.0）に出現させます。
```
(pyenv_rl_demo) $ ros2 run ros_gz_sim create -world arena -file /opt/ros/jazzy/share/turtlebot3_gazebo/models/turtlebot3_waffle/model.sdf -name tb3 -x -2.5 -y -2.5 -z 0.02
```
ROS2とGAZEBOが双方向でコミュニケーションをとるためのブリッジをさせます。
```
(pyenv_rl_demo) $ ros2 run ros_gz_bridge parameter_bridge /cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist &
(pyenv_rl_demo) $ ros2 run ros_gz_bridge parameter_bridge /odom@nav_msgs/msg/Odometry@gz.msgs.Odometry &
```

## PYTHONコードのコンパイルと実行
** コードコンパイル
```
(pyenv_rl_demo) $ colcon build --symlink-install
Starting >>> tb3_rl
Finished <<< tb3_rl [4.17s]
Summary: 1 package finished [4.61s]
```
** 環境読込
```
(pyenv_rl_demo) $ source install/setup.bash
```
** シンプルに実行(ロボットが少し動きます)
```
(pyenv_rl_demo) $ python -m tb3_rl.node_bridge
```
** １から学習
```
(pyenv_rl_demo) $ python -m tb3_rl.train
```
**　チェックポイントの重みから追加の学習
```
(pyenv_rl_demo) $ python -m tb3_rl.resume
```
この場合src/tb3_rl/tb3_rl/resume.pyの以下の一行をファイル名に合わせて変更して、再度BUILDと環境の読み込みを実施してください。
```
def main():
    env = DummyVecEnv([lambda: TB3ArenaEnv()])
    ckpt = "checkpoints/ppo_tb3_arena_11000_steps.zip"  # ←再開したいzip
    model = PPO.load(ckpt, env=env)
```
**　推論
```
(pyenv_rl_demo) $ python -m tb3_rl.resume
```
推論も同様で学習済で使用したい重みを指定してください
```
def main():
    env = DummyVecEnv([lambda: TB3ArenaEnv()])
    model = PPO.load("checkpoints/ppo_tb3_arena_12000_steps.zip", env=env)
    obs = env.reset()
```


