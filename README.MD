# ROS2 JAZZY + GAZEBO Sim on Ubuntu24.04　で簡単な強化学習 

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
# source ~/pyenv_rl_demo
(pyenv_rl_demo)$ pip install -r requirement.txt
```

ROS2の開発場所を作る
```
(pyenv_rl_demo)$ mkdir -p ~/rl_demo/rl_ws/src
(pyenv_rl_demo)$ cd ~/rl_demo/rl_ws/src
```
Pythonパッケージ雛形を作成（ament_python）
```
ros2 pkg create tb3_rl --build-type ament_python --dependencies rclpy geometry_msgs ros_gz_interfaces
```
