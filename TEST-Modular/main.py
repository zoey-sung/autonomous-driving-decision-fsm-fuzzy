# main.py
import os
import cv2
import gymnasium as gym
import highway_env
from ultralytics import YOLO

# 导入我们解耦的各个模块
from config import SimConfig
from scenario import ScenarioManager
from perception import Perception
from decision_engine1 import DecisionEngine
from visualizer import Visualizer

class AutoDriveSystem:
    def __init__(self):
        os.environ['SDL_VIDEO_WINDOW_POS'] = "-5000,-5000"
        self.env = gym.make("highway-v0", render_mode="rgb_array")

        # 使用 config.py 中的宏定义初始化环境
        self.env.unwrapped.configure({
            "lanes_count": SimConfig.LANES_COUNT,
            "vehicles_count": SimConfig.VEHICLES_COUNT,
            "duration": SimConfig.DURATION,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "controlled_vehicles": 1,
            "initial_lane_id": 1,
            "ego_spacing": SimConfig.EGO_SPACING,
            "screen_width": SimConfig.SCREEN_WIDTH,
            "screen_height": SimConfig.SCREEN_HEIGHT,
            "scaling": SimConfig.SCALING,
            "centering_position": [0.5, 0.5],
            "other_vehicles_speed_range": [SimConfig.OTHER_VEHICLE_SPEED, SimConfig.OTHER_VEHICLE_SPEED],
        })

        # 初始化各个子模块
        self.model = YOLO("yolov8n.pt")
        self.scenario = ScenarioManager(self.env)
        self.perception = Perception(self.env)
        self.brain = DecisionEngine(
            test_safe_dist=SimConfig.SAFE_DISTANCE_MSS,
            fuzzy_scale=SimConfig.FUZZY_MARGIN_SCALE,
            target_speed=SimConfig.EGO_INIT_SPEED
        )
        self.viz = Visualizer()

        self.frame_delay = SimConfig.INITIAL_DELAY
        self.paused = False

        # 终端交互与环境重置
        self.scenario.setup_from_cli()
        self.reset_env()

    def reset_env(self):
        """重置环境状态的辅助函数"""
        self.env.reset()
        ego = self.env.unwrapped.vehicle
        ego.speed = SimConfig.EGO_INIT_SPEED
        ego.target_speed = SimConfig.EGO_INIT_SPEED
        self.scenario.spawn_static_obstacles()
        self.brain.reset()

    def run(self):
        print("系统启动：[空格]暂停/恢复 [+]减速 [-]加速 [ESC]退出")
        print(f"=== 基础测试模式启动 ===")

        while True:
            frame = self.env.render()
            if frame is None: continue

            ego = self.env.unwrapped.vehicle
            # 锁定其他车辆速度
            for v in self.env.unwrapped.road.vehicles:
                if v is not ego:
                    v.speed = SimConfig.OTHER_VEHICLE_SPEED
                    if hasattr(v, 'target_speed'):
                        v.target_speed = SimConfig.OTHER_VEHICLE_SPEED

            # 1. 感知
            p_data = self.perception.get_decision_data()
            v_list = self.perception.get_visual_data()
            yolo_results = self.model(frame, classes=[2, 3, 5, 7], verbose=False)

            # 2. 决策
            action, mss, p_safe = self.brain.get_action(
                p_data["v_ego"], p_data["lane_data"], p_data["lane"], p_data["lat_vel"]
            )

            # 3. 可视化
            config = self.env.unwrapped.config
            lane_y_coords = [
                (i, ((i * 4) - ego.position[1]) * config["scaling"] + (config["screen_height"] / 2))
                for i in range(SimConfig.LANES_COUNT)
            ]
            canvas = self.viz.draw_info(
                frame, self.brain.current_state.value, self.frame_delay, p_data["lane_data"]["current"]["dist"],
                mss, p_safe, paused=self.paused, vehicle_list=v_list,
                lane_y_coords=lane_y_coords, ego_x=ego.position[0],
                scaling=config["scaling"], screen_width=config["screen_width"]
            )
            canvas = self.viz.draw_detections(canvas, yolo_results)
            self.viz.show(canvas)

            # 4. 键盘/鼠标交互
            if self.paused and self.viz.clicked_points:
                for px, py in self.viz.clicked_points:
                    self.scenario.spawn_at_pixel(px, py)
                self.viz.clicked_points.clear()
            elif not self.paused:
                self.viz.clicked_points.clear()

            key = cv2.waitKey(self.frame_delay) & 0xFF
            if key == 27: break
            elif key == ord(' '): self.paused = not self.paused
            elif key in [ord('='), ord('+')]: self.frame_delay = min(self.frame_delay + 20, 1000)
            elif key == ord('-'): self.frame_delay = max(self.frame_delay - 20, 10)

            # 5. 执行与防暴冲兜底 + 终极空间锁定修复幽灵倒车
            if not self.paused:
                safe_action = action
                if action == 4:
                    safe_action = 1
                    ego.target_speed = 0.0
                    ego.speed = max(0.0, ego.speed - 5.0)

                # ===================== 【终极必杀：空间锚点】 =====================
                # 步进物理引擎前，记录所有非自车的坐标
                frozen_positions = {v: (v.position[0], v.position[1])
                                    for v in self.env.unwrapped.road.vehicles if v is not ego}
                # =================================================================

                # 执行环境步进
                _, _, terminated, truncated, info = self.env.step(safe_action)

                # 自车防倒车兜底
                if ego.target_speed < 0.0: ego.target_speed = 0.0
                if ego.speed < 0.0: ego.speed = 0.0

                # ===================== 【终极必杀：坐标回溯锁死】 =====================
                # 不管引擎怎么算，强行把其他车辆拉回原位，彻底锁死不动
                for v in self.env.unwrapped.road.vehicles:
                    if v is not ego and v in frozen_positions:
                        v.position[0] = frozen_positions[v][0]  # X轴锁死
                        v.position[1] = frozen_positions[v][1]  # Y轴锁死
                        v.speed = 0.0                           # 速度归零
                        if hasattr(v, 'target_speed'):
                            v.target_speed = 0.0               # 目标速度归零
                # =====================================================================

                # 碰撞/失控判断
                if info.get("crashed") or abs(p_data["ego_heading"]) > 1.5 or terminated or truncated:
                    reason = "碰撞" if info.get("crashed") else "失控/越界"
                    print(f"\n！！！系统重置 [{reason}] ！！！")
                    print(f"快照 -> 自车速: {p_data['v_ego']:.1f}, 距离: {p_data['lane_data']['current']['dist']:.1f}, 航向角: {p_data['ego_heading']:.2f}")
                    self.reset_env()

if __name__ == "__main__":
    AutoDriveSystem().run()