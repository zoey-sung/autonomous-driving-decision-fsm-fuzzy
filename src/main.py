# main.py
import os
import cv2
import numpy as np
import gymnasium as gym
import highway_env
from ultralytics import YOLO

from config import SimConfig
from scenario import ScenarioManager
from perception import Perception
from decision_engine1 import DecisionEngine, DriveState
from visualizer import Visualizer

class AutoDriveSystem:
    def __init__(self):
        os.environ['SDL_VIDEO_WINDOW_POS'] = "-5000,-5000"
        self.env = gym.make("highway-v0", render_mode="rgb_array")

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

        self.model = YOLO("yolov8n.pt")
        self.scenario = ScenarioManager(self.env)
        self.perception = Perception(self.env)
        self.brain = DecisionEngine(target_speed=SimConfig.EGO_INIT_SPEED)
        self.viz = Visualizer()

        self.frame_delay = SimConfig.INITIAL_DELAY
        self.paused = False

        self.scenario.setup_from_cli()
        self.reset_env()

    def reset_env(self):
        self.env.reset()
        ego = self.env.unwrapped.vehicle
        ego.speed = SimConfig.EGO_INIT_SPEED
        ego.target_speed = SimConfig.EGO_INIT_SPEED
        self.scenario.spawn_static_obstacles()
        self.brain.reset()

    def run(self):
        print("系统启动：[空格]暂停/恢复 [+]减速 [-]加速 [ESC]退出 [F]脱困起步")
        print(f"=== 基础测试模式启动 ===")

        while True:
            frame = self.env.render()
            if frame is None: continue

            config = self.env.unwrapped.config
            ego = self.env.unwrapped.vehicle

            # 画面防抖锁死 Y 轴中心
            dy_m = ego.position[1] - SimConfig.CENTER_LANE_Y
            dy_px = int(dy_m * config["scaling"])
            if dy_px != 0:
                M = np.float32([[1, 0, 0], [0, 1, dy_px]])
                bg_color = [int(x) for x in frame[0, 0]]
                frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), borderValue=bg_color)

                # ==========================================
                # 【修复 1 & 2 预处理】强行注入速度，并强制车头回正
                # ==========================================
                for v in self.env.unwrapped.road.vehicles:
                    if v is not ego:
                        v.speed = SimConfig.OTHER_VEHICLE_SPEED
                        v.heading = 0.0  # 绝对笔直，不允许歪头
                        if hasattr(v, 'target_speed'):
                            v.target_speed = SimConfig.OTHER_VEHICLE_SPEED

                        # 👇👇👇 [新增] 彻底抹除障碍车的变道意图 👇👇👇
                        if hasattr(v, 'target_lane_index'):
                            v.target_lane_index = v.lane_index

            p_data = self.perception.get_decision_data()
            v_list = self.perception.get_visual_data()
            yolo_results = self.model(frame, classes=[2, 3, 5, 7], verbose=False)

            action, mss, p_safe = self.brain.get_action(
                p_data["v_ego"], p_data["lane_data"], p_data["lane"], p_data["lat_vel"]
            )

            # 绘制 UI 画布基准线
            lane_y_coords = [
                (i, ((i * SimConfig.LANE_WIDTH) - SimConfig.CENTER_LANE_Y) * config["scaling"] + (config["screen_height"] / 2))
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

            if cv2.getWindowProperty(self.viz.window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("\n[系统] Monitor 窗口已被关闭，退出程序。")
                break

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
            elif key == ord('f') or key == ord('F'):
                print("\n🔄 [手动干预] 触发全局重置！画面已焕然一新！")
                self.reset_env()  # 调用你已经写好的重置函数
                continue  # 关键：跳过本帧剩余的渲染和物理运算，直接进入全新的下一帧

            # 5. 执行与防暴冲兜底
            if not self.paused:
                # ===================== 【核心修复：彻底接管底层油门】 =====================
                # 剥夺 highway-env 对加减速的控制权。
                # 如果决策是加速(3)或刹车(4)，对环境只发送 1(巡航)，加减速由我们手动接管
                safe_action = action
                if action in [3, 4]:
                    safe_action = 1

                frozen_y_positions = {v: v.position[1] for v in self.env.unwrapped.road.vehicles if v is not ego}

                # 🚨 必须放在 env.step 之前！提前改写属性，让引擎用新速度去算物理位移 🚨
                # ===================== 【自车速度强制覆写】 =====================
                if action == 4:
                    # 刹车状态：平滑减速，防止被后车追尾（不再使用瞬间 -5.0 的死亡刹车）
                    ego.target_speed = 0.0
                    ego.speed = max(0.0, ego.speed - 1.0)
                else:
                    # 巡航、加速或变道状态：强行把引擎想要飙升的速度拉回到 Config 设定值
                    ego.target_speed = SimConfig.EGO_INIT_SPEED
                    if ego.speed < SimConfig.EGO_INIT_SPEED:
                        # 如果之前刹车了，现在平滑起步恢复速度
                        ego.speed = min(SimConfig.EGO_INIT_SPEED, ego.speed + 0.5)
                    elif ego.speed > SimConfig.EGO_INIT_SPEED:
                        # 强行压制底层引擎的暴冲
                        ego.speed = SimConfig.EGO_INIT_SPEED

                # 兜底
                if ego.target_speed < 0.0: ego.target_speed = 0.0
                if ego.speed < 0.0: ego.speed = 0.0

                # 🏁 现在再让引擎步进（此时引擎会使用上面设定的 target_speed = 0.0 来演算）
                _, _, terminated, truncated, info = self.env.step(safe_action)

                # ===================== 【其他车辆坐标与速度锁死】 =====================
                for v in self.env.unwrapped.road.vehicles:
                    if v is not ego and v in frozen_y_positions:
                        v.position[1] = frozen_y_positions[v]  # 锁死Y轴防变道
                        v.heading = 0.0  # 锁死航向角防歪头杀
                        # 移除了强行设置v.speed的代码，保留target_speed让IDM模型控制实际速度
                        if hasattr(v, 'target_speed'):
                            v.target_speed = SimConfig.OTHER_VEHICLE_SPEED
                # =====================================================================

                if info.get("crashed") or abs(p_data["ego_heading"]) > 1.5 or terminated or truncated:
                    reason = "碰撞" if info.get("crashed") else "失控/越界"
                    print(f"\n！！！系统重置 [{reason}] ！！！")
                    crash_gap = max(0.0, p_data['lane_data']['current']['dist'] - SimConfig.VEHICLE_LENGTH_COMP)
                    print(
                        f"快照 -> 自车速: {p_data['v_ego']:.1f}, 净距: {crash_gap:.1f}m, 航向角: {p_data['ego_heading']:.2f}")
                    self.reset_env()

if __name__ == "__main__":
    AutoDriveSystem().run()