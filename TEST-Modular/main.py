# main.py
import os
import cv2
import numpy as np  # 新增：用于矩阵变换
import gymnasium as gym
import highway_env
from ultralytics import YOLO

# 导入我们解耦的各个模块
from config import SimConfig
from scenario import ScenarioManager
from perception import Perception
from decision_engine1 import DecisionEngine
from visualizer import Visualizer
# main.py 的开头导入部分
from decision_engine1 import DecisionEngine, DriveState  # 确保加上 DriveState

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

            config = self.env.unwrapped.config
            ego = self.env.unwrapped.vehicle

            # ==========================================
            # 【修改 1】画面防抖：抵消自车变道带来的 Y 轴平移
            # 中间车道(lane 1)的绝对Y坐标是 4.0。通过平移让画面永远锁定在 4.0
            # ==========================================
            dy_m = ego.position[1] - 4.0
            dy_px = int(dy_m * config["scaling"])
            if dy_px != 0:
                M = np.float32([[1, 0, 0], [0, 1, dy_px]])
                # 采样左上角像素（通常是草地背景色）来填充平移后产生的黑边
                bg_color = [int(x) for x in frame[0, 0]]
                frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), borderValue=bg_color)

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
            # ==========================================
            # 【修改 2】画车道号时，Y轴基准也必须锁死在 4.0，而不是 ego.position[1]
            # ==========================================
            lane_y_coords = [
                (i, ((i * 4) - 4.0) * config["scaling"] + (config["screen_height"] / 2))
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

            # ==========================================
            # 【修改 3】监听 OpenCV Monitor 窗口的右上角 X 按钮
            # ==========================================
            if cv2.getWindowProperty(self.viz.window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("\n[系统] Monitor 窗口已被关闭，退出程序。")
                break

            # 4. 键盘/鼠标交互
            if self.paused and self.viz.clicked_points:
                for px, py in self.viz.clicked_points:
                    self.scenario.spawn_at_pixel(px, py)
                self.viz.clicked_points.clear()
            elif not self.paused:
                self.viz.clicked_points.clear()

            key = cv2.waitKey(self.frame_delay) & 0xFF
            if key == 27:
                break
            elif key == ord(' '):
                self.paused = not self.paused
            elif key in [ord('='), ord('+')]:
                self.frame_delay = min(self.frame_delay + 20, 1000)
            elif key == ord('-'):
                self.frame_delay = max(self.frame_delay - 20, 10)
            # ==========================================
            # 【修改 4】新增 F 键，强制解除 ABORT 状态并起步
            # ==========================================
            elif key == ord('f') or key == ord('F'):
                self.brain.current_state = DriveState.KL
                ego.speed = SimConfig.EGO_INIT_SPEED
                ego.target_speed = SimConfig.EGO_INIT_SPEED
                print("\n🚀 [手动干预] 已强制解除 ABORT 状态，注入初始速度重新起步！")

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
                    # 【修改】这里也减去 5.0 显示碰撞瞬间的净距
                    crash_gap = max(0.0, p_data['lane_data']['current']['dist'] - 5.0)
                    print(f"快照 -> 自车速: {p_data['v_ego']:.1f}, 净距: {crash_gap:.1f}m, 航向角: {p_data['ego_heading']:.2f}")
                    self.reset_env()

if __name__ == "__main__":
    AutoDriveSystem().run()