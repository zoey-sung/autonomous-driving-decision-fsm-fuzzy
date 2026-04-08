import os
import gymnasium as gym
import highway_env
from ultralytics import YOLO
import cv2
from decision_engine import DecisionEngine
from visualizer import Visualizer


class AutoDriveSystem:
    def __init__(self):
        # 初始化环境
        os.environ['SDL_VIDEO_WINDOW_POS'] = "-5000,-5000"
        self.env = gym.make("highway-v0", render_mode="rgb_array")
        self.env.unwrapped.configure({
            "lanes_count": 3,
            "vehicles_count": 25,
            "duration": 1000,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "controlled_vehicles": 1,
            "initial_lane_id": 1,
            "ego_spacing": 2,
            "screen_width": 1024,
            "screen_height": 384,
            "scaling": 15,
            "centering_position": [0.5, 0.5],
            "other_vehicles_speed_range": [18, 22],
        })

        self.model = YOLO("yolov8n.pt")
        self.brain = DecisionEngine()
        self.viz = Visualizer()

        self.frame_delay = 400
        self.paused = False
        self.obs, _ = self.env.reset()

    def get_perception_data(self):
        """从环境中提取感知数据，修复侧方盲区问题"""
        ego = self.env.unwrapped.vehicle
        lateral_velocity = ego.velocity[1]
        ego_lane = ego.lane_index[2]

        lane_data = {
            "current": {"dist": 100.0, "v_lead": ego.speed},
            "left": {"dist": 100.0, "v_lead": ego.speed},
            "right": {"dist": 100.0, "v_lead": ego.speed}
        }

        for v in self.env.unwrapped.road.vehicles:
            if v is not ego:
                d = v.position[0] - ego.position[0]
                v_lane = v.lane_index[2]

                # 1. 检测当前车道：依然只关心正前方的车
                if v_lane == ego_lane:
                    if 0 < d < lane_data["current"]["dist"]:
                        lane_data["current"] = {"dist": d, "v_lead": v.speed}

                # 2. 检测相邻车道（左侧或右侧）
                elif v_lane in [ego_lane - 1, ego_lane + 1]:
                    lane_key = "left" if v_lane == ego_lane - 1 else "right"

                    # 【核心修复】如果目标车辆在侧方盲区（例如落后10米到齐头并进）
                    if -10.0 < d <= 0:
                        # 强制把距离设为一个极小值（比如0.1米），警告决策层极度危险！
                        lane_data[lane_key] = {"dist": 0.1, "v_lead": v.speed}

                    # 只有在侧方没有车平齐的情况下，才去关心前方更远的车
                    elif 0 < d < lane_data[lane_key]["dist"]:
                        if lane_data[lane_key]["dist"] != 0.1:  # 防止覆盖盲区报警
                            lane_data[lane_key] = {"dist": d, "v_lead": v.speed}

        return {
            "v_ego": ego.speed,
            "lane_data": lane_data,
            "lane": ego_lane,
            "lat_vel": lateral_velocity,
            "ego_heading": ego.heading
        }

    def get_all_vehicles_data(self):
        vehicles_data = []
        ego = self.env.unwrapped.vehicle
        road = self.env.unwrapped.road
        config = self.env.unwrapped.config
        scaling = config["scaling"]
        center_x = ego.position[0]

        for v in road.vehicles:
            rel_x = (v.position[0] - center_x) * scaling + (config["screen_width"] / 2)
            rel_y = v.position[1] * scaling + (config["screen_height"] / 2)
            vehicles_data.append({
                "x": rel_x,
                "y": rel_y,
                "speed": v.speed * 3.6,
                "is_ego": (v == ego)
            })
        return vehicles_data

    def run(self):
        print("系统启动：[空格]暂停/恢复 [+]减速 [-]加速 [ESC]退出")
        print("系统启动：自车将具备车道探测能力的自动超车系统")

        VIEW_RANGE = 60

        while True:
            frame = self.env.render()
            if frame is None: continue

            ego = self.env.unwrapped.vehicle
            if ego.position[0] > 500:
                offset = 80
                for v in self.env.unwrapped.road.vehicles:
                    v.position[0] -= offset
                print(">>> 循环重置：回到入画点")

            v_list = self.get_all_vehicles_data()
            p_data = self.get_perception_data()
            yolo_results = self.model(frame, classes=[2, 3, 5, 7], verbose=False)

            # 将封装好的 lane_data 传给决策引擎
            action, mss, p_safe = self.brain.get_action(
                p_data["v_ego"],
                p_data["lane_data"],
                p_data["lane"],
                p_data["lat_vel"]
            )

            state_str = self.brain.current_state.value

            # 从字典中提取当前车道的 dist 传递给 UI 渲染
            current_dist = p_data["lane_data"]["current"]["dist"]

            canvas = self.viz.draw_info(
                frame,
                state_str,
                self.frame_delay,
                dist=current_dist,
                mss=mss,
                p_safe=p_safe,
                paused=self.paused,
                vehicle_list=v_list
            )
            canvas = self.viz.draw_detections(canvas, yolo_results)
            self.viz.show(canvas)

            key = cv2.waitKey(self.frame_delay) & 0xFF
            if key == 27:
                break
            elif key == ord(' '):
                self.paused = not self.paused
                print(f"系统{'暂停' if self.paused else '恢复'}")
            elif key in [ord('='), ord('+')]:
                self.frame_delay = min(self.frame_delay + 20, 1000)
                print(f"画面减速，当前延迟：{self.frame_delay}ms")
            elif key == ord('-'):
                self.frame_delay = max(self.frame_delay - 20, 10)
                print(f"画面加速，当前延迟：{self.frame_delay}ms")

            if not self.paused:
                _, _, terminated, truncated, info = self.env.step(action)

                if info.get("crashed") or abs(p_data["ego_heading"]) > 1.5 or terminated or truncated:
                    reason = "碰撞" if info.get("crashed") else "失控/结束"
                    v_lead = p_data['lane_data']['current']['v_lead']
                    dist = p_data['lane_data']['current']['dist']
                    print(f"\n！！！系统重置 [%s] ！！！" % reason)
                    print(f"最后快照 -> 自车速: {p_data['v_ego']:.1f}, 前车速: {v_lead:.1f}, "
                          f"距离: {dist:.1f}, 航向角: {p_data['ego_heading']:.2f}")

                    self.env.reset()
                    self.brain.reset()

        self.viz.close()
        self.env.close()


if __name__ == "__main__":
    AutoDriveSystem().run()