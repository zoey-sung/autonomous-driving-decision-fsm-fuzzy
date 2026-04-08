import os
import gymnasium as gym
import highway_env
from ultralytics import YOLO
import cv2
from decision_engine1 import DecisionEngine
from visualizer1 import Visualizer

TEST_CONFIG = {
    "is_test_mode": True,
    "ego_init_speed": 1.0,
    "other_vehicle_speed": 0,
    "safe_distance_mss": 6.0,  # 同步修改为7.0，匹配决策引擎的默认值
    "fuzzy_margin_scale": 5.0,
    "vehicles_count": 0,  # 【修复】真正改为 0，彻底关闭底层随机刷车机制
    "ego_spacing": 2.0,
    "lanes_count": 3
}


class AutoDriveSystem:
    def __init__(self):
        os.environ['SDL_VIDEO_WINDOW_POS'] = "-5000,-5000"
        self.env = gym.make("highway-v0", render_mode="rgb_array")

        self.env.unwrapped.configure({
            "lanes_count": TEST_CONFIG["lanes_count"],
            "vehicles_count": TEST_CONFIG["vehicles_count"],
            "duration": 1000,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "controlled_vehicles": 1,
            "initial_lane_id": 1,
            "ego_spacing": TEST_CONFIG["ego_spacing"],
            "screen_width": 1024,
            "screen_height": 384,
            "scaling": 15,
            "centering_position": [0.5, 0.5],
            "other_vehicles_speed_range": [TEST_CONFIG["other_vehicle_speed"], TEST_CONFIG["other_vehicle_speed"]],
        })

        self.model = YOLO("yolov8n.pt")
        self.brain = DecisionEngine(
            test_safe_dist=TEST_CONFIG["safe_distance_mss"],
            fuzzy_scale=TEST_CONFIG["fuzzy_margin_scale"],
            target_speed=TEST_CONFIG["ego_init_speed"]
        )
        self.viz = Visualizer()

        self.frame_delay = 100
        self.paused = False

        # 【新增】调用终端交互逻辑，记录用户自定义的障碍物坐标
        self.setup_custom_obstacles()

        self.obs, _ = self.env.reset()
        self.env.unwrapped.vehicle.speed = TEST_CONFIG["ego_init_speed"]
        self.env.unwrapped.vehicle.target_speed = TEST_CONFIG["ego_init_speed"]  # 【关键新增】重置底层控制器的目标速度
        self.spawn_static_obstacles()  # 【新增】初始启动时摆放障碍车

    def setup_custom_obstacles(self):
        """在终端提供交互式指南，接收用户自定义的障碍物位置"""
        self.custom_obstacles = []

        print("\n" + "=" * 50)
        print("🚦 障碍车初始化设置 (交互模式)")
        print(f"自车 (Ego) 初始状态: 默认位于 [1]号车道 (中间), 速度 {TEST_CONFIG['ego_init_speed']} m/s")
        print("车道编号说明:")
        print("  [0] = 左侧车道")
        print("  [1] = 中间车道")
        print("  [2] = 右侧车道")
        print("-" * 50)

        # 1. 先询问需要生成几辆障碍车
        while True:
            try:
                num_str = input("👉 请先输入你要设置的障碍车总数量 (输入 0 表示空旷道路) -> ").strip()
                if not num_str:
                    continue
                num_vehicles = int(num_str)
                if num_vehicles < 0:
                    print("❌ 数量不能为负数，请重新输入！")
                    continue
                break
            except ValueError:
                print("❌ 输入无效！请输入一个整数。")

        if num_vehicles == 0:
            print("⚠️ 未设置任何障碍车，将生成空旷道路。\n" + "=" * 50)
            return

        print(f"\n好的，接下来请依次输入这 {num_vehicles} 辆障碍车的位置。")
        print("格式为：车道号,相对距离 (例如输入 '1,30' 表示在1号车道，前方30米)")
        print("-" * 50)

        # 2. 根据指定的数量，使用精确计数的循环
        count = 1
        while count <= num_vehicles:
            user_in = input(f"请输入第 {count} 辆障碍车 (格式: 车道号,距离) -> ").strip()

            try:
                lane_str, dist_str = user_in.split(',')
                lane_id = int(lane_str.strip())
                dist = float(dist_str.strip())

                if lane_id not in [0, 1, 2]:
                    print("❌ 错误：车道编号必须是 0, 1 或 2！")
                    continue
                if dist <= 0:
                    print("❌ 错误：相对距离必须大于 0！")
                    continue

                self.custom_obstacles.append((lane_id, dist))
                print(f"   [已记录] -> 车道: {lane_id}, 前方距离: {dist} 米")
                count += 1

            except ValueError:
                print("❌ 格式错误！请确保输入形如 '1,30'，并用英文逗号分隔。")

        print(f"\n✅ 配置完成！已成功设置 {num_vehicles} 辆障碍车。正在启动仿真...")
        print("=" * 50 + "\n")

    def spawn_static_obstacles(self):
        from highway_env.vehicle.behavior import IDMVehicle
        road = self.env.unwrapped.road
        ego = self.env.unwrapped.vehicle  # 获取自车对象
        ego_x = ego.position[0]  # 获取自车当前的纵向绝对坐标 (即沿车道的直线位置)

        lane_prefix = ego.lane_index[:2]

        # 【关键修改】不再使用硬编码，直接遍历用户刚才通过键盘输入的列表
        for lane_id, rel_x in self.custom_obstacles:
            lane_index = (*lane_prefix, lane_id)
            absolute_x = ego_x + rel_x  # 【关键】：自车位置 + 相对直线距离 = 生成位置

            v = IDMVehicle.make_on_lane(
                road,
                lane_index,
                longitudinal=absolute_x,  # 喂给环境的是计算后的绝对坐标
                speed=TEST_CONFIG["other_vehicle_speed"]
            )
            v.target_speed = TEST_CONFIG["other_vehicle_speed"]
            road.vehicles.append(v)

    def get_perception_data(self):
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

                if v_lane == ego_lane:
                    if 0 < d < lane_data["current"]["dist"]:
                        lane_data["current"] = {"dist": d, "v_lead": v.speed}
                elif v_lane in [ego_lane - 1, ego_lane + 1]:
                    lane_key = "left" if v_lane == ego_lane - 1 else "right"
                    if -10.0 < d <= 0:
                        lane_data[lane_key] = {"dist": 0.1, "v_lead": v.speed}
                    elif 0 < d < lane_data[lane_key]["dist"]:
                        if lane_data[lane_key]["dist"] != 0.1:
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
        center_y = ego.position[1]  # 【新增】获取自车的实时 Y 坐标

        for v in road.vehicles:
            rel_x = (v.position[0] - center_x) * scaling + (config["screen_width"] / 2)

            # 【关键修复】文字的 Y 坐标减去自车Y坐标，保持相对静止
            rel_y = (v.position[1] - center_y) * scaling + (config["screen_height"] / 2)

            # 【新增】明确计算纯纵向距离（X轴方向的距离）
            longitudinal_dist = v.position[0] - ego.position[0]

            vehicles_data.append({
                "x": rel_x,
                "y": rel_y,
                "speed": v.speed * 3.6,
                "is_ego": (v == ego),
                "long_dist": longitudinal_dist  # 【新增】将纵距传入可视化字典
            })
        return vehicles_data

    # 【新增方法】把鼠标点击的像素点转换成环境里的实车
    def _spawn_vehicle_at_pixel(self, px, py):
        ego = self.env.unwrapped.vehicle
        config = self.env.unwrapped.config
        scaling = config["scaling"]
        center_x = ego.position[0]
        center_y = ego.position[1]

        # 1. 像素坐标 (px, py) 逆推算回真实世界坐标 (X, Y)
        world_x = (px - config["screen_width"] / 2) / scaling + center_x
        world_y = (py - config["screen_height"] / 2) / scaling + center_y

        # 2. 根据 Y 坐标计算属于哪条车道
        # highway-env 的车道中心线通常位于 y = 0, 4, 8 ... (间隔4米)
        lane_idx = int(round(world_y / 4.0))
        # 夹紧在合法车道 0~2 之间，防止点到马路外面报错
        lane_idx = max(0, min(TEST_CONFIG["lanes_count"] - 1, lane_idx))

        from highway_env.vehicle.behavior import IDMVehicle
        road = self.env.unwrapped.road
        lane_index = (*ego.lane_index[:2], lane_idx)

        # 3. 现场创造一辆车塞进系统里
        v = IDMVehicle.make_on_lane(
            road,
            lane_index,
            longitudinal=world_x,
            speed=TEST_CONFIG["other_vehicle_speed"]
        )
        v.target_speed = TEST_CONFIG["other_vehicle_speed"]
        road.vehicles.append(v)

        # 终端也给个反馈
        rel_dist = world_x - ego.position[0]
        print(f"🖱️ [上帝之手] 成功在车道 {lane_idx}, 前方 {rel_dist:.1f}m 处空投了一辆障碍车！")

    def run(self):
        print("系统启动：[空格]暂停/恢复 [+]减速 [-]加速 [ESC]退出")
        print(f"=== 基础测试模式启动 ===")

        while True:
            frame = self.env.render()
            if frame is None: continue

            ego = self.env.unwrapped.vehicle

            for v in self.env.unwrapped.road.vehicles:
                if v is not ego:
                    v.speed = TEST_CONFIG["other_vehicle_speed"]
                    if hasattr(v, 'target_speed'):
                        v.target_speed = TEST_CONFIG["other_vehicle_speed"]

            v_list = self.get_all_vehicles_data()
            p_data = self.get_perception_data()
            yolo_results = self.model(frame, classes=[2, 3, 5, 7], verbose=False)

            action, mss, p_safe = self.brain.get_action(
                p_data["v_ego"],
                p_data["lane_data"],
                p_data["lane"],
                p_data["lat_vel"]
            )

            state_str = self.brain.current_state.value
            current_dist = p_data["lane_data"]["current"]["dist"]

            config = self.env.unwrapped.config
            lane_y_coords = [
                (i, ((i * 4) - ego.position[1]) * config["scaling"] + (config["screen_height"] / 2))
                for i in range(TEST_CONFIG["lanes_count"])
            ]

            # 【修改】调用绘图时，多传三个参数以支持画刻度尺和鼠标提示
            canvas = self.viz.draw_info(
                frame, state_str, self.frame_delay, dist=current_dist,
                mss=mss, p_safe=p_safe, paused=self.paused, vehicle_list=v_list,
                lane_y_coords=lane_y_coords,
                ego_x=ego.position[0], scaling=config["scaling"], screen_width=config["screen_width"]
            )
            canvas = self.viz.draw_detections(canvas, yolo_results)
            self.viz.show(canvas)

            # ==========================================
            # 【新增】鼠标点击空投车辆侦测逻辑
            # ==========================================
            if self.paused and self.viz.clicked_points:
                for px, py in self.viz.clicked_points:
                    self._spawn_vehicle_at_pixel(px, py)
                self.viz.clicked_points.clear()  # 消费完毕清空列表
            elif not self.paused:
                self.viz.clicked_points.clear()  # 没暂停的时候乱点直接清空，不生成
            # ==========================================

            key = cv2.waitKey(self.frame_delay) & 0xFF
            if key == 27:
                break
            elif key == ord(' '):
                self.paused = not self.paused
            elif key in [ord('='), ord('+')]:
                self.frame_delay = min(self.frame_delay + 20, 1000)
            elif key == ord('-'):
                self.frame_delay = max(self.frame_delay - 20, 10)

            if not self.paused:
                # ==========================================
                # 【终极底层拦截】防暴冲保护网
                # ==========================================
                safe_action = action
                if action == 4:
                    safe_action = 1
                    # 【修复】允许彻底刹停，避免幽灵蠕行
                    ego.target_speed = 0.0
                    ego.speed = max(0.0, ego.speed - 5.0)

                _, _, terminated, truncated, info = self.env.step(safe_action)

                # 底层兜底逻辑也改为 0.0
                if ego.target_speed < 0.0:
                    ego.target_speed = 0.0
                if ego.speed < 0.0:
                    ego.speed = 0.0

                if info.get("crashed") or abs(p_data["ego_heading"]) > 1.5 or terminated or truncated:
                    reason = "碰撞" if info.get("crashed") else "失控/越界"
                    v_lead = p_data['lane_data']['current']['v_lead']
                    dist = p_data['lane_data']['current']['dist']
                    print(f"\n！！！系统重置 [%s] ！！！" % reason)
                    print(
                        f"快照 -> 自车速: {p_data['v_ego']:.1f}, 距离: {dist:.1f}, 航向角: {p_data['ego_heading']:.2f}")

                    self.env.reset()
                    self.env.unwrapped.vehicle.speed = TEST_CONFIG["ego_init_speed"]
                    self.env.unwrapped.vehicle.target_speed = TEST_CONFIG["ego_init_speed"]
                    self.spawn_static_obstacles()
                    self.brain.reset()


if __name__ == "__main__":
    AutoDriveSystem().run()