# scenario.py
from config import SimConfig
from highway_env.vehicle.behavior import IDMVehicle


class ScenarioManager:
    def __init__(self, env):
        self.env = env
        self.custom_obstacles = []

    def setup_from_cli(self):
        """在终端提供交互式指南，接收用户自定义的障碍物位置"""
        print("\n" + "=" * 50)
        print("🚦 障碍车初始化设置 (交互模式)")
        print(f"自车默认位于 [1]号车道 (中间), 速度 {SimConfig.EGO_INIT_SPEED} m/s")
        print("车道编号: [0]=左侧, [1]=中间, [2]=右侧")
        print("-" * 50)

        while True:
            try:
                num_str = input("👉 请输入障碍车总数量 (0 表示空旷道路) -> ").strip()
                if not num_str: continue
                num_vehicles = int(num_str)
                if num_vehicles < 0:
                    print("❌ 数量不能为负！")
                    continue
                break
            except ValueError:
                print("❌ 输入无效！请输入整数。")

        if num_vehicles == 0:
            print("⚠️ 未设置任何障碍车。\n" + "=" * 50)
            return

        print(f"\n请依次输入这 {num_vehicles} 辆障碍车的位置 (格式: 车道号,相对距离)。")
        count = 1
        while count <= num_vehicles:
            user_in = input(f"第 {count} 辆 (例: 1,30) -> ").strip()
            try:
                lane_str, dist_str = user_in.split(',')
                lane_id, dist = int(lane_str.strip()), float(dist_str.strip())
                if lane_id not in [0, 1, 2] or dist <= 0:
                    print("❌ 车道须为0-2，距离须>0！")
                    continue
                self.custom_obstacles.append((lane_id, dist))
                print(f"   [已记录] -> 车道: {lane_id}, 前方距离: {dist}m")
                count += 1
            except ValueError:
                print("❌ 格式错误！")
        print(f"\n✅ 配置完成！启动仿真...\n" + "=" * 50)

    def spawn_static_obstacles(self):
        """根据CLI配置生成初始障碍物"""
        road = self.env.unwrapped.road
        ego = self.env.unwrapped.vehicle
        ego_x = ego.position[0]
        lane_prefix = ego.lane_index[:2]

        for lane_id, rel_x in self.custom_obstacles:
            lane_index = (*lane_prefix, lane_id)
            v = IDMVehicle.make_on_lane(
                road, lane_index, longitudinal=ego_x + rel_x, speed=SimConfig.OTHER_VEHICLE_SPEED
            )
            v.target_speed = SimConfig.OTHER_VEHICLE_SPEED
            road.vehicles.append(v)

    def spawn_at_pixel(self, px, py):
        """将鼠标像素点击转换为实车空投"""
        ego = self.env.unwrapped.vehicle
        config = self.env.unwrapped.config

        # X 轴依然跟随自车，这样可以往前/往后扔车
        world_x = (px - config["screen_width"] / 2) / config["scaling"] + ego.position[0]

        # ==========================================
        # 【修改 4】Y 轴计算基于锁死的中心车道(4.0)，不再加上 ego.position[1]
        # ==========================================
        world_y = (py - config["screen_height"] / 2) / config["scaling"] + 4.0

        lane_idx = int(round(world_y / 4.0))
        lane_idx = max(0, min(SimConfig.LANES_COUNT - 1, lane_idx))

        road = self.env.unwrapped.road
        lane_index = (*ego.lane_index[:2], lane_idx)
        v = IDMVehicle.make_on_lane(
            road, lane_index, longitudinal=world_x, speed=SimConfig.OTHER_VEHICLE_SPEED
        )
        v.target_speed = SimConfig.OTHER_VEHICLE_SPEED
        road.vehicles.append(v)

        rel_dist = world_x - ego.position[0]
        print(f"🖱️ [上帝之手] 成功在车道 {lane_idx}, 前方 {rel_dist:.1f}m 处空投了一辆障碍车！")