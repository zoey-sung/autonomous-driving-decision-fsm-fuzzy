# perception.py
from config import SimConfig


class Perception:
    def __init__(self, env):
        self.env = env

    def get_decision_data(self):
        """获取决策引擎需要的数据"""
        ego = self.env.unwrapped.vehicle
        lane_data = {
            "current": {"dist": 100.0, "v_lead": ego.speed},
            "left": {"dist": 100.0, "v_lead": ego.speed},
            "right": {"dist": 100.0, "v_lead": ego.speed}
        }

        for v in self.env.unwrapped.road.vehicles:
            if v is not ego:
                d = v.position[0] - ego.position[0]
                v_lane = v.lane_index[2]

                if v_lane == ego.lane_index[2]:
                    if 0 < d < lane_data["current"]["dist"]:
                        lane_data["current"] = {"dist": d, "v_lead": v.speed}
                elif v_lane in [ego.lane_index[2] - 1, ego.lane_index[2] + 1]:
                    lane_key = "left" if v_lane == ego.lane_index[2] - 1 else "right"
                    if -10.0 < d <= 0:
                        lane_data[lane_key] = {"dist": 0.1, "v_lead": v.speed}
                    elif 0 < d < lane_data[lane_key]["dist"]:
                        if lane_data[lane_key]["dist"] != 0.1:
                            lane_data[lane_key] = {"dist": d, "v_lead": v.speed}

        return {
            "v_ego": ego.speed,
            "lane_data": lane_data,
            "lane": ego.lane_index[2],
            "lat_vel": ego.velocity[1],
            "ego_heading": ego.heading
        }

    def get_visual_data(self):
        """获取可视化画图需要的数据"""
        vehicles_data = []
        ego = self.env.unwrapped.vehicle
        road = self.env.unwrapped.road

        for v in road.vehicles:
            rel_x = (v.position[0] - ego.position[0]) * SimConfig.SCALING + (SimConfig.SCREEN_WIDTH / 2)

            # ==========================================
            # 【修改 3】将 Y 轴的基准从自车(ego.position[1]) 改为固定的中心点(4.0)
            # ==========================================
            rel_y = (v.position[1] - 4.0) * SimConfig.SCALING + (SimConfig.SCREEN_HEIGHT / 2)

            # ==========================================
            # 【新增】将中心距换算为真实物理净距
            center_dist = v.position[0] - ego.position[0]
            if center_dist > 0:
                gap = max(0.0, center_dist - 5.0)  # 前车尾巴 到 自车车头
            else:
                gap = min(0.0, center_dist + 5.0)  # 后车车头 到 自车车尾
            # ==========================================

            vehicles_data.append({
                "x": rel_x,
                "y": rel_y,
                "speed": v.speed * 3.6,
                "is_ego": (v == ego),
                "long_dist": gap  # 传给 UI 的直接是净距了
            })
        return vehicles_data