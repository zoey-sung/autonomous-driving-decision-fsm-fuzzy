# config.py

class SimConfig:
    # ================= 车辆与控制参数 =================
    EGO_INIT_SPEED = 1.0        # 自车初始速度 (m/s) - 提高到5.0以支持正常变道
    OTHER_VEHICLE_SPEED = 0.0   # 障碍车速度 (m/s)
    SAFE_DISTANCE_MSS = 2.0     # 基础防撞底线距离 (m)
    FUZZY_MARGIN_SCALE = 5.0    # 模糊逻辑余量尺度

    # ================= 环境与道路参数 =================
    LANES_COUNT = 3             # 车道总数
    VEHICLES_COUNT = 20          # 随机车辆数 (测试模式下彻底关闭)
    EGO_SPACING = 2.0           # 自车初始间距
    DURATION = 1000             # 仿真最大步数

    # ================= 渲染与UI参数 =================
    SCREEN_WIDTH = 1024         # 画面宽度
    SCREEN_HEIGHT = 384         # 画面高度
    SCALING = 15                # 坐标缩放比例
    INITIAL_DELAY = 100         # 初始帧延迟 (ms)