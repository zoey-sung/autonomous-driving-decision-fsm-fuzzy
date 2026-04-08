# config.py

class SimConfig:
    # ================= 车辆与控制参数 =================
    EGO_INIT_SPEED = 1.0  # 自车初始速度 (m/s)
    OTHER_VEHICLE_SPEED = 0.0  # 障碍车速度 (m/s)
    SAFE_DISTANCE_MSS = 2.0  # 基础防撞底线距离 (m)
    MIN_SPEED_TO_OVERTAKE = 0.5  # 允许触发超车意图的最低车速 (m/s)
    SPEED_MODE_THRESHOLD = 8.0  # 判定为“高速/低速”模式的阈值 (m/s)
    EMERGENCY_DECEL = 5.0  # 强制回撤时的制动减速度幅度 (m/s)

    # ================= 环境与道路参数 =================
    LANES_COUNT = 3  # 车道总数
    VEHICLES_COUNT = 20  # 随机车辆数
    EGO_SPACING = 2.0  # 自车初始间距
    DURATION = 1000  # 仿真最大步数
    CENTER_LANE_Y = 4.0  # 中心车道的标准 Y 坐标锚点 (m)
    LANE_WIDTH = 4.0  # 单根车道的物理宽度 (m)

    # 【新增】终端交互控制
    ENABLE_CLI_SETUP = False  # True(或1): 开启终端交互放车; False(或0): 直接运行
    DEFAULT_OBSTACLES = [(1, 30)]  # 若关闭交互，默认加载的障碍车位置 (车道号, 相对距离)

    # ================= 渲染与UI参数 =================
    SCREEN_WIDTH = 1024  # 画面宽度
    SCREEN_HEIGHT = 384  # 画面高度
    SCALING = 15  # 坐标缩放比例
    INITIAL_DELAY = 100  # 初始帧延迟 (ms)

    # ================= 状态机与决策阈值 =================
    P_SAFE_THRESHOLD = 0.5  # 允许变道的最低安全概率门限
    PLC_MIN_MSS_MULT = 1.8  # 变道跑道锁下限 (需要 1.8 倍 MSS 空间才准打方向盘)
    PLC_MAX_MSS_MULT = 4.0  # 变道跑道上限 (超过此空间自动退回保持状态)
    ABORT_EXIT_MSS_MULT = 2.0  # 危险回撤解除所需的绝对安全空间倍数
    LOW_SPEED_ABORT_MULT = 1.1  # 【低速模式】极限防撞网倍率
    HIGH_SPEED_ABORT_MULT = 1.5  # 【高速模式】防撞网倍率
    ABORT_DEADLOCK_DIST = 8.0  # 极低速下防止刹死僵局的最小安全净距 (m)

    # ================= 冷却时间设置 =================
    COOLDOWN_ABORT = 10  # 终止变道后的策略冷却帧数
    COOLDOWN_LANE_CHANGE = 15  # 成功变道后的策略冷却帧数

    # ================= 模糊逻辑参数 =================
    FUZZY_SCALE_MIN = 2.0  # 模糊余量最低尺度 (极低速下防胆怯)
    FUZZY_SCALE_RATE = 1.0  # 模糊余量随速度的增长系数 (动态扩展)
    FUZZY_REL_SPEED_SCALE = 5.0  # 相对速度模糊化基准范围规范区

    # ================= 物理与系统补偿参数 =================
    VEHICLE_LENGTH_COMP = 5.0  # 车身长度补偿（前后车中心距转为真实净距用）