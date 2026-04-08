from enum import Enum


class DriveState(Enum):
    KL = "车道保持"
    PLC = "准备超车"
    LCL = "左侧超车中"
    LCR = "右侧超车中"
    ABORT = "终止并回撤"


class DecisionEngine:
    def __init__(self, test_safe_dist=7.0, fuzzy_scale=5.0, target_speed=10.0):
        self.current_state = DriveState.KL
        self.MAX_DECEL = 8.5
        self.SAFE_TIME_GAP = 0.8
        self.P_SAFE_THRESHOLD = 0.5
        self.start_lane = None
        self.cooldown = 0

        self.test_safe_dist = test_safe_dist
        self.fuzzy_scale = fuzzy_scale
        self.target_speed = target_speed
        # 【新增】变道动作锁，防止连续发送变道指令导致车辆失控打转
        self.lane_change_initiated = False

    def calculate_mss(self, v_ego, v_lead):
        # 【关键修改】动态安全距离：基础距离 + 速度 * 反应时间(比如 2.0 秒)
        # 这样当车速是 20m/s 时，安全距离会自动扩展到 45 米，给足变道时间
        dynamic_mss = self.test_safe_dist + (v_ego * 2.0)
        return dynamic_mss

    def _fuzzify_margin(self, margin):
        scale = self.fuzzy_scale
        danger = max(0.0, min(1.0, (scale - margin) / scale)) if margin < scale else 0.0
        safe = max(0.0, min(1.0, (margin - scale) / scale)) if margin > scale else 0.0
        warning = max(0.0, min(1.0, 1.0 - abs(margin - scale) / scale))
        return danger, warning, safe

    def _fuzzify_rel_speed(self, v_ego, v_lead):
        delta_v = v_ego - v_lead
        closing = max(0.0, min(1.0, delta_v / 5.0)) if delta_v > 0 else 0.0
        separating = max(0.0, min(1.0, -delta_v / 5.0)) if delta_v < 0 else 0.0
        stable = max(0.0, min(1.0, 1.0 - abs(delta_v) / 5.0))
        return closing, stable, separating

    def calculate_fuzzy_p_safe(self, current_dist, mss, v_ego, v_lead):
        margin = current_dist - mss
        if margin <= 0:
            return 0.0

        d_danger, d_warn, d_safe = self._fuzzify_margin(margin)
        v_close, v_stable, v_sep = self._fuzzify_rel_speed(v_ego, v_lead)

        rule_high_safe = d_safe
        rule_med_safe = min(d_warn, max(v_stable, v_sep))
        rule_low_safe = max(d_danger, min(d_warn, v_close))

        numerator = rule_high_safe * 1.0 + rule_med_safe * 0.6 + rule_low_safe * 0.1
        denominator = rule_high_safe + rule_med_safe + rule_low_safe + 1e-5

        return round(numerator / denominator, 2)

    def get_action(self, v_ego, lane_data, current_lane, lateral_vel):
        # 【新增1】在最开头记录这一帧开始时的状态
        previous_state = self.current_state

        current_dist = lane_data["current"]["dist"]
        v_lead = lane_data["current"]["v_lead"]

        mss = self.calculate_mss(v_ego, v_lead)
        p_safe = self.calculate_fuzzy_p_safe(current_dist, mss, v_ego, v_lead)

        if self.cooldown > 0:
            self.cooldown -= 1

        # 基础巡航与紧急减速指令
        if current_dist < mss * 1.2:
            final_action = 4
        elif v_ego < self.target_speed - 1.0:
            final_action = 3
        elif v_ego > self.target_speed + 1.0:
            final_action = 4
        else:
            final_action = 1

        # ================= 状态机逻辑 =================
        if self.current_state == DriveState.KL:
            # 【修复：增加进入超车状态的门槛】
            # 1. 距离太近（< mss * 1.8）不许想超车，安心刹车即可
            # 2. 车速太慢（< 0.5）不许想超车，先起步再说
            if mss * 1.8 < current_dist < mss * 4.0 and v_ego > 0.5 and self.cooldown == 0:
                self.current_state = DriveState.PLC

        elif self.current_state == DriveState.PLC:
            self.start_lane = current_lane
            best_lane_action = None
            best_p_safe = -1.0
            best_dist = -1.0

            current_mss = self.calculate_mss(v_ego, lane_data["current"]["v_lead"])
            current_p_safe = self.calculate_fuzzy_p_safe(current_dist, current_mss, v_ego,
                                                         lane_data["current"]["v_lead"])

            # ==========================================
            # 【修复1：变道跑道锁】
            # 汽车变道需要前方的物理空间（跑道）。
            # 设定：至少需要 1.8 倍的 MSS 距离，才允许打方向盘变道。
            # 否则，强行变道一定会蹭到前车屁股。
            # ==========================================
            if current_dist > mss * 1.8:
                # 1. 评估左侧车道
                if current_lane > 0:
                    left_dist = lane_data["left"]["dist"]
                    left_mss = self.calculate_mss(v_ego, lane_data["left"]["v_lead"])
                    left_p_safe = self.calculate_fuzzy_p_safe(left_dist, left_mss, v_ego, lane_data["left"]["v_lead"])

                    if left_p_safe > self.P_SAFE_THRESHOLD and (
                            left_p_safe > current_p_safe or left_dist > current_dist + 5.0):
                        best_p_safe = left_p_safe
                        best_dist = left_dist
                        best_lane_action = DriveState.LCL

                # 2. 评估右侧车道
                if current_lane < 2:
                    right_dist = lane_data["right"]["dist"]
                    right_mss = self.calculate_mss(v_ego, lane_data["right"]["v_lead"])
                    right_p_safe = self.calculate_fuzzy_p_safe(right_dist, right_mss, v_ego,
                                                               lane_data["right"]["v_lead"])

                    if right_p_safe > self.P_SAFE_THRESHOLD and (
                            right_p_safe > current_p_safe or right_dist > current_dist + 5.0):
                        if right_p_safe > best_p_safe or (right_p_safe == best_p_safe and right_dist > best_dist):
                            best_p_safe = right_p_safe
                            best_dist = right_dist
                            best_lane_action = DriveState.LCR

            # 3. 结果应用
            if best_lane_action:
                self.current_state = best_lane_action
                self.lane_change_initiated = False
            else:
                # 【修改】如果距离小于 1.8 倍 MSS (约12-15米)，不管旁边多空，直接放弃变道幻想，切入急刹避险！
                if current_dist < mss * 1.8:
                    self.current_state = DriveState.ABORT
                elif current_dist > mss * 4.0:
                    self.current_state = DriveState.KL

        elif self.current_state == DriveState.ABORT:
            # 【关键修复2】使用 target_speed 计算绝对安全退出距离，打破“刹车导致距离缩水”的死循环
            safe_exit_dist = self.calculate_mss(self.target_speed, lane_data["current"]["v_lead"]) * 2.0

            if current_dist > safe_exit_dist:
                self.current_state = DriveState.KL
                self.cooldown = 10
            # 【新增修复】如果车速已经极低（基本刹停），且距离大于物理极限（比如6米以上），切回KL维持刹车，避免死锁
            elif v_ego <= 0.5 and current_dist > 6.0:
                self.current_state = DriveState.KL
                self.cooldown = 10
            else:
                pass

        elif self.current_state in [DriveState.LCL, DriveState.LCR]:
            is_arrived = (current_lane != self.start_lane) and (abs(lateral_vel) < 0.5)

            if is_arrived:
                self.current_state = DriveState.KL
                self.cooldown = 15
                self.lane_change_initiated = False
            else:
                # ==========================================
                # 【关键修复：低速极限超车适配】
                # 动态计算防撞网倍率：高速保持 1.5 倍冗余，低速压缩到 1.1 倍
                # 这样在 5m/s 的低速下，允许车辆贴到 16.5 米处才放弃变道
                # ==========================================
                if v_ego < 8.0:
                    abort_multiplier = 1.1  # 低速模式：极短刹车距离，允许极限微操
                else:
                    abort_multiplier = 1.5  # 高速模式：保持足够的安全回撤空间

                if current_dist < mss * abort_multiplier:
                    self.current_state = DriveState.ABORT
                    self.lane_change_initiated = False
                else:
                    if not self.lane_change_initiated:
                        action = 0 if self.current_state == DriveState.LCL else 2
                        self.lane_change_initiated = True
                        return action, mss, p_safe
                    else:
                        # ==========================================
                        # 【关键修复：屏蔽基础逻辑的过早刹车】
                        # 既然我们允许低速下贴近到 1.1 倍，就绝不能让基础逻辑在 1.2 倍时提前踩死刹车！
                        # ==========================================
                        safe_act = final_action
                        if final_action == 4 and current_dist > mss * abort_multiplier:
                            safe_act = 1  # 强行改为滑行，让它有胆子继续变道
                        return safe_act, mss, p_safe

        # ==========================================================
        # 【新增2】全景快照打印区：一旦状态发生切换，立刻打印全场态势
        # ==========================================================
        if previous_state != self.current_state:
            print("\n" + "!" * 70)
            print(f"🔀 [决策触发] 状态切换: {previous_state.value}  ==>  {self.current_state.value}")
            print(f"🚗 自车状态: 当前车道 [{current_lane}] | 车速: {v_ego:.2f} m/s")
            print("-" * 70)

            # 【提取并打印本车道数据】
            c_dist = lane_data["current"]["dist"]
            c_v = lane_data["current"]["v_lead"]
            c_mss = self.calculate_mss(v_ego, c_v)
            c_p = self.calculate_fuzzy_p_safe(c_dist, c_mss, v_ego, c_v)
            print(
                f"  🎯 [本车道] 纵距: {c_dist:>5.1f}m | 前车速: {c_v:>4.1f}m/s | 安全底线(MSS): {c_mss:>4.1f}m | 概率: {c_p:.2f}")

            # 【提取并打印左车道数据】
            if current_lane > 0:
                l_dist = lane_data["left"]["dist"]
                l_v = lane_data["left"]["v_lead"]
                l_mss = self.calculate_mss(v_ego, l_v)
                l_p = self.calculate_fuzzy_p_safe(l_dist, l_mss, v_ego, l_v)
                print(
                    f"  👈 [左车道] 纵距: {l_dist:>5.1f}m | 前车速: {l_v:>4.1f}m/s | 安全底线(MSS): {l_mss:>4.1f}m | 概率: {l_p:.2f}")
            else:
                print("  👈 [左车道] 无 (自车已在最左侧)")

            # 【提取并打印右车道数据】
            if current_lane < 2:
                r_dist = lane_data["right"]["dist"]
                r_v = lane_data["right"]["v_lead"]
                r_mss = self.calculate_mss(v_ego, r_v)
                r_p = self.calculate_fuzzy_p_safe(r_dist, r_mss, v_ego, r_v)
                print(
                    f"  👉 [右车道] 纵距: {r_dist:>5.1f}m | 前车速: {r_v:>4.1f}m/s | 安全底线(MSS): {r_mss:>4.1f}m | 概率: {r_p:.2f}")
            else:
                print("  👉 [右车道] 无 (自车已在最右侧)")
            print("!" * 70 + "\n")

        # ================= 修复后的 action_map =================
        action_map = {
            DriveState.ABORT: 4,  # 【关键修复1】既然是回撤避险，就直接一脚刹车踩死！不要再听 final_action 的滑行瞎指挥
            DriveState.KL: final_action,
            DriveState.PLC: final_action
        }

        chosen_action = action_map.get(self.current_state, 1)
        return chosen_action, mss, p_safe

    def reset(self):
        self.current_state = DriveState.KL
        self.start_lane = None
        self.cooldown = 0
        self.lane_change_initiated = False