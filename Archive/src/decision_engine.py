from enum import Enum


class DriveState(Enum):
    KL = "车道保持"
    PLC = "准备超车"
    LCL = "左侧超车中"
    LCR = "右侧超车中"
    ABORT = "终止并回撤"


class DecisionEngine:
    def __init__(self):
        self.current_state = DriveState.KL
        self.MAX_DECEL = 8.5
        self.SAFE_TIME_GAP = 0.8
        # 【修改】使用模糊逻辑后，将门限提高至 0.5，评估更平滑准确
        self.P_SAFE_THRESHOLD = 0.5
        self.start_lane = None
        self.cooldown = 0  # 模糊逻辑平滑了决策，cooldown可大幅缩短甚至最终废除

    def calculate_mss(self, v_ego, v_lead):
        if v_ego > v_lead:
            braking_dist = (v_ego ** 2 - v_lead ** 2) / (2 * self.MAX_DECEL)
        else:
            braking_dist = 0.0
        return max(5.0, v_ego * self.SAFE_TIME_GAP + braking_dist)

    # ================= 新增：模糊计算模块 =================
    def _fuzzify_margin(self, margin):
        """将安全余量(margin)模糊化为：危险、警告、安全 三种状态的隶属度 [0.0 - 1.0]"""
        danger = max(0.0, min(1.0, (15.0 - margin) / 15.0)) if margin < 15 else 0.0
        safe = max(0.0, min(1.0, (margin - 15.0) / 15.0)) if margin > 15 else 0.0
        warning = max(0.0, min(1.0, 1.0 - abs(margin - 15.0) / 15.0))
        return danger, warning, safe

    def _fuzzify_rel_speed(self, v_ego, v_lead):
        """将相对速度模糊化为：靠近中、相对静止、远离中"""
        delta_v = v_ego - v_lead
        closing = max(0.0, min(1.0, delta_v / 5.0)) if delta_v > 0 else 0.0
        separating = max(0.0, min(1.0, -delta_v / 5.0)) if delta_v < 0 else 0.0
        stable = max(0.0, min(1.0, 1.0 - abs(delta_v) / 5.0))
        return closing, stable, separating

    def calculate_fuzzy_p_safe(self, current_dist, mss, v_ego, v_lead):
        """替代原有的线性 calculate_p_safe，基于模糊规则输出平滑的安全概率"""
        margin = current_dist - mss
        if margin <= 0:
            return 0.0

        # 1. 模糊化 (Fuzzification)
        d_danger, d_warn, d_safe = self._fuzzify_margin(margin)
        v_close, v_stable, v_sep = self._fuzzify_rel_speed(v_ego, v_lead)

        # 2. 模糊规则评估 (Fuzzy Rules Evaluation)
        # 规则1: 如果余量安全，就是高度安全
        rule_high_safe = d_safe
        # 规则2: 如果余量警告，但前车比我快(远离中)或速度一致，中度安全
        rule_med_safe = min(d_warn, max(v_stable, v_sep))
        # 规则3: 如果余量危险，或者(余量警告且正在快速靠近)，极度不安全
        rule_low_safe = max(d_danger, min(d_warn, v_close))

        # 3. 解模糊 (Defuzzification) - 使用重心法(Center of Gravity)
        # 设定各安全级别的确切代表值：High=1.0, Med=0.6, Low=0.1
        numerator = rule_high_safe * 1.0 + rule_med_safe * 0.6 + rule_low_safe * 0.1
        denominator = rule_high_safe + rule_med_safe + rule_low_safe + 1e-5

        return round(numerator / denominator, 2)

    # =====================================================

    def get_action(self, v_ego, lane_data, current_lane, lateral_vel):
        current_dist = lane_data["current"]["dist"]
        v_lead = lane_data["current"]["v_lead"]

        mss = self.calculate_mss(v_ego, v_lead)
        # 【修改】替换为模糊安全概率计算，传入速度参数
        p_safe = self.calculate_fuzzy_p_safe(current_dist, mss, v_ego, v_lead)

        if self.cooldown > 0:
            self.cooldown -= 1

        if current_dist < mss:
            final_action = 4
        elif v_ego < 35.0:
            final_action = 3
        else:
            final_action = 1

        prev_state = self.current_state

        # ================= 状态转移逻辑 =================
        if self.current_state == DriveState.KL:
            # 【新增】紧急避险逻辑：如果已经突破安全底线，无视冷却期，强制触发变道探查！
            if current_dist < mss:
                self.cooldown = 0  # 强制打破冷静期
                self.current_state = DriveState.PLC
                print(f"[DEBUG 警告] 触发紧急避险！无视冷却期，强制寻路！")

            # 常规逻辑：正常距离下的超车准备，仍受冷却期约束（可后续废除cooldown）
            elif current_dist < mss * 1.5 and self.cooldown == 0:
                self.current_state = DriveState.PLC

            # 前方大路朝天
            elif current_dist > mss * 2.0:
                final_action = 3

        elif self.current_state == DriveState.PLC:
            self.start_lane = current_lane

            # 【新增】用于记录全局最优选择的变量
            best_lane_action = None
            best_p_safe = -1.0
            best_dist = -1.0

            # 1. 评估左侧车道
            if current_lane > 0:
                left_dist = lane_data["left"]["dist"]
                left_mss = self.calculate_mss(v_ego, lane_data["left"]["v_lead"])
                left_p_safe = self.calculate_fuzzy_p_safe(left_dist, left_mss, v_ego,
                                                          lane_data["left"]["v_lead"])

                # 如果左侧安全，暂定为最优解
                if left_p_safe > self.P_SAFE_THRESHOLD:
                    best_p_safe = left_p_safe
                    best_dist = left_dist
                    best_lane_action = DriveState.LCL

            # 2. 评估右侧车道并与当前最优解对比
            if current_lane < 2:
                right_dist = lane_data["right"]["dist"]
                right_mss = self.calculate_mss(v_ego, lane_data["right"]["v_lead"])
                right_p_safe = self.calculate_fuzzy_p_safe(right_dist, right_mss, v_ego,
                                                           lane_data["right"]["v_lead"])

                if right_p_safe > self.P_SAFE_THRESHOLD:
                    # 【核心对比逻辑】：如果右侧安全概率更高，或者一样安全但右侧可用距离更长，则推翻原有决定，选择右侧
                    if right_p_safe > best_p_safe or (right_p_safe == best_p_safe and right_dist > best_dist):
                        best_p_safe = right_p_safe
                        best_dist = right_dist
                        best_lane_action = DriveState.LCR

            # 3. 做出最终决策
            if best_lane_action:
                self.current_state = best_lane_action
            else:
                # 【调优】拓宽滞回区间：左右都没法变道时，只有前方非常空旷（2.0倍MSS）才退回巡航，避免状态频繁跳变
                if current_dist > mss * 2.0:
                    self.current_state = DriveState.KL

        elif self.current_state in [DriveState.LCL, DriveState.LCR]:
            # 【修复连跨两道 Bug】只要车身跨过车道线，立刻判定变道决策完成！
            is_arrived = (current_lane != self.start_lane)
            if is_arrived:
                self.current_state = DriveState.KL
                self.cooldown = 30  # 可后续废除，模糊逻辑已减少抖动
            else:
                action = 0 if self.current_state == DriveState.LCL else 2
                return action, mss, p_safe

        action_map = {
            DriveState.LCL: 0,
            DriveState.LCR: 2,
            DriveState.ABORT: 1,
            DriveState.KL: final_action,
            DriveState.PLC: final_action
        }

        chosen_action = action_map.get(self.current_state, 1)
        return chosen_action, mss, p_safe

    def reset(self):
        self.current_state = DriveState.KL
        self.start_lane = None
        self.cooldown = 0