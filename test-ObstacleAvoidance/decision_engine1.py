from enum import Enum


class DriveState(Enum):
    KL = "车道保持"
    PLC = "准备超车"
    LCL = "左侧超车中"
    LCR = "右侧超车中"
    ABORT = "终止并回撤"


class DecisionEngine:
    def __init__(self, test_safe_dist=5.0, fuzzy_scale=5.0, target_speed=10.0):
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
            if current_dist < mss:
                self.cooldown = 0
                self.current_state = DriveState.PLC
            elif current_dist < mss * 4.0 and self.cooldown == 0:
                self.current_state = DriveState.PLC

        elif self.current_state == DriveState.PLC:
            self.start_lane = current_lane
            can_go_left = False
            can_go_right = False

            if current_lane > 0:
                left_mss = self.calculate_mss(v_ego, lane_data["left"]["v_lead"])
                left_p_safe = self.calculate_fuzzy_p_safe(lane_data["left"]["dist"], left_mss, v_ego,
                                                          lane_data["left"]["v_lead"])
                if left_p_safe > self.P_SAFE_THRESHOLD:
                    can_go_left = True

            if current_lane < 2:
                right_mss = self.calculate_mss(v_ego, lane_data["right"]["v_lead"])
                right_p_safe = self.calculate_fuzzy_p_safe(lane_data["right"]["dist"], right_mss, v_ego,
                                                           lane_data["right"]["v_lead"])
                if right_p_safe > self.P_SAFE_THRESHOLD:
                    can_go_right = True

            if can_go_left:
                self.current_state = DriveState.LCL
                self.lane_change_initiated = False  # 切换状态时重置锁
            elif can_go_right:
                self.current_state = DriveState.LCR
                self.lane_change_initiated = False  # 切换状态时重置锁
            else:
                if current_dist > mss * 4.0:
                    self.current_state = DriveState.KL

        elif self.current_state in [DriveState.LCL, DriveState.LCR]:
            is_arrived = (current_lane != self.start_lane)

            if is_arrived:
                # 变道已跨线完成
                self.current_state = DriveState.KL
                self.cooldown = 15
                self.lane_change_initiated = False
            else:
                # 【关键修复】如果是变道中的第一帧，发送变道动作，然后锁死
                if not self.lane_change_initiated:
                    action = 0 if self.current_state == DriveState.LCL else 2
                    self.lane_change_initiated = True
                    return action, mss, p_safe
                else:
                    # 变道指令已发送，车辆正在侧向漂移中，此时返回巡航动作（保持速度）
                    return final_action, mss, p_safe

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
        self.lane_change_initiated = False