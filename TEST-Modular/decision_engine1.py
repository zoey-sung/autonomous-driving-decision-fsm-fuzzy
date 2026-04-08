from enum import Enum
from config import SimConfig


class DriveState(Enum):
    KL = "车道保持"
    PLC = "准备超车"
    LCL = "左侧超车中"
    LCR = "右侧超车中"
    ABORT = "终止并回撤"


class DecisionEngine:
    def __init__(self, target_speed=10.0):
        self.current_state = DriveState.KL
        self.MAX_DECEL = 8.5
        self.SAFE_TIME_GAP = 0.8
        self.start_lane = None
        self.cooldown = 0

        self.target_speed = target_speed
        self.lane_change_initiated = False

    def calculate_mss(self, v_ego, v_lead):
        # 动态安全距离：基础距离 + 速度 * 反应时间(比如 2.0 秒)
        dynamic_mss = SimConfig.SAFE_DISTANCE_MSS + (v_ego * 2.0)
        return dynamic_mss

    def _fuzzify_margin(self, margin, v_ego):
        # 【方案B】动态模糊余量：至少保留下限，车速快时再放大，解决低速胆怯问题
        scale = max(SimConfig.FUZZY_SCALE_MIN, v_ego * SimConfig.FUZZY_SCALE_RATE)
        danger = max(0.0, min(1.0, (scale - margin) / scale)) if margin < scale else 0.0
        safe = max(0.0, min(1.0, (margin - scale) / scale)) if margin > scale else 0.0
        warning = max(0.0, min(1.0, 1.0 - abs(margin - scale) / scale))
        return danger, warning, safe

    def _fuzzify_rel_speed(self, v_ego, v_lead):
        delta_v = v_ego - v_lead
        scale = SimConfig.FUZZY_REL_SPEED_SCALE
        closing = max(0.0, min(1.0, delta_v / scale)) if delta_v > 0 else 0.0
        separating = max(0.0, min(1.0, -delta_v / scale)) if delta_v < 0 else 0.0
        stable = max(0.0, min(1.0, 1.0 - abs(delta_v) / scale))
        return closing, stable, separating

    def calculate_fuzzy_p_safe(self, current_dist, mss, v_ego, v_lead):
        margin = current_dist - mss
        if margin <= 0:
            return 0.0

        d_danger, d_warn, d_safe = self._fuzzify_margin(margin, v_ego)
        v_close, v_stable, v_sep = self._fuzzify_rel_speed(v_ego, v_lead)

        rule_high_safe = d_safe
        rule_med_safe = min(d_warn, max(v_stable, v_sep))
        rule_low_safe = max(d_danger, min(d_warn, v_close))

        numerator = rule_high_safe * 1.0 + rule_med_safe * 0.6 + rule_low_safe * 0.1
        denominator = rule_high_safe + rule_med_safe + rule_low_safe + 1e-5

        return round(numerator / denominator, 2)

    def get_action(self, v_ego, lane_data, current_lane, lateral_vel):
        previous_state = self.current_state

        current_dist = lane_data["current"]["dist"]
        v_lead = lane_data["current"]["v_lead"]

        mss = self.calculate_mss(v_ego, v_lead)
        p_safe = self.calculate_fuzzy_p_safe(current_dist, mss, v_ego, v_lead)

        if self.cooldown > 0:
            self.cooldown -= 1

        # 基础巡航与紧急减速指令
        if current_dist < (mss * 1.2) + SimConfig.VEHICLE_LENGTH_COMP:
            final_action = 4
        elif v_ego < self.target_speed - 1.0:
            final_action = 3
        elif v_ego > self.target_speed + 1.0:
            final_action = 4
        else:
            final_action = 1

        # ================= 状态机逻辑 =================
        if self.current_state == DriveState.KL:
            if (mss * SimConfig.PLC_MIN_MSS_MULT < current_dist < mss * SimConfig.PLC_MAX_MSS_MULT and
                    v_ego >= SimConfig.MIN_SPEED_TO_OVERTAKE and self.cooldown == 0):
                self.current_state = DriveState.PLC

        elif self.current_state == DriveState.PLC:
            self.start_lane = current_lane
            best_lane_action = None
            best_p_safe = -1.0
            best_dist = -1.0

            current_mss = self.calculate_mss(v_ego, lane_data["current"]["v_lead"])
            current_p_safe = self.calculate_fuzzy_p_safe(current_dist, current_mss, v_ego,
                                                         lane_data["current"]["v_lead"])

            if current_dist > mss * SimConfig.PLC_MIN_MSS_MULT:
                # 评估左侧车道
                if current_lane > 0:
                    left_dist = lane_data["left"]["dist"]
                    left_mss = self.calculate_mss(v_ego, lane_data["left"]["v_lead"])
                    left_p_safe = self.calculate_fuzzy_p_safe(left_dist, left_mss, v_ego, lane_data["left"]["v_lead"])

                    if left_p_safe > SimConfig.P_SAFE_THRESHOLD and (
                            left_p_safe > current_p_safe or left_dist > current_dist + SimConfig.VEHICLE_LENGTH_COMP):
                        best_p_safe = left_p_safe
                        best_dist = left_dist
                        best_lane_action = DriveState.LCL

                # 评估右侧车道
                if current_lane < (SimConfig.LANES_COUNT - 1):
                    right_dist = lane_data["right"]["dist"]
                    right_mss = self.calculate_mss(v_ego, lane_data["right"]["v_lead"])
                    right_p_safe = self.calculate_fuzzy_p_safe(right_dist, right_mss, v_ego,
                                                               lane_data["right"]["v_lead"])

                    if right_p_safe > SimConfig.P_SAFE_THRESHOLD and (
                            right_p_safe > current_p_safe or right_dist > current_dist + SimConfig.VEHICLE_LENGTH_COMP):
                        if right_p_safe > best_p_safe or (right_p_safe == best_p_safe and right_dist > best_dist):
                            best_p_safe = right_p_safe
                            best_dist = right_dist
                            best_lane_action = DriveState.LCR

                if best_lane_action:
                    self.current_state = best_lane_action
                    self.lane_change_initiated = False
                else:
                    absolute_mss = self.calculate_mss(self.target_speed, lane_data["current"]["v_lead"])
                    if current_dist < (mss * SimConfig.PLC_MIN_MSS_MULT) + SimConfig.VEHICLE_LENGTH_COMP:
                        self.current_state = DriveState.ABORT
                    elif current_dist > absolute_mss * SimConfig.PLC_MAX_MSS_MULT:
                        self.current_state = DriveState.KL

        elif self.current_state == DriveState.ABORT:
            safe_exit_dist = self.calculate_mss(self.target_speed,
                                                lane_data["current"]["v_lead"]) * SimConfig.ABORT_EXIT_MSS_MULT

            if current_dist > safe_exit_dist:
                self.current_state = DriveState.KL
                self.cooldown = SimConfig.COOLDOWN_ABORT
            elif v_ego <= SimConfig.MIN_SPEED_TO_OVERTAKE and current_dist > SimConfig.ABORT_DEADLOCK_DIST:
                self.current_state = DriveState.KL
                self.cooldown = SimConfig.COOLDOWN_ABORT
            else:
                pass

        elif self.current_state in [DriveState.LCL, DriveState.LCR]:
            is_arrived = (current_lane != self.start_lane) and (abs(lateral_vel) < 0.5)

            if is_arrived:
                self.current_state = DriveState.KL
                self.cooldown = SimConfig.COOLDOWN_LANE_CHANGE
                self.lane_change_initiated = False
            else:
                abort_multiplier = SimConfig.LOW_SPEED_ABORT_MULT if v_ego < SimConfig.SPEED_MODE_THRESHOLD else SimConfig.HIGH_SPEED_ABORT_MULT

                if current_dist < mss * abort_multiplier:
                    self.current_state = DriveState.ABORT
                    self.lane_change_initiated = False
                else:
                    if not self.lane_change_initiated:
                        action = 0 if self.current_state == DriveState.LCL else 2
                        self.lane_change_initiated = True
                        return action, mss, p_safe
                    else:
                        safe_act = final_action
                        if final_action == 4 and current_dist > mss * abort_multiplier:
                            safe_act = 1
                        return safe_act, mss, p_safe

        # =================全景快照打印区=================
        if previous_state != self.current_state:
            print("\n" + "!" * 70)
            print(f"🔀 [决策触发] 状态切换: {previous_state.value}  ==>  {self.current_state.value}")
            print(f"🚗 自车状态: 当前车道 [{current_lane}] | 车速: {v_ego:.2f} m/s")
            print("-" * 70)

            def to_gap(d):
                return max(0.0, d - SimConfig.VEHICLE_LENGTH_COMP) if d < 99.0 else 100.0

            c_dist = lane_data["current"]["dist"]
            c_v = lane_data["current"]["v_lead"]
            c_mss = self.calculate_mss(v_ego, c_v)
            c_p = self.calculate_fuzzy_p_safe(c_dist, c_mss, v_ego, c_v)
            # 【修复】c_mss 本身就是净距，直接打印，不再套 to_gap
            print(
                f"  🎯 [本车道] 净距: {to_gap(c_dist):>5.1f}m | 前车速: {c_v:>4.1f}m/s | 安全底线: {c_mss:>4.1f}m | 概率: {c_p:.2f}")

            if current_lane > 0:
                l_dist = lane_data["left"]["dist"]
                l_v = lane_data["left"]["v_lead"]
                l_mss = self.calculate_mss(v_ego, l_v)
                l_p = self.calculate_fuzzy_p_safe(l_dist, l_mss, v_ego, l_v)
                # 【修复】
                print(
                    f"  👈 [左车道] 净距: {to_gap(l_dist):>5.1f}m | 前车速: {l_v:>4.1f}m/s | 安全底线: {l_mss:>4.1f}m | 概率: {l_p:.2f}")
            else:
                print("  👈 [左车道] 无 (自车已在最左侧)")

            if current_lane < (SimConfig.LANES_COUNT - 1):
                r_dist = lane_data["right"]["dist"]
                r_v = lane_data["right"]["v_lead"]
                r_mss = self.calculate_mss(v_ego, r_v)
                r_p = self.calculate_fuzzy_p_safe(r_dist, r_mss, v_ego, r_v)
                # 【修复】
                print(
                    f"  👉 [右车道] 净距: {to_gap(r_dist):>5.1f}m | 前车速: {r_v:>4.1f}m/s | 安全底线: {r_mss:>4.1f}m | 概率: {r_p:.2f}")
            else:
                print("  👉 [右车道] 无 (自车已在最右侧)")
            print("!" * 70 + "\n")

        action_map = {
            DriveState.ABORT: 4,
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