import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Visualizer:
    def __init__(self, window_name="Drive Monitor"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name)

        # 【新增】存储鼠标点击的像素坐标
        self.clicked_points = []
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        try:
            self.font = ImageFont.truetype("msyh.ttc", 20)
            self.small_font = ImageFont.truetype("msyh.ttc", 14)  # 【新增】用于刻度尺的小号字体
        except:
            self.font = ImageFont.load_default()
            self.small_font = ImageFont.load_default()

    # 【新增】OpenCV的鼠标点击回调函数
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_points.append((x, y))

    def draw_info(self, frame, state_name, delay_ms, dist, mss, p_safe, paused=False, vehicle_list=None,
                  lane_y_coords=None, ego_x=0.0, scaling=15.0, screen_width=1024):

        canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 【新增】暂停时的 UI 交互提示
        if paused:
            draw.text((screen_width // 2 - 180, 80), "⏸️ 仿真已暂停: 请使用鼠标点击车道放置障碍车", font=self.font,
                      fill=(0, 255, 255))
            draw.text((screen_width // 2 - 180, 110), "提示: 车辆放下后将立刻显示实时纵距", font=self.small_font,
                      fill=(200, 200, 200))

        # ==========================================
        # 【新增】绘制双侧 10 米间隔刻度尺
        # ==========================================
        if lane_y_coords:
            # 确定马路的上下边缘 Y 坐标
            top_y = lane_y_coords[0][1] - 30
            bottom_y = lane_y_coords[-1][1] + 30

            # 计算屏幕当前能显示的最左和最右的“真实世界X坐标”
            start_x = ego_x - (screen_width / 2) / scaling
            end_x = ego_x + (screen_width / 2) / scaling

            # 找到视野内第一个 10 的倍数
            first_tick = int((start_x // 10) * 10)
            for tick in range(first_tick, int(end_x) + 10, 10):
                # 将真实的 10 米节点转换回屏幕的像素坐标
                pixel_x = int((tick - ego_x) * scaling + (screen_width / 2))

                # 画上方刻度线和文字
                draw.line([(pixel_x, top_y), (pixel_x, top_y + 10)], fill=(255, 255, 255), width=2)
                draw.text((pixel_x - 15, top_y - 20), f"{tick}m", font=self.small_font, fill=(200, 200, 200))

                # 画下方刻度线和文字
                draw.line([(pixel_x, bottom_y), (pixel_x, bottom_y - 10)], fill=(255, 255, 255), width=2)
                draw.text((pixel_x - 15, bottom_y + 5), f"{tick}m", font=self.small_font, fill=(200, 200, 200))

        # ==========================================

        info_lines = [
            f"系统状态: {state_name}",
            f"本车道纵距: {dist:.1f}m (安全线: {mss:.1f}m)",
            f"安全概率: {p_safe:.2f} (门限: 0.70)",
            f"帧延迟: {delay_ms}ms (空格暂停, +/- 调速)"
        ]

        if lane_y_coords:
            for lane_id, y_pos in lane_y_coords:
                draw.text((15, y_pos - 15), f"车道 [{lane_id}]", font=self.font, fill=(200, 200, 200))

        for i, line in enumerate(info_lines):
            color = (0, 255, 0) if (i != 1 or dist > mss) else (255, 0, 0)
            draw.text((20, 20 + i * 30), line, font=self.font, fill=color)

        if vehicle_list:
            for v in vehicle_list:
                speed_text = f"{v['speed']:.0f} km/h"
                text_color = (255, 255, 0) if v["is_ego"] else (255, 255, 255)
                draw.text((v["x"] - 30, v["y"] - 45), speed_text, font=self.font, fill=text_color)

                # 【修改】不再屏蔽后方车辆，只要不是自车，被点出来立刻显示距离！
                if not v["is_ego"]:
                    dist_text = f"纵距: {v['long_dist']:.1f} m"
                    draw.text((v["x"] - 30, v["y"] - 20), dist_text, font=self.font, fill=(0, 255, 255))

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def draw_detections(self, canvas, yolo_results):
        for res in yolo_results[0].boxes:
            xmin, ymin, xmax, ymax = map(int, res.xyxy.cpu().numpy()[0])
            cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        return canvas

    def show(self, canvas):
        cv2.imshow(self.window_name, canvas)

    def close(self):
        cv2.destroyAllWindows()