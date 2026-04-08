import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Visualizer:
    def __init__(self, window_name="Drive Monitor"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name)
        try:
            self.font = ImageFont.truetype("msyh.ttc", 20)
        except:
            self.font = ImageFont.load_default()

    def draw_info(self, frame, state_name, delay_ms, dist, mss, p_safe, paused=False, vehicle_list=None,
                  lane_y_coords=None):
        # 移除原先的水平翻转 (cv2.flip)，直接使用原始画面
        canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if paused:
            cv2.putText(canvas, "PAUSED", (450, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # 【修改】强调主面板上显示的也是纵距
        info_lines = [
            f"系统状态: {state_name}",
            f"本车道纵距: {dist:.1f}m (安全线: {mss:.1f}m)",
            f"安全概率: {p_safe:.2f} (门限: 0.70)",
            f"帧延迟: {delay_ms}ms (空格暂停, +/- 调速)"
        ]

        img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 【新增】绘制车道号标签
        if lane_y_coords:
            for lane_id, y_pos in lane_y_coords:
                # 绘制在画面最左侧，透明度稍低一点的灰色避免喧宾夺主
                draw.text((15, y_pos - 15), f"车道 [{lane_id}]", font=self.font, fill=(200, 200, 200))

        # 绘制左上角UI信息
        for i, line in enumerate(info_lines):
            color = (0, 255, 0) if (i != 1 or dist > mss) else (255, 0, 0)
            draw.text((20, 20 + i * 30), line, font=self.font, fill=color)

        # 绘制每辆车的信息
        if vehicle_list:
            for v in vehicle_list:
                # 绘制速度
                speed_text = f"{v['speed']:.0f} km/h"
                text_color = (255, 255, 0) if v["is_ego"] else (255, 255, 255)
                draw.text((v["x"] - 30, v["y"] - 45), speed_text, font=self.font, fill=text_color)

                # 【新增】为前方障碍车绘制实时纵向距离
                if not v["is_ego"] and v.get("long_dist", 0) > 0:
                    dist_text = f"纵距: {v['long_dist']:.1f} m"
                    # 画在车速文字的下方
                    draw.text((v["x"] - 30, v["y"] - 20), dist_text, font=self.font, fill=(0, 255, 255))

        # 转回OpenCV格式并返回
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def draw_detections(self, canvas, yolo_results):
        """绘制 YOLO 检测框"""
        for res in yolo_results[0].boxes:
            xmin, ymin, xmax, ymax = map(int, res.xyxy.cpu().numpy()[0])

            # 因为没有进行翻转，直接使用原始的 xmin 和 xmax 画框即可
            cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        return canvas

    def show(self, canvas):
        cv2.imshow(self.window_name, canvas)

    def close(self):
        cv2.destroyAllWindows()