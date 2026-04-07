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

    def draw_info(self, frame, state_name, delay_ms, dist, mss, p_safe, paused=False, vehicle_list=None):
        # 1. 记录原始宽度用于坐标翻转
        width = frame.shape[1]

        # 2. 水平翻转画面（实现“右边入画，左边出画”）
        frame = cv2.flip(frame, 1)
        canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if paused:
            cv2.putText(canvas, "PAUSED", (450, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # 准备多行中文信息
        info_lines = [
            f"系统状态: {state_name}",
            f"当前距离: {dist:.1f}m (标准安全线: {mss:.1f}m)",
            f"安全概率: {p_safe:.2f} (门限: 0.70)",
            f"帧延迟: {delay_ms}ms (空格暂停, +/- 调速)"
        ]

        img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 绘制左上角UI信息（无需翻转，因为本身在固定位置）
        for i, line in enumerate(info_lines):
            color = (0, 255, 0) if (i != 1 or dist > mss) else (255, 0, 0)
            draw.text((20, 20 + i * 30), line, font=self.font, fill=color)

        # 【新增】绘制每辆车的速度（适配镜像坐标）
        if vehicle_list:
            for v in vehicle_list:
                # 关键：根据原始宽度翻转x坐标，抵消画面水平翻转的影响
                flip_x = width - v["x"]

                # 速度文本格式化
                speed_text = f"{v['speed']:.0f} km/h"

                # 颜色区分：自车（黄色）、其他车辆（白色）
                text_color = (255, 255, 0) if v["is_ego"] else (255, 255, 255)

                # 在车辆坐标上方40像素、水平居中（减30像素抵消文本宽度）绘制
                draw.text((flip_x - 30, v["y"] - 40), speed_text, font=self.font, fill=text_color)

        # 转回OpenCV格式并返回
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def draw_detections(self, canvas, yolo_results):
        """绘制 YOLO 检测框（已修复：适配镜像后的画面）"""
        width = canvas.shape[1]
        for res in yolo_results[0].boxes:
            xmin, ymin, xmax, ymax = map(int, res.xyxy.cpu().numpy()[0])

            # 【关键修复】YOLO 是基于原始未翻转的 frame 预测的
            # canvas 已经被水平翻转了，所以检测框的左右 X 坐标也必须做翻转对应
            flip_xmin = width - xmax
            flip_xmax = width - xmin

            cv2.rectangle(canvas, (flip_xmin, ymin), (flip_xmax, ymax), (0, 255, 0), 2)
        return canvas

    def show(self, canvas):
        cv2.imshow(self.window_name, canvas)

    def close(self):
        cv2.destroyAllWindows()