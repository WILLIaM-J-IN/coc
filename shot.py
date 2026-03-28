import mss
import cv2
import numpy as np
import os
import time


def screenshot_window(rect, save_path="data/screenshot_latest.png"):
    left, top, right, bottom = rect
    width = right - left
    height = bottom - top

    with mss.mss() as sct:
        monitor = {"top": top, "left": left, "width": width, "height": height}
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)
        # 转为 BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img)

    return img


def verify_color(img_crop, template_img):

    mean_crop = cv2.mean(img_crop)[:3]
    mean_template = cv2.mean(template_img)[:3]


    diff = np.sqrt(
        (mean_crop[0] - mean_template[0]) ** 2 +
        (mean_crop[1] - mean_template[1]) ** 2 +
        (mean_crop[2] - mean_template[2]) ** 2
    )
    return diff


def find_template(screen_img, template_path, threshold=0.6, color_diff_threshold=40.0):

    # 1. 读取模板 (强制彩色)
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template is None:
        print(f"[Error] 模板无法读取: {template_path}")
        return None


    try:
        res = cv2.matchTemplate(screen_img, template, cv2.TM_CCOEFF_NORMED)
    except Exception as e:
        # print(f"[Error] 匹配出错 (可能是尺寸问题): {e}")
        return None


    loc = np.where(res >= threshold)
    points = list(zip(*loc[::-1]))  # 转换为 [(x,y), ...]

    if not points:
        return None

    h, w = template.shape[:2]
    best_pos = None
    best_score = -1.0


    for pt in points:
        x, y = pt

        # 边界检查
        if y + h > screen_img.shape[0] or x + w > screen_img.shape[1]:
            continue

        # 抠图
        crop = screen_img[y:y + h, x:x + w]

        # 计算颜色差异
        diff = verify_color(crop, template)

        # 形状分数
        score = res[y, x]

        # 判断：只有颜色差异足够小，才认为是有效目标
        if diff <= color_diff_threshold:
            # 在颜色合格的前提下，取形状最像的
            if score > best_score:
                best_score = score
                best_pos = (x + w // 2, y + h // 2)

    return best_pos
