import cv2
import numpy as np


def match_button(template_img, target_img, threshold=0.6, scales=None):

    if scales is None:
        scales = np.arange(0.5, 1.51, 0.05)

    gray_target   = cv2.cvtColor(target_img,   cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

    best_score = -1
    best_loc   = None
    best_scale = 1.0
    best_size  = (0, 0)

    orig_h, orig_w = gray_template.shape[:2]

    for scale in scales:
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        if new_w > gray_target.shape[1] or new_h > gray_target.shape[0]:
            continue
        if new_w < 10 or new_h < 10:
            continue

        resized = cv2.resize(gray_template, (new_w, new_h))
        res     = cv2.matchTemplate(gray_target, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > best_score:
            best_score = max_val
            best_loc   = max_loc
            best_scale = scale
            best_size  = (new_w, new_h)

    if best_score < threshold:
        print(f"[ButtonMatch] 匹配失败，最高得分: {best_score:.3f} (阈值: {threshold})")
        return None

    bw, bh = best_size
    cx = best_loc[0] + bw // 2
    cy = best_loc[1] + bh // 2
    top_left     = (best_loc[0], best_loc[1])
    bottom_right = (best_loc[0] + bw, best_loc[1] + bh)

    print(f"[ButtonMatch] 匹配成功  得分: {best_score:.3f}  缩放: {best_scale:.2f}  中心: ({cx}, {cy})")
    return {
        'cx':           cx,
        'cy':           cy,
        'score':        best_score,
        'scale':        best_scale,
        'top_left':     top_left,
        'bottom_right': bottom_right,
    }


def match_button_from_path(template_path, target_path, threshold=0.6, scales=None):

    template_img = cv2.imread(template_path)
    target_img   = cv2.imread(target_path)

    if template_img is None:
        print(f"[ButtonMatch] 模板图读取失败: {template_path}")
        return None
    if target_img is None:
        print(f"[ButtonMatch] 目标图读取失败: {target_path}")
        return None

    return match_button(template_img, target_img, threshold, scales)