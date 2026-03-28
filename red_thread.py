import cv2
import numpy as np


def detect_coc_border_from_image(img):
    if img is None:
        return None

    h, w = img.shape[:2]
    game_h = int(h * 0.58)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    is_forest = cv2.inRange(
        hsv[:game_h],
        np.array([35, 80,  50]),
        np.array([85, 255, 200])
    )

    kernel = np.ones((10, 10), np.uint8)
    smooth = cv2.morphologyEx(is_forest, cv2.MORPH_CLOSE, kernel, iterations=2)
    smooth = cv2.morphologyEx(smooth,    cv2.MORPH_OPEN,  kernel, iterations=1)

    not_forest = cv2.bitwise_not(smooth)
    flood = not_forest.copy()
    cv2.floodFill(flood, None, (w // 2, game_h // 2), 128)
    base_mask = (flood == 128).astype(np.uint8) * 255

    contours, _ = cv2.findContours(base_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[边界] 未检测到基地区域")
        return None

    main_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(main_contour)
    pts  = hull[:, 0, :]

    top_pt  = pts[np.argmin(pts[:, 1])]
    left_pt = pts[np.argmin(pts[:, 0])]

    tr_pts = pts[(pts[:, 0] >= top_pt[0]) & (pts[:, 1] < game_h * 0.9)]
    tl_pts = pts[(pts[:, 0] <= top_pt[0]) & (pts[:, 1] < game_h * 0.9)]

    tr_pts = np.vstack([top_pt, tr_pts]) if len(tr_pts) >= 1 else np.vstack([top_pt, top_pt + [100, 70]])
    tl_pts = np.vstack([top_pt, tl_pts]) if len(tl_pts) >= 1 else np.vstack([top_pt, top_pt + [-100, 88]])

    def fit_line(points):
        out = cv2.fitLine(points.reshape(-1, 1, 2).astype(np.float32),
                          cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = out[0,0], out[1,0], out[2,0], out[3,0]
        if abs(vx) < 1e-6:
            return None, None
        m = vy / vx
        return float(m), float(y0 - m * x0)

    def intersect(m1, b1, m2, b2):
        if abs(m1 - m2) < 0.05:   # 斜率太接近，交点不可信
            return None
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
        return int(round(x)), int(round(y))

    def is_valid_pt(pt, max_w, max_h, factor=5):
        if pt is None:
            return False
        return abs(pt[0]) <= max_w * factor and abs(pt[1]) <= max_h * factor

    m_tr, b_tr = fit_line(tr_pts)
    m_tl, b_tl = fit_line(tl_pts)

    if m_tr is None or m_tl is None:
        print("[边界] 直线拟合失败")
        return None

    top_corner    = intersect(m_tr, b_tr, m_tl, b_tl)
    right_corner  = (w - 1, int(round(m_tr * (w - 1) + b_tr)))
    left_corner   = (int(left_pt[0]), int(round(m_tl * left_pt[0] + b_tl)))
    b_bl          = left_corner[1]  - m_tr * left_corner[0]
    b_br          = right_corner[1] - m_tl * right_corner[0]
    bottom_corner = intersect(m_tr, b_bl, m_tl, b_br)

    # 所有顶点合法性检查
    corners = {
        'top':    top_corner,
        'right':  right_corner,
        'bottom': bottom_corner,
        'left':   left_corner,
    }
    for name, pt in corners.items():
        if not is_valid_pt(pt, w, h, factor=5):
            print(f"[边界] 顶点 {name}={pt} 坐标异常，识别失败")
            return None

    print(f"[边界] Top={top_corner}, Right={right_corner}, Bottom={bottom_corner}, Left={left_corner}")
    return corners


def detect_coc_border(image_path, output_path='result_with_border.png'):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    corners = detect_coc_border_from_image(img)
    if corners is None:
        raise RuntimeError("边界检测失败")

    h, w = img.shape[:2]
    result = img.copy()
    pts_arr = np.array([corners['top'], corners['right'],
                        corners['bottom'], corners['left']], dtype=np.int32)
    cv2.polylines(result, [pts_arr], isClosed=True, color=(0, 255, 0), thickness=5)
    for name, pt in corners.items():
        px = max(0, min(pt[0], w-1))
        py = max(0, min(pt[1], h-1))
        cv2.circle(result, (px, py), 18, (0, 0, 255), -1)
        cv2.putText(result, name, (px+12, py+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)

    cv2.imwrite(output_path, result)
    print(f"[边界] 已保存: {output_path}")
    return pts_arr