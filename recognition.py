import cv2
import os
import numpy as np


def is_rect_shape(dst, angle_tol=20, ratio_tol=0.5):

    pts = dst[:, 0, :]  # (4, 2)

    # 计算四个内角
    def angle(p1, vertex, p2):
        v1 = p1 - vertex
        v2 = p2 - vertex
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        return np.degrees(np.arccos(cos_a))

    n = len(pts)
    for i in range(n):
        p1     = pts[(i - 1) % n]
        vertex = pts[i]
        p2     = pts[(i + 1) % n]
        a = angle(p1, vertex, p2)
        if abs(a - 90.0) > angle_tol:
            print(f"[矩形校验] 角 {i} = {a:.1f}°，偏差超过 {angle_tol}°，视为匹配失败")
            return False

    # 检查对边长度比
    side0 = np.linalg.norm(pts[1] - pts[0])
    side1 = np.linalg.norm(pts[2] - pts[1])
    side2 = np.linalg.norm(pts[3] - pts[2])
    side3 = np.linalg.norm(pts[0] - pts[3])

    # 对边0和2、对边1和3
    for s1, s2, name in [(side0, side2, "上/下"), (side1, side3, "左/右")]:
        if max(s1, s2) < 1e-6:
            continue
        ratio = abs(s1 - s2) / max(s1, s2)
        if ratio > ratio_tol:
            print(f"[矩形校验] 对边({name})长度差异 {ratio:.2f} 超过 {ratio_tol}，视为匹配失败")
            return False

    return True


def match_template_sift(template_img, target_img, min_match=10):

    gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    gray_target   = cv2.cvtColor(target_img,   cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_template, None)
    kp2, des2 = sift.detectAndCompute(gray_target,   None)

    if des1 is None or des2 is None:
        return None, []

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=50)
    )
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(good_matches) < min_match:
        print(f"[SIFT] 匹配失败: 仅 {len(good_matches)} 个特征点 (需要 {min_match})")
        return None, good_matches

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None:
        return None, good_matches

    h, w = gray_template.shape
    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # 矩形校验：不是矩形/正方形视为失败
    if not is_rect_shape(dst, angle_tol=20, ratio_tol=0.5):
        print(f"[SIFT] 匹配区域形状异常，视为匹配失败")
        return None, good_matches

    inliers = int(np.sum(mask))
    print(f"[SIFT] 匹配成功: {len(good_matches)} 个特征点，内点 {inliers} 个")
    return dst, good_matches