import cv2
import numpy as np
import os
from recognition import match_template_sift

# ===== 路径配置 =====
target_dir  = r"E:\PythonProject\royaleai\target"
photo_path  = r"E:\PythonProject\royaleai\data\photo1.png"
result_dir  = r"E:\PythonProject\royaleai\result"
os.makedirs(result_dir, exist_ok=True)

# 每个模板最多匹配次数，防止死循环
MAX_MATCH_PER_TEMPLATE = 10


def match_all_occurrences(template_img, photo, template_name, max_count=MAX_MATCH_PER_TEMPLATE):
    """
    在 photo 中反复匹配 template_img，每次匹配成功后遮蔽该区域，
    直到匹配失败或达到 max_count 次。
    返回所有匹配结果列表: [(dst, cx, cy, radius), ...]
    """
    search_img = photo.copy()
    results    = []

    for attempt in range(max_count):
        dst, good_matches = match_template_sift(template_img, search_img)

        if dst is None:
            break

        # 计算中心和半径
        cx = int(np.mean(dst[:, 0, 0]))
        cy = int(np.mean(dst[:, 0, 1]))
        w  = np.linalg.norm(dst[0, 0] - dst[3, 0])
        h  = np.linalg.norm(dst[0, 0] - dst[1, 0])
        radius = max(int(max(w, h) / 2), 20)

        results.append((dst, cx, cy, radius))
        print(f"   ✓ {template_name} 第{attempt + 1}次匹配成功  中心:({cx},{cy})  特征点:{len(good_matches)}")

        # 遮蔽已匹配区域，稍微扩大一圈防止边缘重复匹配
        expansion = int(max(w, h) * 0.1)
        center    = np.array([cx, cy], dtype=np.float32)
        expanded  = []
        for pt in np.int32(dst)[:, 0]:
            direction = pt.astype(np.float32) - center
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm * (norm + expansion)
            expanded.append((center + direction).astype(np.int32))
        expanded = np.array(expanded).reshape(-1, 1, 2)

        cv2.fillPoly(search_img, [expanded], color=(0, 0, 0))

    return results


# ===== 读取底图 =====
photo = cv2.imread(photo_path)
if photo is None:
    print(f"[错误] 底图读取失败: {photo_path}")
    exit()

print(f"底图尺寸: {photo.shape}")
result_img  = photo.copy()
total_count = 0

# ===== 颜色列表，每个模板用不同颜色区分 =====
colors = [
    (0, 0, 255),    (0, 255, 0),    (255, 0, 0),    (0, 255, 255),
    (255, 0, 255),  (255, 255, 0),  (0, 128, 255),  (128, 0, 255),
    (0, 255, 128),  (255, 128, 0),  (128, 255, 0),  (0, 128, 128),
    (128, 128, 0),
]

# ===== 依次匹配 t1 ~ t13 =====
for i in range(1, 14):
    template_path = os.path.join(target_dir, f"t{i}.jpg")
    template_img  = cv2.imread(template_path)

    if template_img is None:
        print(f"\n[跳过] t{i}.jpg 读取失败: {template_path}")
        continue

    print(f"\n{'='*40}")
    print(f"=== 匹配 t{i}.jpg ===")

    color   = colors[(i - 1) % len(colors)]
    results = match_all_occurrences(template_img, photo, f"t{i}")

    if not results:
        print(f"   ✗ t{i} 未找到任何匹配")
        continue

    for idx, (dst, cx, cy, radius) in enumerate(results):
        # 画匹配多边形框
        cv2.polylines(result_img, [np.int32(dst)], True, color, 2, cv2.LINE_AA)

        # 画圆
        cv2.circle(result_img, (cx, cy), radius, color, 3)

        # 标注文字（多次匹配加序号：t1, t1-2, t1-3...）
        label = f"t{i}" if idx == 0 else f"t{i}-{idx + 1}"
        font_scale = 1.0
        thickness  = 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        tx = cx - tw // 2
        ty = cy - radius - 10

        # 文字背景
        cv2.rectangle(result_img,
                      (tx - 4, ty - th - 4),
                      (tx + tw + 4, ty + 4),
                      (0, 0, 0), -1)

        # 写文字
        cv2.putText(result_img, label,
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

    total_count += len(results)
    print(f"   t{i} 共找到 {len(results)} 个匹配")

# ===== 保存结果 =====
output_path = os.path.join(result_dir, "photo1_matched.jpg")
cv2.imwrite(output_path, result_img)

print(f"\n{'='*40}")
print(f"全部模板匹配完成，共标记 {total_count} 个目标")
print(f"结果已保存: {output_path}")