import time
import os
import cv2
import numpy as np
from pynput.mouse import Controller, Button
from getwindow import get_window_rect
from shot import screenshot_window, find_template
from red_thread import detect_coc_border_from_image  # 边界检测
from recognition import match_template_sift
from recognition_for_button import match_button
from ai import read_percentage, evaluate_score, RLAgent

class GameBot:
    def __init__(self, window_name="雷电模拟器"):
        self.window_name = window_name
        self.mouse = Controller()
        self.rect = None
        self.border_corners = None  # 存储当前轮次检测到的菱形边界

        if not os.path.exists("result"):
            os.makedirs("result")

        self.cfg = {
            'threshold_default': 0.6,
            'color_diff_threshold': 45.0,
            'steps': {
                1: "data_for_search/step1.jpg",
                2: "data_for_search/step2.png",
                3: "data_for_search/step3.png",
                4: "data_for_search/step4.png",
                5: "data_for_search/step5.png",
                6: "data_for_search/step6.png",
                7: "data_for_search/step7.png",
            },
            # Step4 后攻击循环专用，与 steps 完全独立
            'attack_templates': {
                2: "compare/example2.jpg",
                3: "compare/example3.jpg",
                4: "compare/example4.jpg",
                5: "compare/example5.jpg",
                6: "compare/example6.jpg",
                7: "compare/example7.jpg",
                8: "compare/example8.jpg",
                9: "compare/example9.jpg",
                10: "compare/example10.jpg",
                11: "compare/example11.jpg",
                12: "compare/example12.jpg",
                13: "compare/example13.jpg",
                14: "compare/example14.jpg",
                15: "compare/example15.jpg",
            }
        }

    # ------------------------------------------------------------------ #
    # 原有函数（不改动）
    # ------------------------------------------------------------------ #

    def update_window_rect(self):
        self.rect = get_window_rect(self.window_name)
        if not self.rect:
            print(f"警告: 未找到窗口 [{self.window_name}]")
            return False
        return True

    def _save_debug_image(self, img, center_pos, template_path, step_name):
        try:
            template = cv2.imread(template_path)
            if template is None: return
            h, w = template.shape[:2]
            cx, cy = center_pos
            top_left     = (int(cx - w/2), int(cy - h/2))
            bottom_right = (int(cx + w/2), int(cy + h/2))
            debug_img = img.copy()
            cv2.rectangle(debug_img, top_left, bottom_right, (0, 0, 255), 3)
            cv2.putText(debug_img, str(step_name), (top_left[0], top_left[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            filename = f"result/match_{step_name}_{int(time.time())}.jpg"
            cv2.imwrite(filename, debug_img)
            print(f"   >>> [调试] 已保存: {filename}")
        except Exception as e:
            print(f"   >>> [调试] 保存出错: {e}")

    def _click(self, x, y, button=Button.left, count=1, interval=0.05):
        self.mouse.position = (int(x), int(y))
        time.sleep(0.05)
        for _ in range(count):
            self.mouse.click(button, 1)
            if count > 1:
                time.sleep(interval)

    def _drag_left(self):
        if not self.rect: return
        x, y, w, h = self.rect
        start_x = int(x + w/2)
        start_y = int(y + h/2)
        end_x   = int(x + w*0.1)
        print("   >>> 执行向左拖拽: 中心 -> 左侧")
        self.mouse.position = (start_x, start_y)
        time.sleep(0.2)
        self.mouse.press(Button.left)
        time.sleep(0.2)
        step_x = (end_x - start_x) / 15
        for i in range(15):
            self.mouse.position = (int(start_x + step_x*(i+1)), start_y)
            time.sleep(0.02)
        time.sleep(0.2)
        self.mouse.release(Button.left)
        time.sleep(0.5)

    def find_and_click_step(self, step_num, timeout=20):
        template_path = self.cfg['steps'].get(step_num)
        if not template_path: return False

        template_img = None
        if step_num > 4:
            template_img = cv2.imread(template_path)
            if template_img is None:
                print(f"   -> Step {step_num} 模板图读取失败: {template_path}")
                return False

        print(f"=== 正在查找 Step {step_num} ===")
        start_time = time.time()

        while time.time() - start_time < timeout:
            if not self.update_window_rect(): return False

            img = screenshot_window(self.rect)
            if img is None:
                time.sleep(0.5)
                continue

            if step_num <= 4:
                pos = find_template(
                    img,
                    template_path,
                    threshold=self.cfg['threshold_default'],
                    color_diff_threshold=self.cfg['color_diff_threshold']
                )
                if pos:
                    self._click(self.rect[0] + pos[0], self.rect[1] + pos[1])
                    print(f"   -> Step {step_num} 点击成功")
                    return True

            else:
                match = match_button(template_img, img, threshold=0.6)
                if match is not None:
                    cx, cy = match['cx'], match['cy']
                    self._click(self.rect[0] + cx, self.rect[1] + cy)
                    print(f"   -> Step {step_num} 点击成功: ({cx},{cy})  得分: {match['score']:.3f}")
                    return True

            time.sleep(0.5)

        print(f"   -> Step {step_num} 未找到 (超时)")
        return False

    def click_right_side_repeatedly(self, count=20):
        if not self.rect: return
        print(f"   >>> 在窗口右侧快速点击 {count} 次...")
        target_x = self.rect[2] - 600
        target_y = int((self.rect[1] + self.rect[3]) / 2)
        self._click(target_x, target_y, count=count, interval=0.04)

    # ------------------------------------------------------------------ #
    # 新增函数
    # ------------------------------------------------------------------ #

    def _screenshot_img(self):
        if not self.update_window_rect():
            return None
        return screenshot_window(self.rect)

    def detect_battle_border(self, save_debug=True):
        img = self._screenshot_img()
        if img is None:
            print("[边界] 截图失败")
            return None

        corners = detect_coc_border_from_image(img)

        if corners is None:
            return None

        # 坐标合法性检查：所有点必须在图像范围的合理倍数内
        h, w = img.shape[:2]
        for name, pt in corners.items():
            if abs(pt[0]) > w * 5 or abs(pt[1]) > h * 5:
                print(f"[边界] 顶点 {name}={pt} 坐标异常，视为识别失败")
                return None

        if save_debug:
            debug = img.copy()
            try:
                pts = np.array([corners['top'], corners['right'],
                                corners['bottom'], corners['left']], dtype=np.int64)
                # 绘制时裁剪到图像范围内
                pts_draw = np.clip(pts, [0, 0], [w - 1, h - 1]).astype(np.int32)
                cv2.polylines(debug, [pts_draw], isClosed=True, color=(0, 255, 0), thickness=4)
                for name, pt in corners.items():
                    px = max(0, min(pt[0], w - 1))
                    py = max(0, min(pt[1], h - 1))
                    cv2.circle(debug, (px, py), 15, (0, 0, 255), -1)
                    cv2.putText(debug, name, (px + 10, py + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            except Exception as e:
                print(f"[边界] 绘制失败: {e}")

            debug_path = os.path.join("result", f"border_{int(time.time())}.jpg")
            cv2.imwrite(debug_path, debug)
            print(f"   >>> 边框图已保存: {debug_path}")

        return corners

    def match_attack_screen(self, template_path, save_debug=True):
        template_img = cv2.imread(template_path)
        if template_img is None:
            print(f"[SIFT] 模板图片读取失败: {template_path}")
            return False

        img = self._screenshot_img()
        if img is None:
            return False

        dst, good_matches = match_template_sift(template_img, img)

        if dst is None:
            return False

        if save_debug:
            debug = img.copy()
            cv2.polylines(debug, [np.int32(dst)], True, (0,0,255), 3, cv2.LINE_AA)
            cv2.imwrite(f"result/sift_{int(time.time())}.jpg", debug)

        return True

    def _img_to_screen(self, img_x, img_y):

        if not self.rect:
            return img_x, img_y
        return self.rect[0] + img_x, self.rect[1] + img_y

    def click_border_corner(self, corner_name, offset_x=0, offset_y=0):

        if not self.border_corners:
            print(f"[点击] 边界未检测，跳过点击 {corner_name}")
            return False
        pt = self.border_corners.get(corner_name)
        if pt is None:
            return False
        sx, sy = self._img_to_screen(pt[0] + offset_x, pt[1] + offset_y)
        print(f"   >>> 点击 {corner_name} 角: 屏幕({sx}, {sy})")
        self._click(sx, sy)
        return True

    # ------------------------------------------------------------------ #
    # 修改 run_loop
    # ------------------------------------------------------------------ #
    def _click_outside_border_multi(self, times=15, offset=80):

        if self.border_corners:
            left_pt = self.border_corners['left']
            sx, sy = self._img_to_screen(
                left_pt[0] - offset,
                left_pt[1]
            )
        else:
            if not self.rect: return
            sx = self.rect[0] + 60
            sy = self.rect[1] + int((self.rect[3] - self.rect[1]) / 2)

        print(f"   >>> 边框外连点 {times} 次: 屏幕({sx}, {sy})")
        for _ in range(times):
            self.mouse.position = (int(sx), int(sy))
            self.mouse.click(Button.left, 1)

    def detect_battle_border(self, save_debug=True):
        img = self._screenshot_img()
        if img is None:
            print("[边界] 截图失败")
            return None

        corners = detect_coc_border_from_image(img)

        if save_debug:
            debug = img.copy()
            if corners:
                pts = np.array([corners['top'], corners['right'],
                                corners['bottom'], corners['left']], dtype=np.int32)
                cv2.polylines(debug, [pts], isClosed=True, color=(0, 255, 0), thickness=4)
                for name, pt in corners.items():
                    px = max(0, min(pt[0], img.shape[1] - 1))
                    py = max(0, min(pt[1], img.shape[0] - 1))
                    cv2.circle(debug, (px, py), 15, (0, 0, 255), -1)
                    cv2.putText(debug, name, (px + 10, py + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # 保存到 result 文件夹
            filename = os.path.join("result", f"border_{int(time.time())}.jpg")
            cv2.imwrite(filename, debug)
            print(f"   >>> 边框图已保存: {filename}")

        return corners

    def find_and_click_attack_once(self, template_num):

        template_path = self.cfg['attack_templates'].get(template_num)
        if not template_path: return False

        template_img = cv2.imread(template_path)
        if template_img is None:
            print(f"   -> 补点 template{template_num} 模板读取失败")
            return False

        if not self.update_window_rect(): return False

        img = screenshot_window(self.rect)
        if img is None: return False

        dst, _ = match_template_sift(template_img, img)
        if dst is not None:
            cx = int(np.mean(dst[:, 0, 0]))
            cy = int(np.mean(dst[:, 0, 1]))
            self._click(self.rect[0] + cx, self.rect[1] + cy)
            print(f"   -> 补点 template{template_num} 成功: 窗口内({cx}, {cy})")
            return True

        print(f"   -> 补点 template{template_num} 未找到，跳过")
        return False




    def find_and_click_attack(self, template_num, timeout=20):

        template_path = self.cfg['attack_templates'].get(template_num)
        if not template_path: return False

        template_img = cv2.imread(template_path)
        if template_img is None:
            print(f"   -> attack template{template_num} 读取失败: {template_path}")
            return False

        print(f"=== 查找 attack template{template_num} (SIFT: {template_path}) ===")
        start_time = time.time()

        while time.time() - start_time < timeout:
            if not self.update_window_rect(): return False

            img = screenshot_window(self.rect)
            if img is None:
                time.sleep(0.5)
                continue

            dst, _ = match_template_sift(template_img, img)

            if dst is not None:
                cx = int(np.mean(dst[:, 0, 0]))
                cy = int(np.mean(dst[:, 0, 1]))
                self._click(self.rect[0] + cx, self.rect[1] + cy)
                print(f"   -> attack template{template_num} 点击成功: 窗口内({cx}, {cy})")
                return True

            time.sleep(0.5)

        print(f"   -> attack template{template_num} 未找到 (超时)")
        return False

    def run_loop(self):
        if not self.update_window_rect(): return
        print("程序启动，启用颜色严格比对...")
        round_count = 1

        while True:
            print(f"\n======== 开始第 {round_count} 轮循环 ========")
            self.border_corners = None

            # Step 1 ~ 3：原逻辑不变
            if self.find_and_click_step(1):
                time.sleep(3)

            if self.find_and_click_step(2):
                time.sleep(3)

            if self.find_and_click_step(3):
                time.sleep(3)

            # ===== Step 4：不做图片匹配，直接执行红线识别 + 攻击循环 =====
            print("=== Step 4：识别红线边框 ===")
            self.border_corners = self.detect_battle_border(save_debug=True)
            if self.border_corners:
                print(f"   边框识别成功: {self.border_corners}")
            else:
                print("   边框识别失败，后续使用降级坐标")

            # 攻击循环 example2~15，每个点击后边框外连点 20 次
            for template_num in range(2, 16):
                print(f"=== 攻击 template{template_num} ===")
                self.find_and_click_attack(template_num)
                self._click_outside_border_multi(times=20, offset=80)
                if template_num == 2:
                    print(f"=== 补点 example2 ===")
                    self.find_and_click_attack_once(2)
                elif template_num == 8:
                    print(f"=== 补点 example8 ===")
                    self.find_and_click_attack_once(8)
                elif template_num == 10:
                    print(f"=== 补点 example10（使用新模板）===")
                    # example10 补点专用新模板
                    template_img = cv2.imread(r"E:\PythonProject\royaleai\compare\example10_new.jpg")
                    if template_img is not None:
                        if not self.update_window_rect(): continue
                        img = screenshot_window(self.rect)
                        if img is not None:
                            dst, _ = match_template_sift(template_img, img)
                            if dst is not None:
                                cx = int(np.mean(dst[:, 0, 0]))
                                cy = int(np.mean(dst[:, 0, 1]))
                                self._click(self.rect[0] + cx, self.rect[1] + cy)
                                print(f"   -> 补点 example10 成功: 窗口内({cx}, {cy})")
                            else:
                                print("   -> 补点 example10 未找到，跳过")
                    else:
                        print("   -> example10_new.jpg 读取失败，跳过补点")

            # 攻击循环结束后，再次识别并补点 example8 和 example10
            print("=== 攻击循环结束===")


            print("=== 攻击完毕，等待 40 秒 ===")
            time.sleep(40)
            # ===== Step 4 结束 =====

            # Step 5 ~ 7：原逻辑不变
            if self.find_and_click_step(5):
                time.sleep(3)

            if self.find_and_click_step(6):
                time.sleep(3)

                # 等3秒后截图识别百分比
                print("=== 识别战斗结果百分比 ===")
                time.sleep(3)
                img = self._screenshot_img()
                if img is not None:
                    pct = read_percentage(img)
                    rating = evaluate_score(pct)
                    print(f"[结果] 本次得分: {pct}%  评级: {rating}")

            if self.find_and_click_step(7):
                print("   Step 7 完成")


            print(f"======== 第 {round_count} 轮结束，休息 5 秒后从 Step 1 重新开始 ========")
            time.sleep(5)
            round_count += 1





if __name__ == "__main__":
    time.sleep(6)
    bot = GameBot("雷电模拟器")
    bot.run_loop()