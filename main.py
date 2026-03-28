import time
import os
import cv2
import numpy as np
from pynput.mouse import Controller, Button
from getwindow import get_window_rect
from shot import screenshot_window, find_template
from red_thread import detect_coc_border_from_image
from recognition import match_template_sift
from recognition_for_button import match_button
from ai import read_percentage, evaluate_score, RLAgent
from window_helper import get_window_area


class GameBot:
    def __init__(self, window_name="雷电模拟器"):
        self.window_name    = window_name
        self.mouse          = Controller()
        self.rect           = None
        self.wa             = None
        self.border_corners = None
        self.agent          = RLAgent(window_rect=None)
        self._depleted_templates = set()

        if not os.path.exists("result"):
            os.makedirs("result")

        self.cfg = {
            'threshold_default':    0.6,
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
            'attack_templates': {
                2:  "compare/example2.jpg",
                3:  "compare/example3.jpg",
                4:  "compare/example4.jpg",
                5:  "compare/example5.jpg",
                6:  "compare/example6.jpg",
                7:  "compare/example7.jpg",
                8:  "compare/example8.jpg",
                9:  "compare/example9.jpg",
                10: "compare/example10.jpg",
                11: "compare/example11.jpg",
                12: "compare/example12.jpg",
                13: "compare/example13.jpg",
                14: "compare/example14.jpg",
                15: "compare/example15.jpg",
            }
        }

    # ------------------------------------------------------------------ #
    # 窗口更新
    # ------------------------------------------------------------------ #

    def update_window_rect(self):
        self.rect = get_window_rect(self.window_name)
        if not self.rect:
            print(f"警告: 未找到窗口 [{self.window_name}]")
            self.wa = None
            return False
        self.wa = get_window_area(self.rect)
        return True

    # ------------------------------------------------------------------ #
    # ★ 所有鼠标点击的唯一出口 ★
    # ------------------------------------------------------------------ #

    def _safe_mouse_click(self, x, y, times=1, button=Button.left):
        """
        所有鼠标点击必须经过这里。
        在移动鼠标前强制将坐标限制在游戏区域内（game area），
        杜绝任何路径点击到窗口外或右侧控制栏。
        """
        x, y = int(x), int(y)

        if self.wa is not None:
            # 强制限制在 game area（排除右侧控制栏）
            x, y = self.wa.clamp_game(x, y)
        elif self.rect is not None:
            # wa 不可用时用 rect 兜底
            xl, yt, xr, yb = self.rect
            margin = 15
            x = max(xl + margin, min(x, xr - margin))
            y = max(yt + margin, min(y, yb - margin))

        self.mouse.position = (x, y)
        for _ in range(times):
            self.mouse.click(button, 1)

    # ------------------------------------------------------------------ #
    # 原有函数（_click 内部走 _safe_mouse_click）
    # ------------------------------------------------------------------ #

    def _click(self, x, y, button=Button.left, count=1, interval=0.05):
        """普通单次点击，经过 _safe_mouse_click 确保坐标安全。"""
        if count == 1:
            self._safe_mouse_click(x, y, times=1, button=button)
        else:
            # 多次点击（带间隔）
            x, y = int(x), int(y)
            if self.wa is not None:
                x, y = self.wa.clamp_game(x, y)
            elif self.rect is not None:
                xl, yt, xr, yb = self.rect
                x = max(xl+15, min(x, xr-15))
                y = max(yt+15, min(y, yb-15))
            self.mouse.position = (x, y)
            time.sleep(0.05)
            for _ in range(count):
                self.mouse.click(button, 1)
                time.sleep(interval)

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

    def _drag_left(self):
        if not self.rect: return
        x, y, x2, y2 = self.rect
        w = x2 - x
        h = y2 - y
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
                    img, template_path,
                    threshold=self.cfg['threshold_default'],
                    color_diff_threshold=self.cfg['color_diff_threshold']
                )
                if pos:
                    # _click → _safe_mouse_click 确保坐标安全
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
    # 辅助函数
    # ------------------------------------------------------------------ #

    def _screenshot_img(self):
        if not self.update_window_rect():
            return None
        return screenshot_window(self.rect)

    def _img_to_screen(self, img_x, img_y):
        if not self.rect:
            return img_x, img_y
        return self.rect[0] + img_x, self.rect[1] + img_y

    def _is_card_gray(self, img, card_cx, card_cy,
                      card_w=60, card_h=60, sat_threshold=30):
        h, w = img.shape[:2]
        x1 = max(0, card_cx - card_w // 2)
        y1 = max(0, card_cy - card_h // 2)
        x2 = min(w, card_cx + card_w // 2)
        y2 = min(h, card_cy + card_h // 2)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        hsv        = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].mean()
        print(f"   >>> 卡牌饱和度: {saturation:.1f}  (阈值 {sat_threshold})")
        return saturation < sat_threshold

    # ------------------------------------------------------------------ #
    # 边框识别
    # ------------------------------------------------------------------ #

    def detect_battle_border(self, save_debug=True):
        img = self._screenshot_img()
        if img is None:
            print("[边界] 截图失败")
            return None

        corners = detect_coc_border_from_image(img)
        if corners is None:
            return None

        h, w = img.shape[:2]
        for name, pt in corners.items():
            if abs(pt[0]) > w * 5 or abs(pt[1]) > h * 5:
                print(f"[边界] 顶点 {name}={pt} 坐标异常，视为识别失败")
                return None

        if save_debug:
            debug = img.copy()
            try:
                pts      = np.array([corners['top'], corners['right'],
                                     corners['bottom'], corners['left']], dtype=np.int64)
                pts_draw = np.clip(pts, [0, 0], [w-1, h-1]).astype(np.int32)
                cv2.polylines(debug, [pts_draw], isClosed=True,
                              color=(0, 255, 0), thickness=4)
                for name, pt in corners.items():
                    px = max(0, min(pt[0], w-1))
                    py = max(0, min(pt[1], h-1))
                    cv2.circle(debug, (px, py), 15, (0, 0, 255), -1)
                    cv2.putText(debug, name, (px+10, py+10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            except Exception as e:
                print(f"[边界] 绘制失败: {e}")

            debug_path = os.path.join("result", f"border_{int(time.time())}.jpg")
            cv2.imwrite(debug_path, debug)
            print(f"   >>> 边框图已保存: {debug_path}")

        return corners

    # ------------------------------------------------------------------ #
    # 攻击函数
    # ------------------------------------------------------------------ #

    def _click_outside_border_default(self, template_num, times=20, offset=80):
        """降级点击：边框左角外侧，经 _safe_mouse_click 限制。"""
        if self.border_corners:
            left_pt = self.border_corners['left']
            sx, sy  = self._img_to_screen(left_pt[0] - offset, left_pt[1])
        else:
            if not self.rect: return
            sx = self.rect[0] + 60
            sy = self.rect[1] + int((self.rect[3] - self.rect[1]) / 2)

        print(f"   >>> 降级连点 {times} 次: 屏幕({sx},{sy})")
        # _safe_mouse_click 会自动 clamp 到 game area
        self._safe_mouse_click(sx, sy, times=times)

    def find_and_click_attack_once(self, template_num, custom_path=None):
        """单次 SIFT 补点，不循环等待。"""
        template_path = custom_path or self.cfg['attack_templates'].get(template_num)
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
            # 卡牌点击走 _click → _safe_mouse_click
            self._click(self.rect[0] + cx, self.rect[1] + cy)
            print(f"   -> 补点 template{template_num} 成功: ({cx},{cy})")
            return True

        print(f"   -> 补点 template{template_num} 未找到，跳过")
        return False

    def _find_card_pos(self, template_num):
        """SIFT 匹配卡牌，返回 (cx, cy, img)，找不到返回 (None, None, img)。"""
        template_path = self.cfg['attack_templates'].get(template_num)
        if not template_path: return None, None, None

        template_img = cv2.imread(template_path)
        if template_img is None: return None, None, None

        if not self.update_window_rect(): return None, None, None
        img = screenshot_window(self.rect)
        if img is None: return None, None, None

        dst, _ = match_template_sift(template_img, img)
        if dst is None:
            return None, None, img

        cx = int(np.mean(dst[:, 0, 0]))
        cy = int(np.mean(dst[:, 0, 1]))
        return cx, cy, img

    def _agent_click(self, template_num, screenshot, times=20):
        """
        agent 决定点击位置，经 _safe_mouse_click（终极保险）连点 times 次。

        template 11-15 → agent.window_rect = map area
        template 2-10  → agent.window_rect = game area
        两者最终都经过 _safe_mouse_click → clamp_game 兜底。
        """
        if self.wa is None:
            self._click_outside_border_default(template_num, times=times)
            return

        # 给 agent 传正确的参考区域（决定归一化坐标缩放到哪个矩形）
        if template_num in {11, 12, 13, 14, 15}:
            self.agent.window_rect = self.wa.as_map_rect()
        else:
            self.agent.window_rect = self.wa.as_game_rect()

        try:
            action = self.agent.select_action(screenshot, self.border_corners)
            raw_x  = action['click_x']
            raw_y  = action['click_y']

            print(f"   >>> [Agent] template{template_num}"
                  f"  原始:({raw_x},{raw_y})")

            # _safe_mouse_click 是最终出口，强制 clamp 到 game area
            self._safe_mouse_click(raw_x, raw_y, times=times)

        except Exception as e:
            print(f"   >>> [Agent] 决策失败({e})，降级")
            self._click_outside_border_default(template_num, times=times)

    def _deploy_template_until_depleted(self, template_num, max_rounds=15):
        """循环点击同一个 template 直到卡牌变灰（耗尽）为止。"""
        print(f"   >>> 开始循环部署 template{template_num}（直到耗尽）")

        for round_i in range(max_rounds):
            print(f"   >>> template{template_num} 第 {round_i+1} 轮")

            # 1. 找卡牌位置
            card_cx, card_cy, img = self._find_card_pos(template_num)
            if card_cx is None:
                print(f"   -> template{template_num} 找不到卡牌，视为耗尽停止")
                self._depleted_templates.add(template_num)
                return

            # 2. 点击卡牌（走 _click → _safe_mouse_click）
            self._click(self.rect[0] + card_cx, self.rect[1] + card_cy)
            print(f"   -> template{template_num} 点击成功: ({card_cx},{card_cy})")

            # 3. agent 决定地图点击位置，连点 20 次
            #    （走 _agent_click → _safe_mouse_click）
            screenshot = self._screenshot_img()
            if screenshot is not None:
                self._agent_click(template_num, screenshot, times=20)
            else:
                self._click_outside_border_default(template_num, times=20)

            # 4. 检测卡牌是否变灰
            time.sleep(0.3)
            img_after = self._screenshot_img()
            if img_after is not None:
                depleted = self._is_card_gray(img_after, card_cx, card_cy)
                if depleted:
                    print(f"   -> template{template_num} 已变灰（耗尽），停止循环")
                    self._depleted_templates.add(template_num)
                    return

        print(f"   -> template{template_num} 达到最大轮数 {max_rounds}，强制停止")
        self._depleted_templates.add(template_num)

    # ------------------------------------------------------------------ #
    # run_loop（不改动）
    # ------------------------------------------------------------------ #

    def run_loop(self):
        if not self.update_window_rect(): return
        print("程序启动，启用颜色严格比对...")
        print(f"窗口区域: {self.wa}")
        round_count = 1

        while True:
            print(f"\n======== 开始第 {round_count} 轮循环 ========")
            self.border_corners      = None
            self._depleted_templates = set()

            # ── Step 1 ~ 3 ──────────────────────────────────────────────
            if self.find_and_click_step(1):
                time.sleep(3)

            if self.find_and_click_step(2):
                time.sleep(3)

            if self.find_and_click_step(3):
                time.sleep(3)

            # ── Step 3 后等待 10 秒，让攻击界面完全加载 ──────────────────
            print("=== Step 3 完成，等待 10 秒后识别红线边框 ===")
            time.sleep(10)

            # ── Step 4：先检测窗口区域，再识别红线边框 ──────────────────
            print("=== Step 4：检测窗口安全区域 ===")
            self.update_window_rect()
            if self.wa:
                print(f"   游戏区域: {self.wa}")

            print("=== Step 4：识别红线边框 ===")
            self.border_corners = self.detect_battle_border(save_debug=True)
            if self.border_corners:
                print(f"   边框识别成功: {self.border_corners}")
            else:
                print("   边框识别失败，后续使用降级坐标")

            # 最左侧固定点击坐标
            if self.wa:
                left_x = self.wa.game_left + 15
                left_y = self.wa.game_top + self.wa.game_h // 2
            else:
                left_x = self.rect[0] + 15
                left_y = self.rect[1] + int((self.rect[3] - self.rect[1]) / 2)

            # ── 第一段：2-10 全部跑完（10放中间，7不补点）────────────────
            print("=== 第一段：部署 template2-10 ===")
            SEQUENCE_2_10 = [2, 3, 4, 5, 10, 6, 7, 8, 9]

            for template_num in SEQUENCE_2_10:
                print(f"\n─── 部署 template{template_num} → 最左侧 ({left_x},{left_y}) ───")

                for _ in range(20):
                    card_cx, card_cy, _ = self._find_card_pos(template_num)
                    if card_cx is None:
                        print(f"   template{template_num} 找不到，视为耗尽")
                        break

                    self._click(self.rect[0] + card_cx, self.rect[1] + card_cy)
                    print(f"   template{template_num} 点击成功")

                    self._safe_mouse_click(left_x, left_y, times=20)

                    # 补点逻辑（7不补点）
                    if template_num == 2:
                        self.find_and_click_attack_once(2)
                    elif template_num == 8:
                        self.find_and_click_attack_once(8)
                    elif template_num == 10:
                        self.find_and_click_attack_once(
                            10,
                            custom_path=r"E:\PythonProject\coc_resource\compare\example10_new.jpg"
                        )

                    time.sleep(0.3)
                    img_after = self._screenshot_img()
                    if img_after is not None and self._is_card_gray(img_after, card_cx, card_cy):
                        print(f"   template{template_num} 已变灰（耗尽）")
                        break

            print("=== 第一段完成 ===")

            # ── 第二段：2-10 全部跑完后，进入 agent（11-15）────────────
            print("\n=== 第二段：agent 驱动 template11-15 ===")

            # 先把 12,13,14,15 各释放一次
            for template_num in [12, 13, 14, 15]:
                print(f"\n─── 释放 template{template_num}（一次）───")
                card_cx, card_cy, _ = self._find_card_pos(template_num)
                if card_cx is None:
                    print(f"   template{template_num} 找不到，跳过")
                    continue

                self._click(self.rect[0] + card_cx, self.rect[1] + card_cy)
                print(f"   template{template_num} 点击成功")

                screenshot = self._screenshot_img()
                if screenshot is not None and self.wa:
                    self.agent.window_rect = self.wa.as_game_rect()
                    action = self.agent.select_action(screenshot, self.border_corners)
                    self._safe_mouse_click(action['click_x'], action['click_y'], times=1)
                    print(f"   释放位置: ({action['click_x']},{action['click_y']})")

            # template11 agent 循环释放直到耗尽
            print("\n─── template11 agent 循环释放 ───")
            last_click_pos = None
            MIN_DIST = 80
            consecutive_miss = 0
            MAX_MISS = 5
            loop_count = 0
            MAX_LOOPS = 50

            while loop_count < MAX_LOOPS:
                loop_count += 1
                print(f"   template11 第 {loop_count} 轮")

                card_cx, card_cy, _ = self._find_card_pos(11)
                if card_cx is None:
                    consecutive_miss += 1
                    print(f"   template11 找不到 ({consecutive_miss}/{MAX_MISS})")
                    if consecutive_miss >= MAX_MISS:
                        print("   template11 已耗尽，结束")
                        break
                    continue

                consecutive_miss = 0
                self._click(self.rect[0] + card_cx, self.rect[1] + card_cy)
                print(f"   template11 点击成功: ({card_cx},{card_cy})")

                screenshot = self._screenshot_img()
                if screenshot is not None and self.wa:
                    self.agent.window_rect = self.wa.as_game_rect()
                    action = self.agent.select_action(screenshot, self.border_corners)
                    click_x = action['click_x']
                    click_y = action['click_y']

                    # 和上次位置不同
                    if last_click_pos is not None:
                        dist = ((click_x - last_click_pos[0]) ** 2 +
                                (click_y - last_click_pos[1]) ** 2) ** 0.5
                        if dist < MIN_DIST:
                            click_y = click_y + MIN_DIST
                            click_x, click_y = self.wa.clamp_game(click_x, click_y)
                            print(f"   位置太近，调整为 ({click_x},{click_y})")

                    self._safe_mouse_click(click_x, click_y, times=1)
                    last_click_pos = (click_x, click_y)
                    print(f"   释放位置: ({click_x},{click_y})")
                else:
                    self._click_outside_border_default(11, times=1)

                time.sleep(0.3)
                img_after = self._screenshot_img()
                if img_after is not None and self._is_card_gray(img_after, card_cx, card_cy):
                    print("   template11 已变灰（耗尽），结束")
                    break

            print(f"\n=== 攻击循环结束，等待 30 秒 ===")
            time.sleep(30)



            # ── Step 5 ──────────────────────────────────────────────────
            if self.find_and_click_step(5):
                time.sleep(3)

            # ── Step 6 + 百分比识别 + agent 更新 ────────────────────────
            if self.find_and_click_step(6):
                time.sleep(3)

                print("=== 识别战斗结果百分比 ===")
                time.sleep(3)
                img = self._screenshot_img()
                if img is not None:
                    pct    = read_percentage(img)
                    rating = evaluate_score(pct)
                    print(f"[结果] 本次得分: {pct}%  评级: {rating}")

                    self.agent.record_reward(pct, done=True)
                    self.agent.update()

            # ── Step 7 ──────────────────────────────────────────────────
            if self.find_and_click_step(7):
                print("   Step 7 完成")

            print(f"\n======== 第 {round_count} 轮结束，休息 5 秒后从 Step 1 重新开始 ========")
            time.sleep(5)
            round_count += 1


if __name__ == "__main__":
    time.sleep(6)
    bot = GameBot("雷电模拟器")
    bot.run_loop()