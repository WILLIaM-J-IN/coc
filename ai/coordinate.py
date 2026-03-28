"""
coordinate.py
=============
1. Screenshots the LDPlayer window
2. Draws a coordinate grid + area zones
3. Saves annotated images to result/
4. Passes the detected game area to the RL agent

Usage:
    python coordinate.py

Output:
    result/coordinate_grid_<ts>.jpg   — coordinate grid
    result/coordinate_areas_<ts>.jpg  — area zones
    Prints key coordinates and passes game area to RLAgent
"""

import os
import cv2
import time
import numpy as np

from getwindow import get_window_rect
from shot import screenshot_window
from window_helper import get_window_area
from ai.transformer_rl import RLAgent

RESULT_DIR  = "result"
WINDOW_NAME = "雷电模拟器"
GRID_STEP   = 100   # grid line every N pixels

# Colors (BGR)
C_GRID      = (180, 180, 180)
C_LABEL     = (0,   230,   0)
C_GAME      = (0,   165, 255)   # orange
C_MAP       = (255,   0, 255)   # magenta
C_SIDEBAR   = (0,     0, 180)   # dark red overlay
C_CARD_UI   = (0,   200, 200)   # yellow overlay
C_BG        = (0,     0,   0)   # text background


def put(img, text, pos, scale=0.4, color=C_LABEL, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x-2, y-th-2), (x+tw+2, y+bl+2), C_BG, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_grid(img):
    """Draw coordinate grid and axis labels."""
    out = img.copy()
    h, w = out.shape[:2]

    for gx in range(0, w, GRID_STEP):
        cv2.line(out, (gx, 0), (gx, h), C_GRID, 1)
        if gx % 200 == 0 and gx > 0:
            put(out, str(gx), (gx + 2, 15), scale=0.4, color=(0, 210, 255))

    for gy in range(0, h, GRID_STEP):
        cv2.line(out, (0, gy), (w, gy), C_GRID, 1)
        if gy % 200 == 0 and gy > 0:
            put(out, str(gy), (2, gy + 14), scale=0.4, color=(0, 210, 255))

    # Origin
    cv2.circle(out, (0, 0), 8, (0, 0, 255), -1)
    put(out, "(0,0)", (10, 28), scale=0.5, color=(0, 0, 255))

    # Center crosshair
    cx, cy = w // 2, h // 2
    cv2.line(out, (cx-20, cy), (cx+20, cy), C_LABEL, 2)
    cv2.line(out, (cx, cy-20), (cx, cy+20), C_LABEL, 2)
    put(out, f"center ({cx},{cy})", (cx+5, cy-5), scale=0.45)

    put(out, f"size: {w}x{h}px", (5, h-8), scale=0.45,
        color=(255, 255, 255))
    return out


def draw_areas(img, rect, wa):
    """
    Highlight game area, map area, and excluded zones.
    All coordinates in screenshot space (origin = window top-left).
    """
    out     = img.copy()
    overlay = img.copy()
    h, w    = out.shape[:2]
    wx, wy  = rect[0], rect[1]

    def s(x, y):   # screen → screenshot
        return int(x - wx), int(y - wy)

    if wa:
        # Red overlay: excluded right sidebar
        cv2.rectangle(overlay,
                      s(wa.game_right, wa.game_top),
                      s(rect[2], wa.game_bottom),
                      (0, 0, 160), -1)

        # Yellow overlay: excluded card UI bar
        cv2.rectangle(overlay,
                      s(wa.map_left,  wa.map_bottom),
                      s(wa.map_right, wa.game_bottom),
                      (0, 200, 200), -1)

    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)

    if wa:
        # Game area border
        cv2.rectangle(out,
                      s(wa.game_left, wa.game_top),
                      s(wa.game_right, wa.game_bottom),
                      C_GAME, 3)
        put(out,
            f"GAME AREA  {wa.game_w}x{wa.game_h}px"
            f"  screen:({wa.game_left},{wa.game_top})"
            f"→({wa.game_right},{wa.game_bottom})",
            (s(wa.game_left, wa.game_top)[0] + 5,
             s(wa.game_left, wa.game_top)[1] + 22),
            scale=0.5, color=C_GAME)

        # Map area border
        cv2.rectangle(out,
                      s(wa.map_left, wa.map_top),
                      s(wa.map_right, wa.map_bottom),
                      C_MAP, 3)
        put(out,
            f"MAP AREA (template 11-15)  {wa.map_w}x{wa.map_h}px"
            f"  screen:({wa.map_left},{wa.map_top})"
            f"→({wa.map_right},{wa.map_bottom})",
            (s(wa.map_left, wa.map_top)[0] + 5,
             s(wa.map_left, wa.map_top)[1] + 44),
            scale=0.5, color=C_MAP)

    # Lightweight grid on areas image
    for gx in range(0, w, 200):
        cv2.line(out, (gx, 0), (gx, h), (100, 100, 100), 1)
        put(out, str(gx), (gx+2, 13), scale=0.35, color=(150, 150, 150))
    for gy in range(0, h, 200):
        cv2.line(out, (0, gy), (w, gy), (100, 100, 100), 1)
        put(out, str(gy), (2, gy+13), scale=0.35, color=(150, 150, 150))

    # Legend
    items = [
        (C_GAME,        "Game area — template 2-10 click range"),
        (C_MAP,         "Map area  — template 11-15 free click"),
        ((0,  0,  160), "Excluded: right sidebar"),
        ((0,  200, 200),"Excluded: card UI bar"),
    ]
    lx, ly = 10, h - 10 - len(items) * 22
    for color, label in items:
        cv2.rectangle(out, (lx, ly-12), (lx+18, ly+4), color, -1)
        put(out, label, (lx+24, ly), scale=0.48, color=(240, 240, 240))
        ly += 22

    return out


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    # ── 1. Get window ──────────────────────────────────────────────────
    rect = get_window_rect(WINDOW_NAME)
    if rect is None:
        print(f"[错误] 未找到窗口: {WINDOW_NAME}")
        return

    wa = get_window_area(rect)
    print(f"窗口坐标  : {rect}")
    print(f"区域信息  : {wa}")

    # ── 2. Screenshot ─────────────────────────────────────────────────
    img = screenshot_window(rect)
    if img is None:
        print("[错误] 截图失败")
        return

    print(f"截图尺寸  : {img.shape[1]}x{img.shape[0]}px")
    ts = int(time.time())

    # ── 3. Save grid image ────────────────────────────────────────────
    grid_img  = draw_grid(img)
    grid_path = os.path.join(RESULT_DIR, f"coordinate_grid_{ts}.jpg")
    cv2.imwrite(grid_path, grid_img)
    print(f"坐标网格图: {grid_path}")

    # ── 4. Save areas image ───────────────────────────────────────────
    areas_img  = draw_areas(img, rect, wa)
    areas_path = os.path.join(RESULT_DIR, f"coordinate_areas_{ts}.jpg")
    cv2.imwrite(areas_path, areas_img)
    print(f"区域标注图: {areas_path}")

    # ── 5. Pass game area to RL agent ─────────────────────────────────
    # The agent receives the game area as (left, top, w, h).
    # When the agent outputs normalised click coords (0-1),
    # they are scaled within this rectangle, then clamped by window_helper.
    #
    # template 2-10  → agent.window_rect = wa.as_game_rect()
    # template 11-15 → agent.window_rect = wa.as_map_rect()
    #
    # Both are set dynamically inside main.py _agent_click().
    # Here we just print the values for verification.

    if wa:
        print("\n── 传给 AI 模型的区域参数 ──")
        print(f"  game_rect (template 2-10)  : {wa.as_game_rect()}")
        print(f"  map_rect  (template 11-15) : {wa.as_map_rect()}")

        # Instantiate agent and log the area it will use
        agent = RLAgent(window_rect=wa.as_game_rect())
        print(f"\n  RLAgent 已初始化")
        print(f"  agent.window_rect = {agent.window_rect}")
        print(f"\n  模型只会在以下区域内点击:")
        print(f"    template 2-10  → 屏幕 ({wa.game_left},{wa.game_top})"
              f" → ({wa.game_right},{wa.game_bottom})")
        print(f"    template 11-15 → 屏幕 ({wa.map_left},{wa.map_top})"
              f" → ({wa.map_right},{wa.map_bottom})")

    print("\n── 关键坐标（截图坐标系，原点=窗口左上角）──")
    if wa:
        print(f"  游戏区域 : (0,0) → ({wa.game_w},{wa.game_h})")
        print(f"  地图区域 : (0,0) → ({wa.map_w},{wa.map_h})")
        print(f"  右侧栏宽 : {rect[2] - wa.game_right}px")
        print(f"  底部UI高 : {wa.game_bottom - wa.map_bottom}px")


if __name__ == "__main__":
    main()