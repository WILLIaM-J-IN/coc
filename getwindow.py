import win32gui




def get_window_rect(window_name="雷电模拟器"):
    """
    获取指定窗口的客户端区域坐标（去掉标题栏和边框）
    返回 (left, top, right, bottom)
    """
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd == 0:
        print(f"窗口未找到: {window_name}")
        return None


    left, top, right, bottom = win32gui.GetWindowRect(hwnd)


    TITLE_BAR_HEIGHT = 30  # 标题栏高度
    BORDER_WIDTH = 8  # 窗口边框宽度

    game_left = left + BORDER_WIDTH
    game_top = top + TITLE_BAR_HEIGHT
    game_right = right + 90
    game_bottom = bottom + 50

    rect = (game_left, game_top, game_right, game_bottom)
    print("窗口坐标（去掉边框和标题栏）:", rect)
    return rect