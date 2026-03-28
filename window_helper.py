"""
window_helper.py
================
Detects LDPlayer window layout and provides safe click areas.

LDPlayer layout:
  ┌──────────────────────────────────┬────────┐
  │                                  │  side  │
  │          MAP AREA                │  bar   │
  │      (template 11-15)            │        │
  ├──────────────────────────────────┤        │
  │         CARD UI BAR              │        │
  └──────────────────────────────────┴────────┘

Coordinate origin = top-left of the window rect.
"""

SIDEBAR_W        = 50    # right-side emulator control bar (px)
MAP_BOTTOM_RATIO = 0.72  # map ends at 72% of window height
CLICK_MARGIN     = 15    # minimum distance from any boundary


class WindowArea:
    """
    All clickable areas within LDPlayer, in SCREEN coordinates.

    Attributes (all screen coords):
        game_left/top/right/bottom  — full game area (sidebar excluded)
        map_left/top/right/bottom   — map area (card UI excluded)
        game_w, game_h              — game area dimensions
        map_w,  map_h               — map area dimensions
    """

    def __init__(self, rect):
        """
        Args:
            rect: (x_left, y_top, x_right, y_bottom) from get_window_rect
        """
        x_left, y_top, x_right, y_bottom = rect

        # Game area: full window minus right sidebar
        self.game_left   = x_left
        self.game_top    = y_top
        self.game_right  = x_right  - SIDEBAR_W
        self.game_bottom = y_bottom

        self.game_w = self.game_right  - self.game_left
        self.game_h = self.game_bottom - self.game_top

        # Map area: game area minus bottom card UI bar
        self.map_left   = self.game_left
        self.map_top    = self.game_top
        self.map_right  = self.game_right
        self.map_bottom = self.game_top + int(self.game_h * MAP_BOTTOM_RATIO)

        self.map_w = self.map_right  - self.map_left
        self.map_h = self.map_bottom - self.map_top

    # ── Clamping ──────────────────────────────────────────────────────

    def clamp_game(self, x, y):
        """Clamp screen coords to game area."""
        x = max(self.game_left + CLICK_MARGIN,
                min(int(x), self.game_right  - CLICK_MARGIN))
        y = max(self.game_top  + CLICK_MARGIN,
                min(int(y), self.game_bottom - CLICK_MARGIN))
        return x, y

    def clamp_map(self, x, y):
        """Clamp screen coords to map area (template 11-15)."""
        x = max(self.map_left + CLICK_MARGIN,
                min(int(x), self.map_right  - CLICK_MARGIN))
        y = max(self.map_top  + CLICK_MARGIN,
                min(int(y), self.map_bottom - CLICK_MARGIN))
        return x, y

    # ── Normalised → absolute ─────────────────────────────────────────

    def norm_to_game(self, nx, ny):
        """Convert normalised (0-1) to absolute game area screen coords."""
        x = self.game_left + int(nx * self.game_w)
        y = self.game_top  + int(ny * self.game_h)
        return self.clamp_game(x, y)

    def norm_to_map(self, nx, ny):
        """Convert normalised (0-1) to absolute map area screen coords."""
        x = self.map_left + int(nx * self.map_w)
        y = self.map_top  + int(ny * self.map_h)
        return self.clamp_map(x, y)

    # ── Agent-friendly tuple ──────────────────────────────────────────

    def as_game_rect(self):
        """Return game area as (left, top, w, h) for agent.window_rect."""
        return (self.game_left, self.game_top, self.game_w, self.game_h)

    def as_map_rect(self):
        """Return map area as (left, top, w, h) for agent.window_rect."""
        return (self.map_left, self.map_top, self.map_w, self.map_h)

    def __repr__(self):
        return (f"WindowArea(\n"
                f"  game=({self.game_left},{self.game_top})"
                f"→({self.game_right},{self.game_bottom})"
                f"  {self.game_w}x{self.game_h}px\n"
                f"  map =({self.map_left},{self.map_top})"
                f"→({self.map_right},{self.map_bottom})"
                f"  {self.map_w}x{self.map_h}px\n)")


def get_window_area(rect):
    """Build WindowArea from a raw window rect tuple."""
    if rect is None:
        return None
    return WindowArea(rect)