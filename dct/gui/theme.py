"""Dark theme colours, font scale and global stylesheet."""
from __future__ import annotations

BG     = "#1E1E1E"
PANEL  = "#252526"
BORDER = "#3C3C3C"
ACCENT = "#007ACC"
TEXT   = "#D4D4D4"
DIM    = "#858585"
OK     = "#4EC9B0"
WARN   = "#CE9178"
ERR    = "#F44747"

DRONE   = "#4EC9B0"
TRAIL   = "#2D7DD2"
LOCALIZER = "#D7BA7D"
LOC_TRAIL = "#8B7355"
# Second localizer (RC sticks) in Liftoff+RC mode — distinct from sim (gold).
LOCALIZER_RC = "#569CD6"
LOC_TRAIL_RC = "#3D6FA3"
# Третий PF (legacy sticks, тот же круг что BF-эталон)
LOCALIZER_LEGACY = "#B5CEA8"
LOC_TRAIL_LEGACY = "#6A9955"
GATE    = "#CE9178"
GATE_SF = "#4EC9B0"

STICK_T    = "#4EC9B0"
STICK_Y    = "#569CD6"
STICK_P    = "#CE9178"
STICK_R    = "#F44747"

# ---------------------------------------------------------------------------
# Font scale (pixels). Stay in px so HiDPI rounding does not surprise us.
# ---------------------------------------------------------------------------
FONT_BASE = 12
FONT_DIM  = 11
FONT_HEAD = 14
FONT_HUD  = 16
FONT_HUD_RACE = 22

QSS = f"""
QMainWindow, QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: {FONT_BASE}px;
}}
QSplitter::handle {{ background-color: {BORDER}; width: 2px; height: 2px; }}

QPushButton {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 14px;
    font-weight: 600;
    min-width: 72px;
}}
QPushButton:hover  {{ background-color: {ACCENT}; border-color: {ACCENT}; }}
QPushButton:pressed {{ background-color: #005a9e; }}
QPushButton:disabled {{ color: {DIM}; background-color: #2A2A2A; border-color: #2A2A2A; }}
QPushButton:checked {{ background-color: {ACCENT}; border-color: {ACCENT}; color: #fff; }}

QPushButton#btn_start {{
    background-color: #1b3d1e; border-color: {OK}; color: {OK};
}}
QPushButton#btn_start:hover {{ background-color: #265929; }}
QPushButton#btn_stop {{
    background-color: #3d1b1b; border-color: {ERR}; color: {ERR};
}}
QPushButton#btn_stop:hover {{ background-color: #592626; }}
QPushButton[role="big"] {{
    min-height: 36px; font-size: {FONT_HEAD}px; padding: 6px 16px;
}}
QPushButton[role="icon"] {{
    min-width: 28px; padding: 2px 6px;
}}

QComboBox {{
    background-color: {PANEL}; color: {TEXT};
    border: 1px solid {BORDER}; border-radius: 4px;
    padding: 4px 8px; min-width: 90px;
}}
QComboBox:hover {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background-color: {PANEL}; color: {TEXT};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
}}

QLabel {{ color: {TEXT}; }}
QLabel[role="dim"]   {{ color: {DIM};  font-size: {FONT_DIM}px; }}
QLabel[role="value"] {{ color: {TEXT}; font-size: {FONT_BASE}px; font-weight: 600; }}
QLabel[role="head"]  {{ color: {TEXT}; font-size: {FONT_HEAD}px; font-weight: 700; }}
QLabel[role="hud"]   {{ color: {TEXT}; font-size: {FONT_HUD}px;  font-weight: 600; }}
QLabel[role="ok"]    {{ color: {OK};   font-weight: 600; }}
QLabel[role="warn"]  {{ color: {WARN}; font-weight: 600; }}
QLabel[role="err"]   {{ color: {ERR};  font-weight: 600; }}

QLabel#lbl_dim  {{ color: {DIM};  font-size: {FONT_DIM}px; }}
QLabel#lbl_ok   {{ color: {OK};   font-weight: 600; }}
QLabel#lbl_warn {{ color: {WARN}; font-weight: 600; }}
QLabel#lbl_err  {{ color: {ERR};  font-weight: 600; }}

QGroupBox {{
    border: 1px solid {BORDER}; border-radius: 4px;
    margin-top: 10px; padding-top: 6px;
    color: {DIM}; font-size: {FONT_DIM}px;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}

QSlider::groove:horizontal {{
    background: {BORDER}; height: 4px; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT}; width: 12px; height: 12px;
    margin: -4px 0; border-radius: 6px;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}

QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {PANEL}; color: {TEXT};
    border: 1px solid {BORDER}; border-radius: 4px; padding: 4px 8px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}

QScrollBar:vertical {{
    background: {BG}; width: 8px; border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER}; border-radius: 4px; min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QFrame[role="card"] {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}

QFrame[role="topbar"], QFrame[role="bottombar"] {{
    background-color: {PANEL};
    border-top: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
}}

QFrame#sidebar_root {{
    background-color: {PANEL};
    border-left: 1px solid {BORDER};
}}

QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {BORDER}; border-radius: 3px;
    background: {PANEL};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT}; border-color: {ACCENT};
}}
QRadioButton {{ spacing: 6px; }}

QStackedWidget {{ background-color: {BG}; }}
QTabBar::tab {{
    background: {PANEL}; color: {TEXT};
    padding: 4px 10px; border: 1px solid {BORDER};
    border-bottom-color: {PANEL};
}}
QTabBar::tab:selected {{ background: {ACCENT}; color: #fff; }}
"""
