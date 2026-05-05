"""Dark theme colours and global stylesheet."""

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
LOCALIZER = "#D7BA7D"   # оценка позиции по стикам (отличить от GT)
LOC_TRAIL = "#8B7355"
GATE    = "#CE9178"
GATE_SF = "#4EC9B0"

STICK_T    = "#4EC9B0"   # throttle — teal
STICK_Y    = "#569CD6"   # yaw     — blue
STICK_P    = "#CE9178"   # pitch   — orange
STICK_R    = "#F44747"   # roll    — red

QSS = f"""
QMainWindow, QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 12px;
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
QLabel#lbl_dim  {{ color: {DIM};  font-size: 11px; }}
QLabel#lbl_ok   {{ color: {OK};   font-weight: 600; }}
QLabel#lbl_warn {{ color: {WARN}; font-weight: 600; }}
QLabel#lbl_err  {{ color: {ERR};  font-weight: 600; }}

QGroupBox {{
    border: 1px solid {BORDER}; border-radius: 4px;
    margin-top: 10px; padding-top: 6px;
    color: {DIM}; font-size: 11px;
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

QLineEdit {{
    background-color: {PANEL}; color: {TEXT};
    border: 1px solid {BORDER}; border-radius: 4px; padding: 4px 8px;
}}
QLineEdit:focus {{ border-color: {ACCENT}; }}

QScrollBar:vertical {{
    background: {BG}; width: 8px; border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER}; border-radius: 4px; min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""
