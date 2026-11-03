"""Design tokens. Every colour used anywhere in the GUI comes from these two dictionaries."""

from __future__ import annotations

DARK = {
    "bg": "#121417", "surface": "#181b20", "elevated": "#22272e", "border": "#30363d", "text": "#e6edf3",
    "muted": "#8b949e", "accent": "#4c9aff", "accent_2": "#2f81f7", "success": "#3fb950", "warning": "#d29922",
    "danger": "#f85149", "info": "#a371f7", "nan": "#2a2f36", "overlay_focus": "#3fb950", "overlay_line": "#4c9aff",
    "overlay_profile": "#f0883e", "grid": "#30363d",
}
LIGHT = {
    "bg": "#f6f8fa", "surface": "#ffffff", "elevated": "#eaeef2", "border": "#d0d7de", "text": "#1f2328",
    "muted": "#57606a", "accent": "#0969da", "accent_2": "#0550ae", "success": "#1a7f37", "warning": "#9a6700",
    "danger": "#cf222e", "info": "#8250df", "nan": "#d8dee4", "overlay_focus": "#1a7f37", "overlay_line": "#0969da",
    "overlay_profile": "#bc4c00", "grid": "#d0d7de",
}
FONT_SANS = "'Inter', 'Segoe UI', 'Cantarell', 'Ubuntu', sans-serif"
FONT_MONO = "'JetBrains Mono', 'Cascadia Mono', 'Ubuntu Mono', 'DejaVu Sans Mono', monospace"
SPACING = 8
RADIUS = 6


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def stylesheet(t: dict) -> str:
    return f"""
    QWidget {{ background: {t['bg']}; color: {t['text']}; font-family: {FONT_SANS}; font-size: 13px; }}
    QMainWindow::separator {{ background: {t['border']}; width: 1px; height: 1px; }}
    QDockWidget {{ titlebar-close-icon: none; }}
    QDockWidget::title {{ background: {t['surface']}; padding: 6px 8px; font-weight: 600; }}
    QFrame#card, QGroupBox {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: {RADIUS}px; }}
    QGroupBox {{ margin-top: 14px; padding: 8px; font-weight: 600; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {t['muted']}; }}
    QLabel {{ background: transparent; }}
    QLabel#muted {{ color: {t['muted']}; }}
    QLabel#title {{ font-size: 15px; font-weight: 700; }}
    QLabel#mono {{ font-family: {FONT_MONO}; }}
    QLabel#rule {{ background: {t['border']}; }}
    QLabel#guide {{ background: {t['elevated']}; border: 1px solid {t['border']}; border-left: 4px solid {t['accent']};
        border-radius: {RADIUS}px; padding: 8px 12px; font-size: 14px; }}
    QPushButton, QToolButton {{ background: {t['elevated']}; border: 1px solid {t['border']}; border-radius: {RADIUS}px;
        padding: 6px 12px; font-weight: 600; }}
    QPushButton:hover, QToolButton:hover {{ border-color: {t['accent']}; }}
    QPushButton:pressed {{ background: {t['border']}; }}
    QPushButton:disabled {{ color: {t['muted']}; }}
    QPushButton#primary {{ background: {t['accent_2']}; color: #ffffff; border-color: {t['accent_2']}; }}
    QPushButton#step {{ text-align: left; padding: 8px 10px; }}
    QPushButton#step:checked {{ border: 1px solid {t['accent']}; background: {t['surface']}; }}
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {t['elevated']}; border: 1px solid {t['border']};
        border-radius: {RADIUS}px; padding: 4px 8px; font-family: {FONT_MONO}; }}
    QComboBox QAbstractItemView {{ background: {t['surface']}; selection-background-color: {t['accent_2']}; }}
    QTabWidget::pane {{ border: 1px solid {t['border']}; border-radius: {RADIUS}px; background: {t['surface']}; top: -1px; }}
    QTabBar::tab {{ background: transparent; color: {t['muted']}; padding: 8px 16px; border: none;
        border-bottom: 2px solid transparent; font-weight: 600; }}
    QTabBar::tab:selected {{ color: {t['text']}; border-bottom: 2px solid {t['accent']}; }}
    QTabBar::tab:hover {{ color: {t['text']}; }}
    QProgressBar {{ background: {t['elevated']}; border: 1px solid {t['border']}; border-radius: {RADIUS}px;
        text-align: center; height: 18px; }}
    QProgressBar::chunk {{ background: {t['accent_2']}; border-radius: {RADIUS - 1}px; }}
    QSlider::groove:horizontal {{ height: 4px; background: {t['border']}; border-radius: 2px; }}
    QSlider::handle:horizontal {{ background: {t['accent']}; width: 14px; margin: -6px 0; border-radius: 7px; }}
    QTextBrowser, QTextEdit, QPlainTextEdit {{ background: {t['surface']}; border: 1px solid {t['border']};
        border-radius: {RADIUS}px; }}
    QPlainTextEdit {{ font-family: {FONT_MONO}; font-size: 12px; }}
    QCheckBox {{ background: transparent; spacing: 8px; }}
    QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {t['muted']}; border-radius: 4px;
        background: {t['elevated']}; }}
    QCheckBox::indicator:checked {{ background: {t['accent_2']}; border-color: {t['accent_2']};
        image: none; }}
    QCheckBox::indicator:hover {{ border-color: {t['accent']}; }}
    QToolTip {{ background: {t['elevated']}; color: {t['text']}; border: 1px solid {t['border']}; padding: 6px; }}
    QStatusBar {{ background: {t['surface']}; color: {t['muted']}; }}
    QScrollArea {{ border: none; }}
    """
