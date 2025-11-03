"""Centralized theming utilities for a modern-looking Tk application."""

import tkinter as tk
from tkinter import ttk, font as tkfont


PALETTE = {
    "bg": "#1E1F25",
    "bg_alt": "#262830",
    "fg_primary": "#F2F2F5",
    "fg_muted": "#B5B8C4",
    "accent": "#4C8DFF",
    "accent_hover": "#6BA4FF",
    "accent_active": "#316FDB",
    "border": "#2E3038",
    "highlight": "#3A3D46",
    "success": "#2ECC71",
    "warning": "#F5A623",
}


def _configure_fonts(root):
    """Set a consistent font family across widgets."""
    default_family = "Segoe UI"
    default_size = 10
    font_names = [
        "TkDefaultFont",
        "TkTextFont",
        "TkMenuFont",
        "TkFixedFont",
        "TkHeadingFont",
        "TkCaptionFont",
        "TkSmallCaptionFont",
        "TkTooltipFont",
        "TkIconFont",
        "TkStatusFont",
    ]
    for name in font_names:
        try:
            tkfont.nametofont(name).configure(family=default_family, size=default_size)
        except tk.TclError:
            continue
    root._app_fonts = {
        "accent": tkfont.Font(root=root, family=default_family, size=default_size, weight="bold"),
        "regular_bold": tkfont.Font(root=root, family=default_family, size=default_size, weight="bold"),
    }


def _configure_styles(root):
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(
        ".",
        background=PALETTE["bg"],
        foreground=PALETTE["fg_primary"],
        fieldbackground=PALETTE["bg_alt"],
        troughcolor=PALETTE["highlight"],
        bordercolor=PALETTE["border"],
        lightcolor=PALETTE["highlight"],
        darkcolor=PALETTE["border"],
    )

    style.configure("TFrame", background=PALETTE["bg"])
    style.configure("Sidebar.TFrame", background=PALETTE["bg_alt"])
    style.configure("TLabelframe", background=PALETTE["bg"], bordercolor=PALETTE["border"])
    style.configure(
        "TLabelframe.Label",
        background=PALETTE["bg"],
        foreground=PALETTE["fg_muted"],
        padding=(6, 2, 6, 2),
    )
    style.configure("TLabel", background=PALETTE["bg"], foreground=PALETTE["fg_muted"])
    style.configure(
        "Status.TLabel",
        background=PALETTE["highlight"],
        foreground=PALETTE["fg_primary"],
        padding=(10, 4),
    )

    style.configure(
        "Accent.TButton",
        font=root._app_fonts["accent"],
        padding=(14, 8),
        foreground=PALETTE["fg_primary"],
        background=PALETTE["accent"],
        bordercolor=PALETTE["accent"],
    )
    style.map(
        "Accent.TButton",
        background=[
            ("active", PALETTE["accent_active"]),
            ("hover", PALETTE["accent_hover"]),
        ],
        bordercolor=[
            ("active", PALETTE["accent_active"]),
            ("hover", PALETTE["accent_hover"]),
        ],
    )

    style.configure(
        "Secondary.TButton",
        padding=(12, 6),
        background=PALETTE["highlight"],
        foreground=PALETTE["fg_primary"],
        bordercolor=PALETTE["highlight"],
    )
    style.map(
        "Secondary.TButton",
        background=[("active", PALETTE["accent"]), ("hover", PALETTE["accent_hover"])],
        foreground=[("active", PALETTE["fg_primary"]), ("hover", PALETTE["fg_primary"])],
    )

    style.configure(
        "Warning.TButton",
        padding=(12, 6),
        background=PALETTE["warning"],
        foreground=PALETTE["bg"],
        bordercolor=PALETTE["warning"],
    )
    style.map(
        "Warning.TButton",
        background=[("active", PALETTE["warning"]), ("hover", PALETTE["warning"])],
        foreground=[("active", PALETTE["bg"]), ("hover", PALETTE["bg"])],
    )

    style.configure("Transparent.TPanedwindow", background=PALETTE["bg"], borderwidth=0)

    style.configure("Horizontal.TScale", background=PALETTE["bg"], troughcolor=PALETTE["highlight"])
    style.configure(
        "Modern.Horizontal.TScale",
        background=PALETTE["bg"],
        troughcolor=PALETTE["highlight"],
        bordercolor=PALETTE["highlight"],
    )
    style.configure(
        "Vertical.TScrollbar",
        background=PALETTE["bg"],
        bordercolor=PALETTE["border"],
        troughcolor=PALETTE["highlight"],
        arrowcolor=PALETTE["fg_primary"],
    )
    style.configure(
        "Horizontal.TScrollbar",
        background=PALETTE["bg"],
        bordercolor=PALETTE["border"],
        troughcolor=PALETTE["highlight"],
        arrowcolor=PALETTE["fg_primary"],
    )

    style.configure(
        "TNotebook",
        background=PALETTE["bg"],
        borderwidth=0,
    )
    style.configure(
        "TNotebook.Tab",
        padding=(12, 6),
        background=PALETTE["bg"],
        foreground=PALETTE["fg_muted"],
        focuscolor=PALETTE["accent"],
        bordercolor=PALETTE["bg"],
    )
    style.map(
        "TNotebook.Tab",
        background=[
            ("selected", PALETTE["bg_alt"]),
            ("active", PALETTE["accent_hover"]),
            ("!disabled", PALETTE["bg"]),
        ],
        foreground=[
            ("selected", PALETTE["fg_primary"]),
            ("active", PALETTE["fg_primary"]),
            ("!disabled", PALETTE["fg_muted"]),
        ],
        bordercolor=[("selected", PALETTE["accent"]), ("!disabled", PALETTE["bg"])],
    )

    style.configure("Treeview", background=PALETTE["bg_alt"], fieldbackground=PALETTE["bg_alt"])
    style.configure(
        "Modern.Horizontal.TProgressbar",
        troughcolor=PALETTE["bg_alt"],
        background=PALETTE["accent"],
        bordercolor=PALETTE["bg_alt"],
    )
    style.map(
        "Modern.Horizontal.TProgressbar",
        background=[("active", PALETTE["accent_hover"]), ("!disabled", PALETTE["accent"])],
    )

    style.configure(
        "TCombobox",
        fieldbackground=PALETTE["bg_alt"],
        background=PALETTE["bg_alt"],
        foreground=PALETTE["fg_primary"],
        bordercolor=PALETTE["border"],
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", PALETTE["bg_alt"])],
        foreground=[("readonly", PALETTE["fg_primary"])],
        background=[("readonly", PALETTE["bg_alt"])],
    )


def apply_theme(root):
    """Apply the custom dark theme to the Tk root window."""
    root.configure(bg=PALETTE["bg"])
    _configure_fonts(root)
    _configure_styles(root)
    root.option_add("*TCombobox*Listbox.background", PALETTE["bg_alt"])
    root.option_add("*TCombobox*Listbox.foreground", PALETTE["fg_primary"])
    root.option_add("*TCombobox*Listbox.selectBackground", PALETTE["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", PALETTE["fg_primary"])
