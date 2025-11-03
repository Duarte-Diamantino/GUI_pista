"""Tkinter helper widgets used throughout the GUI."""

import tkinter as tk
from tkinter import ttk

from .theme import PALETTE


class ScrollableFrame(ttk.Frame):
    """Scrollable frame that exposes a body frame for content."""

    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0, background=PALETTE["bg"])
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.vbar.set, xscrollcommand=self.hbar.set)
        self.vbar.pack(side="right", fill="y")
        self.hbar.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.body = ttk.Frame(self.canvas, style="Sidebar.TFrame")
        self.window = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", self._on_body_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)
        self.canvas.bind("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))
        self.canvas.bind("<Shift-Button-4>", lambda e: self.canvas.xview_scroll(-1, "units"))
        self.canvas.bind("<Shift-Button-5>", lambda e: self.canvas.xview_scroll(1, "units"))

    def _on_body_configure(self, _):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.window, width=event.width)

    def _on_mousewheel(self, event):
        delta = -1 * int(event.delta / 120) if event.delta else (
            1 if getattr(event, "num", 0) == 5 else -1
        )
        self.canvas.yview_scroll(delta, "units")

    def _on_shift_mousewheel(self, event):
        delta = -1 * int(event.delta / 120) if event.delta else (
            1 if getattr(event, "num", 0) == 5 else -1
        )
        self.canvas.xview_scroll(delta, "units")
