"""Matplotlib drawing helpers for the main canvas."""

import math
import time

import matplotlib
import numpy as np

from ..theme import PALETTE


class RenderingMixin:
    """Handles canvas redraw operations."""

    def redraw(self, also_paths=None, overlay=None):
        if not hasattr(self, "_pending_redraw_args"):
            self._pending_redraw_args = None
        if not hasattr(self, "_redraw_pending"):
            self._redraw_pending = False

        self._pending_redraw_args = (also_paths, overlay)
        if self._redraw_pending:
            return

        delay_ms = 0.0
        min_interval = getattr(self, "_redraw_min_interval_ms", 0.0) or 0.0
        last_ts = getattr(self, "_last_redraw_ts", 0.0) or 0.0
        if min_interval and last_ts:
            elapsed = (time.perf_counter() - last_ts) * 1000.0
            if elapsed < min_interval:
                delay_ms = min_interval - elapsed

        self._redraw_pending = True
        self._schedule_redraw(delay_ms)

    def _schedule_redraw(self, delay_ms):
        if delay_ms <= 0:
            after_id = getattr(self, "_redraw_after_id", None)
            if after_id is not None:
                try:
                    self.after_cancel(after_id)
                except Exception:
                    pass
                self._redraw_after_id = None
            try:
                self.after_idle(self._flush_redraw)
            except Exception:
                self._flush_redraw()
            return

        delay_ms = max(1, int(math.ceil(delay_ms)))
        after_id = getattr(self, "_redraw_after_id", None)
        if after_id is not None:
            try:
                self.after_cancel(after_id)
            except Exception:
                pass
        self._redraw_after_id = self.after(delay_ms, self._flush_redraw)

    def _flush_redraw(self):
        also_paths, overlay = getattr(self, "_pending_redraw_args", (None, None))
        self._pending_redraw_args = None
        self._redraw_pending = False
        self._redraw_after_id = None
        self._perform_redraw(also_paths, overlay)
        try:
            self._last_redraw_ts = time.perf_counter()
        except Exception:
            self._last_redraw_ts = 0.0

    def _perform_redraw(self, also_paths=None, overlay=None):
        self.ax.clear()
        self.ax.set_facecolor(PALETTE["bg_alt"])
        width = height = None
        if self.image is not None:
            self.ax.imshow(self.image)
            width, height = self.image.size
            self.ax.set_xlim(0, width)
            self.ax.set_ylim(height, 0)
        self.ax.axis("off")

        grid_color = PALETTE["highlight"]
        meter_color = PALETTE["accent"]

        if self.grid_spacing and width is not None and height is not None:
            xs = np.arange(0, width, self.grid_spacing, dtype=float)
            ys = np.arange(0, height, self.grid_spacing, dtype=float)
            if xs.size:
                self.ax.vlines(xs, ymin=0, ymax=height, linestyles="--", linewidth=0.4, colors=grid_color, alpha=0.35)
            if ys.size:
                self.ax.hlines(ys, xmin=0, xmax=width, linestyles="--", linewidth=0.4, colors=grid_color, alpha=0.35)

        if self.show_meter_grid.get() and width is not None and height is not None:
            step = max(8.0, self.m2px(1.0))
            if step > 0:
                xs = np.arange(0, width + 1, step, dtype=float)
                ys = np.arange(0, height + 1, step, dtype=float)
                if xs.size:
                    self.ax.vlines(xs, ymin=0, ymax=height, colors=meter_color, linewidth=0.4, alpha=0.28)
                if ys.size:
                    self.ax.hlines(ys, xmin=0, xmax=width, colors=meter_color, linewidth=0.4, alpha=0.28)

        if self.start_point is not None:
            sx, sy = self.start_point
            self.ax.plot(
                sx,
                sy,
                "o",
                markersize=11,
                markeredgecolor=PALETTE["fg_primary"],
                markerfacecolor=PALETTE["success"],
                markeredgewidth=2,
            )
            length = self.m2px(0.35)
            self.ax.arrow(
                sx,
                sy,
                length * math.cos(self.start_theta),
                length * math.sin(self.start_theta),
                head_width=8,
                head_length=5,
                fc=PALETTE["accent"],
                ec=PALETTE["accent"],
                alpha=0.9,
            )

        if self.obstacle_points:
            pts = np.asarray(self.obstacle_points, dtype=float)
            if pts.ndim == 1:
                pts = pts.reshape(-1, 2)
            if pts.size:
                self.ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    s=28,
                    c=PALETTE["warning"],
                    edgecolors=PALETTE["border"],
                    linewidths=0.6,
                    alpha=0.85,
                )

        for idx, (gx, gy, radius) in enumerate(self.goals):
            circ = matplotlib.patches.Circle(
                (gx, gy),
                radius,
                color=PALETTE["accent"],
                alpha=0.28,
                edgecolor=PALETTE["accent"],
                linewidth=2,
            )
            self.ax.add_patch(circ)
            self.ax.text(
                gx,
                gy,
                str(idx + 1),
                color=PALETTE["fg_primary"],
                fontsize=12,
                ha="center",
                va="center",
                fontweight="bold",
            )

        if width is not None and self.grid_spacing:
            bar = self.m2px(1.0)
            y0 = height - 15
            x0 = 15
            self.ax.plot([x0, x0 + bar], [y0, y0], color=PALETTE["fg_primary"], linewidth=4)
            self.ax.text(
                x0 + bar / 2,
                y0 - 8,
                "1 m",
                ha="center",
                va="bottom",
                fontsize=10,
                color=PALETTE["fg_primary"],
                fontweight="bold",
                bbox=dict(facecolor=PALETTE["bg"], alpha=0.7, edgecolor="none"),
            )

        if also_paths:
            for color, path, lw, alpha in also_paths:
                if path and len(path) > 1:
                    xs = [p[0] for p in path]
                    ys = [p[1] for p in path]
                    self.ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha)

        if overlay:
            x, y, theta, color = overlay
            head = self.m2px(0.22)
            tail = self.m2px(0.12)
            left = theta + 2.6
            right = theta - 2.6
            self.ax.fill(
                [
                    x + head * math.cos(theta),
                    x - tail * math.cos(left),
                    x - tail * math.cos(right),
                ],
                [
                    y + head * math.sin(theta),
                    y - tail * math.sin(left),
                    y - tail * math.sin(right),
                ],
                color=color,
                alpha=0.95,
                edgecolor="black",
                linewidth=1.0,
            )

        self._draw_overlay()

        if self.last_measure:
            (x1, y1), (x2, y2), dpx, dm = self.last_measure
            self.ax.plot([x1, x2], [y1, y2], color=PALETTE["accent"], linewidth=2.5, alpha=0.9)
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            self.ax.text(
                mx,
                my,
                f"{dm:.3f} m",
                color=PALETTE["fg_primary"],
                fontsize=10,
                ha="center",
                va="bottom",
                bbox=dict(facecolor=PALETTE["bg"], alpha=0.75, edgecolor="none"),
            )

        self._draw_canvas()

    def _draw_overlay(self):
        if self.draw_points and len(self.draw_points) > 1:
            xs = [p[0] for p in self.draw_points]
            ys = [p[1] for p in self.draw_points]
            self.ax.plot(xs, ys, linewidth=3.0, alpha=0.95, color="cyan")
        # Preview circle while adding obstacle by drag
        if getattr(self, "mode", None) == "resize_obstacle":
            center = getattr(self, "_obs_tmp_center", None)
            radius = getattr(self, "_obs_tmp_radius", None)
            if center is not None and radius is not None and radius > 0:
                circ = matplotlib.patches.Circle(
                    center,
                    radius,
                    color=PALETTE["warning"],
                    alpha=0.18,
                    edgecolor=PALETTE["warning"],
                    linewidth=2,
                )
                self.ax.add_patch(circ)

    def _draw_canvas(self):
        canvas = getattr(self, "canvas", None)
        if canvas is None:
            return
        draw_idle = getattr(canvas, "draw_idle", None)
        if callable(draw_idle):
            draw_idle()
        else:
            canvas.draw()
