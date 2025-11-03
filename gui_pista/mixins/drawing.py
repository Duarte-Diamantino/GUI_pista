"""Manual drawing workflow for the planner."""

import math

import numpy as np
from tkinter import messagebox


class DrawingMixin:
    """Implements brush-based manual path editing."""

    def start_draw_mode(self):
        self.mode = "draw_free"
        self.draw_mode = True
        self.draw_active_drag = False
        self.update_status("Desenho à mão — arrasta com o rato.")
        self.log_msg("Desenho: modo ativo.")

    def stop_draw_mode(self):
        self.draw_mode = False
        if self.mode == "draw_free":
            self.mode = None
        self.draw_active_drag = False
        self.update_status("Desenho parado.")
        self.log_msg("Desenho: parado.")

    def clear_draw(self):
        self.draw_points.clear()
        self.manual_path = None
        self.redraw()
        self.log_msg("Desenho: limpo.")

    def _append_draw_point(self, x, y):
        if x is None or y is None:
            return False
        if not self.draw_points:
            self.draw_points.append((x, y))
            return True
        last_x, last_y = self.draw_points[-1]
        if math.hypot(x - last_x, y - last_y) >= float(self.draw_sampling_px):
            self.draw_points.append((x, y))
            return True
        return False

    def _smooth_polyline(self, points, iters=2):
        if len(points) < 3:
            return points[:]
        current = points[:]
        for _ in range(iters):
            refined = [current[0]]
            for i in range(len(current) - 1):
                p = np.array(current[i])
                q = np.array(current[i + 1])
                refined.append(tuple(0.75 * p + 0.25 * q))
                refined.append(tuple(0.25 * p + 0.75 * q))
            refined.append(current[-1])
            current = refined
        return current

    def _polyline_to_path(self, points_px):
        if len(points_px) < 2:
            return None
        if bool(self.draw_smooth_on_use.get()):
            points_px = self._smooth_polyline(points_px, iters=2)
        converted = []
        for idx, (x, y) in enumerate(points_px):
            if idx == 0:
                nx, ny = points_px[idx + 1]
                theta = math.atan2(ny - y, nx - x)
            else:
                px, py = points_px[idx - 1]
                theta = math.atan2(y - py, x - px)
            converted.append((float(x), float(y), float(theta), float(self.max_speed)))
        return converted

    def _prepare_manual_path(self, title):
        if len(self.draw_points) < 2:
            messagebox.showwarning(title, "Desenha primeiro (m�nimo dois pontos).")
            return None
        path = self._polyline_to_path(self.draw_points)
        if not path:
            messagebox.showwarning(title, "N�o consegui converter o desenho em caminho.")
            return None
        self.manual_path = path
        self.run_state = dict(full_path=path)
        return path

    def use_draw_as_path(self):
        path = self._prepare_manual_path("Desenho")
        if not path:
            return
        self.redraw(also_paths=[("cyan", [(x, y) for (x, y, _, _) in path], 3, 0.95)])
        total_m = sum(
            self.px2m(
                np.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            )
            for i in range(1, len(path))
        )
        self.log_msg(f"Desenho: {len(path)} pts | comprimento ~ {total_m:.2f} m")

    def trace_raceline_from_draw(self):
        path = self._prepare_manual_path("Race line")
        if not path:
            return
        self.log_msg("Race line: a gerar a partir do desenho atual.")
        self._build_and_draw_raceline(path)

    def trace_fast_raceline_from_draw(self):
        path = self._prepare_manual_path("Race line rápida")
        if not path:
            return
        self.log_msg("Race line (rápida): a gerar a partir do desenho atual.")
        self.build_fast_raceline_current()

    def publish_drawn_path(self):
        path = self._prepare_manual_path("Publicar desenho")
        if not path:
            return
        self.publish_reference_path()
