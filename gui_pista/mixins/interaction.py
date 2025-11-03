"""Mouse interaction logic for canvas operations."""

import math

import numpy as np
from tkinter import messagebox


class InteractionMixin:
    """Encapsulates pointer-based interaction modes."""

    # Start/origin placement ------------------------------------------------
    def set_start_point(self):
        self._deactivate_goal_modes()
        self._deactivate_obstacle_modes()
        if self.start_point is None:
            self.mode = "set_start"
            self.update_status(
                "Clique para posição da origem; mova o rato para orientar; clique de novo para confirmar."
            )
        else:
            self.update_status("Clique em cima da origem e arraste para rodar; ou use 'Recolocar Origem'.")
            self.mode = None

    def relocate_start(self):
        self._deactivate_goal_modes()
        self._deactivate_obstacle_modes()
        self.mode = "relocate_start"
        self.update_status(
            "Clique em novo ponto para recolocar a origem; depois mova o rato para orientar e clique para confirmar."
        )

    # Limits (image flood) --------------------------------------------------
    def add_limit_mode(self):
        self._deactivate_goal_modes()
        self._set_obstacle_remove_state(False)
        if not self.image:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
            self._set_obstacle_toggle_state(False)
            return
        self.mode = "add_limit"
        self._set_obstacle_toggle_state(True)
        self.update_status("Clique numa zona (flood) para marcar obstáculo.")

    def remove_limit_mode(self):
        self._deactivate_goal_modes()
        self._set_obstacle_toggle_state(False)
        self.mode = "remove_limit"
        self._set_obstacle_remove_state(True)
        self.update_status("Clique num ponto vermelho para remover obstáculo.")

    def clear_all_limits(self):
        self._deactivate_obstacle_modes()
        self.obstacle_points.clear()
        self.redraw()

    # Goals -----------------------------------------------------------------
    def add_goal_mode(self):
        self._deactivate_obstacle_modes()
        self._set_goal_toggle_state(True)
        self._set_goal_delete_state(False)
        self.selected_goal = None
        self.mode = "add_goal"
        self.update_status("Clique para adicionar objetivo; arraste a borda para redimensionar.")

    def remove_goal_mode(self):
        self._deactivate_obstacle_modes()
        self._set_goal_delete_state(True)
        self._set_goal_toggle_state(False)
        self.selected_goal = None
        self.mode = "remove_goal"
        if self.goals:
            self.update_status("Modo remover objetivo: clique num objetivo para apagar.")
        else:
            self.update_status("Sem objetivos para remover.")

    # Measure ---------------------------------------------------------------
    def start_measure(self):
        self._deactivate_goal_modes()
        self._deactivate_obstacle_modes()
        self.measure_p1 = None
        self.measure_p2 = None
        self.last_measure = None
        self.mode = "measure"
        self.update_status("Medição: clica no ponto 1 e depois no ponto 2.")

    def _finish_measure(self):
        if self.measure_p1 and self.measure_p2:
            x1, y1 = self.measure_p1
            x2, y2 = self.measure_p2
            d_px = float(math.hypot(x2 - x1, y2 - y1))
            d_m = float(self.px2m(d_px))
            self.last_measure = (self.measure_p1, self.measure_p2, d_px, d_m)
            self.log_msg(f"Medição: d = {d_px:.1f} px  |  {d_m:.3f} m")
        self.mode = None
        self.update_status("Medição concluída.")
        self.redraw()

    # Mouse interactions ----------------------------------------------------
    def on_mouse_press(self, event):
        if event.xdata is None or event.ydata is None:
            return

        if self.mode == "measure":
            if self.measure_p1 is None:
                self.measure_p1 = (event.xdata, event.ydata)
                self.update_status("Medição: escolhe o ponto 2.")
            else:
                self.measure_p2 = (event.xdata, event.ydata)
                self._finish_measure()
            return

        if self.mode == "draw_free":
            self.draw_active_drag = True
            self._append_draw_point(event.xdata, event.ydata)
            self.redraw()
            return

        if self.mode is None and self.start_point is not None:
            sx, sy = self.start_point
            if (event.xdata - sx) ** 2 + (event.ydata - sy) ** 2 <= (12 ** 2):
                self.mode = "rotate_start"
                self.update_status("A rodar a orientação inicial...")
                return

        if self.mode in {"set_start", "relocate_start"}:
            self.start_point = (event.xdata, event.ydata)
            self.start_theta = 0.0
            self.mode = "set_start_heading"
            self.redraw()
            return

        if self.mode == "set_start_heading":
            self.mode = None
            self.update_status("Origem definida.")
            return

        if self.mode == "add_limit":
            if not self.grid_spacing:
                return
            cx = int(event.xdata // self.grid_spacing)
            cy = int(event.ydata // self.grid_spacing)
            for point in self.flood_fill_grid(cx, cy):
                if point not in self.obstacle_points:
                    self.obstacle_points.append(point)
            self.redraw()
            return

        if self.mode == "remove_limit":
            if not self.obstacle_points or not self.grid_spacing:
                return
            px, py = event.xdata, event.ydata
            for idx, (cx, cy) in enumerate(self.obstacle_points):
                if abs(cx - px) < self.grid_spacing / 2 and abs(cy - py) < self.grid_spacing / 2:
                    self.obstacle_points.pop(idx)
                    break
            self.redraw()
            return

        if self.mode == "add_obstacle":
            self._obs_tmp_center = (event.xdata, event.ydata)
            base_r = getattr(self, "goal_radius", 18) or 18
            self._obs_tmp_radius = max(6, int(base_r))
            self.mode = "resize_obstacle"
            self.update_status("Arraste para definir o raio do obstáculo.")
            self.redraw()
            return

        if self.mode == "remove_obstacle":
            if not self.obstacle_points:
                return
            px, py = float(event.xdata), float(event.ydata)
            r = float(self.grid_spacing or 20)
            r2 = r * r
            keep = []
            for (ox, oy) in self.obstacle_points:
                if (ox - px) * (ox - px) + (oy - py) * (oy - py) > r2:
                    keep.append((ox, oy))
            removed = len(self.obstacle_points) - len(keep)
            self.obstacle_points = keep
            if removed:
                try:
                    self.log_msg(f"{removed} pontos de obstáculo removidos.")
                except Exception:
                    pass
            self.redraw()
            return

        if self.mode == "remove_goal" and bool(getattr(self, "remove_goal_active", False)):
            idx = self._goal_index_at(event.xdata, event.ydata)
            if idx is not None:
                self.goals.pop(idx)
                self.selected_goal = None
                self.update_status("Objetivo removido.")
                try:
                    self.log_msg("Objetivo removido.")
                except Exception:
                    pass
                self.redraw()
            else:
                self.update_status("Nenhum objetivo encontrado nesse ponto.")
            return

        if self.mode == "add_goal":
            existing_idx = self._goal_index_at(event.xdata, event.ydata)
            if existing_idx is not None:
                self.selected_goal = existing_idx
                self.mode = "resize_goal"
                self.update_status("Arraste para alterar o raio do objetivo.")
                return
            self.goals.append([event.xdata, event.ydata, self.goal_radius])
            keep_active = bool(getattr(self, "add_goal_active", False))
            if keep_active:
                self.update_status("Modo objetivo ativo - clique para adicionar outro.")
            else:
                self._set_goal_toggle_state(False)
                self.mode = None
                self.update_status("Objetivo adicionado.")
            self.redraw()
            return

        if bool(getattr(self, "add_goal_active", False)):
            idx = self._goal_index_at(event.xdata, event.ydata)
            if idx is not None:
                self.selected_goal = idx
                self.mode = "resize_goal"
                self.update_status("Arraste para alterar o raio do objetivo.")

    def on_mouse_move(self, event):
        if event.xdata is None or event.ydata is None:
            return

        if self.mode == "draw_free" and self.draw_active_drag:
            if self._append_draw_point(event.xdata, event.ydata):
                self.redraw()
            return

        if self.mode == "set_start_heading" and self.start_point is not None:
            sx, sy = self.start_point
            self.start_theta = math.atan2(event.ydata - sy, event.xdata - sx)
            self.redraw()
            return

        if self.mode == "rotate_start" and self.start_point is not None:
            sx, sy = self.start_point
            self.start_theta = math.atan2(event.ydata - sy, event.xdata - sx)
            self.redraw()
            return

        if self.mode == "resize_obstacle" and getattr(self, "_obs_tmp_center", None) is not None:
            cx, cy = self._obs_tmp_center
            self._obs_tmp_radius = max(6, int(np.hypot(event.xdata - cx, event.ydata - cy)))
            self.redraw()
            return

        add_active = bool(getattr(self, "add_goal_active", False))
        if self.selected_goal is not None and add_active and self.mode == "resize_goal":
            gx, gy, _ = self.goals[self.selected_goal]
            self.goals[self.selected_goal][2] = max(6, int(np.hypot(event.xdata - gx, event.ydata - gy)))
            self.redraw()
            return
        if self.selected_goal is not None and not add_active:
            self.selected_goal = None

    def on_mouse_release(self, _event):
        if self.mode == "draw_free":
            self.draw_active_drag = False
            return
        if self.mode == "rotate_start":
            self.mode = None
            self.update_status("Origem orientada.")
        self.selected_goal = None
        if self.mode == "resize_goal":
            if bool(getattr(self, "add_goal_active", False)):
                self.mode = "add_goal"
            else:
                self.mode = None
        if self.mode == "resize_obstacle":
            cx, cy = getattr(self, "_obs_tmp_center", (None, None))
            r = getattr(self, "_obs_tmp_radius", None)
            self._obs_tmp_center = None
            self._obs_tmp_radius = None
            keep_active = bool(getattr(self, "add_obstacle_active", False))
            if cx is not None and r is not None and r > 0:
                pts = self.circle_fill_grid(cx, cy, r)
                added = 0
                for p in pts:
                    if p not in self.obstacle_points:
                        self.obstacle_points.append(p)
                        added += 1
                try:
                    self.log_msg(f"Obstáculo adicionado: {added} pontos.")
                except Exception:
                    pass
                self.redraw()
            if keep_active:
                self.mode = "add_obstacle"
                self.update_status("Modo obstáculo ativo - clique para adicionar outro.")
            else:
                self._set_obstacle_toggle_state(False)
                self.mode = None

    # Helpers ---------------------------------------------------------------
    def _goal_index_at(self, x, y):
        if not self.goals:
            return None
        for idx, (gx, gy, radius) in enumerate(self.goals):
            if (x - gx) ** 2 + (y - gy) ** 2 < radius ** 2:
                return idx
        return None

    def _set_goal_toggle_state(self, active):
        try:
            active = bool(active)
        except Exception:
            active = False
        try:
            self.add_goal_active = active
        except Exception:
            pass
        btn = getattr(self, "goal_toggle_btn", None)
        if btn is not None:
            state_txt = "ON" if active else "OFF"
            try:
                btn.config(text=f"Adicionar objetivo ({state_txt})")
            except Exception:
                pass
            try:
                if active:
                    btn.state(["pressed"])
                else:
                    btn.state(["!pressed"])
            except Exception:
                pass

    def _set_goal_delete_state(self, active):
        try:
            active = bool(active)
        except Exception:
            active = False
        try:
            self.remove_goal_active = active
        except Exception:
            pass
        btn = getattr(self, "goal_delete_btn", None)
        if btn is not None:
            state_txt = "ON" if active else "OFF"
            try:
                btn.config(text=f"Remover objetivo ({state_txt})")
            except Exception:
                pass
            try:
                if active:
                    btn.state(["pressed"])
                else:
                    btn.state(["!pressed"])
            except Exception:
                pass

    def _set_obstacle_toggle_state(self, active):
        try:
            active = bool(active)
        except Exception:
            active = False
        try:
            self.add_obstacle_active = active
        except Exception:
            pass
        btn = getattr(self, "obstacle_toggle_btn", None)
        if btn is not None:
            state_txt = "ON" if active else "OFF"
            try:
                btn.config(text=f"Adicionar obstaculo ({state_txt})")
            except Exception:
                pass
            try:
                if active:
                    btn.state(["pressed"])
                else:
                    btn.state(["!pressed"])
            except Exception:
                pass

    def _set_obstacle_remove_state(self, active):
        try:
            active = bool(active)
        except Exception:
            active = False
        try:
            self.remove_obstacle_active = active
        except Exception:
            pass
        btn = getattr(self, "obstacle_delete_btn", None)
        if btn is not None:
            state_txt = "ON" if active else "OFF"
            try:
                btn.config(text=f"Remover obstaculo ({state_txt})")
            except Exception:
                pass
            try:
                if active:
                    btn.state(["pressed"])
                else:
                    btn.state(["!pressed"])
            except Exception:
                pass

    # New: obstacle modes like goals
    def add_obstacle_mode(self):
        self._deactivate_goal_modes()
        self._set_obstacle_remove_state(False)
        if not self.image:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
            self._set_obstacle_toggle_state(False)
            return
        self.mode = "add_obstacle"
        self._set_obstacle_toggle_state(True)
        self._obs_tmp_center = None
        self._obs_tmp_radius = None
        self.update_status("Clique para adicionar obstáculo; arraste para definir o raio.")

    def remove_obstacle_mode(self):
        self._deactivate_goal_modes()
        self._set_obstacle_toggle_state(False)
        self.mode = "remove_obstacle"
        self._set_obstacle_remove_state(True)
        self.update_status("Modo remover obstáculo: clique para apagar uma zona.")

    def _deactivate_goal_modes(self):
        self._set_goal_toggle_state(False)
        self._set_goal_delete_state(False)
        if getattr(self, "mode", None) in {"add_goal", "remove_goal", "resize_goal"}:
            self.mode = None
        self.selected_goal = None

    def _deactivate_obstacle_modes(self):
        self._set_obstacle_toggle_state(False)
        self._set_obstacle_remove_state(False)
        if getattr(self, "mode", None) in {"add_limit", "remove_limit", "add_obstacle", "remove_obstacle", "resize_obstacle"}:
            self.mode = None

