"""Full planning pipeline (Hybrid A*, DWA, MPC and race line)."""

import heapq
import math

import matplotlib
import numpy as np
from matplotlib.collections import LineCollection
from tkinter import messagebox

from ..utils import ang_norm, clamp


class PlanningMixin:
    """Implements the Hybrid A* → DWA → MPC pipeline."""

    def stop_all(self):
        if self.anim_timer:
            try:
                self.after_cancel(self.anim_timer)
            except Exception:
                pass
        self.anim_timer = None
        self.run_state = None
        self.update_status("Parado.")

    def run_all(self):
        if not self.image or not self.start_point or not self.goals:
            messagebox.showwarning(
                "Aviso", "Carrega imagem, define origem+orientação e adiciona objetivos."
            )
            return
        self.stop_all()
        self.log.delete("1.0", "end")
        self.progress["value"] = 0
        self.run_state = dict(
            seg_index=0,
            seg_count=len(self.goals),
            seg_start_pose=(self.start_point[0], self.start_point[1], self.start_theta),
            full_path=[],
            attempt=0,
            max_attempts=3,
            entry_pts=None,
            entry_pt=None,
            phase="hybrid",
            ha_open=[],
            ha_counter=0,
            ha_closed=set(),
            ha_came={},
            ha_g={},
            ha_grid_res=max(20, self.grid_spacing or 30),
            ha_angle_res=math.pi / 4,
            ha_current_best=None,
            ha_iter=0,
            ha_max_iter=4000,
            dwa_state=None,
            dwa_path=[],
            dwa_steps=0,
            dwa_stuck=0,
            mpc_iter=0,
            mpc_max_iter=60,
            mpc_best=None,
            mpc_seed=None,
            path_hybrid=None,
            path_dwa=None,
            path_mpc=None,
        )
        self.manual_path = None
        self._start_segment()

    def _entry_points(self, gx, gy, radius, sx, sy, count=6):
        base_ang = math.atan2(sy - gy, sx - gx)
        points = []
        for i in range(count):
            ang = base_ang + (i - (count - 1) / 2.0) * (math.pi / (count + 1))
            x = gx + (radius - self.m2px(0.1)) * math.cos(ang)
            y = gy + (radius - self.m2px(0.1)) * math.sin(ang)
            points.append((x, y, ang))
        return sorted(points, key=lambda p: abs(ang_norm(p[2] - base_ang)))

    def _start_segment(self):
        state = self.run_state
        if state["seg_index"] >= state["seg_count"]:
            if state["full_path"]:
                self.redraw(
                    also_paths=[
                        ("blue", [(x, y) for (x, y, _, _) in state["full_path"]], 3, 0.8)
                    ]
                )
                self._build_and_draw_raceline(state["full_path"])
                self.update_status(
                    "Concluído — todos os objetivos atingidos + linha de corrida feita."
                )
                self.progress["value"] = 100
            else:
                self.update_status("Sem caminho.")
            return

        if state["attempt"] == 0:
            gx, gy, radius = self.goals[state["seg_index"]]
            sx, sy, _ = state["seg_start_pose"]
            state["entry_pts"] = self._entry_points(gx, gy, radius, sx, sy, count=6)

        if state["attempt"] >= state["max_attempts"]:
            self.log_msg(
                f"[Seg {state['seg_index']+1}] Falhou após {state['max_attempts']} tentativas → a saltar alvo."
            )
            state["seg_index"] += 1
            self._start_segment()
            return

        idx = min(state["attempt"], len(state["entry_pts"]) - 1)
        state["entry_pt"] = (state["entry_pts"][idx][0], state["entry_pts"][idx][1])
        state["phase"] = "hybrid"
        state["path_hybrid"] = state["path_dwa"] = state["path_mpc"] = None
        state["ha_open"].clear()
        state["ha_closed"].clear()
        state["ha_came"].clear()
        state["ha_g"].clear()
        state["ha_iter"] = 0
        state["ha_current_best"] = None
        state["ha_counter"] = 0
        state["ha_grid_res"] = max(16, (self.grid_spacing or 30) * (0.9 ** state["attempt"]))
        state["ha_max_iter"] = 4000 + 1500 * state["attempt"]

        gx, gy, radius = self.goals[state["seg_index"]]
        state["cur_goal"] = (gx, gy, radius)
        sx, sy, sth = state["seg_start_pose"]
        state["ha_start"] = (sx, sy, sth)
        state["ha_goal"] = (gx, gy)
        heapq.heappush(state["ha_open"], (0.0, state["ha_counter"], state["ha_start"]))
        state["ha_counter"] += 1
        state["ha_g"][state["ha_start"]] = 0.0

        self.log_msg(f"[Seg {state['seg_index']+1}] Tentativa {state['attempt']+1}: Hybrid A* → entrada dirigida.")
        self.update_status(
            f"Segmento {state['seg_index']+1} — tentativa {state['attempt']+1}: Hybrid A*"
        )
        self._tick_hybrid()

    def _ha_disc(self, x, y, theta, grid_res, ang_res):
        return (int(x / grid_res), int(y / grid_res), int((theta + math.pi) / ang_res))

    def _gen_primitives(self, x, y, theta, cur_goal):
        speeds = [0.6, 1.0, 1.6]
        steering = [
            -self.max_steering_angle,
            -self.max_steering_angle / 2,
            0,
            self.max_steering_angle / 2,
            self.max_steering_angle,
        ]
        primitives = []
        for velocity in speeds:
            for steer in steering:
                cx, cy, ct = x, y, theta
                traj = [(cx, cy, ct)]
                ok = True
                for _ in range(8):
                    px, py = cx, cy
                    cx, cy, ct = self.bicycle_step(cx, cy, ct, velocity, steer, self.dt)
                    if not self.is_valid(cx, cy):
                        ok = False
                        break
                    if self.early_entry_forbidden((px, py), (cx, cy), cur_goal):
                        ok = False
                        break
                    traj.append((cx, cy, ct))
                if ok and len(traj) > 1:
                    dist_m = self.px2m(math.hypot(traj[-1][0] - x, traj[-1][1] - y))
                    cost = dist_m + 0.3 * abs(steer)
                    primitives.append(dict(end=(cx, cy, ct), traj=traj, cost=cost))
        return primitives

    def _ha_reconstruct(self, came, current):
        path = []
        state = current
        while state in came:
            path.extend(reversed(came[state]["traj"]))
            state = came[state]["parent"]
        path.reverse()
        return path

    def _tick_hybrid(self):
        if not self.run_state or self.run_state["phase"] != "hybrid":
            return
        state = self.run_state
        grid_res = state["ha_grid_res"]
        ang_res = state["ha_angle_res"]
        iteration = state["ha_iter"] + 1
        state["ha_iter"] = iteration

        if not state["ha_open"] or iteration > state["ha_max_iter"]:
            self.log_msg("Hybrid A*: terminou (sem solução clara).")
            state["path_hybrid"] = state["ha_current_best"] or None
            self._start_dwa()
            return

        _, _, current = heapq.heappop(state["ha_open"])
        cd = self._ha_disc(*current, grid_res, ang_res)
        if cd in state["ha_closed"]:
            self.anim_timer = self.after(1, self._tick_hybrid)
            return
        state["ha_closed"].add(cd)

        gx, gy, radius = state["cur_goal"]
        ex, ey = state["entry_pt"]
        dist_center = math.hypot(current[0] - gx, current[1] - gy)
        use_tx, use_ty = (
            (ex, ey)
            if dist_center > radius + self.m2px(self.early_entry_band_m * 0.8)
            else (gx, gy)
        )

        if dist_center < grid_res * 1.5:
            path = self._ha_reconstruct(state["ha_came"], current)
            if not path:
                path = [
                    (state["ha_start"][0], state["ha_start"][1], state["ha_start"][2]),
                    (gx, gy, 0.0),
                ]
            state["path_hybrid"] = [(p[0], p[1], p[2], self.max_speed) for p in path]
            self.log_msg(
                f"Hybrid A*: solução (tent {state['attempt']+1}) com {len(path)} pontos em {iteration} it."
            )
            self.redraw(also_paths=[("orange", [(p[0], p[1]) for p in state["path_hybrid"]], 3, 0.9)])
            self._start_dwa()
            return

        for primitive in self._gen_primitives(*current, cur_goal=state["cur_goal"]):
            nxt = primitive["end"]
            nd = self._ha_disc(*nxt, grid_res, ang_res)
            if nd in state["ha_closed"]:
                continue
            ng = state["ha_g"][current] + primitive["cost"]
            if nxt not in state["ha_g"] or ng < state["ha_g"][nxt]:
                state["ha_g"][nxt] = ng
                heuristic = math.hypot(nxt[0] - use_tx, nxt[1] - use_ty)
                heuristic += 0.4 * self.goal_avoid_penalty(nxt[0], nxt[1], state["seg_index"])
                total = ng + heuristic
                heapq.heappush(state["ha_open"], (total, state["ha_counter"], nxt))
                state["ha_counter"] += 1
                state["ha_came"][nxt] = dict(parent=current, traj=primitive["traj"])

        state["ha_current_best"] = self._ha_reconstruct(state["ha_came"], current)
        show_best = (
            [(p[0], p[1]) for p in state["ha_current_best"]]
            if state["ha_current_best"]
            else None
        )
        self.redraw(also_paths=[("orange", show_best, 2, 0.9)] if show_best else None)
        self.update_status(
            f"Seg {state['seg_index']+1} tent {state['attempt']+1}: Hybrid A* — it {iteration}/{state['ha_max_iter']}"
        )
        self.anim_timer = self.after(1, self._tick_hybrid)

    def _start_dwa(self):
        if not self.run_state:
            return
        state = self.run_state
        state["phase"] = "dwa"
        sx, sy, sth = state["seg_start_pose"]
        state["dwa_state"] = dict(x=sx, y=sy, th=sth, v=self.max_speed * 0.6)
        state["dwa_path"] = [(sx, sy, sth, state["dwa_state"]["v"])]
        state["dwa_steps"] = 0
        state["dwa_stuck"] = 0
        self.log_msg("DWA: a avançar para entrar na região (entrada dirigida, sem entrada precoce).")
        self.update_status(f"Seg {state['seg_index']+1} tent {state['attempt']+1}: DWA")
        self._tick_dwa()

    def _tick_dwa(self):
        if not self.run_state or self.run_state["phase"] != "dwa":
            return
        state = self.run_state
        gx, gy, radius = state["cur_goal"]
        ex, ey = state["entry_pt"]
        dwa_state = state["dwa_state"]
        dist_center = math.hypot(dwa_state["x"] - gx, dwa_state["y"] - gy)
        tx, ty = (
            (ex, ey)
            if dist_center > radius + self.m2px(self.early_entry_band_m * 0.8)
            else (gx, gy)
        )

        if self.goal_hit(dwa_state["x"], dwa_state["y"], (gx, gy, radius)):
            state["path_dwa"] = state["dwa_path"][:]
            self.log_msg(f"DWA: terminou com {len(state['path_dwa'])} amostras (tent {state['attempt']+1}).")
            self._start_mpc_seed()
            return

        if state["dwa_steps"] > 1600:
            self.log_msg("DWA: bloqueado por passos excessivos.")
            self._retry_segment()
            return

        prev_dist = math.hypot(state["dwa_path"][-1][0] - tx, state["dwa_path"][-1][1] - ty)
        rich = state["attempt"] >= 1
        allow_reverse = state["attempt"] >= 2
        v = dwa_state["v"]
        v_min = -0.5 if allow_reverse else 0.0
        v_max = self.max_speed
        acc = 2.2 if rich else 2.0
        v_candidates = np.linspace(
            clamp(v - acc * self.dt, v_min, v_max),
            clamp(v + acc * self.dt, v_min, v_max),
            5 if rich else 4,
        )
        steer_candidates = np.linspace(
            -self.max_steering_angle, self.max_steering_angle, 13 if rich else 9
        )
        best = None
        for vv in v_candidates:
            for steer in steer_candidates:
                rx, ry, rth = dwa_state["x"], dwa_state["y"], dwa_state["th"]
                ok = True
                closest = 1e9
                for _ in range(6 + (2 if rich else 0)):
                    px, py = rx, ry
                    rx, ry, rth = self.bicycle_step(rx, ry, rth, vv, steer, self.dt * 0.5)
                    if not self.is_valid(rx, ry):
                        ok = False
                        break
                    if self.early_entry_forbidden((px, py), (rx, ry), (gx, gy, radius)):
                        ok = False
                        break
                    for (ox, oy) in self.obstacle_points:
                        distance = math.hypot(rx - ox, ry - oy)
                        closest = min(closest, distance)
                if not ok:
                    continue
                dist = math.hypot(rx - tx, ry - ty)
                cost = dist + 0.12 * abs(steer) + 0.18 * (v_max - max(0.0, vv))
                if closest < 25:
                    cost += (25 - closest) * 0.9
                cost += self.goal_avoid_penalty(rx, ry, state["seg_index"])
                if best is None or cost < best[0]:
                    best = (cost, vv, steer)
        if best is None:
            self.log_msg("DWA: sem movimento viável → retry.")
            self._retry_segment()
            return

        _, v_new, steer = best
        dwa_state["x"], dwa_state["y"], dwa_state["th"] = self.bicycle_step(
            dwa_state["x"], dwa_state["y"], dwa_state["th"], v_new, steer, self.dt
        )
        if self.early_entry_forbidden(
            (state["dwa_path"][-1][0], state["dwa_path"][-1][1]),
            (dwa_state["x"], dwa_state["y"]),
            (gx, gy, radius),
        ):
            dwa_state["x"], dwa_state["y"], dwa_state["th"] = self.bicycle_step(
                state["dwa_path"][-1][0],
                state["dwa_path"][-1][1],
                state["dwa_path"][-1][2],
                0.4,
                0.0,
                self.dt,
            )
        dwa_state["v"] = v_new
        state["dwa_path"].append((dwa_state["x"], dwa_state["y"], dwa_state["th"], dwa_state["v"]))
        state["dwa_steps"] += 1

        new_dist = math.hypot(dwa_state["x"] - tx, dwa_state["y"] - ty)
        if new_dist >= prev_dist - self.m2px(0.02):
            state["dwa_stuck"] += 1
        else:
            state["dwa_stuck"] = 0
        if state["dwa_stuck"] > 120:
            self.log_msg("DWA: sem progresso → retry com outra entrada.")
            self._retry_segment()
            return

        overlays = []
        if state.get("path_hybrid"):
            overlays.append(("orange", [(p[0], p[1]) for p in state["path_hybrid"]], 2, 0.6))
        overlays.append(("purple", [(p[0], p[1]) for p in state["dwa_path"]], 3, 0.9))
        self.redraw(also_paths=overlays, overlay=(dwa_state["x"], dwa_state["y"], dwa_state["th"], "purple"))
        self.update_status(
            f"Seg {state['seg_index']+1} tent {state['attempt']+1}: DWA — passo {state['dwa_steps']}"
        )
        self.anim_timer = self.after(16, self._tick_dwa)

    def _retry_segment(self):
        state = self.run_state
        state["attempt"] += 1
        self.log_msg(f"→ Retry {state['attempt']+1} / {state['max_attempts']}")
        self._start_segment()

    def _start_mpc_seed(self):
        if not self.run_state:
            return
        state = self.run_state
        state["phase"] = "mpc"
        c_h = self._traj_cost(state.get("path_hybrid"))
        c_d = self._traj_cost(state.get("path_dwa"))
        seed = (
            state.get("path_dwa") if (state.get("path_dwa") and c_d <= c_h) else state.get("path_hybrid")
        )
        state["mpc_seed"] = seed
        state["mpc_best"] = seed
        state["mpc_best_cost"] = self._traj_cost(seed)
        state["mpc_iter"] = 0
        self.log_msg(f"MPC: seed custo {state['mpc_best_cost']:.2f}. A otimizar…")
        self.update_status(f"Seg {state['seg_index']+1} tent {state['attempt']+1}: MPC")
        self._tick_mpc()

    def _traj_cost(self, path):
        if not path or len(path) < 2:
            return 1e9
        state = self.run_state
        distance = 0.0
        smooth = 0.0
        obs_pen = 0.0
        early_pen = 0.0
        gx = gy = radius = None
        if state and "cur_goal" in state and state["cur_goal"]:
            gx, gy, radius = state["cur_goal"]
        allow_tail = 8
        for idx in range(1, len(path)):
            x1, y1 = path[idx - 1][0], path[idx - 1][1]
            x2, y2 = path[idx][0], path[idx][1]
            distance += self.px2m(math.hypot(x2 - x1, y2 - y1))
            if idx >= 2:
                v1 = (x1 - path[idx - 2][0], y1 - path[idx - 2][1])
                v2 = (x2 - x1, y2 - y1)
                a1 = math.atan2(v1[1], v1[0])
                a2 = math.atan2(v2[1], v2[0])
                smooth += abs(ang_norm(a2 - a1))
            for (ox, oy) in self.obstacle_points[:: max(1, len(self.obstacle_points) // 80 or 1)]:
                d = math.hypot(x2 - ox, y2 - oy)
                if d < 28:
                    obs_pen += (28 - d) * 0.06
            cur_idx = state["seg_index"] if state else 0
            obs_pen += self.goal_avoid_penalty(x2, y2, cur_idx)
            if gx is not None and idx < len(path) - allow_tail:
                if self.early_entry_forbidden((x1, y1), (x2, y2), (gx, gy, radius)):
                    early_pen += 5.0
        return distance + 0.2 * smooth + obs_pen + early_pen

    def _curvature(self, points):
        curvature = np.zeros(len(points))
        if len(points) < 3:
            return curvature
        for idx in range(1, len(points) - 1):
            x1, y1 = points[idx - 1]
            x2, y2 = points[idx]
            x3, y3 = points[idx + 1]
            a = np.hypot(x2 - x1, y2 - y1)
            b = np.hypot(x3 - x2, y3 - y2)
            c = np.hypot(x3 - x1, y3 - y1)
            if a * b * c == 0:
                curvature[idx] = 0
                continue
            s = (a + b + c) / 2.0
            area = max(1e-9, np.sqrt(max(0.0, s * (s - a) * (s - b) * (s - c))))
            r = (a * b * c) / (4.0 * area)
            curvature[idx] = 0.0 if r == 0 else 1.0 / self.px2m(r)
        curvature[0] = curvature[1]
        curvature[-1] = curvature[-2]
        return curvature

    def _estimate_time(self, path):
        if not path or len(path) < 2:
            return 1e9
        pts = [(x, y) for (x, y, _, _) in path]
        ds = [
            self.px2m(np.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]))
            for i in range(len(pts) - 1)
        ]
        ds.append(ds[-1] if ds else 0.1)
        curv = self._curvature(pts)
        ay_max = self.mu * 9.81
        v_k = [min(self.max_speed, np.sqrt(ay_max / max(1e-6, abs(curv[i])))) for i in range(len(pts))]
        velocities = [0.0] * len(pts)
        velocities[0] = min(v_k[0], self.max_speed)
        for i in range(len(pts) - 1):
            v_hat = np.sqrt(max(0.0, velocities[i] ** 2 + 2 * self.ax_max * ds[i]))
            velocities[i + 1] = min(v_hat, v_k[i + 1], self.max_speed)
        for i in range(len(pts) - 2, -1, -1):
            v_hat = np.sqrt(max(0.0, velocities[i + 1] ** 2 + 2 * self.brake_max * ds[i]))
            velocities[i] = min(velocities[i], v_hat, v_k[i], self.max_speed)
        return sum(ds[i] / max(0.2, velocities[i]) for i in range(len(pts)))

    def _smooth_path(self, path, iters=2):
        if not path or len(path) < 3:
            return path[:]
        pts = [(x, y) for (x, y, _, _) in path]
        for _ in range(iters):
            new_pts = [pts[0]]
            for i in range(len(pts) - 1):
                p = np.array(pts[i])
                q = np.array(pts[i + 1])
                new_pts.append(tuple(0.75 * p + 0.25 * q))
                new_pts.append(tuple(0.25 * p + 0.75 * q))
            new_pts.append(pts[-1])
            pts = new_pts
        output = []
        for idx, (x, y) in enumerate(pts):
            if idx == 0:
                theta = math.atan2(pts[1][1] - y, pts[1][0] - x)
            else:
                theta = math.atan2(y - pts[idx - 1][1], x - pts[idx - 1][0])
            output.append((x, y, theta, self.max_speed))
        return output

    def _nearest_obstacle_dist(self, x, y):
        if not self.obstacle_points:
            return 1e9
        step = max(1, len(self.obstacle_points) // 300)
        dmin = 1e9
        for (ox, oy) in self.obstacle_points[::step]:
            d = math.hypot(x - ox, y - oy)
            if d < dmin:
                dmin = d
        return self.px2m(dmin)

    def _raceline_profile(self, path):
        if not path or len(path) < 2:
            return None
        pts = [(x, y) for (x, y, _, _) in path]
        ds = [
            self.px2m(np.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]))
            for i in range(len(pts) - 1)
        ]
        ds.append(ds[-1] if ds else 0.1)
        curv = self._curvature(pts)
        ay_max = self.mu * 9.81
        v_k = np.array([min(self.max_speed, np.sqrt(ay_max / max(1e-6, abs(curv[i])))) for i in range(len(pts))])
        v_obs = []
        for (x, y) in pts:
            d = max(0.01, self._nearest_obstacle_dist(x, y) - self.safety_margin_m)
            factor = clamp(d / (d + 0.7), 0.35, 1.0)
            v_obs.append(self.max_speed * factor)
        v_lim = np.minimum(v_k, np.array(v_obs))

        velocities = [0.0] * len(pts)
        velocities[0] = min(v_lim[0], self.max_speed)
        for i in range(len(pts) - 1):
            v_hat = np.sqrt(max(0.0, velocities[i] ** 2 + 2 * self.ax_max * ds[i]))
            velocities[i + 1] = min(v_hat, v_lim[i + 1], self.max_speed)
        for i in range(len(pts) - 2, -1, -1):
            v_hat = np.sqrt(max(0.0, velocities[i + 1] ** 2 + 2 * self.brake_max * ds[i]))
            velocities[i] = min(velocities[i], v_hat, v_lim[i], self.max_speed)

        total_time = sum(ds[i] / max(0.2, velocities[i]) for i in range(len(pts)))
        avg_v = (sum(ds[:-1]) / max(1e-6, total_time))
        return dict(points=pts, speeds=velocities, ds=ds, time=total_time, vavg=avg_v)

    def _three_point_filter(self, pts, passes=4):
        if len(pts) < 3:
            return np.array(pts, dtype=float)
        work = np.array(pts, dtype=float)
        for _ in range(max(1, passes)):
            updated = work.copy()
            updated[1:-1] = (work[:-2] + work[1:-1] + work[2:]) / 3.0
            work = updated
        return work

    def _fast_raceline_profile(self, path):
        if not path or len(path) < 3:
            return None
        pts = []
        for node in path:
            if isinstance(node, (list, tuple)) and len(node) >= 2:
                pts.append((float(node[0]), float(node[1])))
        if len(pts) < 3:
            return None
        filtered = self._three_point_filter(pts, passes=5)
        ds = [
            self.px2m(
                np.hypot(filtered[i + 1][0] - filtered[i][0], filtered[i + 1][1] - filtered[i][1])
            )
            for i in range(len(filtered) - 1)
        ]
        ds.append(ds[-1] if ds else 0.0)
        ay_max = max(0.1, float(self.mu) * 9.81)
        min_turning_m = max(0.1, float(self.min_turning_radius))
        speeds = []
        for idx in range(len(filtered)):
            if idx == 0 or idx == len(filtered) - 1:
                speeds.append(float(self.max_speed))
                continue
            radius_px = self._radius_from_points(
                filtered[idx - 1], filtered[idx], filtered[idx + 1]
            )
            radius_m = max(min_turning_m, self.px2m(radius_px))
            vmax = math.sqrt(max(ay_max * radius_m, 1e-6))
            speeds.append(min(float(self.max_speed), max(0.5, vmax)))
        total_time = 0.0
        for i in range(len(filtered) - 1):
            seg_speed = min(speeds[i], speeds[i + 1])
            seg_speed = max(seg_speed, 0.5)
            total_time += ds[i] / seg_speed
        if total_time <= 0:
            total_time = sum(ds[:-1]) / max(0.5, float(self.max_speed))
        return dict(
            points=[(float(x), float(y)) for (x, y) in filtered],
            speeds=speeds,
            ds=ds,
            time=total_time,
            vavg=(sum(ds[:-1]) / total_time) if total_time > 1e-6 else 0.0,
        )

    def _radius_from_points(self, p0, p1, p2):
        a = np.asarray(p0, dtype=float)
        b = np.asarray(p1, dtype=float)
        c = np.asarray(p2, dtype=float)
        ab = np.linalg.norm(a - b)
        bc = np.linalg.norm(b - c)
        ac = np.linalg.norm(a - c)
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        denom = 2 * abs(cross)
        if denom < 1e-6 or ab < 1e-6 or bc < 1e-6 or ac < 1e-6:
            return 1e9
        radius = (ab * bc * ac) / denom
        return float(max(radius, 1e-6))

    def _circle_center(self, p0, p1, p2):
        (x1, y1) = p0
        (x2, y2) = p1
        (x3, y3) = p2
        d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if abs(d) < 1e-6:
            return None
        ux = (
            (x1**2 + y1**2) * (y2 - y3)
            + (x2**2 + y2**2) * (y3 - y1)
            + (x3**2 + y3**2) * (y1 - y2)
        ) / d
        uy = (
            (x1**2 + y1**2) * (x3 - x2)
            + (x2**2 + y2**2) * (x1 - x3)
            + (x3**2 + y3**2) * (x2 - x1)
        ) / d
        return np.array([ux, uy], dtype=float)

    def _resample_polyline(self, pts, spacing_px):
        if len(pts) < 2 or spacing_px <= 0:
            return [tuple(p) for p in pts]
        lengths = [0.0]
        cumulative = 0.0
        for i in range(1, len(pts)):
            seg_len = np.linalg.norm(
                np.asarray(pts[i], dtype=float) - np.asarray(pts[i - 1], dtype=float)
            )
            cumulative += seg_len
            lengths.append(cumulative)
        total = cumulative
        if total < 1e-6:
            return [tuple(pts[0])]
        samples = [0.0]
        cur = spacing_px
        while cur < total:
            samples.append(cur)
            cur += spacing_px
        samples.append(total)
        resampled = []
        idx = 0
        pts_arr = [np.asarray(p, dtype=float) for p in pts]
        for s in samples:
            while idx < len(lengths) - 1 and lengths[idx + 1] < s:
                idx += 1
            if idx >= len(lengths) - 1:
                resampled.append(tuple(pts_arr[-1]))
                continue
            span = lengths[idx + 1] - lengths[idx]
            if span <= 1e-9:
                resampled.append(tuple(pts_arr[idx]))
                continue
            t = (s - lengths[idx]) / span
            interp = pts_arr[idx] * (1 - t) + pts_arr[idx + 1] * t
            resampled.append((float(interp[0]), float(interp[1])))
        return resampled

    def _enforce_curvature(self, pts, min_radius_px, max_iter=40, step_px=5.0):
        if len(pts) < 3:
            return pts[:]
        work = [np.asarray(p, dtype=float) for p in pts]
        for _ in range(max_iter):
            changed = False
            for i in range(1, len(work) - 1):
                radius = self._radius_from_points(work[i - 1], work[i], work[i + 1])
                if radius >= min_radius_px:
                    continue
                center = self._circle_center(work[i - 1], work[i], work[i + 1])
                if center is None:
                    continue
                direction = work[i] - center
                norm = np.linalg.norm(direction)
                if norm < 1e-6:
                    continue
                unit = direction / norm
                delta = min(step_px, (min_radius_px - radius) * 0.6)
                candidate = work[i] + unit * delta
                if not self.is_valid(float(candidate[0]), float(candidate[1])):
                    continue
                work[i] = candidate
                changed = True
            if not changed:
                break
        return [(float(p[0]), float(p[1])) for p in work]

    def _points_to_path(self, pts, base_speed):
        if not pts:
            return []
        path = []
        base_speed = max(0.5, float(base_speed))
        for idx, (x, y) in enumerate(pts):
            if idx == len(pts) - 1 and idx > 0:
                px, py = pts[idx - 1]
                theta = math.atan2(y - py, x - px)
            elif idx < len(pts) - 1:
                nx, ny = pts[idx + 1]
                theta = math.atan2(ny - y, nx - x)
            else:
                theta = 0.0
            path.append((float(x), float(y), float(theta), base_speed))
        return path

    def _optimize_points_for_speed(self, path):
        base_pts = [(float(node[0]), float(node[1])) for node in path if len(node) >= 2]
        if len(base_pts) < 3:
            return None
        spacing_px = max(self.m2px(0.3), 10.0)
        ay_max = max(0.1, float(self.mu) * 9.81)
        target_speed = float(self.max_speed)
        best_pts = None
        best_time = float("inf")
        for factor in (1.0, 0.95, 0.9, 0.85):
            desired_speed = max(0.5, target_speed * factor)
            min_radius_m = max(self.min_turning_radius, (desired_speed ** 2) / ay_max)
            min_radius_px = self.m2px(min_radius_m)
            candidate = self._enforce_curvature(
                self._resample_polyline(base_pts, spacing_px), min_radius_px
            )
            if len(candidate) < 3:
                continue
            invalid = False
            for i in range(len(candidate) - 1):
                if not self.los_clear(candidate[i], candidate[i + 1]):
                    invalid = True
                    break
            if invalid:
                continue
            candidate_path = self._points_to_path(candidate, desired_speed)
            profile = self._fast_raceline_profile(candidate_path)
            if not profile:
                continue
            if profile["time"] < best_time:
                best_time = profile["time"]
                best_pts = candidate
        return best_pts

    def _draw_raceline(self, raceline):
        pts = raceline["points"]
        speeds = raceline["speeds"]
        segments = [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)]
        lc = LineCollection(segments, array=np.array(speeds[:-1]), cmap="plasma", linewidths=4)
        if self._raceline_lc is not None:
            try:
                self._raceline_lc.remove()
            except Exception:
                pass
            self._raceline_lc = None
        self.ax.add_collection(lc)
        self._raceline_lc = lc
        if self._raceline_cbar is not None:
            try:
                self._raceline_cbar.remove()
            except Exception:
                pass
            self._raceline_cbar = None
        self._raceline_cbar = self.fig.colorbar(lc, ax=self.ax, fraction=0.03, pad=0.01)
        self._raceline_cbar.set_label("Velocidade [m/s]")
        self.canvas.draw()

    def _build_and_draw_raceline(self, full_path):
        smooth = self._smooth_path(full_path, iters=2)
        raceline = self._raceline_profile(smooth)
        if raceline is None:
            self.log_msg("Linha de corrida: falhou a gerar perfil.")
            return
        self.redraw()
        self._draw_raceline(raceline)
        self.log_msg(f"Race line: tempo {raceline['time']:.2f} s | v̄ = {raceline['vavg']:.2f} m/s")

    def _build_and_draw_fast_raceline(self, full_path, profile=None):
        if profile is None:
            profile = self._fast_raceline_profile(full_path)
        if profile is None:
            messagebox.showwarning("Race line", "Precisas de um caminho com pelo menos 3 pontos.")
            return
        base_coords = []
        for node in full_path:
            if isinstance(node, (list, tuple)) and len(node) >= 2:
                base_coords.append((float(node[0]), float(node[1])))
        self.redraw(also_paths=[("cyan", base_coords, 2, 0.6)] if base_coords else None)
        self._draw_raceline(profile)
        self.log_msg(
            f"Race line (rápida): dist ~ {sum(profile['ds'][:-1]):.2f} m | tempo {profile['time']:.2f} s"
        )

    def _astar_pixels(self, start, goal, step_px):
        sx, sy = start
        gx, gy = goal
        if step_px < 4:
            step_px = 4
        quant = lambda v: int(round(v / step_px))
        dequant = lambda q: float(q * step_px)
        start_cell = (quant(sx), quant(sy))
        goal_cell = (quant(gx), quant(gy))

        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0.0, start_cell))
        g_cost = {start_cell: 0.0}
        came = {}
        visited = set()

        neighbors = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)
            if heuristic(current, goal_cell) <= 1.0:
                path = []
                node = current
                while node in came:
                    x = dequant(node[0])
                    y = dequant(node[1])
                    path.append((x, y))
                    node = came[node]
                path.append((dequant(start_cell[0]), dequant(start_cell[1])))
                path.reverse()
                path.append((float(gx), float(gy)))
                return path
            cx, cy = current
            current_pt = (dequant(cx), dequant(cy))
            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                next_cell = (nx, ny)
                next_pt = (dequant(nx), dequant(ny))
                if not self.is_valid(next_pt[0], next_pt[1]):
                    continue
                if not self.los_clear(current_pt, next_pt):
                    continue
                tentative = g_cost[current] + math.hypot(dx, dy)
                if tentative < g_cost.get(next_cell, float("inf")):
                    g_cost[next_cell] = tentative
                    priority = tentative + heuristic(next_cell, goal_cell)
                    heapq.heappush(open_set, (priority, next_cell))
                    came[next_cell] = current
        return None

    def _plan_map_path(self):
        if not self.start_point:
            return None
        if not self.image:
            return None
        step_px = int(max(self.grid_spacing or 20, 20) // 2)
        waypoints = []
        goal_targets = []
        start_pt = (float(self.start_point[0]), float(self.start_point[1]))
        waypoints.append(start_pt)
        goal_targets.append((start_pt, 0.0))
        for goal in self.goals:
            gx, gy, radius = goal
            waypoints.append((float(gx), float(gy)))
            goal_targets.append(((float(gx), float(gy)), float(radius)))
        if len(waypoints) == 1:
            return None
        path_pts = [waypoints[0]]
        radii = [goal_targets[0][1]]
        for idx in range(1, len(waypoints)):
            segment = self._astar_pixels(waypoints[idx - 1], waypoints[idx], step_px)
            if not segment:
                return None
            if path_pts:
                segment = segment[1:]
            path_pts.extend(segment)
            radii.append(goal_targets[idx][1])
        if len(path_pts) < 2:
            return None
        path_pts = self._relax_goal_points(path_pts, radii)
        return self._points_to_path(path_pts, self.max_speed)

    def _synthetic_raceline_path(self):
        if not self.start_point:
            return None
        anchors = [
            (float(self.start_point[0]), float(self.start_point[1])),
        ]
        if self.goals:
            for gx, gy, _ in self.goals[:2]:
                anchors.append((float(gx), float(gy)))
        start_theta = getattr(self, "start_theta", 0.0)
        if len(anchors) < 2:
            step = max(self.m2px(5.0), 40.0)
            anchors.append(
                (
                    anchors[0][0] + math.cos(start_theta) * step,
                    anchors[0][1] + math.sin(start_theta) * step,
                )
            )
        if len(anchors) < 3:
            step = max(self.m2px(10.0), 80.0)
            anchors.append(
                (
                    anchors[0][0] + math.cos(start_theta) * step,
                    anchors[0][1] + math.sin(start_theta) * step,
                )
            )

        dense = []
        for idx in range(len(anchors) - 1):
            x0, y0 = anchors[idx]
            x1, y1 = anchors[idx + 1]
            seg_len_px = math.hypot(x1 - x0, y1 - y0)
            steps = max(12, int(self.px2m(seg_len_px) * 20))
            for k in range(steps):
                t = k / steps
                dense.append((x0 * (1 - t) + x1 * t, y0 * (1 - t) + y1 * t))
        dense.append(anchors[-1])

        path = []
        for idx, (x, y) in enumerate(dense):
            if idx < len(dense) - 1:
                nx, ny = dense[idx + 1]
            else:
                nx, ny = dense[idx]
            theta = math.atan2(ny - y, nx - x) if (nx != x or ny != y) else start_theta
            path.append((float(x), float(y), float(theta), float(self.max_speed)))
        return path

    def _project_point_to_circle(self, point, center, radius_px, prev_vec=None):
        if radius_px <= 1:
            return point
        cx, cy = center
        vx = point[0] - cx
        vy = point[1] - cy
        dist = math.hypot(vx, vy)
        if dist < 1e-6:
            if prev_vec is not None:
                vx, vy = prev_vec
                dist = math.hypot(vx, vy)
                if dist < 1e-6:
                    return (cx + radius_px, cy)
            else:
                return (cx + radius_px, cy)
        scale = radius_px / dist
        return (cx + vx * scale, cy + vy * scale)

    def _relax_goal_points(self, pts, radii_px, passes=2):
        if not radii_px or len(radii_px) != len(pts):
            return pts
        relaxed = list(pts)
        for _ in range(max(1, passes)):
            for idx in range(1, len(relaxed)):
                radius = radii_px[idx]
                if radius <= 0:
                    continue
                center = pts[idx]
                prev_vec = None
                if idx > 0:
                    prev_vec = (
                        relaxed[idx - 1][0] - center[0],
                        relaxed[idx - 1][1] - center[1],
                    )
                relaxed[idx] = self._project_point_to_circle(relaxed[idx], center, radius, prev_vec)
        return relaxed

    def build_fast_raceline_current(self):
        path = self._choose_best_available_path()
        synthetic_used = False
        if not path:
            path = self._plan_map_path()
            if not path:
                path = self._synthetic_raceline_path()
                if not path:
                    messagebox.showwarning(
                        "Race line",
                        "Define a origem e, idealmente, adiciona alguns objetivos antes de gerar a race line.",
                    )
                    return
                synthetic_used = True
        best_path = path
        best_profile = self._fast_raceline_profile(path)
        optimized_pts = self._optimize_points_for_speed(path)
        if optimized_pts:
            radii = []
            if self.goals:
                radii = [0.0]
                radii.extend([float(g[2]) for g in self.goals])
                if len(radii) < len(optimized_pts):
                    radii.extend([radii[-1]] * (len(optimized_pts) - len(radii)))
                optimized_pts = self._relax_goal_points(optimized_pts, radii)
            optimized_path = self._points_to_path(optimized_pts, self.max_speed)
            optimized_profile = self._fast_raceline_profile(optimized_path)
            if optimized_profile and (
                best_profile is None or optimized_profile["time"] < best_profile["time"]
            ):
                best_path = optimized_path
                best_profile = optimized_profile
        if best_profile is None:
            messagebox.showwarning("Race line", "Falha ao gerar perfil rápido.")
            return
        self.manual_path = best_path
        self.run_state = dict(full_path=best_path)
        if synthetic_used:
            self.log_msg("Race line (rápida): caminho sintético gerado a partir de origem/objetivos.")
        elif best_path is not path:
            self.log_msg("Race line (rápida): caminho otimizado para alta velocidade.")
        self._build_and_draw_fast_raceline(best_path, profile=best_profile)

    def _tick_mpc(self):
        if not self.run_state or self.run_state["phase"] != "mpc":
            return
        state = self.run_state
        iteration = state["mpc_iter"] + 1
        state["mpc_iter"] = iteration
        if iteration > state["mpc_max_iter"]:
            state["path_mpc"] = state["mpc_best"]
            self._finish_segment_and_continue()
            return

        base = state["mpc_best"]
        if not base or len(base) < 3:
            state["path_mpc"] = base
            self._finish_segment_and_continue()
            return

        trials = 40
        best_local = None
        best_cost = 1e9
        for _ in range(trials):
            candidate = []
            for idx, point in enumerate(base):
                x, y, theta, vel = point
                if 0 < idx < len(base) - 1:
                    x += np.random.normal(0, 3.0)
                    y += np.random.normal(0, 3.0)
                candidate.append((x, y, theta, vel))
            ok = True
            for idx in range(1, len(candidate)):
                if not self.los_clear(
                    (candidate[idx - 1][0], candidate[idx - 1][1]),
                    (candidate[idx][0], candidate[idx][1]),
                ):
                    ok = False
                    break
            if not ok:
                continue
            cost = self._traj_cost(candidate)
            if cost < best_cost:
                best_cost = cost
                best_local = candidate

        if best_local is not None and best_cost < state["mpc_best_cost"]:
            state["mpc_best"] = best_local
            state["mpc_best_cost"] = best_cost

        overlays = []
        if state.get("mpc_seed"):
            overlays.append(("lightgray", [(p[0], p[1]) for p in state["mpc_seed"]], 2, 0.6))
        if state.get("path_hybrid"):
            overlays.append(("orange", [(p[0], p[1]) for p in state["path_hybrid"]], 1.5, 0.5))
        if state.get("path_dwa"):
            overlays.append(("purple", [(p[0], p[1]) for p in state["path_dwa"]], 1.5, 0.5))
        overlays.append(("magenta", [(p[0], p[1]) for p in state["mpc_best"]], 3, 0.9))
        self.redraw(also_paths=overlays)
        self.update_status(
            f"Seg {state['seg_index']+1} tent {state['attempt']+1}: MPC — it {iteration}/{state['mpc_max_iter']}"
        )
        self.anim_timer = self.after(1, self._tick_mpc)

    def _finish_segment_and_continue(self):
        state = self.run_state
        gx, gy, radius = state["cur_goal"]

        def enters(path):
            return bool(path) and self.goal_hit(path[-1][0], path[-1][1], (gx, gy, radius))

        candidates = []
        if enters(state.get("path_hybrid")):
            candidates.append(("Hybrid A*", state["path_hybrid"]))
        if enters(state.get("path_dwa")):
            candidates.append(("DWA", state["path_dwa"]))
        if enters(state.get("path_mpc")):
            candidates.append(("MPC", state["path_mpc"]))

        if not candidates:
            self.log_msg(f"[Seg {state['seg_index']+1}] Nenhum caminho entrou no objetivo → nova tentativa.")
            self._retry_segment()
            return

        ranked = []
        for name, path in candidates:
            time_est = self._estimate_time(path)
            cost = self._traj_cost(path)
            ranked.append((time_est, cost, name, path))
        ranked.sort(key=lambda z: (z[0], z[1]))
        time_best, cost_best, name, best = ranked[0]
        self.log_msg(
            f"[Seg {state['seg_index']+1}] Melhor: {name}  (tempo {time_best:.2f}s | custo {cost_best:.2f})"
        )

        if not state["full_path"]:
            state["full_path"].extend(best)
        else:
            state["full_path"].extend(best[1:])
        xb, yb, thb, _ = best[-1]
        state["seg_start_pose"] = (xb, yb, thb)

        state["attempt"] = 0
        state["seg_index"] += 1
        self.redraw(
            also_paths=[("blue", [(x, y) for (x, y, _, _) in state["full_path"]], 3, 0.9)]
        )
        self._start_segment()
