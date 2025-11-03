"""Environment, kinematics, and flood-fill helpers."""

import math
from collections import deque

import numpy as np

from ..utils import ang_norm


class EnvironmentMixin:
    """Handles kinematics, collision, goals, and flood-fill support."""

    def bicycle_step(self, x, y, theta, velocity, steer, dt):
        dx = self.m2px(velocity * math.cos(theta) * dt)
        dy = self.m2px(velocity * math.sin(theta) * dt)
        x2, y2 = x + dx, y + dy
        theta2 = ang_norm(theta + (velocity / self.wheelbase) * math.tan(steer) * dt)
        return x2, y2, theta2

    def is_valid(self, x, y):
        if not self.image:
            return True
        width, height = self.image.size
        if not (0 <= x < width and 0 <= y < height):
            return False

        margin = max(10, self.grid_spacing / 3 if self.grid_spacing else 10)
        for ox, oy in self.obstacle_points:
            if abs(x - ox) < margin and abs(y - oy) < margin:
                if math.hypot(x - ox, y - oy) < margin:
                    return False
        return True

    def los_clear(self, point_a, point_b):
        x1, y1 = point_a
        x2, y2 = point_b
        distance = math.hypot(x2 - x1, y2 - y1)
        if distance == 0:
            return True
        step = max(2, int(distance / max(2, (self.grid_spacing or 20) / 2)))
        for i in range(step + 1):
            t = i / step
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            if not self.is_valid(x, y):
                return False
        return True

    def goal_hit(self, x, y, goal):
        gx, gy, radius = goal
        extra = max(12, int(self.grid_spacing or 12))
        return (x - gx) ** 2 + (y - gy) ** 2 <= (radius + extra) ** 2

    def goal_avoid_penalty(self, x, y, current_idx):
        if not self.goals:
            return 0.0
        penalty_px = 0.0
        margin_px = self.m2px(self.goal_avoid_margin_m)
        for idx, (gx, gy, radius) in enumerate(self.goals):
            if idx == current_idx:
                continue
            distance = math.hypot(x - gx, y - gy)
            threshold = radius + margin_px
            if distance < threshold:
                penalty_px += (threshold - distance)
        return self.px2m(penalty_px) * float(self.goal_avoid_weight)

    def early_entry_forbidden(self, prev_xy, new_xy, goal):
        gx, gy, radius = goal
        d_prev = math.hypot(prev_xy[0] - gx, prev_xy[1] - gy)
        d_new = math.hypot(new_xy[0] - gx, new_xy[1] - gy)
        band = self.m2px(self.early_entry_band_m)
        return (d_new <= radius) and (d_prev > radius + band)

    def flood_fill_grid(self, cell_x, cell_y):
        if self.image_array is None or self.grid_spacing is None:
            return []
        height, width = self.image_array.shape[:2]
        grid_w = int(width // self.grid_spacing)
        grid_h = int(height // self.grid_spacing)
        if not (0 <= cell_x < grid_w and 0 <= cell_y < grid_h):
            return []

        visited = set()
        queue = deque([(cell_x, cell_y)])
        points = []
        px = int(cell_x * self.grid_spacing + self.grid_spacing // 2)
        py = int(cell_y * self.grid_spacing + self.grid_spacing // 2)
        if px < 0 or px >= width or py < 0 or py >= height:
            return []
        reference = self.image_array[py, px]

        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) in visited or not (0 <= cx < grid_w and 0 <= cy < grid_h):
                continue
            visited.add((cx, cy))
            px = int(cx * self.grid_spacing + self.grid_spacing // 2)
            py = int(cy * self.grid_spacing + self.grid_spacing // 2)
            if px < 0 or px >= width or py < 0 or py >= height:
                continue
            color = self.image_array[py, px]
            if len(self.image_array.shape) == 3:
                ok = np.all(np.abs(color.astype(int) - reference.astype(int)) <= self.color_tolerance)
            else:
                ok = abs(int(color) - int(reference)) <= self.color_tolerance
            if ok:
                points.append((px, py))
                for dx, dy in [
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                    (-1, -1),
                    (-1, 1),
                    (1, -1),
                    (1, 1),
                ]:
                    queue.append((cx + dx, cy + dy))
        return points

    def circle_fill_grid(self, center_x, center_y, radius_px):
        """Return grid cell center points within a circle in pixel space.

        The returned points are pixel coordinates at grid centers, compatible
        with obstacle_points usage elsewhere.
        """
        if self.image is None or self.grid_spacing is None:
            return []
        width, height = self.image.size
        if radius_px <= 0:
            return []
        gs = float(self.grid_spacing)
        # compute bounds in grid index space
        gx0 = max(0, int((center_x - radius_px) // gs))
        gy0 = max(0, int((center_y - radius_px) // gs))
        gx1 = min(int(width // gs), int((center_x + radius_px) // gs) + 1)
        gy1 = min(int(height // gs), int((center_y + radius_px) // gs) + 1)
        r2 = float(radius_px) * float(radius_px)
        pts = []
        # iterate grid cell indices and add center if within circle
        for gy in range(gy0, gy1):
            py = gy * gs + gs / 2.0
            for gx in range(gx0, gx1):
                px = gx * gs + gs / 2.0
                if (px - center_x) * (px - center_x) + (py - center_y) * (py - center_y) <= r2:
                    # clamp within image just in case
                    if 0 <= px < width and 0 <= py < height:
                        pts.append((px, py))
        return pts
