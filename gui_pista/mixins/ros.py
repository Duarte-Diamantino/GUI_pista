"""ROS2 publishing helpers."""

import math
from tkinter import messagebox

import numpy as np

try:
    import rclpy
    from geometry_msgs.msg import Point, Pose, PoseStamped
    from nav_msgs.msg import MapMetaData, OccupancyGrid, Path
    from visualization_msgs.msg import Marker, MarkerArray

    HAVE_ROS2 = True
except Exception:
    HAVE_ROS2 = False


class ROSMixin:
    """Handles ROS2 path publication."""

    def _yaw_to_quat(self, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        return (0.0, 0.0, sy, cy)

    def _px_to_world(self, x_px, y_px):
        vx, vy = self.axis_x_vec
        ux, uy = self.axis_y_vec
        xp = vx * x_px + vy * y_px
        yp = ux * x_px + uy * y_px
        xm = self.px2m(xp) + float(self.origin_x_m)
        ym = self.px2m(yp) + float(self.origin_y_m)
        return xm, ym

    def _set_origin_from_pixel(self, x_px, y_px, label="start"):
        vx, vy = self.axis_x_vec
        ux, uy = self.axis_y_vec
        xp0 = vx * x_px + vy * y_px
        yp0 = ux * x_px + uy * y_px
        self.origin_x_m = -self.px2m(xp0)
        self.origin_y_m = -self.px2m(yp0)
        if hasattr(self, "offx_var"):
            self.offx_var.set(self.origin_x_m)
        if hasattr(self, "offy_var"):
            self.offy_var.set(self.origin_y_m)
        self.log_msg(f"Origin @{label}: offsets = ({self.origin_x_m:.3f}, {self.origin_y_m:.3f}) m")
        return True

    def _ensure_ros2(self):
        if not HAVE_ROS2:
            messagebox.showerror("ROS2", "ROS2 não está disponível.")
            return False
        if not self.ros_inited:
            try:
                rclpy.init(args=None)
                self.ros_node = rclpy.create_node("traj_gui_publisher")
                self.ros_pub = self.ros_node.create_publisher(Path, self.ref_path_topic, 10)
                self.ros_map_pub = self.ros_node.create_publisher(OccupancyGrid, self.map_topic, 10)
                self.ros_limit_pub = self.ros_node.create_publisher(MarkerArray, self.limit_topic, 10)
                self.ros_inited = True
                self.log_msg(
                    f"ROS2: publishers criados em '{self.ref_path_topic}', '{self.map_topic}' e '{self.limit_topic}'."
                )
            except Exception as exc:
                messagebox.showerror("ROS2", f"Falha a iniciar ROS2: {exc}")
                self.ros_node = None
                self.ros_pub = None
                self.ros_map_pub = None
                self.ros_limit_pub = None
                self.ros_inited = False
                return False
        return True

    def _build_nav_path(self, path_pts):
        if not path_pts or len(path_pts) < 2:
            return None
        msg = Path()
        frame = self.frame_id_var.get().strip() or "map"
        msg.header.frame_id = frame
        if self.ros_node is not None:
            msg.header.stamp = self.ros_node.get_clock().now().to_msg()
        poses = []
        for (x, y, theta, _) in path_pts:
            px, py = self._px_to_world(x, y)
            qx, qy, qz, qw = self._yaw_to_quat(theta)
            pose = PoseStamped()
            pose.header.frame_id = frame
            if self.ros_node is not None:
                pose.header.stamp = self.ros_node.get_clock().now().to_msg()
            pose.pose.position.x = float(px)
            pose.pose.position.y = float(py)
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            poses.append(pose)
        msg.poses = poses
        return msg

    def _build_occupancy_grid(self):
        if self.image_array is None:
            return None
        arr = np.array(self.image_array)
        if arr.size == 0:
            return None
        if arr.ndim == 3:
            gray = arr.mean(axis=2)
        else:
            gray = arr.astype(np.float32)
        threshold = float(getattr(self, "map_occ_threshold", 140.0))
        occ = np.where(gray <= threshold, 100, 0).astype(np.int16)
        if getattr(self, "obstacle_points", None):
            radius = max(1, int((self.grid_spacing or 12) // 4) or 1)
            height, width = occ.shape
            for ox, oy in self.obstacle_points:
                ix = int(round(ox))
                iy = int(round(oy))
                if 0 <= ix < width and 0 <= iy < height:
                    x0 = max(0, ix - radius)
                    x1 = min(width, ix + radius + 1)
                    y0 = max(0, iy - radius)
                    y1 = min(height, iy + radius + 1)
                    occ[y0:y1, x0:x1] = 100
        occ = np.flipud(occ)

        msg = OccupancyGrid()
        msg.header.frame_id = self.frame_id
        if self.ros_node is not None:
            stamp = self.ros_node.get_clock().now().to_msg()
            msg.header.stamp = stamp
        info = MapMetaData()
        if self.ros_node is not None:
            info.map_load_time = self.ros_node.get_clock().now().to_msg()
        res = float(self.px2m(1.0))
        if not math.isfinite(res) or res <= 0:
            res = 0.05
        height, width = occ.shape
        info.resolution = res
        info.width = int(width)
        info.height = int(height)
        yaw = math.atan2(self.axis_x_vec[1], self.axis_x_vec[0])
        qx, qy, qz, qw = self._yaw_to_quat(yaw)
        origin_pose = Pose()
        origin_x, origin_y = self._px_to_world(-0.5, float(height) - 0.5)
        origin_pose.position.x = float(origin_x)
        origin_pose.position.y = float(origin_y)
        origin_pose.position.z = 0.0
        origin_pose.orientation.x = qx
        origin_pose.orientation.y = qy
        origin_pose.orientation.z = qz
        origin_pose.orientation.w = qw
        info.origin = origin_pose
        msg.info = info
        msg.data = [int(val) for val in occ.reshape(-1)]
        return msg

    def _build_limit_marker_array(self):
        points_px = getattr(self, "obstacle_points", None)
        if not points_px:
            return None

        stamp = None
        if self.ros_node is not None:
            stamp = self.ros_node.get_clock().now().to_msg()

        marker_array = MarkerArray()

        limit_marker = self._create_limit_marker(points_px, stamp=stamp)
        if limit_marker is None:
            return None
        marker_array.markers.append(limit_marker)

        path_marker = self._build_path_obstacle_marker(
            stamp=stamp, marker_id=len(marker_array.markers), points_px=points_px
        )
        if path_marker is not None:
            marker_array.markers.append(path_marker)

        goal_markers = self._build_goal_markers(stamp=stamp, start_id=len(marker_array.markers))
        if goal_markers:
            marker_array.markers.extend(goal_markers)

        return marker_array

    def _create_limit_marker(self, points_px, stamp=None):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        if stamp is not None:
            marker.header.stamp = stamp
        marker.ns = "limits"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0

        scale_xy = self.px2m(self.grid_spacing or 12) * 0.8 if self.grid_spacing else 0.2
        marker.scale.x = float(max(scale_xy, 0.02))
        marker.scale.y = marker.scale.x
        marker.scale.z = 0.02
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.9

        max_points = getattr(self, "limit_max_points", 2000)
        step = max(1, len(points_px) // max_points) if max_points else 1
        for ox, oy in points_px[::step]:
            wx, wy = self._px_to_world(ox, oy)
            pt = Point()
            pt.x = float(wx)
            pt.y = float(wy)
            pt.z = 0.0
            marker.points.append(pt)

        if not marker.points:
            return None
        return marker

    def _build_path_obstacle_marker(self, stamp=None, marker_id=1, points_px=None):
        points_px = points_px if points_px is not None else getattr(self, "obstacle_points", None)
        if not points_px:
            return None

        path_pts = self._choose_best_available_path()
        if not path_pts:
            return None

        step = max(1, len(path_pts) // 400)
        sampled_path = path_pts[::step]
        thresh = getattr(self, "path_obstacle_radius_px", int(max(self.grid_spacing or 30, 30)))
        reduced = []
        for ox, oy in points_px:
            for px, py, *_ in sampled_path:
                if math.hypot(px - ox, py - oy) <= thresh:
                    reduced.append((ox, oy))
                    break

        if not reduced:
            return None

        marker = Marker()
        marker.header.frame_id = self.frame_id
        if stamp is not None:
            marker.header.stamp = stamp
        marker.ns = "path_obstacles"
        marker.id = int(marker_id)
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0

        scale_xy = self.px2m(self.grid_spacing or 12) * 0.6 if self.grid_spacing else 0.2
        marker.scale.x = float(max(scale_xy, 0.02))
        marker.scale.y = marker.scale.x
        marker.scale.z = 0.02
        marker.color.r = 1.0
        marker.color.g = 0.55
        marker.color.b = 0.0
        marker.color.a = 0.9

        max_points = getattr(self, "limit_max_points", 2000)
        step = max(1, len(reduced) // max_points) if max_points else 1
        for ox, oy in reduced[::step]:
            wx, wy = self._px_to_world(ox, oy)
            pt = Point()
            pt.x = float(wx)
            pt.y = float(wy)
            pt.z = 0.0
            marker.points.append(pt)

        if not marker.points:
            return None
        return marker

    def _build_goal_markers(self, stamp=None, start_id=0):
        goals = getattr(self, "goals", None)
        if not goals:
            return []

        markers = []
        for idx, (gx, gy, radius) in enumerate(goals):
            marker = Marker()
            marker.header.frame_id = self.frame_id
            if stamp is not None:
                marker.header.stamp = stamp
            marker.ns = "objectives"
            marker.id = int(start_id + idx)
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            wx, wy = self._px_to_world(gx, gy)
            marker.pose.position.x = float(wx)
            marker.pose.position.y = float(wy)
            marker.pose.position.z = 0.0
            radius_m = max(float(self.px2m(radius)), 0.05)
            marker.scale.x = float(max(radius_m * 2.0, 0.05))
            marker.scale.y = marker.scale.x
            marker.scale.z = float(max(radius_m * 0.4, 0.05))
            marker.color.r = 0.1
            marker.color.g = 0.8
            marker.color.b = 0.2
            marker.color.a = 0.7
            markers.append(marker)
        return markers

    def _choose_best_available_path(self):
        if self.manual_path and len(self.manual_path) >= 2:
            return self.manual_path
        state = self.run_state or {}
        if state.get("full_path"):
            return state["full_path"]
        for key in ("path_mpc", "path_dwa", "path_hybrid"):
            if state.get(key):
                return state[key]
        return None

    def publish_reference_path(self):
        path_pts = self._choose_best_available_path()
        if not path_pts:
            messagebox.showwarning("Publicar Path", "Não há caminho para publicar ainda.")
            return
        self.frame_id = self.frame_id_var.get().strip() or "map"
        if bool(self.anchor_origin_start.get()):
            x0, y0, _, _ = path_pts[0]
            self._set_origin_from_pixel(x0, y0, label="path start")
        else:
            self.origin_x_m = float(self.offx_var.get())
            self.origin_y_m = float(self.offy_var.get())

        if not self._ensure_ros2():
            return
        try:
            msg = self._build_nav_path(path_pts)
            if msg is None:
                messagebox.showwarning("Publicar Path", "Caminho inválido/curto.")
                return
            x0, y0, _, _ = path_pts[0]
            wx0, wy0 = self._px_to_world(x0, y0)
            self.log_msg(
                f"Debug origem: primeiro ponto GUI ({x0:.1f},{y0:.1f}) → mundo ({wx0:.3f},{wy0:.3f}) m"
            )
            self.ros_pub.publish(msg)
            self.log_msg(
                f"ROS2: publicado {len(msg.poses)} poses em '{self.ref_path_topic}' (frame_id='{msg.header.frame_id}')."
            )
            self.update_status(f"Publicado em {self.ref_path_topic}")
        except Exception as exc:
            messagebox.showerror("Publicar Path", f"Erro ao publicar: {exc}")

    def publish_occupancy_grid(self):
        if self.image_array is None:
            messagebox.showwarning("Publicar mapa", "Carrega uma imagem primeiro.")
            return
        if getattr(self, "start_point", None) is None:
            messagebox.showerror("Publicar mapa", "Defina a origem/orienta��o do ve�culo antes de publicar o mapa.")
            return
        self.frame_id = self.frame_id_var.get().strip() or "map"
        if bool(self.anchor_origin_start.get()):
            sx, sy = self.start_point
            self._set_origin_from_pixel(sx, sy, label="start")
        else:
            self.origin_x_m = float(self.offx_var.get())
            self.origin_y_m = float(self.offy_var.get())

        if not self._ensure_ros2():
            return
        try:
            msg = self._build_occupancy_grid()
            if msg is None:
                messagebox.showwarning("Publicar mapa", "Falha ao gerar grid de ocupação.")
                return
            if self.ros_map_pub is None:
                self.ros_map_pub = self.ros_node.create_publisher(OccupancyGrid, self.map_topic, 10)
            self.ros_map_pub.publish(msg)
            self.log_msg(
                f"ROS2: mapa {msg.info.width}x{msg.info.height} publicado em '{self.map_topic}' (res={msg.info.resolution:.3f} m)."
            )
            self.update_status(f"Mapa publicado em {self.map_topic}")
        except Exception as exc:
            messagebox.showerror("Publicar mapa", f"Erro ao publicar: {exc}")

    def publish_limits(self):
        if not getattr(self, "obstacle_points", None):
            messagebox.showwarning("Publicar limites", "Nenhum limite definido.")
            return
        if getattr(self, "start_point", None) is None:
            messagebox.showerror("Publicar limites", "Defina a origem/orienta��o do ve�culo antes de publicar limites.")
            return

        self.frame_id = self.frame_id_var.get().strip() or "map"
        if bool(self.anchor_origin_start.get()):
            sx, sy = self.start_point
            self._set_origin_from_pixel(sx, sy, label="start")
        else:
            self.origin_x_m = float(self.offx_var.get())
            self.origin_y_m = float(self.offy_var.get())

        if not self._ensure_ros2():
            return

        try:
            if self.ros_limit_pub is None:
                self.ros_limit_pub = self.ros_node.create_publisher(MarkerArray, self.limit_topic, 10)
            marker_array = self._build_limit_marker_array()
            if marker_array is None:
                messagebox.showwarning("Publicar limites", "Falha ao gerar marcacao dos limites.")
                return
            self.ros_limit_pub.publish(marker_array)
            total_points = sum(len(marker.points) for marker in marker_array.markers)
            self.log_msg(
                f"ROS2: limites ({total_points} pts) publicados em '{self.limit_topic}'."
            )
            self.update_status(f"Limites publicados em {self.limit_topic}")
        except Exception as exc:
            messagebox.showerror("Publicar limites", f"Erro ao publicar: {exc}")

    def publish_obstacles_marker(self):
        points = getattr(self, "obstacle_points", None)
        if not points:
            messagebox.showwarning("Publicar obstaculos", "Nenhum obstaculo definido.")
            return
        if getattr(self, "start_point", None) is None:
            messagebox.showerror("Publicar obstaculos", "Defina a origem/orientacao do veiculo antes de publicar.")
            return

        self.frame_id = self.frame_id_var.get().strip() or "map"
        if bool(self.anchor_origin_start.get()):
            sx, sy = self.start_point
            self._set_origin_from_pixel(sx, sy, label="start")
        else:
            self.origin_x_m = float(self.offx_var.get())
            self.origin_y_m = float(self.offy_var.get())

        if not self._ensure_ros2():
            return

        try:
            if self.ros_limit_pub is None:
                self.ros_limit_pub = self.ros_node.create_publisher(MarkerArray, self.limit_topic, 10)
            stamp = self.ros_node.get_clock().now().to_msg() if self.ros_node is not None else None
            marker = self._build_path_obstacle_marker(stamp=stamp, marker_id=0)
            if marker is None or not getattr(marker, "points", None):
                messagebox.showwarning(
                    "Publicar obstaculos", "Sem obstaculos proximos ao caminho para publicar."
                )
                return
            marker_array = MarkerArray()
            marker_array.markers.append(marker)
            self.ros_limit_pub.publish(marker_array)
            total_points = len(marker.points)
            self.log_msg(
                f"ROS2: obstaculos ({total_points} pts) publicados em '{self.limit_topic}'."
            )
            self.update_status(f"Obstaculos publicados em {self.limit_topic}")
        except Exception as exc:
            messagebox.showerror("Publicar obstaculos", f"Erro ao publicar: {exc}")

    def publish_objectives_marker(self):
        goals = getattr(self, "goals", None)
        if not goals:
            messagebox.showwarning("Publicar objetivos", "Nenhum objetivo definido.")
            return
        if getattr(self, "start_point", None) is None:
            messagebox.showerror("Publicar objetivos", "Defina a origem/orientacao do veiculo antes de publicar.")
            return

        self.frame_id = self.frame_id_var.get().strip() or "map"
        if bool(self.anchor_origin_start.get()):
            sx, sy = self.start_point
            self._set_origin_from_pixel(sx, sy, label="start")
        else:
            self.origin_x_m = float(self.offx_var.get())
            self.origin_y_m = float(self.offy_var.get())

        if not self._ensure_ros2():
            return

        try:
            if self.ros_limit_pub is None:
                self.ros_limit_pub = self.ros_node.create_publisher(MarkerArray, self.limit_topic, 10)
            stamp = self.ros_node.get_clock().now().to_msg() if self.ros_node is not None else None
            goal_markers = self._build_goal_markers(stamp=stamp, start_id=0)
            if not goal_markers:
                messagebox.showwarning("Publicar objetivos", "Falha ao gerar marcadores de objetivos.")
                return
            marker_array = MarkerArray()
            marker_array.markers.extend(goal_markers)
            self.ros_limit_pub.publish(marker_array)
            self.log_msg(
                f"ROS2: objetivos ({len(goal_markers)} marcadores) publicados em '{self.limit_topic}'."
            )
            self.update_status(f"Objetivos publicados em {self.limit_topic}")
        except Exception as exc:
            messagebox.showerror("Publicar objetivos", f"Erro ao publicar: {exc}")
