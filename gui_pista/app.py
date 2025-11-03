"""Main application entry point."""

import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
import numpy as np
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .mixins.drawing import DrawingMixin
from .mixins.environment import EnvironmentMixin
from .mixins.interaction import InteractionMixin
from .mixins.planning import PlanningMixin
from .mixins.rendering import RenderingMixin
from .mixins.ros import HAVE_ROS2, ROSMixin
from .mixins.scale import ScaleMixin
from .theme import PALETTE, apply_theme
from .widgets import ScrollableFrame
matplotlib.use("TkAgg")


class TrajApp(
    ScaleMixin,
    EnvironmentMixin,
    RenderingMixin,
    InteractionMixin,
    PlanningMixin,
    DrawingMixin,
    ROSMixin,
    tk.Frame,
):
    """Main GUI application frame."""

    def __init__(self, master=None, initial_squares=24):
        super().__init__(master)

        self.grid_count = initial_squares
        self.grid_spacing = None
        self.image = None
        self.image_array = None
        self.image_original = None
        self.image_scale_x = 1.0
        self.image_scale_y = 1.0

        self.mode = None
        self.obstacle_points = []
        self.color_tolerance = 30

        self.start_point = None
        self.start_theta = 0.0

        self.scale_mode = tk.StringVar(value="sqm")
        self.squares_per_meter = 5.0
        self.meters_per_square = 0.20
        self.show_meter_grid = tk.BooleanVar(value=True)

        self.wheelbase = 0.324
        self.max_steering_angle = math.radians(30)
        self.max_speed = 2.0
        self.dt = 0.1
        self.min_turning_radius = self.wheelbase / math.tan(self.max_steering_angle)

        self.goals = []
        self.selected_goal = None
        self.goal_radius = 18

        self.goal_avoid_margin_m = 0.6
        self.goal_avoid_weight = 0.6
        self.early_entry_band_m = 0.5

        self.run_state = None
        self.anim_timer = None

        self.mu = 0.9
        self.ax_max = 3.0
        self.brake_max = 5.0
        self.safety_margin_m = 0.30

        self.ros_node = None
        self.ros_pub = None
        self.ros_map_pub = None
        self.ros_limit_pub = None
        self.ros_inited = False
        self.ref_path_topic = "reference_path"
        self.map_topic = "occupancy_grid"
        self.limit_topic = "obstacles_marker"
        self.limit_max_points = 2000
        self.frame_id = "map"
        self.origin_x_m = 0.0
        self.origin_y_m = 0.0
        self.map_occ_threshold = 140

        self.axis_options = ["Direita", "Esquerda", "Cima", "Baixo"]
        self.axis_x_vec = (1.0, 0.0)
        self.axis_y_vec = (0.0, -1.0)
        self.pending_axis_x = "Direita"
        self.pending_axis_y = "Cima"

        self.anchor_origin_start = tk.BooleanVar(value=True)
        self.frame_id_var = tk.StringVar(value=self.frame_id)
        self.offx_var = tk.DoubleVar(value=self.origin_x_m)
        self.offy_var = tk.DoubleVar(value=self.origin_y_m)
        self.add_goal_active = False
        self.remove_goal_active = False
        self.add_obstacle_active = False
        self.remove_obstacle_active = False

        self.draw_mode = False
        self.draw_active_drag = False
        self.draw_points = []
        self.draw_sampling_px = 1.0
        self.draw_smooth_on_use = tk.BooleanVar(value=True)
        self.manual_path = None

        self._raceline_cbar = None
        self._raceline_lc = None

        self.measure_p1 = None
        self.measure_p2 = None
        self.last_measure = None

        self._handling_grid_slider = False
        self._handling_speed_slider = False
        self._handling_steering_slider = False
        self._handling_stretch_x = False
        self._handling_stretch_y = False
        self._redraw_pending = False
        self._pending_redraw_args = None
        self._redraw_after_id = None
        self._last_redraw_ts = 0.0
        self._redraw_min_interval_ms = 16.0

        self.create_ui()
        self.pack(fill=tk.BOTH, expand=True)

    # UI setup ---------------------------------------------------------




    def create_ui(self):
        self.speed_var = tk.DoubleVar(value=self.max_speed)
        self.steering_var = tk.DoubleVar(value=math.degrees(self.max_steering_angle))

        container = tk.Frame(self, bg=PALETTE["bg"])
        container.pack(fill=tk.BOTH, expand=True)

        pane = ttk.Panedwindow(container, orient=tk.HORIZONTAL, style="Transparent.TPanedwindow")
        pane.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(pane, padding=(6, 8, 4, 8))
        right = ttk.Frame(pane, padding=(0, 8, 8, 8))
        pane.add(left, weight=0)
        pane.add(right, weight=1)

        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        tabs = ttk.Notebook(left)
        tabs.grid(row=0, column=0, sticky="nsew")

        self.status_label = ttk.Label(left, text="Status: Pronto", style="Status.TLabel", anchor="w")
        self.status_label.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        map_scroll = ScrollableFrame(tabs)
        vehicle_scroll = ScrollableFrame(tabs)
        goals_scroll = ScrollableFrame(tabs)
        run_scroll = ScrollableFrame(tabs)
        draw_scroll = ScrollableFrame(tabs)

        tab_map = map_scroll.body
        tab_vehicle = vehicle_scroll.body
        tab_goals = goals_scroll.body
        tab_run = run_scroll.body
        tab_draw = draw_scroll.body

        for tab in (tab_map, tab_vehicle, tab_goals, tab_run, tab_draw):
            tab.configure(padding=(10, 12))

        tabs.add(map_scroll, text="Mapa & Grid")
        tabs.add(vehicle_scroll, text="Veiculo & Escala")
        tabs.add(goals_scroll, text="Objetivos")
        tabs.add(run_scroll, text="Execucao & Logs")
        tabs.add(draw_scroll, text="Desenhar")

        def make_label(parent, text, **kwargs):
            kwargs.setdefault("foreground", PALETTE["fg_primary"])
            kwargs.setdefault("justify", "left")
            kwargs.setdefault("wraplength", 220)
            return ttk.Label(parent, text=text, **kwargs)

        # --- tab_map ---
        tab_map.columnconfigure(1, weight=1)
        row = 0
        ttk.Button(tab_map, text="Carregar Imagem", command=self.load_image, style="Accent.TButton")            .grid(row=row, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 8))
        row += 1
        make_label(tab_map, "Resolucao da grid (linhas):").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        self.grid_slider_var = tk.DoubleVar(value=self.grid_count)
        self.grid_slider = ttk.Scale(
            tab_map,
            from_=15,
            to=120,
            orient="horizontal",
            variable=self.grid_slider_var,
            command=self._on_grid_slider,
            style="Modern.Horizontal.TScale",
        )
        self.grid_slider.grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        self.grid_value_label = ttk.Label(tab_map, text=f"{self.grid_count} linhas", foreground=PALETTE["fg_muted"])
        self.grid_value_label.grid(row=row, column=2, sticky="w", padx=6, pady=4)
        row += 1

        self.stretch_x_var = tk.DoubleVar(value=1.0)
        self.stretch_y_var = tk.DoubleVar(value=1.0)

        axes_box = ttk.LabelFrame(tab_map, text="Orientacao dos eixos (aplica ao validar)")
        axes_box.grid(row=row, column=0, columnspan=3, sticky="ew", padx=6, pady=(6, 8))
        axes_box.columnconfigure(1, weight=1)
        axes_box.columnconfigure(3, weight=1)
        make_label(axes_box, "X+ ->", wraplength=120).grid(row=0, column=0, padx=(8, 4), pady=6, sticky="w")
        self.axis_x_var = tk.StringVar(value=self.pending_axis_x)
        ttk.Combobox(axes_box, values=self.axis_options, textvariable=self.axis_x_var, state="readonly", width=12)            .grid(row=0, column=1, padx=4, pady=6, sticky="ew")
        make_label(axes_box, "Y+ ->", wraplength=120).grid(row=0, column=2, padx=(12, 4), pady=6, sticky="w")
        self.axis_y_var = tk.StringVar(value=self.pending_axis_y)
        ttk.Combobox(axes_box, values=self.axis_options, textvariable=self.axis_y_var, state="readonly", width=12)            .grid(row=0, column=3, padx=4, pady=6, sticky="ew")
        ttk.Button(axes_box, text="Validar", style="Secondary.TButton", command=self.validate_axes)            .grid(row=0, column=4, padx=(12, 8), pady=6, sticky="w")
        row += 1

        make_label(tab_map, "Limites e obstaculos:").grid(row=row, column=0, sticky="w", padx=6, pady=(4, 2))
        row += 1
        ttk.Button(tab_map, text="Adicionar limite (flood)", command=self.add_limit_mode, style="Secondary.TButton")            .grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Button(tab_map, text="Remover limite", command=self.remove_limit_mode, style="Secondary.TButton")            .grid(row=row, column=1, sticky="w", padx=6, pady=4)
        row += 1
        ttk.Button(tab_map, text="Limpar limites", command=self.clear_all_limits, style="Secondary.TButton")            .grid(row=row, column=0, sticky="w", padx=6, pady=4)
        row += 1
        ttk.Button(tab_map, text="Medir 2 pontos", command=self.start_measure, style="Secondary.TButton")            .grid(row=row, column=0, sticky="w", padx=6, pady=(4, 10))
        row += 1

        origin_box = ttk.LabelFrame(tab_map, text="Origem e orientacao do veiculo")
        origin_box.grid(row=row, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 10))
        origin_box.columnconfigure(0, weight=1)
        origin_box.columnconfigure(1, weight=1)
        ttk.Button(
            origin_box,
            text="Definir origem + orientacao",
            command=self.set_start_point,
            style="Accent.TButton",
        ).grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Button(
            origin_box,
            text="Recolocar origem",
            command=self.relocate_start,
            style="Secondary.TButton",
        ).grid(row=0, column=1, sticky="w", padx=6, pady=6)
        row += 1

        stretch_box = ttk.LabelFrame(tab_map, text="Deformar imagem (escala)")
        stretch_box.grid(row=row, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 10))
        stretch_box.columnconfigure(1, weight=1)
        stretch_box.columnconfigure(4, weight=1)
        make_label(stretch_box, "Eixo X:", wraplength=120).grid(row=0, column=0, sticky="w", padx=(8, 4), pady=6)
        ttk.Scale(
            stretch_box,
            from_=0.5,
            to=2.0,
            orient="horizontal",
            variable=self.stretch_x_var,
            command=self._on_stretch_x,
            style="Modern.Horizontal.TScale",
        ).grid(row=0, column=1, columnspan=3, sticky="ew", padx=4, pady=6)
        self.stretch_x_readout = ttk.Label(stretch_box, text="100%", foreground=PALETTE["fg_muted"])
        self.stretch_x_readout.grid(row=0, column=4, sticky="e", padx=(4, 8), pady=6)

        make_label(stretch_box, "Eixo Y:", wraplength=120).grid(row=1, column=0, sticky="w", padx=(8, 4), pady=6)
        ttk.Scale(
            stretch_box,
            from_=0.5,
            to=2.0,
            orient="horizontal",
            variable=self.stretch_y_var,
            command=self._on_stretch_y,
            style="Modern.Horizontal.TScale",
        ).grid(row=1, column=1, columnspan=3, sticky="ew", padx=4, pady=6)
        self.stretch_y_readout = ttk.Label(stretch_box, text="100%", foreground=PALETTE["fg_muted"])
        self.stretch_y_readout.grid(row=1, column=4, sticky="e", padx=(4, 8), pady=6)
        self._update_stretch_labels()
        row += 1

        scale = ttk.LabelFrame(tab_map, text="Escala de comprimento (baseada na grid)")
        scale.grid(row=row, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 6))
        scale.columnconfigure(1, weight=1)
        scale.columnconfigure(3, weight=1)
        ttk.Radiobutton(scale, text="1 m = N quadriculas", value="sqm", variable=self.scale_mode,
                        command=self.update_scale).grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=(6, 2))
        ttk.Radiobutton(scale, text="1 quadricula = S metros", value="mps", variable=self.scale_mode,
                        command=self.update_scale).grid(row=0, column=2, columnspan=2, sticky="w", padx=6, pady=(6, 2))
        make_label(scale, "N quadriculas por 1 m:").grid(row=1, column=0, sticky="w", padx=(8, 4), pady=4)
        self.sqm_var = tk.DoubleVar(value=self.squares_per_meter)
        ttk.Spinbox(scale, from_=0.5, to=200, increment=0.5, width=8, textvariable=self.sqm_var,
                    command=lambda: self._set_sqm(self.sqm_var.get()))            .grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        make_label(scale, "S metros por 1 quadricula:").grid(row=2, column=0, sticky="w", padx=(8, 4), pady=4)
        self.mps_var = tk.DoubleVar(value=self.meters_per_square)
        ttk.Spinbox(scale, from_=0.01, to=50.0, increment=0.01, width=8, textvariable=self.mps_var,
                    command=lambda: self._set_mps(self.mps_var.get()))            .grid(row=2, column=1, sticky="ew", padx=4, pady=4)
        ttk.Checkbutton(scale, text="Mostrar grelha de 1 m", variable=self.show_meter_grid,
                        command=self.redraw).grid(row=3, column=0, columnspan=2, sticky="w", padx=6, pady=(6, 6))
        self.lbl_scale_info = ttk.Label(scale, text="", foreground=PALETTE["fg_primary"])
        self.lbl_scale_info.grid(row=3, column=2, columnspan=2, sticky="e", padx=6, pady=(6, 6))

        row += 1
        ros_map_frame = ttk.LabelFrame(tab_map, text="Publicar mapa e marcadores (ROS2)")
        ros_map_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=6, pady=(6, 10))
        ros_map_frame.columnconfigure(1, weight=1)
        ros_map_frame.columnconfigure(3, weight=1)
        make_label(ros_map_frame, "frame_id:", wraplength=100).grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(ros_map_frame, textvariable=self.frame_id_var, width=12).grid(row=0, column=1, sticky="w", padx=4, pady=4)
        make_label(ros_map_frame, "offset X (m):", wraplength=100).grid(row=0, column=2, sticky="w", padx=(12, 6), pady=4)
        ttk.Spinbox(
            ros_map_frame,
            from_=-1000,
            to=1000,
            increment=0.1,
            width=9,
            textvariable=self.offx_var,
            command=lambda: setattr(self, "origin_x_m", float(self.offx_var.get())),
        ).grid(row=0, column=3, sticky="w", padx=4, pady=4)
        make_label(ros_map_frame, "offset Y (m):", wraplength=100).grid(row=0, column=4, sticky="w", padx=(12, 6), pady=4)
        ttk.Spinbox(
            ros_map_frame,
            from_=-1000,
            to=1000,
            increment=0.1,
            width=9,
            textvariable=self.offy_var,
            command=lambda: setattr(self, "origin_y_m", float(self.offy_var.get())),
        ).grid(row=0, column=5, sticky="w", padx=4, pady=4)
        ttk.Checkbutton(
            ros_map_frame,
            text="Ancorar origem no start (0,0)",
            variable=self.anchor_origin_start,
        ).grid(row=1, column=0, columnspan=6, sticky="w", padx=6, pady=(2, 8))
        ttk.Button(
            ros_map_frame,
            text="Publicar mapa (ROS2)",
            command=self.publish_occupancy_grid,
            style="Accent.TButton",
        ).grid(row=2, column=0, columnspan=6, sticky="ew", padx=6, pady=(0, 4))
        ttk.Button(
            ros_map_frame,
            text="Publicar limites (ROS2)",
            command=self.publish_limits,
            style="Secondary.TButton",
        ).grid(row=3, column=0, columnspan=6, sticky="ew", padx=6, pady=(0, 8))

        # --- tab_vehicle ---
        tab_vehicle.columnconfigure(0, weight=1)
        vrow = 0
        params = ttk.LabelFrame(tab_vehicle, text="Modelo bicicleta")
        params.grid(row=vrow, column=0, sticky="ew", padx=6, pady=(0, 10))
        params.columnconfigure(2, weight=1)
        self.lbl_wb = make_label(params, f"Wheelbase: {self.wheelbase*1000:.0f} mm", wraplength=200)
        self.lbl_wb.grid(row=0, column=0, sticky="w", padx=6, pady=(6, 2))
        self.lbl_sa = make_label(params, f"Direcao max: {math.degrees(self.max_steering_angle):.0f} graus", wraplength=200)
        self.lbl_sa.grid(row=1, column=0, sticky="w", padx=6, pady=2)
        self.lbl_rt = make_label(params, f"Raio minimo: {self.min_turning_radius:.2f} m", wraplength=200)
        self.lbl_rt.grid(row=2, column=0, sticky="w", padx=6, pady=(2, 6))
        make_label(params, "Velocidade max (m/s):", wraplength=160).grid(row=0, column=1, sticky="w", padx=6, pady=(6, 2))
        self.speed_slider = ttk.Scale(params, from_=0.5, to=6.0, variable=self.speed_var,
                                      command=self._on_speed_change, style="Modern.Horizontal.TScale")
        self.speed_slider.grid(row=0, column=2, sticky="ew", padx=6, pady=(6, 2))
        self.speed_readout = ttk.Label(params, text=f"{self.max_speed:.1f} m/s", foreground=PALETTE["fg_primary"])
        self.speed_readout.grid(row=0, column=3, sticky="w", padx=(0, 6), pady=(6, 2))
        make_label(params, "Angulo max direcao (graus):", wraplength=160).grid(row=1, column=1, sticky="w", padx=6, pady=2)
        self.steering_slider = ttk.Scale(params, from_=10, to=45, variable=self.steering_var,
                                         command=self._on_steering_change, style="Modern.Horizontal.TScale")
        self.steering_slider.grid(row=1, column=2, sticky="ew", padx=6, pady=2)
        self.steering_readout = ttk.Label(params, text=f"{math.degrees(self.max_steering_angle):.0f} graus",
                                          foreground=PALETTE["fg_primary"])
        self.steering_readout.grid(row=1, column=3, sticky="w", padx=(0, 6), pady=2)

        # --- tab_goals ---
        tab_goals.columnconfigure(0, weight=1)
        goal_controls = ttk.Frame(tab_goals)
        goal_controls.grid(row=0, column=0, sticky="ew", padx=6, pady=(0, 10))
        goal_controls.columnconfigure(0, weight=1)
        goal_controls.columnconfigure(1, weight=1)

        self.goal_toggle_btn = ttk.Button(
            goal_controls,
            text="Adicionar objetivo (OFF)",
            command=self._toggle_add_goal,
            style="Accent.TButton",
        )
        self.goal_toggle_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.goal_delete_btn = ttk.Button(
            goal_controls,
            text="Remover objetivo (OFF)",
            command=self._toggle_remove_goal,
            style="Secondary.TButton",
        )
        self.goal_delete_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self._set_goal_toggle_state(False)
        self._set_goal_delete_state(False)

        obstacle_controls = ttk.Frame(tab_goals)
        obstacle_controls.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 10))
        obstacle_controls.columnconfigure(0, weight=1)
        obstacle_controls.columnconfigure(1, weight=1)

        self.obstacle_toggle_btn = ttk.Button(
            obstacle_controls,
            text="Adicionar obstaculo (OFF)",
            command=self._toggle_add_obstacle,
            style="Warning.TButton",
        )
        self.obstacle_toggle_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.obstacle_delete_btn = ttk.Button(
            obstacle_controls,
            text="Remover obstaculo (OFF)",
            command=self._toggle_remove_obstacle,
            style="Secondary.TButton",
        )
        self.obstacle_delete_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self._set_obstacle_toggle_state(False)
        self._set_obstacle_remove_state(False)

        avoid_frame = ttk.LabelFrame(tab_goals, text="Evitar objetivos nao ativos")
        avoid_frame.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 10))
        make_label(avoid_frame, "Margem (m):", wraplength=160).grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.avoid_margin_var = tk.DoubleVar(value=self.goal_avoid_margin_m)
        ttk.Spinbox(avoid_frame, from_=0.0, to=3.0, increment=0.1, textvariable=self.avoid_margin_var, width=7,
                    command=lambda: setattr(self, "goal_avoid_margin_m", float(self.avoid_margin_var.get())))            .grid(row=0, column=1, sticky="w", padx=4, pady=4)
        make_label(avoid_frame, "Peso:", wraplength=100).grid(row=0, column=2, sticky="w", padx=(12, 6), pady=4)
        self.avoid_weight_var = tk.DoubleVar(value=self.goal_avoid_weight)
        ttk.Spinbox(avoid_frame, from_=0.0, to=3.0, increment=0.1, textvariable=self.avoid_weight_var, width=7,
                    command=lambda: setattr(self, "goal_avoid_weight", float(self.avoid_weight_var.get())))            .grid(row=0, column=3, sticky="w", padx=4, pady=4)

        entry_frame = ttk.LabelFrame(tab_goals, text="Entrada no objetivo corrente")
        entry_frame.grid(row=3, column=0, sticky="ew", padx=6, pady=(0, 10))
        make_label(entry_frame, "Banda de chegada (m):", wraplength=180).grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.band_var = tk.DoubleVar(value=self.early_entry_band_m)
        ttk.Spinbox(entry_frame, from_=0.0, to=2.0, increment=0.1, textvariable=self.band_var, width=7,
                    command=lambda: setattr(self, "early_entry_band_m", float(self.band_var.get())))            .grid(row=0, column=1, sticky="w", padx=4, pady=4)

        ros_goals_frame = ttk.LabelFrame(tab_goals, text="Publicar ROS2")
        ros_goals_frame.grid(row=4, column=0, sticky="ew", padx=6, pady=(0, 10))
        ros_goals_frame.columnconfigure(0, weight=1)
        ros_goals_frame.columnconfigure(1, weight=1)
        ttk.Button(
            ros_goals_frame,
            text="Publicar objetivos (ROS2)",
            command=self.publish_objectives_marker,
            style="Secondary.TButton",
        ).grid(row=0, column=0, sticky="ew", padx=(6, 4), pady=6)
        ttk.Button(
            ros_goals_frame,
            text="Publicar obstaculos (ROS2)",
            command=self.publish_obstacles_marker,
            style="Secondary.TButton",
        ).grid(row=0, column=1, sticky="ew", padx=(4, 6), pady=6)

        # --- tab_run ---
        tab_run.columnconfigure(0, weight=1)
        tab_run.rowconfigure(2, weight=1)
        run_bar = ttk.Frame(tab_run)
        run_bar.grid(row=0, column=0, sticky="ew", padx=6, pady=(0, 8))
        ttk.Button(run_bar, text="Executar (Hybrid A* + DWA + MPC)", command=self.run_all, style="Accent.TButton")            .grid(row=0, column=0, sticky="w", padx=(0, 4))
        ttk.Button(run_bar, text="Parar", command=self.stop_all, style="Secondary.TButton")            .grid(row=0, column=1, sticky="w", padx=(4, 0))
        ttk.Button(run_bar, text="Race line (rápida)", command=self.build_fast_raceline_current,
                   style="Secondary.TButton").grid(row=0, column=2, sticky="w", padx=(4, 0))

        self.progress = ttk.Progressbar(tab_run, mode="determinate", maximum=100,
                                         style="Modern.Horizontal.TProgressbar")
        self.progress.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 10))

        log_frame = ttk.Frame(tab_run)
        log_frame.grid(row=2, column=0, sticky="nsew", padx=6, pady=0)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(
            log_frame,
            height=12,
            wrap="none",
            bg=PALETTE["bg_alt"],
            fg=PALETTE["fg_primary"],
            insertbackground=PALETTE["accent"],
            highlightthickness=0,
            bd=0,
            relief=tk.FLAT,
        )
        self.log.grid(row=0, column=0, sticky="nsew")
        ybar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        ybar.grid(row=0, column=1, sticky="ns")
        xbar = ttk.Scrollbar(log_frame, orient="horizontal", command=self.log.xview)
        xbar.grid(row=1, column=0, sticky="ew")
        self.log.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)

        export_frame = ttk.LabelFrame(tab_run, text="Publicar caminho (ROS2)")
        export_frame.grid(row=3, column=0, sticky="ew", padx=6, pady=(10, 0))
        export_frame.columnconfigure(1, weight=1)
        make_label(export_frame, "frame_id:", wraplength=100).grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(export_frame, textvariable=self.frame_id_var, width=12).grid(row=0, column=1, sticky="w", padx=4, pady=4)
        make_label(export_frame, "offset X (m):", wraplength=100).grid(row=0, column=2, sticky="w", padx=(12, 6), pady=4)
        ttk.Spinbox(export_frame, from_=-1000, to=1000, increment=0.1, width=9,
                    textvariable=self.offx_var, command=lambda: setattr(self, "origin_x_m", float(self.offx_var.get())))            .grid(row=0, column=3, sticky="w", padx=4, pady=4)
        make_label(export_frame, "offset Y (m):", wraplength=100).grid(row=0, column=4, sticky="w", padx=(12, 6), pady=4)
        ttk.Spinbox(export_frame, from_=-1000, to=1000, increment=0.1, width=9,
                    textvariable=self.offy_var, command=lambda: setattr(self, "origin_y_m", float(self.offy_var.get())))            .grid(row=0, column=5, sticky="w", padx=4, pady=4)
        ttk.Checkbutton(export_frame, text="Ancorar origem no start (0,0)", variable=self.anchor_origin_start)            .grid(row=1, column=0, columnspan=6, sticky="w", padx=6, pady=(2, 8))
        ttk.Button(export_frame, text="Publicar Path (ROS2)", command=self.publish_reference_path,
                   style="Accent.TButton").grid(row=2, column=0, columnspan=6, sticky="ew", padx=6, pady=(0, 8))

        # --- tab_draw ---
        tab_draw.columnconfigure(0, weight=1)
        tab_draw.columnconfigure(1, weight=1)
        tab_draw.columnconfigure(2, weight=1)
        drow = 0
        make_label(tab_draw, "Desenha no canvas a direita mantendo o botao esquerdo premido e a arrastar.",
                   wraplength=260).grid(row=drow, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 10))
        drow += 1
        ttk.Button(tab_draw, text="Iniciar desenho", command=self.start_draw_mode, style="Accent.TButton")            .grid(row=drow, column=0, sticky="w", padx=6, pady=4)
        ttk.Button(tab_draw, text="Parar", command=self.stop_draw_mode, style="Secondary.TButton")            .grid(row=drow, column=1, sticky="w", padx=6, pady=4)
        ttk.Button(tab_draw, text="Limpar", command=self.clear_draw, style="Secondary.TButton")            .grid(row=drow, column=2, sticky="w", padx=6, pady=4)
        drow += 1
        make_label(tab_draw, "Amostragem (px entre pontos):", wraplength=200).grid(row=drow, column=0, sticky="w", padx=6, pady=6)
        self.draw_sample_var = tk.DoubleVar(value=self.draw_sampling_px)
        ttk.Spinbox(tab_draw, from_=0.5, to=10.0, increment=0.5, width=7, textvariable=self.draw_sample_var,
                    command=lambda: setattr(self, "draw_sampling_px", float(self.draw_sample_var.get())))            .grid(row=drow, column=1, sticky="w", padx=4, pady=6)
        ttk.Checkbutton(tab_draw, text="Suavizar quando usar (Chaikin leve)", variable=self.draw_smooth_on_use)            .grid(row=drow, column=2, sticky="w", padx=6, pady=6)
        drow += 1
        ttk.Button(tab_draw, text="Usar desenho como caminho", command=self.use_draw_as_path,
                   style="Accent.TButton").grid(row=drow, column=0, columnspan=3, sticky="ew", padx=6, pady=(10, 6))
        drow += 1
        ttk.Button(tab_draw, text="Race Line (clássico)", command=self.trace_raceline_from_draw,
                   style="Accent.TButton").grid(row=drow, column=0, sticky="ew", padx=6, pady=(0, 6))
        ttk.Button(tab_draw, text="Race Line (rápida)", command=self.trace_fast_raceline_from_draw,
                   style="Accent.TButton").grid(row=drow, column=1, sticky="ew", padx=6, pady=(0, 6))
        ttk.Button(tab_draw, text="Publicar desenho (ROS2)", command=self.publish_drawn_path,
                   style="Secondary.TButton").grid(row=drow, column=2, sticky="ew", padx=6, pady=(0, 6))
        # --- right canvas ---
        self.fig = Figure(figsize=(10, 8), facecolor=PALETTE["bg"])
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(PALETTE["bg_alt"])
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.configure(bg=PALETTE["bg"], highlightthickness=0, bd=0)
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        # events and initial state
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)

        self._on_grid_slider(self.grid_slider_var.get())
        self._on_speed_change(self.speed_var.get())
        self._on_steering_change(self.steering_var.get())
        self.update_scale()
    def _on_grid_slider(self, value):
        if self._handling_grid_slider:
            return
        val = int(round(float(value)))
        self._handling_grid_slider = True
        try:
            self.grid_slider_var.set(val)
            self.grid_value_label.config(text=f"{val} linhas")
            self.update_grid(val)
        finally:
            self._handling_grid_slider = False

    def _on_speed_change(self, value):
        if self._handling_speed_slider:
            return
        val = round(float(value), 1)
        val = min(max(val, 0.5), 6.0)
        self._handling_speed_slider = True
        try:
            self.speed_var.set(val)
            self.speed_readout.config(text=f"{val:.1f} m/s")
            self.update_max_speed(val)
        finally:
            self._handling_speed_slider = False

    def _on_steering_change(self, value):
        if self._handling_steering_slider:
            return
        val = round(float(value))
        val = min(max(val, 10), 45)
        self._handling_steering_slider = True
        try:
            self.steering_var.set(val)
            self.steering_readout.config(text=f"{val:.0f}°")
            self.update_max_steering(val)
        finally:
            self._handling_steering_slider = False

    def _toggle_add_obstacle(self):
        new_state = not bool(getattr(self, "add_obstacle_active", False))
        self._set_obstacle_toggle_state(new_state)
        if new_state:
            self._set_obstacle_remove_state(False)
            self.add_obstacle_mode()
            if self.mode != "add_obstacle":
                self._set_obstacle_toggle_state(False)
        else:
            if self.mode == "add_obstacle":
                self.mode = None
                self.update_status("Modo de obstaculo desligado.")
            self.redraw()

    def _toggle_remove_obstacle(self):
        new_state = not bool(getattr(self, "remove_obstacle_active", False))
        self._set_obstacle_remove_state(new_state)
        if new_state:
            self._set_obstacle_toggle_state(False)
            self.remove_obstacle_mode()
            if self.mode != "remove_obstacle":
                self._set_obstacle_remove_state(False)
        else:
            if self.mode == "remove_obstacle":
                self.mode = None
                self.update_status("Modo remover obstaculo desligado.")
            self.redraw()

    def _toggle_add_goal(self):
        new_state = not bool(getattr(self, "add_goal_active", False))
        self._set_goal_toggle_state(new_state)
        if new_state:
            self._set_goal_delete_state(False)
            self.selected_goal = None
            self.add_goal_mode()
        else:
            if self.mode in {"add_goal", "resize_goal"}:
                self.mode = None
            self.selected_goal = None
            self.update_status("Modo de objetivo desligado.")
            self.redraw()

    def _toggle_remove_goal(self):
        new_state = not bool(getattr(self, "remove_goal_active", False))
        self._set_goal_delete_state(new_state)
        if new_state:
            self._set_goal_toggle_state(False)
            self.selected_goal = None
            self.remove_goal_mode()
        else:
            if self.mode == "remove_goal":
                self.mode = None
                self.update_status("Modo remover objetivo desligado.")

    # General helpers --------------------------------------------------
    def update_status(self, status):
        self.status_label.config(text=f"Status: {status}")
        self.update()

    def log_msg(self, message):
        self.log.config(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def validate_axes(self):
        name_x = self.axis_x_var.get()
        name_y = self.axis_y_var.get()
        vec_map = {"Direita": (1.0, 0.0), "Esquerda": (-1.0, 0.0), "Cima": (0.0, -1.0), "Baixo": (0.0, 1.0)}
        vx, vy = vec_map[name_x]
        ux, uy = vec_map[name_y]
        if abs(vx * ux + vy * uy) != 0:
            messagebox.showerror(
                "Orientação inválida",
                f"X+ = {name_x}, Y+ = {name_y} não são ortogonais.",
            )
            return
        self.axis_x_vec = (vx, vy)
        self.axis_y_vec = (ux, uy)
        self.pending_axis_x = name_x
        self.pending_axis_y = name_y
        self.log_msg(f"✅ Eixos aplicados: X+→{name_x}, Y+→{name_y}")

    def _update_stretch_labels(self):
        try:
            if hasattr(self, "stretch_x_readout"):
                self.stretch_x_readout.config(text=f"{self.stretch_x_var.get() * 100:.0f}%")
            if hasattr(self, "stretch_y_readout"):
                self.stretch_y_readout.config(text=f"{self.stretch_y_var.get() * 100:.0f}%")
        except Exception:
            pass

    def _scale_canvas_entities(self, sx, sy):
        if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6:
            return

        if self.start_point is not None:
            self.start_point = (self.start_point[0] * sx, self.start_point[1] * sy)
        if self.measure_p1 is not None:
            self.measure_p1 = (self.measure_p1[0] * sx, self.measure_p1[1] * sy)
        if self.measure_p2 is not None:
            self.measure_p2 = (self.measure_p2[0] * sx, self.measure_p2[1] * sy)
        # Preserve and update the measurement overlay across scaling
        try:
            if self.measure_p1 is not None and self.measure_p2 is not None:
                x1, y1 = self.measure_p1
                x2, y2 = self.measure_p2
                d_px = float(math.hypot(x2 - x1, y2 - y1))
                d_m = float(self.px2m(d_px))
                self.last_measure = ((x1, y1), (x2, y2), d_px, d_m)
        except Exception:
            pass

        if self.obstacle_points:
            self.obstacle_points = [(ox * sx, oy * sy) for (ox, oy) in self.obstacle_points]

        if self.goals:
            radius_scale = (sx + sy) / 2.0
            if radius_scale <= 0:
                radius_scale = 1.0
            self.goals = [
                [gx * sx, gy * sy, max(4.0, radius * radius_scale)]
                for (gx, gy, radius) in self.goals
            ]

        if self.draw_points:
            self.draw_points = [(px * sx, py * sy) for (px, py) in self.draw_points]

        if self.manual_path:
            self.manual_path = [
                (x * sx, y * sy, theta, v) for (x, y, theta, v) in self.manual_path
            ]

        self.goal_radius = max(4, int(round(self.goal_radius * ((sx + sy) / 2.0))))

        if self.run_state:
            for key in ("full_path", "path_hybrid", "path_dwa", "path_mpc"):
                path = self.run_state.get(key)
                if path:
                    self.run_state[key] = [
                        (x * sx, y * sy, theta, v) for (x, y, theta, v) in path
                    ]
            if self.run_state.get("seg_start_pose"):
                x, y, th = self.run_state["seg_start_pose"]
                self.run_state["seg_start_pose"] = (x * sx, y * sy, th)
            if self.run_state.get("entry_pts"):
                self.run_state["entry_pts"] = [
                    (ex * sx, ey * sy, ang) for (ex, ey, ang) in self.run_state["entry_pts"]
                ]
            if self.run_state.get("cur_goal"):
                gx, gy, radius = self.run_state["cur_goal"]
                self.run_state["cur_goal"] = (gx * sx, gy * sy, radius * ((sx + sy) / 2.0))
            if self.run_state.get("ha_start"):
                sx0, sy0, sth = self.run_state["ha_start"]
                self.run_state["ha_start"] = (sx0 * sx, sy0 * sy, sth)
            if self.run_state.get("ha_goal"):
                gx0, gy0 = self.run_state["ha_goal"]
                self.run_state["ha_goal"] = (gx0 * sx, gy0 * sy)
            if self.run_state.get("dwa_path"):
                self.run_state["dwa_path"] = [
                    (x * sx, y * sy, theta, v) for (x, y, theta, v) in self.run_state["dwa_path"]
                ]

        if self._raceline_lc is not None:
            try:
                self._raceline_lc.remove()
            except Exception:
                pass
            self._raceline_lc = None
        if self._raceline_cbar is not None:
            try:
                self._raceline_cbar.remove()
            except Exception:
                pass
            self._raceline_cbar = None

    def _apply_image_scaling(self):
        if self.image_original is None:
            return

        try:
            scale_x = float(self.stretch_x_var.get())
            scale_y = float(self.stretch_y_var.get())
        except Exception:
            return

        base_w, base_h = self.image_original.size
        old_w, old_h = self.image.size if self.image is not None else self.image_original.size
        # Preserve current pixels-per-meter if this call is driven by vertical stretch only
        prev_ppm = None
        try:
            if bool(getattr(self, "_handling_stretch_y", False)) and not bool(getattr(self, "_handling_stretch_x", False)):
                prev_ppm = self.ppm()
        except Exception:
            prev_ppm = None
        new_w = max(1, int(round(base_w * scale_x)))
        new_h = max(1, int(round(base_h * scale_y)))

        if new_w == old_w and new_h == old_h:
            self.image_scale_x = new_w / base_w if base_w else 1.0
            self.image_scale_y = new_h / base_h if base_h else 1.0
            self._update_stretch_labels()
            return

        sx = new_w / old_w if old_w else 1.0
        sy = new_h / old_h if old_h else 1.0

        self.stop_all()
        self._scale_canvas_entities(sx, sy)

        self.image = self.image_original.resize((new_w, new_h), Image.BILINEAR)
        self.image_array = np.array(self.image)
        self.image_scale_x = new_w / base_w if base_w else 1.0
        self.image_scale_y = new_h / base_h if base_h else 1.0

        self.grid_spacing = new_h / self.grid_count if self.grid_count else None
        # If only Y was stretched, keep horizontal distances (ppm) unchanged by
        # compensating the scale parameter (sqm or mps) for the grid_spacing change.
        if prev_ppm is not None and self.grid_spacing:
            try:
                if self.scale_mode.get() == "sqm":
                    new_sqm = float(prev_ppm) / float(self.grid_spacing)
                    self.squares_per_meter = new_sqm
                    if hasattr(self, "sqm_var") and self.sqm_var is not None:
                        self.sqm_var.set(new_sqm)
                else:
                    new_mps = float(self.grid_spacing) / float(prev_ppm)
                    self.meters_per_square = new_mps
                    if hasattr(self, "mps_var") and self.mps_var is not None:
                        self.mps_var.set(new_mps)
            except Exception:
                pass
        self.update_scale()
        self._update_stretch_labels()

    def _on_stretch_x(self, value):
        if self._handling_stretch_x:
            return
        self._handling_stretch_x = True
        try:
            self._update_stretch_labels()
            self._apply_image_scaling()
        finally:
            self._handling_stretch_x = False

    def _on_stretch_y(self, value):
        if self._handling_stretch_y:
            return
        self._handling_stretch_y = True
        try:
            self._update_stretch_labels()
            self._apply_image_scaling()
        finally:
            self._handling_stretch_y = False

    # Image & grid -----------------------------------------------------
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if not path:
            return
        try:
            loaded = Image.open(path).convert("RGB")
            self.image_original = loaded
            self.image_scale_x = 1.0
            self.image_scale_y = 1.0
            self.image = self.image_original.copy()
            self.image_array = np.array(self.image)
            _, height = self.image.size
            self.grid_spacing = height / self.grid_count if self.grid_count else None

            self.stop_all()
            self.obstacle_points.clear()
            self.start_point = None
            self.start_theta = 0.0
            self.goals.clear()
            self.selected_goal = None
            self.draw_points.clear()
            self.manual_path = None
            self.measure_p1 = None
            self.measure_p2 = None
            self.last_measure = None

            if hasattr(self, "stretch_x_var"):
                self._handling_stretch_x = True
                self.stretch_x_var.set(1.0)
                self._handling_stretch_x = False
            if hasattr(self, "stretch_y_var"):
                self._handling_stretch_y = True
                self.stretch_y_var.set(1.0)
                self._handling_stretch_y = False
            self._update_stretch_labels()

            self.update_scale()
            self.update_status("Imagem carregada")
        except Exception as exc:
            messagebox.showerror("Erro", f"Erro ao carregar imagem: {exc}")

    def update_grid(self, value):
        try:
            self.grid_count = int(value)
            if self.image:
                _, height = self.image.size
                self.grid_spacing = height / self.grid_count
                self.update_scale()
        except Exception:
            pass


def main():
    root = tk.Tk()
    root.title("Planner: Hybrid A* + DWA + MPC + Race Line + Desenhar (pincel)")
    root.geometry("1550x950")
    apply_theme(root)
    app = TrajApp(master=root)

    def on_close():
        try:
            app.stop_all()
        except Exception:
            pass
        try:
            if app._raceline_cbar is not None:
                try:
                    app._raceline_cbar.remove()
                except Exception:
                    pass
                app._raceline_cbar = None
            if app._raceline_lc is not None:
                try:
                    app._raceline_lc.remove()
                except Exception:
                    pass
                app._raceline_lc = None
        except Exception:
            pass
        try:
            if getattr(app, "ros_inited", False) and app.ros_node is not None:
                app.ros_node.destroy_node()
                if HAVE_ROS2:
                    import rclpy

                    rclpy.shutdown()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    app.mainloop()


__all__ = ["TrajApp", "main"]
