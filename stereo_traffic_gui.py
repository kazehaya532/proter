#!/usr/bin/env python3
"""
Stereo Traffic Monitor - GUI
============================
Three modes (tabs on the right):

  LIVE      choose left/right sources (camera index or video file), a calibration
            file and a YOLO model, then watch detection + tracking + speed overlays
            in real time. Optionally record CSV + rectified videos + metadata.

  ANALYSIS  open a recording folder (or videos + CSV) and play it back with the
            overlay drawn from tracking_data.csv: play/pause, scrub, step frame by
            frame, change speed, loop, left / right / side-by-side view.

  CALIBRATE run stereo calibration from paired chessboard photos without the
            command line; the result feeds the Live tab automatically.

Run:   python stereo_traffic_gui.py        (or:  python stereo_traffic_monitor.py gui)
Keys:  Space = play/pause    Left/Right = step one frame (analysis)
"""

import base64
import json
import os
import queue
import sys
import threading
import time
import traceback

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # pragma: no cover
    sys.exit("tkinter is not available in this Python install.\n"
             "  Debian/Ubuntu: sudo apt install python3-tk\n"
             "  Windows/macOS: reinstall Python from python.org (include 'tcl/tk').")

import cv2

import stereo_traffic_monitor as stm

VIDEO_TYPES = [("Video files", "*.mp4 *.avi *.mkv *.mov *.m4v *.wmv"), ("All files", "*.*")]


# --------------------------------------------------------------------------- #
# Default file locations (work both as script and as a PyInstaller exe)
# --------------------------------------------------------------------------- #
def app_dir():
    """Folder to look in for bundled files (next to the exe when frozen)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def default_model_path():
    """Existing YOLO weights: cwd first, then next to the app; else the bundled path."""
    for base in (os.getcwd(), app_dir()):
        p = os.path.join(base, "yolov8n.pt")
        if os.path.isfile(p):
            return p
    return os.path.join(app_dir(), "yolov8n.pt")


def default_calib_path():
    """Existing calibration.json: cwd first, then next to the app."""
    for base in (os.getcwd(), app_dir()):
        p = os.path.join(base, "calibration.json")
        if os.path.isfile(p):
            return p
    for base in (os.getcwd(), app_dir()):
        p = os.path.join(base, "Calibration",
                         "manual_calibration_experimental_multidistance.json")
        if os.path.isfile(p):
            return p
    return os.path.join(app_dir(), "calibration.json")


# --------------------------------------------------------------------------- #
# Background worker for live processing
# --------------------------------------------------------------------------- #
class LiveWorker(threading.Thread):
    """Reads both sources, runs the pipeline, optionally records, and hands the
    newest frame to the GUI through a small queue (old frames are dropped)."""

    def __init__(self, cfg, out_q):
        super().__init__(daemon=True)
        self.cfg, self.q = cfg, out_q
        self.stop_event = threading.Event()

    def _put(self, item):
        try:
            self.q.put_nowait(item)
        except queue.Full:
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(item)
            except queue.Full:
                pass

    def run(self):
        cfg = self.cfg
        src = rec = None
        frame_id = 0
        try:
            self._put(("status", "Loading calibration and model (first run may download weights)..."))
            with open(cfg["calibration"]) as f:
                rig = stm.StereoRig(json.load(f))
            pipe = stm.StereoPipeline(rig, cfg["model"], cfg["conf"], max_disp=cfg["max_disp"],
                                      min_z=cfg["min_z"], max_z=cfg["max_z"])
            src = stm.StereoSource(cfg["left"], cfg["right"])
            if cfg["record"]:
                rec = stm.Recorder(cfg["out_dir"], rig, src, cfg["left"], cfg["right"],
                                    {"model": cfg["model"], "conf": cfg["conf"],
                                     "classes": list(stm.VEHICLE_CLASSES.values())},
                                    save_video=cfg["save_video"],
                                    stereo_settings={"max_disparity_px": cfg["max_disp"],
                                                     "min_depth_m": cfg["min_z"],
                                                     "max_depth_m": cfg["max_z"]})
            trails = stm.LiveTrails()
            frame_dt = 1.0 / src.fps
            t_prev = time.perf_counter()
            fps = 0.0
            while not self.stop_event.is_set():
                t0 = time.perf_counter()
                ok, fl, fr, t_ns = src.read()
                if not ok:
                    break
                frame_id += 1
                rl, rr, dets = pipe.process(fl, fr, t_ns, frame_id)
                if rec:
                    rec.write(frame_id, t_ns, rl, rr, dets)
                trails.update(dets, frame_id)

                if cfg["realtime"] and not src.live:  # don't run files faster than real time
                    rest = frame_dt - (time.perf_counter() - t0)
                    if rest > 0:
                        time.sleep(rest)
                now = time.perf_counter()
                inst = 1.0 / max(now - t_prev, 1e-6)
                t_prev = now
                fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
                self._put(("frame", {
                    "left": rl, "right": rr, "dets": dets, "frame_id": frame_id,
                    "t_ns": t_ns, "fps": fps,
                    "capture_sizes": src.capture_sizes,
                    "experimental_fit": rig.depth_fit is not None,
                    "trails_l": trails.trails("left"), "trails_r": trails.trails("right"),
                }))
            msg = f"Finished after {frame_id} frames."
            if rec:
                msg += f" Saved to {rec.dir}"
            self._put(("done", msg))
        except Exception as e:  # report to the GUI instead of dying silently
            traceback.print_exc()
            self._put(("error", f"{type(e).__name__}: {e}"))
        finally:
            if rec:
                rec.close()
            if src:
                src.release()


def to_photo(bgr):
    """BGR numpy image -> tk.PhotoImage (PPM is decoded natively by Tk, no PIL needed)."""
    ok, buf = cv2.imencode(".ppm", bgr)
    try:
        return tk.PhotoImage(data=buf.tobytes())
    except tk.TclError:
        ok, png = cv2.imencode(".png", bgr)
        return tk.PhotoImage(data=base64.b64encode(png.tobytes()))


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stereo Traffic Monitor")
        self.geometry("1400x820")
        self.minsize(1050, 620)

        self.q = queue.Queue(maxsize=3)
        self.worker = None
        self.playback = None
        self.playing = False
        self.cur_idx = 0
        self.selected_id = None
        self.last_img = None
        self.photo = None
        self._slider_guard = False
        self._cal_running = False

        self._build_ui()
        self.bind("<space>", self._key_space)
        self.bind("<Left>", lambda e: self._key_step(-1))
        self.bind("<Right>", lambda e: self._key_step(1))
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(15, self._poll_live)

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)
        left = ttk.Frame(root)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(root, width=350)
        right.pack(side="right", fill="y", padx=(6, 6), pady=6)
        right.pack_propagate(False)

        # video canvas
        self.canvas = tk.Canvas(left, bg="#111111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.img_item = self.canvas.create_image(0, 0, anchor="center")
        self.hint = self.canvas.create_text(
            0, 0, fill="#888888", font=("TkDefaultFont", 13),
            text="Choose sources and press Start  (Live)\nor open a recording  (Analysis)")
        self.canvas.bind("<Configure>", lambda e: self._redraw())

        # transport bar (analysis)
        self.transport = ttk.Frame(left)
        self.btn_start = ttk.Button(self.transport, text="|<", width=3, command=lambda: self._seek(0))
        self.btn_prev = ttk.Button(self.transport, text="<", width=3, command=lambda: self._step(-1))
        self.btn_play = ttk.Button(self.transport, text="Play", width=7, command=self._toggle_play)
        self.btn_next = ttk.Button(self.transport, text=">", width=3, command=lambda: self._step(1))
        for b in (self.btn_start, self.btn_prev, self.btn_play, self.btn_next):
            b.pack(side="left", padx=2, pady=4)
        self.slider = ttk.Scale(self.transport, from_=0, to=1, orient="horizontal",
                                command=self._on_slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=8)
        self.time_var = tk.StringVar(value="0:00.000 / 0:00.000")
        ttk.Label(self.transport, textvariable=self.time_var, width=20).pack(side="left")
        self.speed_var = tk.StringVar(value="1x")
        ttk.Combobox(self.transport, textvariable=self.speed_var, width=6, state="readonly",
                     values=["0.25x", "0.5x", "1x", "2x", "4x", "max"]).pack(side="left", padx=4)
        self.loop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.transport, text="Loop", variable=self.loop_var).pack(side="left", padx=4)

        # status bar
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w").pack(
            side="bottom", fill="x")

        # right panel
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill="x")
        self.tab_live = ttk.Frame(self.nb, padding=8)
        self.tab_an = ttk.Frame(self.nb, padding=8)
        self.tab_cal = ttk.Frame(self.nb, padding=8)
        self.nb.add(self.tab_live, text="Live")
        self.nb.add(self.tab_an, text="Analysis")
        self.nb.add(self.tab_cal, text="Calibrate")
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab)
        self._build_live_tab()
        self._build_analysis_tab()
        self._build_calibrate_tab()
        self._build_display_box(right)
        self._build_table(right)

    def _build_live_tab(self):
        t = self.tab_live
        t.columnconfigure(1, weight=1)
        self.left_var = tk.StringVar(value="0")
        self.right_var = tk.StringVar(value="1")
        self.calib_var = tk.StringVar(value=default_calib_path())
        self.model_var = tk.StringVar(value=default_model_path())
        self.conf_var = tk.DoubleVar(value=0.35)
        self.record_var = tk.BooleanVar(value=False)
        self.savevid_var = tk.BooleanVar(value=True)
        self.realtime_var = tk.BooleanVar(value=True)
        self.out_var = tk.StringVar(value="traffic_monitoring")
        self.maxdisp_var = tk.IntVar(value=750)
        self.minz_var = tk.DoubleVar(value=2.0)
        self.maxz_var = tk.DoubleVar(value=80.0)

        ttk.Label(t, text="Left source").grid(row=0, column=0, sticky="w")
        self.cb_left = ttk.Combobox(t, textvariable=self.left_var, values=["0", "1"], width=18)
        self.cb_left.grid(row=0, column=1, sticky="ew", padx=3, pady=2)
        ttk.Button(t, text="File...", width=7,
                   command=lambda: self._browse_video(self.left_var)).grid(row=0, column=2)

        ttk.Label(t, text="Right source").grid(row=1, column=0, sticky="w")
        self.cb_right = ttk.Combobox(t, textvariable=self.right_var, values=["0", "1"], width=18)
        self.cb_right.grid(row=1, column=1, sticky="ew", padx=3, pady=2)
        ttk.Button(t, text="File...", width=7,
                   command=lambda: self._browse_video(self.right_var)).grid(row=1, column=2)

        ttk.Button(t, text="Scan cameras", command=self._scan_cameras).grid(
            row=2, column=1, sticky="w", padx=3)
        ttk.Label(t, text="(a number = camera index)", foreground="#777").grid(
            row=2, column=2, sticky="e")

        ttk.Label(t, text="Calibration").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(t, textvariable=self.calib_var).grid(row=3, column=1, sticky="ew", padx=3,
                                                       pady=(8, 0))
        ttk.Button(t, text="Browse", width=7, command=self._browse_calib).grid(row=3, column=2,
                                                                               pady=(8, 0))
        ttk.Label(t, text="YOLO model").grid(row=4, column=0, sticky="w")
        ttk.Entry(t, textvariable=self.model_var).grid(row=4, column=1, sticky="ew", padx=3, pady=2)
        ttk.Button(t, text="Browse", width=7, command=self._browse_model).grid(row=4, column=2)

        ttk.Label(t, text="Confidence").grid(row=5, column=0, sticky="w")
        tk.Scale(t, variable=self.conf_var, from_=0.05, to=0.95, resolution=0.05,
                 orient="horizontal", showvalue=True).grid(row=5, column=1, columnspan=2,
                                                           sticky="ew", padx=3)

        adv = ttk.LabelFrame(t, text="Stereo limits", padding=4)
        adv.grid(row=6, column=0, columnspan=3, sticky="ew", pady=4)
        for i, (lbl, var) in enumerate((("Max disparity px", self.maxdisp_var),
                                        ("Min Z m", self.minz_var), ("Max Z m", self.maxz_var))):
            ttk.Label(adv, text=lbl).grid(row=0, column=2 * i, sticky="w", padx=(0, 2))
            ttk.Entry(adv, textvariable=var, width=6).grid(row=0, column=2 * i + 1, padx=(0, 6))

        ttk.Checkbutton(t, text="Play video files at real-time speed",
                        variable=self.realtime_var).grid(row=7, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(t, text="Record session (CSV + metadata)",
                        variable=self.record_var).grid(row=8, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(t, text="   also save rectified videos (needed for Analysis)",
                        variable=self.savevid_var).grid(row=9, column=0, columnspan=3, sticky="w")
        ttk.Label(t, text="Output folder").grid(row=10, column=0, sticky="w")
        ttk.Entry(t, textvariable=self.out_var).grid(row=10, column=1, sticky="ew", padx=3, pady=2)
        ttk.Button(t, text="Browse", width=7, command=self._browse_out).grid(row=10, column=2)

        btns = ttk.Frame(t)
        btns.grid(row=11, column=0, columnspan=3, pady=(10, 0), sticky="ew")
        self.btn_go = ttk.Button(btns, text="Start", command=self._start_live)
        self.btn_go.pack(side="left", expand=True, fill="x", padx=2)
        self.btn_stop = ttk.Button(btns, text="Stop", command=self._stop_live, state="disabled")
        self.btn_stop.pack(side="left", expand=True, fill="x", padx=2)
        self._live_widgets = [self.cb_left, self.cb_right]

    def _build_analysis_tab(self):
        t = self.tab_an
        ttk.Button(t, text="Open recording folder...", command=self._open_recording).pack(
            fill="x", pady=2)
        ttk.Button(t, text="Open videos + CSV manually...", command=self._open_files).pack(
            fill="x", pady=2)
        self.info_var = tk.StringVar(value="No recording loaded.")
        ttk.Label(t, textvariable=self.info_var, wraplength=310, justify="left",
                  foreground="#555").pack(fill="x", pady=(8, 0))
        ttk.Label(t, text="A recording folder contains left_camera.mp4, right_camera.mp4 and "
                          "tracking_data.csv (created by Live > Record, or the 'run' command).",
                  wraplength=310, justify="left", foreground="#777").pack(fill="x", pady=(8, 0))

    def _build_calibrate_tab(self):
        t = self.tab_cal
        t.columnconfigure(1, weight=1)
        self.cal_left_var = tk.StringVar()
        self.cal_right_var = tk.StringVar()
        self.cal_board_var = tk.StringVar(value="9x6")
        self.cal_square_var = tk.StringVar(value="0.025")
        self.cal_out_var = tk.StringVar(value="calibration.json")

        ttk.Label(t, text="Left photo folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(t, textvariable=self.cal_left_var).grid(row=0, column=1, sticky="ew", padx=3)
        ttk.Button(t, text="...", width=4,
                   command=lambda: self._browse_dir(self.cal_left_var)).grid(row=0, column=2)

        ttk.Label(t, text="Right photo folder").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(t, textvariable=self.cal_right_var).grid(row=1, column=1, sticky="ew",
                                                           padx=3, pady=(4, 0))
        ttk.Button(t, text="...", width=4,
                   command=lambda: self._browse_dir(self.cal_right_var)).grid(row=1, column=2,
                                                                              pady=(4, 0))

        grid = ttk.Frame(t)
        grid.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Label(grid, text="Board (COLSxROWS)").pack(side="left")
        ttk.Entry(grid, textvariable=self.cal_board_var, width=7).pack(side="left", padx=(2, 8))
        ttk.Label(grid, text="Square (m)").pack(side="left")
        ttk.Entry(grid, textvariable=self.cal_square_var, width=7).pack(side="left", padx=2)

        ttk.Label(t, text="Output file").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(t, textvariable=self.cal_out_var).grid(row=3, column=1, sticky="ew",
                                                         padx=3, pady=(6, 0))
        ttk.Button(t, text="...", width=4,
                   command=lambda: self._browse_calib_save()).grid(row=3, column=2, pady=(6, 0))

        self.btn_cal = ttk.Button(t, text="Run calibration", command=self._run_calibration)
        self.btn_cal.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ttk.Label(t, textvariable=self.status, wraplength=310, justify="left",
                  foreground="#555").grid(row=5, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Label(t, text="Use 10-15 paired photos of a chessboard taken simultaneously by "
                          "both cameras. Do not move the cameras afterwards. Aim for an RMS "
                          "error below ~1.0 px.",
                  wraplength=310, justify="left", foreground="#777").grid(
            row=6, column=0, columnspan=3, sticky="ew", pady=(6, 0))

    def _build_display_box(self, parent):
        box = ttk.LabelFrame(parent, text="Display", padding=6)
        box.pack(fill="x", pady=(8, 0))
        self.view_var = tk.StringVar(value="both")
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, text="View").pack(side="left")
        for text, val in (("Left", "left"), ("Right", "right"), ("Both", "both")):
            ttk.Radiobutton(row, text=text, value=val, variable=self.view_var).pack(
                side="left", padx=4)
        self.boxes_var = tk.BooleanVar(value=True)
        self.labels_var = tk.BooleanVar(value=True)
        self.trails_var = tk.BooleanVar(value=True)
        row2 = ttk.Frame(box)
        row2.pack(fill="x")
        ttk.Checkbutton(row2, text="Boxes", variable=self.boxes_var).pack(side="left")
        ttk.Checkbutton(row2, text="Labels", variable=self.labels_var).pack(side="left", padx=6)
        ttk.Checkbutton(row2, text="Trails", variable=self.trails_var).pack(side="left")
        row3 = ttk.Frame(box)
        row3.pack(fill="x", pady=(2, 0))
        ttk.Label(row3, text="Speed unit").pack(side="left")
        self.unit_var = tk.StringVar(value="km/h")
        ttk.Combobox(row3, textvariable=self.unit_var, values=list(stm.SPEED_UNITS), width=6,
                     state="readonly").pack(side="left", padx=6)
        ttk.Label(row3, text="Trail length").pack(side="left", padx=(8, 0))
        self.trail_var = tk.IntVar(value=60)
        tk.Scale(row3, variable=self.trail_var, from_=5, to=200, orient="horizontal",
                 showvalue=False, length=80, command=lambda v: self._refresh()).pack(side="left")
        for v in (self.view_var, self.boxes_var, self.labels_var, self.trails_var, self.unit_var):
            v.trace_add("write", lambda *a: self._refresh())

    def _build_table(self, parent):
        box = ttk.LabelFrame(parent, text="Vehicles in frame  (click a row to highlight)",
                             padding=4)
        box.pack(fill="both", expand=True, pady=(8, 0))
        cols = ("id", "class", "speed", "x", "z", "conf")
        self.tree = ttk.Treeview(box, columns=cols, show="headings", height=8,
                                 selectmode="browse")
        for c, w, txt in (("id", 40, "ID"), ("class", 70, "Class"), ("speed", 65, "Speed"),
                          ("x", 50, "X m"), ("z", 50, "Z m"), ("conf", 45, "Conf")):
            self.tree.heading(c, text=txt)
            self.tree.column(c, width=w, anchor="center")
        sb = ttk.Scrollbar(box, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tree.bind("<ButtonRelease-1>", self._on_tree_click)

    # --------------------------------------------------------- shared bits --
    def _opts(self):
        return stm.OverlayOptions(view=self.view_var.get(), boxes=self.boxes_var.get(),
                                  labels=self.labels_var.get(), trails=self.trails_var.get(),
                                  trail_len=int(self.trail_var.get()),
                                  speed_unit=self.unit_var.get(), selected_id=self.selected_id)

    def _display(self, bgr):
        self.last_img = bgr
        self._redraw()

    def _redraw(self):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.canvas.coords(self.hint, cw / 2, ch / 2)
        if self.last_img is None or cw < 10 or ch < 10:
            return
        h, w = self.last_img.shape[:2]
        s = min(cw / w, ch / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        small = cv2.resize(self.last_img, (nw, nh),
                           interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        self.photo = to_photo(small)  # keep a reference or Tk garbage-collects it
        self.canvas.itemconfig(self.img_item, image=self.photo)
        self.canvas.coords(self.img_item, cw / 2, ch / 2)
        self.canvas.itemconfig(self.hint, state="hidden")

    def _fill_table(self, dets):
        unit = self.unit_var.get()
        f = stm.SPEED_UNITS.get(unit, 3.6)
        self.tree.delete(*self.tree.get_children())
        for d in sorted(dets, key=lambda d: d["object_id"]):
            self.tree.insert("", "end", iid=str(d["object_id"]), values=(
                d["object_id"], d["class"], f"{d['speed_ms'] * f:.1f} {unit}",
                f"{d['world_x_m']:+.1f}", f"{d['world_z_m']:.1f}", f"{d['confidence']:.2f}"))
        if self.selected_id is not None and self.tree.exists(str(self.selected_id)):
            self.tree.selection_set(str(self.selected_id))

    def _on_tree_click(self, event):
        row = self.tree.identify_row(event.y)
        new = int(row) if row else None
        self.selected_id = None if new == self.selected_id else new
        if not row or self.selected_id is None:
            self.tree.selection_remove(*self.tree.selection())
        self._refresh()

    def _refresh(self):
        """Re-render after an option change when nothing else is driving the display."""
        if self.playback and not self.playing and self._tab() == "Analysis":
            self._show_index(self.cur_idx, sequential=False)

    def _tab(self):
        return self.nb.tab(self.nb.select(), "text")

    def _on_tab(self, _e=None):
        if self._tab() == "Analysis":
            self.transport.pack(fill="x")
        else:
            self._pause()
            self.transport.pack_forget()

    def _typing(self):
        return isinstance(self.focus_get(), (tk.Entry, ttk.Entry, ttk.Combobox, tk.Spinbox))

    def _key_space(self, _e):
        if self._tab() == "Analysis" and not self._typing():
            self._toggle_play()

    def _key_step(self, d):
        if self._tab() == "Analysis" and not self._typing():
            self._step(d)

    # ---------------------------------------------------------- browsing ----
    def _browse_video(self, var):
        p = filedialog.askopenfilename(title="Select video", filetypes=VIDEO_TYPES)
        if p:
            var.set(p)

    def _browse_calib(self):
        p = filedialog.askopenfilename(title="Calibration JSON",
                                       filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if p:
            self.calib_var.set(p)

    def _browse_model(self):
        p = filedialog.askopenfilename(title="YOLO weights",
                                       filetypes=[("Weights", "*.pt *.onnx *.engine"),
                                                  ("All files", "*.*")])
        if p:
            self.model_var.set(p)

    def _browse_out(self):
        p = filedialog.askdirectory(title="Output folder")
        if p:
            self.out_var.set(p)

    def _browse_dir(self, var):
        p = filedialog.askdirectory(title="Photo folder")
        if p:
            var.set(p)

    def _browse_calib_save(self):
        p = filedialog.asksaveasfilename(title="Save calibration as", defaultextension=".json",
                                         filetypes=[("JSON", "*.json")])
        if p:
            self.cal_out_var.set(p)

    def _scan_cameras(self):
        self.status.set("Scanning camera indices 0-5...")
        self.update_idletasks()
        found = []
        for i in range(6):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                found.append(str(i))
            cap.release()
        for cb in (self.cb_left, self.cb_right):
            cb.configure(values=found or ["0", "1"])
        self.status.set("Cameras found: " + (", ".join(found) if found else "none"))

    # ------------------------------------------------------------- live -----
    def _start_live(self):
        if self.worker and self.worker.is_alive():
            return
        left, right = self.left_var.get().strip(), self.right_var.get().strip()
        calib = self.calib_var.get().strip()
        if not left or not right:
            messagebox.showwarning("Sources", "Select both a left and a right source.")
            return
        if not os.path.isfile(calib):
            messagebox.showwarning("Calibration",
                                   "Calibration file not found.\nCreate one with:\n"
                                   "  python stereo_traffic_monitor.py calibrate ...")
            return
        try:
            cfg = dict(left=left, right=right, calibration=calib,
                       model=self.model_var.get().strip(), conf=float(self.conf_var.get()),
                       max_disp=int(self.maxdisp_var.get()), min_z=float(self.minz_var.get()),
                       max_z=float(self.maxz_var.get()), realtime=self.realtime_var.get(),
                       record=self.record_var.get(), save_video=self.savevid_var.get(),
                       out_dir=self.out_var.get().strip() or "traffic_monitoring")
        except (tk.TclError, ValueError):
            messagebox.showwarning("Settings", "One of the numeric settings is invalid.")
            return
        self._pause()
        self.selected_id = None
        self.last_img = None
        self.worker = LiveWorker(cfg, self.q)
        self.worker.start()
        self._set_live_running(True)

    def _stop_live(self):
        if self.worker:
            self.worker.stop_event.set()
            self.status.set("Stopping...")

    # ------------------------------------------------------- calibration ----
    def _run_calibration(self):
        if self._cal_running:
            return
        left, right = self.cal_left_var.get().strip(), self.cal_right_var.get().strip()
        if not left or not right:
            messagebox.showwarning("Calibration", "Select both photo folders.")
            return
        if not os.path.isdir(left) or not os.path.isdir(right):
            messagebox.showwarning("Calibration", "One of the folders does not exist.")
            return
        try:
            board = "x".join(str(int(v)) for v in self.cal_board_var.get().strip().split("x"))
            square = float(self.cal_square_var.get())
            out = self.cal_out_var.get().strip() or "calibration.json"
        except ValueError:
            messagebox.showwarning("Calibration",
                                   "Board must look like 9x6 and square size must be a number.")
            return
        self._cal_running = True
        self.btn_cal.configure(state="disabled")
        self.status.set("Calibrating... (this can take a minute)")

        def work():
            try:
                calib, rms, baseline = stm.stereo_calibration(
                    left, right, board, square,
                    progress=lambda m: self.q.put(("status", "Calibrate " + m)))
                with open(out, "w") as f:
                    json.dump(calib, f, indent=2)
                self.q.put(("cal_done", (out, rms, baseline)))
            except Exception as e:
                self.q.put(("error", "Calibration failed: " + str(e)))
            finally:
                self._cal_running = False

        threading.Thread(target=work, daemon=True).start()

    def _set_live_running(self, running):
        self.btn_go.configure(state="disabled" if running else "normal")
        self.btn_stop.configure(state="normal" if running else "disabled")

    def _poll_live(self):
        latest, done, error, cal_done = None, None, None, None
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "frame":
                    latest = payload
                elif kind == "status":
                    self.status.set(payload)
                elif kind == "done":
                    done = payload
                elif kind == "error":
                    error = payload
                elif kind == "cal_done":
                    cal_done = payload
        except queue.Empty:
            pass
        if latest is not None:
            self._show_live(latest)
        if done is not None:
            self.status.set(done)
            self._set_live_running(False)
        if cal_done is not None:
            path, rms, baseline = cal_done
            self.calib_var.set(os.path.abspath(path))
            self.status.set(f"Calibration saved: {path} (RMS {rms:.2f} px)")
            self.btn_cal.configure(state="normal")
            messagebox.showinfo(
                "Calibration complete",
                f"Saved: {path}\nRMS error: {rms:.3f} px\nBaseline: {baseline:.3f} m\n\n"
                "The Live tab now points at this calibration file.")
        if error is not None:
            self.status.set("Error: " + error)
            self._set_live_running(False)
            self.btn_cal.configure(state="normal")
            messagebox.showerror("Processing error", error)
        self.after(15, self._poll_live)

    def _show_live(self, p):
        hud = f"LIVE  frame {p['frame_id']}  {p['fps']:.1f} fps  vehicles {len(p['dets'])}"
        sizes = p.get("capture_sizes")
        if sizes:
            hud += "  input " + "/".join(f"{w}x{h}" for w, h in sizes.values())
        if p.get("experimental_fit"):
            hud += "  EXPERIMENTAL DEPTH FIT"
        img = stm.compose_view(p["left"], p["right"], p["dets"], self._opts(),
                               p["trails_l"], p["trails_r"], hud)
        self._display(img)
        self._fill_table(p["dets"])
        self.status.set(hud)

    # ---------------------------------------------------------- analysis ----
    def _open_recording(self):
        d = filedialog.askdirectory(title="Select recording folder (e.g. recording_001)")
        if d:
            self._load_playback(lambda: stm.Playback.from_folder(d))

    def _open_files(self):
        left = filedialog.askopenfilename(title="Left video", filetypes=VIDEO_TYPES)
        if not left:
            return
        right = filedialog.askopenfilename(title="Right video (Cancel to skip)",
                                           filetypes=VIDEO_TYPES)
        csvp = filedialog.askopenfilename(title="Tracking CSV (Cancel to skip)",
                                          filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        self._load_playback(lambda: stm.Playback(left, right or None, csvp or None))

    def _load_playback(self, factory):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Live is running", "Stop the live session first.")
            return
        try:
            pb = factory()
        except Exception as e:
            messagebox.showerror("Cannot open recording", str(e))
            return
        self._pause()
        if self.playback:
            self.playback.close()
        self.playback = pb
        self.selected_id = None
        self.cur_idx = 0
        self.slider.configure(to=max(1, pb.n_frames - 1))
        bits = [f"{pb.n_frames} frames @ {pb.fps:.1f} fps, {pb.size[0]}x{pb.size[1]}",
                "right camera: " + ("yes" if pb.has_right else "no"),
                "tracking data: " + (f"yes ({len(pb.by_frame)} frames with rows)"
                                     if pb.has_data else "NO - video only")]
        self.info_var.set("\n".join(bits))
        self._show_index(0, sequential=False)

    def _fmt_t(self, sec):
        return f"{int(sec // 60)}:{sec % 60:06.3f}"

    def _show_index(self, idx, sequential=True):
        pb = self.playback
        if pb is None:
            return False
        idx = max(0, min(pb.n_frames - 1, idx))
        frames = pb.get(idx)
        if frames is None:
            return False
        left, right = frames
        dets = pb.dets(idx)
        opts = self._opts()
        tl = pb.trails(idx, dets, "left", opts.trail_len)
        tr = pb.trails(idx, dets, "right", opts.trail_len)
        hud = f"ANALYSIS  frame {idx + 1}/{pb.n_frames}  vehicles {len(dets)}"
        self._display(stm.compose_view(left, right, dets, opts, tl, tr, hud))
        self._fill_table(dets)
        self.cur_idx = idx
        self._slider_guard = True
        self.slider.set(idx)
        self._slider_guard = False
        self.time_var.set(f"{self._fmt_t(idx / pb.fps)} / {self._fmt_t(pb.n_frames / pb.fps)}")
        self.status.set(hud)
        return True

    def _seek(self, idx):
        self._show_index(idx, sequential=False)

    def _step(self, d):
        if self.playback:
            self._pause()
            self._show_index(self.cur_idx + d, sequential=False)

    def _on_slider(self, val):
        if self._slider_guard or not self.playback:
            return
        idx = int(float(val))
        if idx != self.cur_idx:
            self._show_index(idx, sequential=False)

    def _toggle_play(self):
        if not self.playback:
            return
        if self.playing:
            self._pause()
        else:
            if self.cur_idx >= self.playback.n_frames - 1:
                self._show_index(0, sequential=False)
            self.playing = True
            self.btn_play.configure(text="Pause")
            self._play_tick()

    def _pause(self):
        self.playing = False
        self.btn_play.configure(text="Play")

    def _speed(self):
        s = self.speed_var.get()
        return None if s == "max" else float(s.rstrip("x"))

    def _play_tick(self):
        if not self.playing or not self.playback:
            return
        t0 = time.perf_counter()
        nxt = self.cur_idx + 1
        if nxt >= self.playback.n_frames:
            if self.loop_var.get():
                nxt = 0
            else:
                self._pause()
                return
        if not self._show_index(nxt):
            self._pause()
            return
        sp = self._speed()
        if sp is None:
            delay = 1
        else:
            delay = max(1, int(1000.0 / (self.playback.fps * sp) - (time.perf_counter() - t0) * 1000))
        self.after(delay, self._play_tick)

    # --------------------------------------------------------------- exit ---
    def _on_close(self):
        self.playing = False
        if self.worker and self.worker.is_alive():
            self.worker.stop_event.set()
            self.worker.join(timeout=3)
        if self.playback:
            self.playback.close()
        self.destroy()


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
