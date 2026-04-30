"""
Image Point Annotation Tool —— PyQt5 本地版
对应原 FastAPI + index.html 的功能：
  - 在图像上 create / drag 点对（T0 红 -> T1 绿，中间虚线）
  - drag 模式：根据 points_data 在 networkx 图中找 loss 最小节点 -> 队列 -> 切图
  - CoTracker 轨迹推理（后台线程，不阻塞 UI）
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from collections import deque

import numpy as np
import networkx as nx
import torch
from PIL import Image

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QRectF
from PyQt5.QtGui import QPixmap, QPen, QBrush, QColor, QPainter
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QTextEdit, QGroupBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsEllipseItem, QGraphicsLineItem, QMessageBox,
)

import models

# ----------------------------------------------------------------------
# 路径配置（与原 main.py 一致）
# ----------------------------------------------------------------------
BASE_PATH = "nuero"
models.BASE_DIR = BASE_PATH
models.GRAPH_PATH = os.path.join(BASE_PATH, "graph.txt")
models.IMAGES_DIR = os.path.join(BASE_PATH, "images")
models.TRACKS_DIR = os.path.join(BASE_PATH, "track")
models.load_graph()

IMAGE_BASE_REL_DIR = os.path.join(BASE_PATH, "images", "video_00014")
IMAGE_BASE_PHYSICAL_DIR = os.path.abspath(IMAGE_BASE_REL_DIR)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

FPS = 30
MOUSE_FPS=10

DEPTH_LIMIT=10
QUEUE_LEN=10
MAX_QUEUE_ADD = 10


# ----------------------------------------------------------------------
# 启动时把磁盘上已有的 track JSON 预加载到 NODE_POSITIONS
# 等价于：CoTracker 跑完后那段 `for node in models.G.nodes:
#                                     NODE_POSITIONS[node] = get_node_position(node)`
# 的逻辑，但提前到启动阶段，使首次进入 drag 模式即可看到缓存点位。
# ----------------------------------------------------------------------
def _preload_track_caches():
    if models.G is None:
        return
    loaded = 0
    for node in models.G.nodes:
        # 已有数据就不覆盖（尊重 load_graph 内部可能已做的初始化）
        if models.NODE_POSITIONS.get(node):
            loaded += 1
            continue
        try:
            pos = models.get_node_position(node)   # 读 track JSON -> [[x,y], ...]
            if pos:
                models.NODE_POSITIONS[node] = pos
                loaded += 1
        except Exception:
            pass
    print(f"✅ track 缓存预加载完成：{loaded} 个节点有坐标数据")


_preload_track_caches()

# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def calculate_loss(points, node_positions):
    """与原 main.py 相同：均方误差损失"""
    if not points or not node_positions or len(points) != len(node_positions):
        return float("inf")
    p_np = np.array(points, dtype=np.float32)
    n_np = np.array(node_positions, dtype=np.float32)
    return float(np.mean((p_np - n_np) ** 2))


def generate_image_paths():
    if not os.path.exists(IMAGE_BASE_PHYSICAL_DIR):
        print(f"❌ 图片物理目录不存在: {IMAGE_BASE_PHYSICAL_DIR}")
        return []
    rels = []
    for fn in os.listdir(IMAGE_BASE_PHYSICAL_DIR):
        if fn.lower().endswith(IMAGE_EXTENSIONS):
            rels.append(Path(IMAGE_BASE_REL_DIR, fn).as_posix())
    rels.sort()
    return rels


# ----------------------------------------------------------------------
# CoTracker 后台线程：避免推理时冻结 UI
# ----------------------------------------------------------------------
class CoTrackerWorker(QThread):
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, t0_points, current_image_path):
        super().__init__()
        self.t0_points = t0_points
        self.current_image_path = current_image_path

    def run(self):
        try:
            queries = torch.tensor(
                [[[0, float(x), float(y)] for x, y in self.t0_points]],
                dtype=torch.float32, device=models.DEVICE,
            )

            start_node = self.current_image_path.lstrip("/")
            paths = models.bfs_collect_paths(models.G, start_node=start_node, max_depth=20)

            results_summary = []
            saved_files_count = 0

            for i, path in enumerate(paths):
                video_pil_list = []
                valid = True
                for node_path in path:
                    if not os.path.exists(node_path):
                        print(f"⚠️ 文件不存在: {node_path}")
                        valid = False
                        break
                    try:
                        video_pil_list.append(Image.open(node_path).convert("RGB"))
                    except Exception as e:
                        print(f"⚠️ 加载失败 {node_path}: {e}")
                        valid = False
                        break

                if not valid or len(video_pil_list) < 2:
                    continue

                try:
                    pred_tracks = models.run_cotracker(video_pil_list, queries)
                    tracks_np = pred_tracks[0].cpu().numpy()
                    T_frames, N_points, _ = tracks_np.shape

                    for frame_idx, node_path in enumerate(path):
                        if frame_idx >= T_frames:
                            break
                        track_path = node_path.replace("images/", "track/", 1)
                        track_dir = os.path.dirname(track_path)
                        track_basename = os.path.splitext(os.path.basename(track_path))[0]
                        full_track_path = os.path.join(track_dir, f"{track_basename}.json")
                        Path(track_dir).mkdir(parents=True, exist_ok=True)

                        with open(full_track_path, "w", encoding="utf-8") as f:
                            json.dump({"track": tracks_np[frame_idx].tolist()},
                                      f, ensure_ascii=False, indent=2)

                        if os.path.exists(full_track_path):
                            saved_files_count += 1
                            print(f"✅ 保存: {full_track_path}")

                    results_summary.append({
                        "path_index": i,
                        "frames": T_frames,
                        "points": N_points,
                        "saved_files": min(T_frames, len(path)),
                    })
                    del pred_tracks
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    continue

            # 重新加载所有节点位置
            for node in models.G.nodes:
                models.NODE_POSITIONS[node] = models.get_node_position(node)

            self.finished_signal.emit({
                "status": "success",
                "input_points_count": len(self.t0_points),
                "paths_processed": len(results_summary),
                "saved_files_total": saved_files_count,
                "details": results_summary,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(str(e))



# ----------------------------------------------------------------------
# ListenPoints 后台线程：drag 模式下轮询 points 变化 -> 触发 add_image_to_queue
# 替代原 QTimer 方案，使 queue 写入与主线程读取并发安全。
# ----------------------------------------------------------------------
class ListenPointsWorker(QThread):
    """
    以固定频率（1000 // MOUSE_FPS ms）发出 should_add_image 信号，
    不再依赖 points 是否变化，队列未满时无条件触发 add_image_to_queue。
    由主线程槽执行实际的队列写入，线程本身不直接操作队列。
    """
    should_add_image = pyqtSignal(dict)   # 携带 points_data

    def __init__(self, get_points_fn, get_queue_len_fn,
                 interval_ms=1000 // MOUSE_FPS, parent=None):
        super().__init__(parent)
        self._get_points = get_points_fn       # callable -> dict
        self._get_queue_len = get_queue_len_fn # callable -> int
        self._interval = interval_ms / 1000.0
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            if self._get_queue_len() <= QUEUE_LEN:
                pd = self._get_points()
                self.should_add_image.emit(pd)
            time.sleep(self._interval)

    def stop(self):
        self._running = False
        self.wait()


class ImageCanvas(QGraphicsView):
    points_changed = pyqtSignal(bool)   # bool=trigger_image
    status_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QBrush(QColor("#fafafa")))

        self.pixmap_item = None
        self.points_t0 = []      # [(x, y), ...] 在图像坐标系下
        self.points_t1 = []
        self.current_mode = "create"
        self.selected_index = -1
        self.is_dragging = False
        self.base_radius = 6
        self.selected_radius = 10

        # 缓存 graphics items
        self._line_items = []
        self._t0_items = []
        self._t1_items = []

    # ----- 图像加载 -----
    def load_image(self, abs_path):
        if not os.path.exists(abs_path):
            print(f"❌ 图片不存在: {abs_path}")
            return False
        pixmap = QPixmap(abs_path)
        if pixmap.isNull():
            print(f"❌ QPixmap 加载失败: {abs_path}")
            return False

        # 清空场景但保留点位数据
        self._scene.clear()
        self._line_items.clear()
        self._t0_items.clear()
        self._t1_items.clear()

        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.pixmap_item.setZValue(-1)
        self._scene.addItem(self.pixmap_item)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self.render_points()
        return True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap_item:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    # ----- 模式与样式 -----
    def set_mode(self, mode):
        self.current_mode = mode
        self.selected_index = -1
        self.is_dragging = False
        self.render_points()
        self.viewport().setCursor(Qt.CrossCursor if mode == "create" else Qt.ArrowCursor)

    def set_point_size(self, size):
        self.base_radius = size
        self.selected_radius = max(size + 2, int(size * 1.6))
        self.render_points()

    def reset_points(self):
        self.points_t0.clear()
        self.points_t1.clear()
        self.selected_index = -1
        self.is_dragging = False
        self.render_points()

    # ----- 鼠标事件（对应原 SVG 的 mousedown / mousemove） -----
    def _scene_xy(self, event):
        sp = self.mapToScene(event.pos())
        return sp.x(), sp.y()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self.pixmap_item is None:
            return super().mousePressEvent(event)

        x, y = self._scene_xy(event)

        if self.current_mode == "create":
            self.points_t0.append((x, y))
            self.points_t1.append((x, y))
            self.render_points()
            self.points_changed.emit(False)
            return

        # drag 模式
        if self.selected_index != -1:
            # 第二次点击 -> 确认
            self.selected_index = -1
            self.is_dragging = False
            self.status_changed.emit("Position confirmed")
            self.render_points()
            self.points_changed.emit(True)
        else:
            threshold = max(10, self.base_radius * 2.5)
            found = -1
            for i in range(len(self.points_t1) - 1, -1, -1):
                px, py = self.points_t1[i]
                if ((x - px) ** 2 + (y - py) ** 2) ** 0.5 < threshold:
                    found = i
                    break
            if found != -1:
                self.selected_index = found
                self.status_changed.emit(f"Selected Point #{found}, move mouse to adjust")
                self.render_points()
            else:
                self.status_changed.emit("No point selected")

    def mouseMoveEvent(self, event):
        if self.current_mode != "drag" or self.selected_index == -1:
            return super().mouseMoveEvent(event)
        x, y = self._scene_xy(event)
        self.points_t1[self.selected_index] = (x, y)
        self.render_points()
        if not self.is_dragging:
            self.is_dragging = True
            self.status_changed.emit(f"Moving Point #{self.selected_index}...")
        self.points_changed.emit(True)

    # ----- 渲染（对应 renderPoints） -----
    def render_points(self):
        for it in self._line_items + self._t0_items + self._t1_items:
            self._scene.removeItem(it)
        self._line_items.clear()
        self._t0_items.clear()
        self._t1_items.clear()

        # 轨迹线
        for i, (p0, p1) in enumerate(zip(self.points_t0, self.points_t1)):
            line = QGraphicsLineItem(p0[0], p0[1], p1[0], p1[1])
            if i == self.selected_index:
                pen = QPen(QColor("#ffaa00"), 3)
            else:
                pen = QPen(QColor("#ff0000"), 2)
                pen.setStyle(Qt.DashLine)
                pen.setDashPattern([5, 5])
            line.setPen(pen)
            self._scene.addItem(line)
            self._line_items.append(line)

        # T0（红）
        for p in self.points_t0:
            r = self.base_radius
            e = QGraphicsEllipseItem(p[0] - r, p[1] - r, 2 * r, 2 * r)
            e.setBrush(QBrush(QColor("red")))
            e.setPen(QPen(QColor("white"), 1.5))
            self._scene.addItem(e)
            self._t0_items.append(e)

        # T1（绿，选中黄）
        for i, p in enumerate(self.points_t1):
            sel = (i == self.selected_index)
            r = self.selected_radius if sel else self.base_radius
            e = QGraphicsEllipseItem(p[0] - r, p[1] - r, 2 * r, 2 * r)
            if sel:
                e.setBrush(QBrush(QColor("#ffcc00")))
                e.setPen(QPen(QColor("#d35400"), 2))
            else:
                e.setBrush(QBrush(QColor("green")))
                e.setPen(QPen(QColor("black"), 1.5))
            self._scene.addItem(e)
            self._t1_items.append(e)


# ----------------------------------------------------------------------
# 主窗口
# ----------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Point Annotation Tool (PyQt5)")
        self.resize(1100, 900)

        # 状态（对应原 FastAPI 全局变量）
        self.current_image_path = Path(IMAGE_BASE_REL_DIR, "000000.png").as_posix()
        self.image_path_queue = deque()
        self.queue_lock = threading.Lock()   # 保护 image_path_queue 的并发读写
        self.cotracker_thread = None
        self.listen_worker = None            # ListenPointsWorker 实例

        self._build_ui()

        # 加载初始图像
        self._load_image_from_rel(self.current_image_path)

        # drag 模式下切图（替代前端 setInterval(fetchCurrentImage, 100)）
        self.image_update_timer = QTimer(self)
        self.image_update_timer.timeout.connect(self._fetch_current_image)

        # ListenPointsWorker 在 set_mode('drag') 时按需启动，此处仅预声明

    # ----- UI 构建 -----
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # 顶部控制栏
        bar = QHBoxLayout()
        self.btn_create = QPushButton("🔴 Create")
        self.btn_create.setCheckable(True)
        self.btn_create.setChecked(True)
        self.btn_create.clicked.connect(lambda: self.set_mode("create"))

        self.btn_drag = QPushButton("🟢 Drag")
        self.btn_drag.setCheckable(True)
        self.btn_drag.clicked.connect(lambda: self.set_mode("drag"))

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setStyleSheet(
            "background:#dc3545;color:white;padding:6px 12px;border-radius:4px;font-weight:bold;")
        self.btn_reset.clicked.connect(self._reset_points)

        self.btn_track = QPushButton("📡 CoTracker Track")
        self.btn_track.setStyleSheet(
            "background:#6f42c1;color:white;padding:6px 12px;border-radius:4px;font-weight:bold;")
        self.btn_track.clicked.connect(self._send_track_points)

        for b in (self.btn_create, self.btn_drag, self.btn_reset, self.btn_track):
            bar.addWidget(b)

        bar.addSpacing(15)
        bar.addWidget(QLabel("Point Size:"))
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(3)
        self.size_slider.setMaximum(20)
        self.size_slider.setValue(6)
        self.size_slider.setMaximumWidth(150)
        self.size_slider.valueChanged.connect(self._on_size_changed)
        self.size_label = QLabel("6 px")
        self.size_label.setMinimumWidth(40)
        bar.addWidget(self.size_slider)
        bar.addWidget(self.size_label)

        bar.addStretch()
        self.status_label = QLabel("Current Mode: Create")
        self.status_label.setStyleSheet("color:#555;")
        bar.addWidget(self.status_label)

        root.addLayout(bar)

        self.hint_label = QLabel(
            "Tip: Click anywhere on the image to add a new point pair.")
        self.hint_label.setStyleSheet("color:#888;font-size:12px;")
        root.addWidget(self.hint_label)

        # 画布
        self.canvas = ImageCanvas()
        self.canvas.points_changed.connect(self._on_points_changed)
        self.canvas.status_changed.connect(self.status_label.setText)
        root.addWidget(self.canvas, 1)

        # 数据面板
        box = QGroupBox("Coordinate Data (Real-time)")
        bl = QVBoxLayout(box)
        self.data_output = QTextEdit()
        self.data_output.setReadOnly(True)
        self.data_output.setStyleSheet("font-family:Consolas,monospace;font-size:12px;")
        self.data_output.setMaximumHeight(180)
        self.data_output.setPlainText("Waiting for operation...")
        bl.addWidget(self.data_output)
        root.addWidget(box)

    # ----- 模式切换 -----
    def set_mode(self, mode):
        self.btn_create.setChecked(mode == "create")
        self.btn_drag.setChecked(mode == "drag")
        self.canvas.set_mode(mode)
        if mode == "create":
            self.status_label.setText("Current Mode: Create")
            self.hint_label.setText(
                "Tip: Click anywhere on the image to add a new point pair.")
            if self.image_update_timer.isActive():
                self.image_update_timer.stop()
            # 退出 drag 模式时停止监听线程
            self._stop_listen_worker()
        else:
            self.status_label.setText("Current Mode: Drag")
            self.hint_label.setText(
                "Tip: 1.Click green point to select -> 2.Move mouse to adjust -> 3.Click again to confirm.")
            self._apply_cached_track_to_canvas(self.current_image_path)
            if not self.image_update_timer.isActive():
                self.image_update_timer.start(1000 // FPS)
            self._fetch_current_image()
            # 进入 drag 模式时启动监听线程
            self._start_listen_worker()

    def _on_size_changed(self, value):
        self.size_label.setText(f"{value} px")
        self.canvas.set_point_size(value)

    def _reset_points(self):
        if not self.canvas.points_t0:
            return
        if QMessageBox.question(self, "Confirm Reset",
                                "Are you sure you want to delete all points?",
                                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.canvas.reset_points()
            self._update_data_text()
            self.status_label.setText("Reset completed")

    def _on_points_changed(self, _trigger_image):
        self._update_data_text()

    def _update_data_text(self):
        t0, t1 = self.canvas.points_t0, self.canvas.points_t1
        if not t0:
            self.data_output.setPlainText("No data")
            return
        lines = [f"# Total Points: {len(t0)}", ""]
        for i, (p0, p1) in enumerate(zip(t0, t1)):
            mark = " <-- [Editing]" if i == self.canvas.selected_index else ""
            lines.append(f"Point {i}: [{p0[0]:.1f}, {p0[1]:.1f}] -> "
                         f"[{p1[0]:.1f}, {p1[1]:.1f}]{mark}")
        self.data_output.setPlainText("\n".join(lines))

    # ----- 加载相对路径图像 -----
    def _load_image_from_rel(self, rel_path):
        abs_path = os.path.abspath(rel_path)
        if self.canvas.load_image(abs_path):
            self.current_image_path = Path(rel_path).as_posix()
        else:
            print(f"❌ 加载图像失败: {abs_path}")

    def _apply_cached_track_to_canvas(self, image_path):
        """
        从 NODE_POSITIONS 取指定图片的缓存点位写入画布。
        对应 index.html 中 fetchCurrentImage() 收到 t0_points 后那段：
            pointsT0 = result.t0_points.map(...);
            if (pointsT1.length === 0) pointsT1 = [...pointsT0];
            renderPoints();

        触发场景：
          1) set_mode('drag')：首次或重复点击 drag 按钮
          2) _fetch_current_image()：drag 模式队列推进切到新图
          3) reset 后再次点击 drag：canvas 已被清空，这里把缓存重新填回去
        """
        node_key = Path(image_path).as_posix().lstrip("/")

        # 运行期可能新写入了 track 文件（CoTracker 刚跑完），再懒加载一次
        if not models.NODE_POSITIONS.get(node_key):
            try:
                pos = models.get_node_position(node_key)
                if pos:
                    models.NODE_POSITIONS[node_key] = pos
            except Exception:
                pass

        t0_positions = models.NODE_POSITIONS.get(node_key, [])
        if t0_positions:
            self.canvas.points_t0 = [(float(p[0]), float(p[1])) for p in t0_positions]
            # 仿照 index.html: pointsT1 为空时用 t0 填充；非空时保留用户拖拽进度
            if not self.canvas.points_t1:
                self.canvas.points_t1 = list(self.canvas.points_t0)
            self.canvas.render_points()
            self._update_data_text()
            self.status_label.setText(
                f"Drag | Image: {image_path} | Cached T0 points: {len(t0_positions)}")
            print(f"📌 已从缓存加载点位: {node_key} -> {len(t0_positions)} 个点")
        else:
            print(f"ℹ️  无 track 缓存: {node_key}")

    # ----- 与原 listen_points_update / add_image_to_queue 等价 -----
    def _get_points_data(self):
        """对应原 points_data：用字符串保留 1 位小数，让 hash 稳定"""
        return {
            "t0": [[f"{x:.1f}", f"{y:.1f}"] for x, y in self.canvas.points_t0],
            "t1": [[f"{x:.1f}", f"{y:.1f}"] for x, y in self.canvas.points_t1],
        }

    # ----- ListenPointsWorker 生命周期 -----
    def _start_listen_worker(self):
        """仅在 drag 模式下启动监听线程（幂等）"""
        if self.listen_worker and self.listen_worker.isRunning():
            return
        self.listen_worker = ListenPointsWorker(
            get_points_fn=self._get_points_data,
            get_queue_len_fn=lambda: len(self.image_path_queue),
        )
        self.listen_worker.should_add_image.connect(self._on_listen_points)
        self.listen_worker.start()
        print("▶️  ListenPointsWorker 启动")

    def _stop_listen_worker(self):
        """停止监听线程（幂等）"""
        if self.listen_worker and self.listen_worker.isRunning():
            self.listen_worker.stop()
            self.listen_worker = None
            print("⏹️  ListenPointsWorker 停止")

    def _on_listen_points(self, pd: dict):
        """
        由 ListenPointsWorker 在检测到 points 变化时通过信号触发，
        运行在主线程（Qt 信号槽跨线程投递），可安全操作 UI 及队列。
        """
        self._add_image_to_queue(pd)

    def _add_image_to_queue(self, points_data, path=None):
        """与原 add_image_to_queue 逻辑一致，queue 操作用 lock 保护"""
        depth_limit = DEPTH_LIMIT
        if path:
            with self.queue_lock:
                self.image_path_queue.append(Path(path.lstrip("/")).as_posix())
            return

        with self.queue_lock:
            start_node = (self.image_path_queue[-1]
                          if self.image_path_queue else self.current_image_path)

        t1_pts = [[float(x), float(y)] for x, y in points_data["t1"]]

        node_loss = {}
        for node in nx.bfs_tree(models.G, start_node, depth_limit=depth_limit):
            node_pos = models.NODE_POSITIONS.get(node, [])
            node_loss[node] = calculate_loss(t1_pts, node_pos)

        target_node, min_loss = min(node_loss.items(), key=lambda kv: kv[1])
        print(f"🎯 loss最小节点: {target_node} (loss={min_loss:.4f})")

        try:
            shortest = nx.shortest_path(models.G, start_node, target_node)
        except nx.NetworkXNoPath:
            print(f"⚠️ 无路径到达 {target_node}")
            shortest = [start_node]

        with self.queue_lock:
            for n in shortest[:MAX_QUEUE_ADD]:
                if len(self.image_path_queue) < QUEUE_LEN:
                    self.image_path_queue.append(n)
            print(f"📥 队列: {list(self.image_path_queue)}")

    def _fetch_current_image(self):
        """对应 GET /api/current-image：drag 模式下定期推进队列、切图"""
        with self.queue_lock:
            if not self.image_path_queue:
                return
            new_path = self.image_path_queue.popleft()
        if new_path == self.current_image_path:
            return

        self._load_image_from_rel(new_path)
        print(f"📤 切图: {new_path}")

        # 复用同一段缓存加载逻辑（对应 index.html fetchCurrentImage 中 image_src 变化时的更新）
        self._apply_cached_track_to_canvas(new_path)

    # ----- CoTracker -----
    def _send_track_points(self):
        if not self.canvas.points_t0:
            QMessageBox.warning(self, "No Points", "Please create at least one point first!")
            return
        if self.cotracker_thread and self.cotracker_thread.isRunning():
            QMessageBox.information(self, "Running",
                                    "CoTracker is already running, please wait...")
            return

        t0_points = list(self.canvas.points_t0)  # 快照，避免线程间共享可变状态
        self.btn_track.setEnabled(False)
        self.status_label.setText("⏳ Running CoTracker...")

        self.cotracker_thread = CoTrackerWorker(t0_points, self.current_image_path)
        self.cotracker_thread.finished_signal.connect(self._on_cotracker_finished)
        self.cotracker_thread.error_signal.connect(self._on_cotracker_error)
        self.cotracker_thread.start()

    def _on_cotracker_finished(self, result):
        self.btn_track.setEnabled(True)
        msg = (f"✅ Sent {result['input_points_count']} points | "
               f"Paths: {result['paths_processed']} | "
               f"Saved files: {result['saved_files_total']}")
        self.status_label.setText(msg)
        print(msg)

    def _on_cotracker_error(self, err_str):
        self.btn_track.setEnabled(True)
        self.status_label.setText("❌ CoTracker error")
        QMessageBox.critical(self, "CoTracker Error", err_str)

    # ----- 关闭时清理 -----
    def closeEvent(self, event):
        self._stop_listen_worker()
        if self.cotracker_thread and self.cotracker_thread.isRunning():
            self.cotracker_thread.wait(2000)
        event.accept()


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
def main():
    print("🔍 路径验证:")
    print(f"  - 基础目录: {BASE_PATH} -> {os.path.abspath(BASE_PATH)} -> "
          f"{'存在' if os.path.exists(os.path.abspath(BASE_PATH)) else '不存在'}")
    print(f"  - 图片目录: {IMAGE_BASE_REL_DIR} -> {IMAGE_BASE_PHYSICAL_DIR} -> "
          f"{'存在' if os.path.exists(IMAGE_BASE_PHYSICAL_DIR) else '不存在'}")
    paths = generate_image_paths()
    if paths:
        print(f"📝 找到 {len(paths)} 张图片，示例: {paths[0]}")
    else:
        print("❌ 未找到任何图片！")

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
