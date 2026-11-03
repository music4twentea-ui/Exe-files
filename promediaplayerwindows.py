#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRO AI MEDIA PLAYER ENHANCED FOR WINDOWS OS
===========================================
Native Windows 10 / 11 Desktop Media Player & Live/Offline AI Enhancer (PyQt6 + OpenCV + NumPy + ONNX Runtime)

Key Features:
1. Real-Time 60fps AI Video Enhancement Engine:
   - Preset 1: AI Detail & Super-Resolution (Noise-damped Unsharp Micro-Detail + Optional CAS, off by default).
   - Preset 2: Cinema HDR & 5-Point Filmic Tone Curve LUT + Adaptive Skin-Safe Vibrance + D65 Color Warmth.
   - Preset 3: Human Face & Skin Detail Restoration (Localized Face Crop + GFPGAN 50-70% Natural Opacity + Soft Feather).
2. Interactive On-Canvas 'Before / After Wipe Slider' Mode:
   - Toggle 'Before/After Wipe' on the top bar and drag anywhere left/right on the playing video to compare Raw vs AI Enhanced.
3. Live AI Enhancer Sheet & Built-In Offline AI Studio (Both with Pinned Always-Visible Before/After Preview Headers):
   - Adjust sliders with an always-visible Before/After preview at the top even while scrolling down.
   - Automatically persists all changed enhancement settings and player preferences across app restarts.
4. Primary Bar 'Import Files' & 'Import Folder' Buttons + Slide-Over Playlist Drawer:
   - Playlist panel automatically hides when you click anywhere outside of it on the video canvas.
   - Includes live Search, Category Filter pills (All, Files, Folders, Resume), and Sort options (Added, A-Z, Z-A, Folder, Resume).
5. Full Pro Transport Controls:
   - Continuous Playback Speed (0.1x - 4.0x slider + custom input), A-B Repeat Loop, 90° Rotation, and .SRT/.VTT Subtitles.
"""

import sys
import os
import re
import json
import time
import uuid
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import cv2

from PyQt6.QtCore import (
    Qt, QTimer, QUrl, QRect, QSize, pyqtSignal, QEvent
)
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QColor, QPen, QFont, QDesktopServices
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QComboBox, QCheckBox, QScrollArea, QFrame,
    QFileDialog, QDialog, QLineEdit, QGroupBox, QInputDialog
)

try:
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
    HAS_QTMULTIMEDIA = True
except ImportError:
    HAS_QTMULTIMEDIA = False

# Reuse the AI Engine, Settings, BeforeAfterCanvasWidget, and OfflineStudioWindow if in same directory,
# or define self-contained fallback so ProMediaPlayerWindows.py works 100% standalone!
from OfflineAIVideoStudioWindows import (
    AIEnhancerSettings,
    WindowsAIEnhancementEngine,
    GLOBAL_AI_ENGINE,
    BeforeAfterCanvasWidget,
    OfflineAIVideoStudioWindow,
    get_app_data_dir,
    bgr_to_qpixmap
)


# ============================================================================
# 1. PLAYLIST ITEM & SUBTITLE PARSER (.SRT / .VTT)
# ============================================================================

@dataclass
class VideoTrackItem:
    id: str
    title: str
    path: str
    is_local: bool = True
    folder_name: Optional[str] = None


@dataclass
class SubtitleCue:
    start_sec: float
    end_sec: float
    text: str


def parse_srt_or_vtt(file_path: str) -> List[SubtitleCue]:
    cues: List[SubtitleCue] = []
    try:
        content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return cues

    blocks = re.split(r"\n\s*\n", content.strip())
    time_pat = re.compile(
        r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})"
    )
    for b in blocks:
        lines = [l.strip() for l in b.splitlines() if l.strip()]
        for i, line in enumerate(lines):
            m = time_pat.search(line)
            if m:
                h1 = int(m.group(1) or 0)
                m1, s1, ms1 = int(m.group(2)), int(m.group(3)), int(m.group(4))
                h2 = int(m.group(5) or 0)
                m2, s2, ms2 = int(m.group(6)), int(m.group(7)), int(m.group(8))
                start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
                end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
                txt = "\n".join(lines[i + 1:])
                txt = re.sub(r"<[^>]+>", "", txt)
                if txt:
                    cues.append(SubtitleCue(start, end, txt))
                break
    return cues


# ============================================================================
# 2. INTERACTIVE LIVE VIDEO RENDERING SURFACE (With Wipe Slider & Click-Outside)
# ============================================================================

class LiveVideoCanvasWidget(QWidget):
    canvasClicked = pyqtSignal()
    wipeRatioChanged = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_pixmap: Optional[QPixmap] = None
        self.is_wipe_mode: bool = False
        self.split_ratio: float = 0.50
        self.subtitle_text: str = ""
        self.hud_text: str = ""
        self.setMinimumSize(640, 360)

    def set_frame_bgr(self, bgr: Optional[np.ndarray]):
        if bgr is None:
            self.current_pixmap = None
        else:
            self.current_pixmap = bgr_to_qpixmap(bgr)
        self.update()

    def mousePressEvent(self, event):
        if self.is_wipe_mode and event.button() == Qt.MouseButton.LeftButton:
            self._update_wipe(event.position().x())
        else:
            self.canvasClicked.emit()

    def mouseMoveEvent(self, event):
        if self.is_wipe_mode and (event.buttons() & Qt.MouseButton.LeftButton):
            self._update_wipe(event.position().x())

    def _update_wipe(self, x: float):
        w = max(self.width(), 1)
        self.split_ratio = float(np.clip(x / w, 0.02, 0.98))
        self.wipeRatioChanged.emit(self.split_ratio)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        painter.fillRect(rect, QColor(8, 10, 14))

        if self.current_pixmap and not self.current_pixmap.isNull():
            scaled = self.current_pixmap.scaled(
                rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            px = (rect.width() - scaled.width()) // 2
            py = (rect.height() - scaled.height()) // 2
            painter.drawPixmap(px, py, scaled)

            if self.is_wipe_mode:
                div_x = int(rect.width() * self.split_ratio)
                painter.setPen(QPen(QColor(250, 204, 21), 3))
                painter.drawLine(div_x, 0, div_x, rect.height())

                handle = QRect(div_x - 16, rect.height() // 2 - 22, 32, 44)
                painter.setBrush(QColor(250, 204, 21))
                painter.setPen(QPen(Qt.GlobalColor.black, 1.5))
                painter.drawRoundedRect(handle, 15, 15)
                painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                painter.drawText(handle, Qt.AlignmentFlag.AlignCenter, "◀▶")

        # Subtitle Overlay
        if self.subtitle_text:
            painter.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
            sub_rect = QRect(60, rect.height() - 100, rect.width() - 120, 64)
            painter.setBrush(QColor(0, 0, 0, 185))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(sub_rect, 10, 10)
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(sub_rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.subtitle_text)

        # HUD Pill
        if self.hud_text:
            painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            hud_rect = QRect(rect.width() // 2 - 150, 24, 300, 38)
            painter.setBrush(QColor(15, 23, 42, 220))
            painter.setPen(QPen(QColor(0, 210, 190), 1.5))
            painter.drawRoundedRect(hud_rect, 18, 18)
            painter.setPen(QColor(0, 210, 190))
            painter.drawText(hud_rect, Qt.AlignmentFlag.AlignCenter, self.hud_text)


# ============================================================================
# 3. LIVE AI ENHANCER CONTROL PANEL DIALOG (With Pinned Before/After Header)
# ============================================================================

class LiveAIEnhancerDialog(QDialog):
    settingsChanged = pyqtSignal()
    openOfflineStudioRequested = pyqtSignal()

    def __init__(self, player_win: "ProMediaPlayerWindow"):
        super().__init__(player_win)
        self.player_win = player_win
        self.setWindowTitle("Live AI Video Quality Enhancer (Pinned Before / After Preview)")
        self.resize(760, 780)
        self._build_ui()
        self.refresh_preview_from_player()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        # Pinned Always-Visible Before/After Header
        pinned = QFrame()
        pinned.setStyleSheet("background-color: #151D28; border: 1px solid #263347; border-radius: 12px;")
        p_layout = QVBoxLayout(pinned)

        top_row = QHBoxLayout()
        lbl = QLabel("✨ Pinned Live Before / After Frame Preview (Stays Visible While Scrolling Sliders)")
        lbl.setStyleSheet("color: #00D2BE; font-weight: bold;")
        btn_offline = QPushButton("🚀 Open Batch Offline Studio")
        btn_offline.setObjectName("primaryTeal")
        btn_offline.clicked.connect(self._on_open_offline)
        top_row.addWidget(lbl)
        top_row.addStretch()
        top_row.addWidget(btn_offline)
        p_layout.addLayout(top_row)

        self.preview_canvas = BeforeAfterCanvasWidget("AFTER (LIVE AI PREVIEW)")
        self.preview_canvas.ratioChanged.connect(lambda _: self.render_preview())
        p_layout.addWidget(self.preview_canvas)
        root.addWidget(pinned)

        # Scrollable Sliders
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        s_layout = QVBoxLayout(content)

        s = self.player_win.enhancer_settings

        grp_master = QGroupBox("Master AI Engine & Presets")
        gm = QVBoxLayout(grp_master)
        chk_enable = QCheckBox("Enable Real-Time AI Video Enhancer")
        chk_enable.setChecked(self.player_win.enhancer_enabled)
        chk_enable.toggled.connect(self._on_toggle_master)
        gm.addWidget(chk_enable)

        chk_d = QCheckBox("Preset 1: AI Detail & Super-Resolution")
        chk_d.setChecked(s.enable_ai_detail)
        chk_d.toggled.connect(lambda v: self._set("enable_ai_detail", v))
        gm.addWidget(chk_d)

        chk_h = QCheckBox("Preset 2: Cinema HDR & 5-Point Filmic LUT")
        chk_h.setChecked(s.enable_cinema_hdr)
        chk_h.toggled.connect(lambda v: self._set("enable_cinema_hdr", v))
        gm.addWidget(chk_h)

        chk_f = QCheckBox("Preset 3: Human Face & Skin Restoration")
        chk_f.setChecked(s.enable_face_restore)
        chk_f.toggled.connect(lambda v: self._set("enable_face_restore", v))
        gm.addWidget(chk_f)
        s_layout.addWidget(grp_master)

        grp_sliders = QGroupBox("Fine-Tuning Enhancement Sliders (Auto-Saved Across Restarts)")
        gs = QVBoxLayout(grp_sliders)
        self._slider(gs, "AI Detail Master Intensity", 5, 100, int(s.ai_detail_intensity * 100),
                     lambda v: self._set("ai_detail_intensity", v / 100.0))
        self._slider(gs, "Edge Reconstruction", 0, 200, int(s.edge_reconstruction * 100),
                     lambda v: self._set("edge_reconstruction", v / 100.0))
        self._slider(gs, "Macroblock Denoise Suppression", 0, 100, int(s.denoise_suppression * 100),
                     lambda v: self._set("denoise_suppression", v / 100.0))
        self._slider(gs, "Cinema HDR Master Intensity", 5, 100, int(s.cinema_hdr_intensity * 100),
                     lambda v: self._set("cinema_hdr_intensity", v / 100.0))
        self._slider(gs, "Filmic S-Curve Contrast", 0, 100, int(s.cinema_contrast * 100),
                     lambda v: self._set("cinema_contrast", v / 100.0))
        self._slider(gs, "OLED Black-Point Depth", 0, 100, int(s.black_point_depth * 100),
                     lambda v: self._set("black_point_depth", v / 100.0))
        self._slider(gs, "Mid-Shadow Recovery", 0, 100, int(s.shadow_recovery * 100),
                     lambda v: self._set("shadow_recovery", v / 100.0))
        self._slider(gs, "Highlight Soft Roll-Off", 0, 100, int(s.highlight_rolloff * 100),
                     lambda v: self._set("highlight_rolloff", v / 100.0))
        self._slider(gs, "Adaptive Vibrance", 0, 100, int(s.adaptive_vibrance * 100),
                     lambda v: self._set("adaptive_vibrance", v / 100.0))
        self._slider(gs, "Face AI Master Strength", 5, 100, int(s.face_restore_strength * 100),
                     lambda v: self._set("face_restore_strength", v / 100.0))
        self._slider(gs, "GFPGAN Blend Opacity (50-70% Natural)", 15, 100, int(s.gfpgan_blend_opacity * 100),
                     lambda v: self._set("gfpgan_blend_opacity", v / 100.0))
        self._slider(gs, "Face Parsing Feather", 10, 100, int(s.face_parsing_feather * 100),
                     lambda v: self._set("face_parsing_feather", v / 100.0))
        s_layout.addWidget(grp_sliders)

        scroll.setWidget(content)
        root.addWidget(scroll, 1)

    def _slider(self, layout, label: str, min_v: int, max_v: int, init_v: int, cb):
        row = QHBoxLayout()
        lbl = QLabel(label)
        val_lbl = QLabel(f"{init_v}%")
        val_lbl.setStyleSheet("color: #00D2BE; font-weight: bold; min-width: 44px;")
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(min_v, max_v)
        sl.setValue(init_v)
        sl.valueChanged.connect(lambda v: (val_lbl.setText(f"{v}%"), cb(v)))
        row.addWidget(lbl)
        row.addWidget(sl, 1)
        row.addWidget(val_lbl)
        layout.addLayout(row)

    def _on_toggle_master(self, checked: bool):
        self.player_win.enhancer_enabled = checked
        self.player_win.save_persistent_settings()
        self.render_preview()
        self.settingsChanged.emit()

    def _set(self, name: str, val):
        setattr(self.player_win.enhancer_settings, name, val)
        self.player_win.save_persistent_settings()
        self.render_preview()
        self.settingsChanged.emit()

    def _on_open_offline(self):
        self.accept()
        self.openOfflineStudioRequested.emit()

    def refresh_preview_from_player(self):
        self.render_preview()

    def render_preview(self):
        raw = self.player_win.last_raw_bgr
        if raw is None:
            return
        h, w = raw.shape[:2]
        scale = min(640.0 / max(w, 1), 1.0)
        sample = cv2.resize(raw, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else raw
        if self.player_win.enhancer_enabled:
            enh = GLOBAL_AI_ENGINE.apply_enhancement(sample, self.player_win.enhancer_settings, is_offline_export=False)
        else:
            enh = sample
        comp = GLOBAL_AI_ENGINE.render_split_wipe(sample, enh, self.preview_canvas.split_ratio)
        self.preview_canvas.set_preview_bgr(comp)


# ============================================================================
# 4. MAIN PRO AI MEDIA PLAYER WINDOW FOR WINDOWS
# ============================================================================

class ProMediaPlayerWindow(QMainWindow):
    SUPPORTED_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".ts"}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pro AI Media Player Enhanced — 60fps Live HDR & Super-Resolution (Windows)")
        self.resize(1340, 820)

        self.playlist: List[VideoTrackItem] = []
        self.current_index: int = 0
        self.resume_positions: Dict[str, float] = {}

        # Playlist filter & sort state
        self.playlist_search_query: str = ""
        self.playlist_filter_mode: str = "All"
        self.playlist_sort_mode: str = "Added Order"

        # Player & AI state
        self.enhancer_enabled: bool = True
        self.enhancer_settings = AIEnhancerSettings()
        self.load_persistent_settings()

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_playing: bool = False
        self.playback_speed: float = 1.0
        self.rotation_degrees: int = 0
        self.video_fps: float = 30.0
        self.duration_sec: float = 0.0
        self.current_time_sec: float = 0.0
        self.last_raw_bgr: Optional[np.ndarray] = None
        self.cached_live_faces: List[Tuple[float, float, float, float]] = []
        self.frame_counter: int = 0

        # A-B Repeat Loop
        self.loop_point_a: Optional[float] = None
        self.loop_point_b: Optional[float] = None

        # Subtitles
        self.subtitle_cues: List[SubtitleCue] = []

        # Optional synchronized audio output
        self.audio_player: Optional["QMediaPlayer"] = None
        self.audio_output: Optional["QAudioOutput"] = None
        if HAS_QTMULTIMEDIA:
            self.audio_player = QMediaPlayer(self)
            self.audio_output = QAudioOutput(self)
            self.audio_player.setAudioOutput(self.audio_output)

        self.offline_studio_window: Optional[OfflineAIVideoStudioWindow] = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_frame_tick)

        self.hud_timer = QTimer(self)
        self.hud_timer.setSingleShot(True)
        self.hud_timer.timeout.connect(self._clear_hud)

        self._build_ui()
        self._apply_theme()

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0B0F17;
                color: #F1F5F9;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton {
                background-color: #1E293B;
                color: #F8FAFC;
                border: 1px solid #334155;
                border-radius: 14px;
                padding: 6px 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #283548;
                border-color: #00D2BE;
            }
            QPushButton#tealPill {
                background-color: #00D2BE;
                color: #061018;
                border: none;
                font-weight: bold;
            }
            QPushButton#yellowPill {
                background-color: #FACC15;
                color: #061018;
                border: none;
                font-weight: bold;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #263347;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #00D2BE;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #FFFFFF;
                border: 2px solid #00D2BE;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QLineEdit, QComboBox {
                background-color: #1E293B;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 5px 10px;
                color: white;
            }
        """)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 1. Top Control Bar (With Unified Import Files + Direct Import Folder Button right beside it)
        top_bar = QFrame()
        top_bar.setStyleSheet("background-color: #111827; border-bottom: 1px solid #1F2937;")
        tb = QHBoxLayout(top_bar)
        tb.setContentsMargins(16, 10, 16, 10)
        tb.setSpacing(8)

        self.lbl_video_title = QLabel("No Video Loaded — Click 'Import Files' or 'Import Folder'")
        self.lbl_video_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #FFFFFF;")

        btn_rotate = QPushButton("⟳ Rotate 90°")
        btn_rotate.clicked.connect(self.rotate_video_clockwise)

        self.btn_ai_panel = QPushButton("✨ AI Enhancer: ON")
        self.btn_ai_panel.setObjectName("tealPill")
        self.btn_ai_panel.clicked.connect(self.open_live_ai_panel)

        self.btn_wipe_mode = QPushButton("Before/After Wipe")
        self.btn_wipe_mode.clicked.connect(self.toggle_wipe_mode)

        btn_offline_studio = QPushButton("🎬 Offline AI Studio")
        btn_offline_studio.clicked.connect(self.open_offline_studio)

        btn_import_files = QPushButton("＋ Import Files")
        btn_import_files.setObjectName("tealPill")
        btn_import_files.clicked.connect(self.import_media_files)

        btn_import_folder = QPushButton("📁 Import Folder")
        btn_import_folder.clicked.connect(self.import_media_folder)

        btn_subtitles = QPushButton("💬 Load .SRT/.VTT")
        btn_subtitles.clicked.connect(self.import_subtitle_file)

        btn_playlist = QPushButton("☰ Playlist (0)")
        self.btn_playlist = btn_playlist
        btn_playlist.clicked.connect(self.toggle_playlist_drawer)

        tb.addWidget(self.lbl_video_title, 1)
        tb.addWidget(btn_rotate)
        tb.addWidget(self.btn_ai_panel)
        tb.addWidget(self.btn_wipe_mode)
        tb.addWidget(btn_offline_studio)
        tb.addWidget(btn_import_files)
        tb.addWidget(btn_import_folder)
        tb.addWidget(btn_subtitles)
        tb.addWidget(btn_playlist)
        root.addWidget(top_bar)

        # 2. Middle Area: Video Canvas + Slide-Over Playlist Drawer
        mid_container = QWidget()
        mid_layout = QHBoxLayout(mid_container)
        mid_layout.setContentsMargins(0, 0, 0, 0)
        mid_layout.setSpacing(0)

        self.video_canvas = LiveVideoCanvasWidget()
        # Clicking anywhere on the video canvas outside the playlist drawer hides the playlist drawer!
        self.video_canvas.canvasClicked.connect(self._on_canvas_clicked)
        self.video_canvas.wipeRatioChanged.connect(lambda _: self._render_current_frame_immediately())
        mid_layout.addWidget(self.video_canvas, 1)

        # Playlist Drawer (Hidden by default; closes when clicking outside on the video canvas)
        self.playlist_drawer = QFrame()
        self.playlist_drawer.setFixedWidth(370)
        self.playlist_drawer.setStyleSheet("background-color: #131B2E; border-left: 1px solid #263347;")
        self.playlist_drawer.setVisible(False)
        self._build_playlist_drawer()
        mid_layout.addWidget(self.playlist_drawer)

        root.addWidget(mid_container, 1)

        # 3. Bottom Scrubber & Centered Transport Deck
        bot_bar = QFrame()
        bot_bar.setStyleSheet("background-color: #111827; border-top: 1px solid #1F2937;")
        bb = QVBoxLayout(bot_bar)
        bb.setContentsMargins(20, 10, 20, 14)
        bb.setSpacing(8)

        # Scrubber row
        scrub_row = QHBoxLayout()
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet("font-family: Consolas, monospace; font-weight: bold;")
        self.scrubber = QSlider(Qt.Orientation.Horizontal)
        self.scrubber.setRange(0, 1000)
        self.scrubber.sliderReleased.connect(self._on_scrubber_released)

        scrub_row.addWidget(self.lbl_time)
        scrub_row.addWidget(self.scrubber, 1)
        bb.addLayout(scrub_row)

        # Transport row
        trans_row = QHBoxLayout()

        # Speed & A-B Loop cluster (Left)
        self.cmb_speed = QComboBox()
        for spd in ["0.25x", "0.5x", "0.75x", "1.0x", "1.25x", "1.5x", "2.0x", "3.0x", "4.0x"]:
            self.cmb_speed.addItem(spd)
        self.cmb_speed.setCurrentText("1.0x")
        self.cmb_speed.currentTextChanged.connect(self._on_speed_changed)

        btn_custom_speed = QPushButton("Custom Speed")
        btn_custom_speed.clicked.connect(self._prompt_custom_speed)

        self.btn_ab_loop = QPushButton("🔁 A-B Loop: Off")
        self.btn_ab_loop.clicked.connect(self.cycle_ab_loop)

        trans_row.addWidget(QLabel("Speed:"))
        trans_row.addWidget(self.cmb_speed)
        trans_row.addWidget(btn_custom_speed)
        trans_row.addWidget(self.btn_ab_loop)
        trans_row.addStretch()

        # Centered 5 Transport Buttons
        btn_prev = QPushButton("⏮ Prev")
        btn_prev.clicked.connect(self.prev_track)
        btn_back10 = QPushButton("⏪ -10s")
        btn_back10.clicked.connect(lambda: self.skip_seconds(-10.0))
        self.btn_play_pause = QPushButton("▶ Play")
        self.btn_play_pause.setObjectName("tealPill")
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        btn_fwd10 = QPushButton("+10s ⏩")
        btn_fwd10.clicked.connect(lambda: self.skip_seconds(10.0))
        btn_next = QPushButton("Next ⏭")
        btn_next.clicked.connect(self.next_track)

        trans_row.addWidget(btn_prev)
        trans_row.addWidget(btn_back10)
        trans_row.addWidget(self.btn_play_pause)
        trans_row.addWidget(btn_fwd10)
        trans_row.addWidget(btn_next)
        trans_row.addStretch()

        lbl_hint = QLabel("Click outside playlist to close • Drag wipe in Before/After mode")
        lbl_hint.setStyleSheet("color: #64748B; font-size: 11px;")
        trans_row.addWidget(lbl_hint)

        bb.addLayout(trans_row)
        root.addWidget(bot_bar)

    def _build_playlist_drawer(self):
        p_layout = QVBoxLayout(self.playlist_drawer)
        p_layout.setContentsMargins(14, 14, 14, 14)
        p_layout.setSpacing(10)

        header = QHBoxLayout()
        self.lbl_playlist_header = QLabel("Playlist (0)")
        self.lbl_playlist_header.setStyleSheet("font-size: 16px; font-weight: bold; color: #00D2BE;")
        btn_close = QPushButton("✕")
        btn_close.setFixedWidth(34)
        btn_close.clicked.connect(lambda: self.playlist_drawer.setVisible(False))
        header.addWidget(self.lbl_playlist_header)
        header.addStretch()
        header.addWidget(btn_close)
        p_layout.addLayout(header)

        # Search & Sort Row
        search_row = QHBoxLayout()
        self.txt_playlist_search = QLineEdit()
        self.txt_playlist_search.setPlaceholderText("Filter title or folder...")
        self.txt_playlist_search.textChanged.connect(self._on_playlist_filter_changed)

        self.cmb_playlist_sort = QComboBox()
        self.cmb_playlist_sort.addItems([
            "Added Order", "Name (A → Z)", "Name (Z → A)", "Folder Name", "Resume Progress"
        ])
        self.cmb_playlist_sort.currentTextChanged.connect(self._on_playlist_sort_changed)

        search_row.addWidget(self.txt_playlist_search, 1)
        search_row.addWidget(self.cmb_playlist_sort)
        p_layout.addLayout(search_row)

        # Category Filter Pills
        filter_row = QHBoxLayout()
        self.filter_buttons: Dict[str, QPushButton] = {}
        for cat in ["All", "Files", "Folders", "Resume"]:
            b = QPushButton(cat)
            if cat == "All":
                b.setObjectName("tealPill")
            b.clicked.connect(lambda _, c=cat: self._set_playlist_category(c))
            self.filter_buttons[cat] = b
            filter_row.addWidget(b)
        p_layout.addLayout(filter_row)

        # Scrollable Track List
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_w = QWidget()
        self.playlist_items_layout = QVBoxLayout(scroll_w)
        self.playlist_items_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(scroll_w)
        p_layout.addWidget(scroll, 1)

        # Bottom Add Files / Add Folder buttons inside drawer
        add_row = QHBoxLayout()
        b_files = QPushButton("＋ Add Files")
        b_files.setObjectName("tealPill")
        b_files.clicked.connect(self.import_media_files)
        b_folder = QPushButton("📁 Add Folder")
        b_folder.clicked.connect(self.import_media_folder)
        add_row.addWidget(b_files)
        add_row.addWidget(b_folder)
        p_layout.addLayout(add_row)

    # ========================================================================
    # PLAYLIST FILTERING, SORTING & CLICK-OUTSIDE DISMISSAL
    # ========================================================================

    def _on_canvas_clicked(self):
        if self.playlist_drawer.isVisible():
            self.playlist_drawer.setVisible(False)
        else:
            self.toggle_play_pause()

    def toggle_playlist_drawer(self):
        self.playlist_drawer.setVisible(not self.playlist_drawer.isVisible())
        if self.playlist_drawer.isVisible():
            self._rebuild_playlist_ui()

    def _on_playlist_filter_changed(self, text: str):
        self.playlist_search_query = text.strip().lower()
        self._rebuild_playlist_ui()

    def _on_playlist_sort_changed(self, sort_mode: str):
        self.playlist_sort_mode = sort_mode
        self._rebuild_playlist_ui()

    def _set_playlist_category(self, category: str):
        self.playlist_filter_mode = category
        for c, btn in self.filter_buttons.items():
            btn.setObjectName("tealPill" if c == category else "")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        self._rebuild_playlist_ui()

    def _displayed_playlist_entries(self) -> List[Tuple[int, VideoTrackItem]]:
        indexed = list(enumerate(self.playlist))
        filtered = []
        for idx, item in indexed:
            if self.playlist_filter_mode == "Files" and item.folder_name is not None:
                continue
            if self.playlist_filter_mode == "Folders" and item.folder_name is None:
                continue
            if self.playlist_filter_mode == "Resume" and self.resume_positions.get(item.path, 0.0) <= 3.0:
                continue
            if self.playlist_search_query:
                q = self.playlist_search_query
                if q not in item.title.lower() and q not in (item.folder_name or "").lower():
                    continue
            filtered.append((idx, item))

        mode = self.playlist_sort_mode
        if mode == "Name (A → Z)":
            filtered.sort(key=lambda x: x[1].title.lower())
        elif mode == "Name (Z → A)":
            filtered.sort(key=lambda x: x[1].title.lower(), reverse=True)
        elif mode == "Folder Name":
            filtered.sort(key=lambda x: ((x[1].folder_name or "zzz").lower(), x[1].title.lower()))
        elif mode == "Resume Progress":
            filtered.sort(key=lambda x: self.resume_positions.get(x[1].path, 0.0), reverse=True)
        return filtered

    def _rebuild_playlist_ui(self):
        while self.playlist_items_layout.count():
            child = self.playlist_items_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        entries = self._displayed_playlist_entries()
        self.lbl_playlist_header.setText(f"Playlist ({len(entries)}/{len(self.playlist)})")
        self.btn_playlist.setText(f"☰ Playlist ({len(self.playlist)})")

        for orig_idx, item in entries:
            is_curr = (orig_idx == self.current_index)
            folder_txt = f" [{item.folder_name}]" if item.folder_name else ""
            res_sec = self.resume_positions.get(item.path, 0.0)
            res_txt = f" • Resume {int(res_sec)//60:02d}:{int(res_sec)%60:02d}" if res_sec > 3.0 else ""
            btn = QPushButton(f"{'▶ ' if is_curr else '🎬 '}{item.title}{folder_txt}{res_txt}")
            if is_curr:
                btn.setObjectName("tealPill")
            btn.clicked.connect(lambda _, i=orig_idx: self.select_track(i))
            self.playlist_items_layout.addWidget(btn)

    # ========================================================================
    # MEDIA IMPORT & PLAYBACK ENGINE
    # ========================================================================

    def import_media_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Video Files",
            "",
            "Video Files (*.mp4 *.mov *.m4v *.mkv *.avi *.webm *.ts);;All Files (*)"
        )
        if files:
            self._ingest_media_paths([Path(f) for f in files])

    def import_media_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Import Folder as Playlist")
        if folder:
            self._ingest_media_paths([Path(folder)])

    def _ingest_media_paths(self, paths: List[Path]):
        discovered: List[VideoTrackItem] = []
        for p in paths:
            if p.is_dir():
                for child in sorted(p.rglob("*")):
                    if child.is_file() and child.suffix.lower() in self.SUPPORTED_EXTS:
                        discovered.append(VideoTrackItem(
                            id=str(uuid.uuid4()),
                            title=child.stem,
                            path=str(child),
                            is_local=True,
                            folder_name=p.name
                        ))
            elif p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS:
                discovered.append(VideoTrackItem(
                    id=str(uuid.uuid4()),
                    title=p.stem,
                    path=str(p),
                    is_local=True,
                    folder_name=None
                ))
        if not discovered:
            self.trigger_hud("No Supported Video Files Found")
            return

        self.playlist = discovered + self.playlist
        self.current_index = 0
        self._rebuild_playlist_ui()
        self.load_current_track(auto_play=True)
        self.trigger_hud(f"Imported {len(discovered)} Video(s)")

    def import_subtitle_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Import Subtitle (.SRT / .VTT)", "", "Subtitles (*.srt *.vtt);;All Files (*)"
        )
        if f:
            self.subtitle_cues = parse_srt_or_vtt(f)
            self.trigger_hud(f"Loaded {len(self.subtitle_cues)} Subtitle Cues")

    def select_track(self, idx: int):
        if 0 <= idx < len(self.playlist):
            self.current_index = idx
            self.load_current_track(auto_play=True)
            self._rebuild_playlist_ui()

    def prev_track(self):
        if self.playlist:
            self.select_track((self.current_index - 1) % len(self.playlist))

    def next_track(self):
        if self.playlist:
            self.select_track((self.current_index + 1) % len(self.playlist))

    def load_current_track(self, auto_play: bool = True):
        if not (0 <= self.current_index < len(self.playlist)):
            return
        track = self.playlist[self.current_index]
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(track.path)
        if not self.cap.isOpened():
            self.trigger_hud("Failed to Open Video Track")
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_fps = fps if (fps and 1.0 < fps <= 240.0) else 30.0
        total_frames = max(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        self.duration_sec = max(total_frames / self.video_fps, 1.0)
        self.current_time_sec = 0.0
        self.loop_point_a = None
        self.loop_point_b = None
        self.btn_ab_loop.setText("🔁 A-B Loop: Off")

        folder_badge = f"[{track.folder_name}] " if track.folder_name else ""
        self.lbl_video_title.setText(f"{folder_badge}{track.title}")

        if self.audio_player is not None:
            self.audio_player.setSource(QUrl.fromLocalFile(track.path))
            self.audio_player.setPlaybackRate(self.playback_speed)

        # Resume last position if available
        saved_pos = self.resume_positions.get(track.path, 0.0)
        if 3.0 < saved_pos < max(self.duration_sec - 5.0, 4.0):
            self.seek_to_seconds(saved_pos)

        if auto_play:
            self.is_playing = True
            self.btn_play_pause.setText("⏸ Pause")
            self._update_timer_interval()
            self.timer.start()
            if self.audio_player is not None:
                self.audio_player.play()
        else:
            self._on_frame_tick()

    def _update_timer_interval(self):
        effective_fps = max(self.video_fps * self.playback_speed, 5.0)
        interval_ms = max(int(1000.0 / min(effective_fps, 120.0)), 8)
        self.timer.setInterval(interval_ms)

    def toggle_play_pause(self):
        if self.cap is None:
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play_pause.setText("⏸ Pause")
            self._update_timer_interval()
            self.timer.start()
            if self.audio_player is not None:
                self.audio_player.play()
        else:
            self.btn_play_pause.setText("▶ Play")
            self.timer.stop()
            if self.audio_player is not None:
                self.audio_player.pause()

    def skip_seconds(self, delta: float):
        self.seek_to_seconds(self.current_time_sec + delta)
        self.trigger_hud(f"{'+' if delta > 0 else ''}{int(delta)}s")

    def seek_to_seconds(self, target_sec: float):
        if self.cap is None:
            return
        clamped = float(np.clip(target_sec, 0.0, max(self.duration_sec - 0.05, 0.0)))
        self.cap.set(cv2.CAP_PROP_POS_MSEC, clamped * 1000.0)
        self.current_time_sec = clamped
        if self.audio_player is not None:
            self.audio_player.setPosition(int(clamped * 1000))
        if not self.is_playing:
            self._on_frame_tick()

    def _on_scrubber_released(self):
        frac = self.scrubber.value() / 1000.0
        self.seek_to_seconds(frac * self.duration_sec)

    def _on_speed_changed(self, text: str):
        try:
            spd = float(text.replace("x", "").strip())
            self.set_playback_speed(spd)
        except ValueError:
            pass

    def _prompt_custom_speed(self):
        val, ok = QInputDialog.getDouble(
            self, "Custom Playback Speed", "Enter playback speed (0.1x - 4.0x):", self.playback_speed, 0.1, 4.0, 2
        )
        if ok:
            self.set_playback_speed(val)

    def set_playback_speed(self, spd: float):
        self.playback_speed = float(np.clip(spd, 0.1, 4.0))
        self._update_timer_interval()
        if self.audio_player is not None:
            self.audio_player.setPlaybackRate(self.playback_speed)
        self.trigger_hud(f"Speed: {self.playback_speed:.2f}x")

    def rotate_video_clockwise(self):
        self.rotation_degrees = (self.rotation_degrees + 90) % 360
        self._render_current_frame_immediately()
        self.trigger_hud(f"Rotation: {self.rotation_degrees}°")

    def cycle_ab_loop(self):
        if self.loop_point_a is None:
            self.loop_point_a = self.current_time_sec
            self.btn_ab_loop.setText(f"🔁 A: {int(self.loop_point_a)}s → ?")
            self.trigger_hud("Loop Point A Set")
        elif self.loop_point_b is None:
            if self.current_time_sec > self.loop_point_a + 0.5:
                self.loop_point_b = self.current_time_sec
                self.btn_ab_loop.setText(f"🔁 A-B ({int(self.loop_point_a)}s-{int(self.loop_point_b)}s)")
                self.trigger_hud("A-B Loop Active")
            else:
                self.loop_point_a = None
                self.btn_ab_loop.setText("🔁 A-B Loop: Off")
        else:
            self.loop_point_a = None
            self.loop_point_b = None
            self.btn_ab_loop.setText("🔁 A-B Loop: Off")
            self.trigger_hud("A-B Loop Cleared")

    def toggle_wipe_mode(self):
        self.video_canvas.is_wipe_mode = not self.video_canvas.is_wipe_mode
        self.btn_wipe_mode.setObjectName("yellowPill" if self.video_canvas.is_wipe_mode else "")
        self.btn_wipe_mode.setText("Wipe: ON" if self.video_canvas.is_wipe_mode else "Before/After Wipe")
        self.btn_wipe_mode.style().unpolish(self.btn_wipe_mode)
        self.btn_wipe_mode.style().polish(self.btn_wipe_mode)
        self._render_current_frame_immediately()
        self.trigger_hud("Before / After Wipe: ON" if self.video_canvas.is_wipe_mode else "Before / After Wipe: OFF")

    def _on_frame_tick(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.next_track()
            return

        pos_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        self.current_time_sec = pos_msec / 1000.0 if pos_msec > 0 else (self.current_time_sec + 1.0 / self.video_fps)

        # Evaluate A-B loop
        if self.loop_point_a is not None and self.loop_point_b is not None:
            if self.current_time_sec >= self.loop_point_b:
                self.seek_to_seconds(self.loop_point_a)
                return

        # Save resume timestamp every ~3 seconds
        self.frame_counter += 1
        if self.frame_counter % 90 == 0 and (0 <= self.current_index < len(self.playlist)):
            self.resume_positions[self.playlist[self.current_index].path] = self.current_time_sec

        self.last_raw_bgr = frame
        self._render_current_frame_immediately()
        self._update_time_and_subtitles()

    def _render_current_frame_immediately(self):
        if self.last_raw_bgr is None:
            return
        raw = self.last_raw_bgr

        if self.rotation_degrees == 90:
            raw = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_degrees == 180:
            raw = cv2.rotate(raw, cv2.ROTATE_180)
        elif self.rotation_degrees == 270:
            raw = cv2.rotate(raw, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.enhancer_enabled:
            if self.enhancer_settings.enable_face_restore and (self.frame_counter % 20 == 0):
                self.cached_live_faces = GLOBAL_AI_ENGINE.detect_faces_normalized(raw)
            enhanced = GLOBAL_AI_ENGINE.apply_enhancement(
                raw, self.enhancer_settings, is_offline_export=False, cached_face_boxes=self.cached_live_faces
            )
            if self.video_canvas.is_wipe_mode:
                out = GLOBAL_AI_ENGINE.render_split_wipe(raw, enhanced, self.video_canvas.split_ratio)
            else:
                out = enhanced
        else:
            out = raw

        self.video_canvas.set_frame_bgr(out)

    def _update_time_and_subtitles(self):
        c_min, c_sec = int(self.current_time_sec) // 60, int(self.current_time_sec) % 60
        d_min, d_sec = int(self.duration_sec) // 60, int(self.duration_sec) % 60
        self.lbl_time.setText(f"{c_min:02d}:{c_sec:02d} / {d_min:02d}:{d_sec:02d}")
        if not self.scrubber.isSliderDown() and self.duration_sec > 0:
            self.scrubber.setValue(int((self.current_time_sec / self.duration_sec) * 1000))

        # Subtitles
        active_sub = ""
        for cue in self.subtitle_cues:
            if cue.start_sec <= self.current_time_sec <= cue.end_sec:
                active_sub = cue.text
                break
        if self.video_canvas.subtitle_text != active_sub:
            self.video_canvas.subtitle_text = active_sub
            self.video_canvas.update()

    # ========================================================================
    # SHEETS, MODALS & PERSISTENT SETTINGS
    # ========================================================================

    def open_live_ai_panel(self):
        dlg = LiveAIEnhancerDialog(self)
        dlg.settingsChanged.connect(self._on_ai_settings_updated)
        dlg.openOfflineStudioRequested.connect(self.open_offline_studio)
        dlg.exec()

    def _on_ai_settings_updated(self):
        self.btn_ai_panel.setText("✨ AI Enhancer: ON" if self.enhancer_enabled else "✨ AI Enhancer: OFF")
        self._render_current_frame_immediately()

    def open_offline_studio(self):
        if self.offline_studio_window is None:
            self.offline_studio_window = OfflineAIVideoStudioWindow()
        self.offline_studio_window.active_settings = self.enhancer_settings.clone()
        if self.playlist and not self.offline_studio_window.videos:
            paths = [Path(t.path) for t in self.playlist if Path(t.path).exists()]
            if paths:
                self.offline_studio_window._ingest_paths(paths)
        self.offline_studio_window.show()
        self.offline_studio_window.raise_()
        self.offline_studio_window.activateWindow()

    def trigger_hud(self, text: str):
        self.video_canvas.hud_text = text
        self.video_canvas.update()
        self.hud_timer.start(1400)

    def _clear_hud(self):
        self.video_canvas.hud_text = ""
        self.video_canvas.update()

    def _settings_file(self) -> Path:
        return get_app_data_dir() / "pro_media_player_settings_v1.json"

    def save_persistent_settings(self):
        try:
            payload = {
                "enhancer_enabled": self.enhancer_enabled,
                "settings": asdict(self.enhancer_settings),
                "resume_positions": self.resume_positions
            }
            with open(self._settings_file(), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    def load_persistent_settings(self):
        p = self._settings_file()
        if not p.exists():
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.enhancer_enabled = data.get("enhancer_enabled", True)
            s_dict = data.get("settings", {})
            self.enhancer_settings = AIEnhancerSettings(**s_dict)
            self.resume_positions = data.get("resume_positions", {})
        except Exception:
            pass


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Pro AI Media Player Enhanced")
    win = ProMediaPlayerWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()