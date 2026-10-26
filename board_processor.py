"""
board_processor.py
--------------------
Ports the VirtualBoard engine from main.py into a streamlit-webrtc
VideoProcessor so it can run entirely in the browser (via WebRTC) instead
of a native cv2.imshow() window.

Everything that made the original app work is preserved 1:1:
  - HandTracker (MediaPipe) for detection + 21 landmarks
  - fingers_up() -> gesture classification -> SELECT / DRAW / IDLE
  - One Euro Filter + Catmull-Rom smoothing for marker-like ink
  - Undo stack, taper, jump guard, in-frame toolbar with hover-to-select
  - VoiceController running in the background, polled once per frame

New in this version: a HandwritingRecognizer runs on its own background
thread and turns the ink on the canvas into real text (see
handwriting_recognizer.py). streamlit_app.py polls `recognized_text` each
rerun and renders it in the sidebar with a live status + download button,
exactly the same debounced "recognize a moment after you stop writing"
behavior as the desktop app in main.py.

The only structural difference from a native app: instead of a
`while True` cv2 loop, the same per-frame logic runs inside `recv()`,
which streamlit-webrtc calls for every incoming browser video frame.
Streamlit sidebar widgets talk to a running processor instance through
plain attribute writes guarded by a lock (the pattern streamlit-webrtc
recommends for two-way control).
"""

import math
import os
import threading
import time
from collections import deque

import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase

from hand_tracker import HandTracker
from handwriting_recognizer import HandwritingRecognizer
from smoothing import OneEuroFilter, catmull_rom_segment
from voice_control import VoiceController

TOOLBAR_HEIGHT = 90
DEFAULT_BRUSH = 6
ERASER_BRUSH = 44
SELECTION_DEBOUNCE = 12

ONE_EURO_MIN_CUTOFF = 0.8
ONE_EURO_BETA = 0.4
ONE_EURO_D_CUTOFF = 1.0

CURVE_STEPS = 8
TAPER_FRAMES = 6
TAPER_MIN_FRACTION = 0.45
MAX_JUMP_PX = 120
GESTURE_STABLE_FRAMES = 3
MAX_UNDO_STEPS = 20

RECOGNIZE_DEBOUNCE_SEC = 0.9  # pen idle time before a live handwriting-to-text pass fires

WHITE = (255, 255, 255)
TOOLBAR_BG = (40, 40, 40)

PREVIEW_W, PREVIEW_H = 200, 112
PREVIEW_MARGIN = 14

# name, BGR color (None => eraser, or a string tag for action buttons)
TOOLS = [
    ("Red", (60, 60, 230)),
    ("Green", (60, 170, 60)),
    ("Blue", (220, 110, 40)),
    ("Yellow", (30, 200, 240)),
    ("Purple", (170, 60, 150)),
    ("Black", (35, 35, 35)),
    ("Eraser", None),
    ("Undo", "UNDO"),
    ("Clear", "CLEAR"),
    ("Save", "SAVE"),
    ("Text", "TEXT"),
]

OUTPUT_DIR = "saved_boards"
TEXT_OUTPUT_DIR = "recognized_text"


class BoardProcessor(VideoProcessorBase):
    """One instance of this lives for the lifetime of a browser session's
    webrtc connection. Streamlit sidebar controls mutate its public
    attributes (guarded by self.lock); recv() reads them each frame."""

    def __init__(self):
        self.lock = threading.Lock()

        self.tracker = HandTracker(max_hands=1)
        self.canvas = None  # allocated lazily once we know the frame size

        self.draw_color = TOOLS[0][1]
        self.brush_thickness = DEFAULT_BRUSH
        self.eraser_mode = False
        self.show_preview = True
        self.show_toolbar = True

        self.prev_x, self.prev_y = 0, 0
        self.filter_x = OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
        self.filter_y = OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
        self.point_history = deque(maxlen=4)
        self._stroke_point_count = 0
        self.action_cooldown = 0
        self._stroke_started = False

        self._pending_mode = "IDLE"
        self._pending_count = 0
        self.current_mode = "IDLE"

        self.undo_stack = []
        self.slots = []

        # one-shot action requests set by sidebar buttons, consumed in recv()
        self.request_clear = False
        self.request_undo = False
        self.request_save = False
        self.last_saved_path = None

        # status surfaced back to the Streamlit sidebar
        self.status_mode = "IDLE"
        self.status_fps = 0.0
        self._prev_time = 0.0

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(TEXT_OUTPUT_DIR, exist_ok=True)

        self.voice = VoiceController()
        self.voice_ok = self.voice.start()

        # ---- Handwriting -> Text -------------------------------------
        # Same background-threaded engine as the desktop app: model load
        # and every inference call happen off the WebRTC callback thread
        # so incoming video frames never stall behind a model forward pass.
        self.recognizer = HandwritingRecognizer()
        self.recognizer.start()
        self.text_mode = False
        self.live_text_mode = True
        self.canvas_dirty = False
        self.last_activity_time = 0.0
        self.recognized_text = ""
        self.request_recognize = False  # one-shot "recognize now" from the sidebar

    # ------------------------------------------------------------------
    # Toolbar layout / drawing (mirrors main.py's VirtualBoard)
    # ------------------------------------------------------------------
    def _build_toolbar_geometry(self, width):
        n = len(TOOLS)
        slot_w = width // n
        self.slots = []
        for i, (name, value) in enumerate(TOOLS):
            x1, x2 = i * slot_w, (i + 1) * slot_w
            self.slots.append({"name": name, "value": value, "x1": x1, "x2": x2})

    def _tool_at(self, x, y):
        if y > TOOLBAR_HEIGHT:
            return None
        for slot in self.slots:
            if slot["x1"] <= x < slot["x2"]:
                return slot
        return None

    def _draw_toolbar(self, board, hover_slot=None):
        w = board.shape[1]
        cv2.rectangle(board, (0, 0), (w, TOOLBAR_HEIGHT), TOOLBAR_BG, -1)
        cv2.line(board, (0, TOOLBAR_HEIGHT), (w, TOOLBAR_HEIGHT), (15, 15, 15), 2)

        for slot in self.slots:
            cx = (slot["x1"] + slot["x2"]) // 2
            cy = TOOLBAR_HEIGHT // 2 - 6
            is_hover = hover_slot is slot

            if slot["value"] in ("CLEAR", "SAVE", "UNDO"):
                self._icon_button(board, cx, cy, slot["name"], (70, 70, 70), is_hover)
            elif slot["value"] == "TEXT":
                color = (0, 130, 60) if self.text_mode else (70, 70, 70)
                self._icon_button(board, cx, cy, "Text", color, is_hover)
            elif slot["name"] == "Eraser":
                r = 20 if not is_hover else 23
                cv2.circle(board, (cx, cy), r, WHITE, -1, cv2.LINE_AA)
                cv2.circle(board, (cx, cy), r, (90, 90, 90), 2, cv2.LINE_AA)
                if self.eraser_mode:
                    cv2.circle(board, (cx, cy), r + 4, (0, 220, 255), 3, cv2.LINE_AA)
                cv2.putText(board, "Eraser", (cx - 24, cy + 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
            else:
                color = slot["value"]
                r = 20 if not is_hover else 23
                cv2.circle(board, (cx, cy), r, color, -1, cv2.LINE_AA)
                cv2.circle(board, (cx, cy), r, (15, 15, 15), 1, cv2.LINE_AA)
                if color == self.draw_color and not self.eraser_mode:
                    cv2.circle(board, (cx, cy), r + 4, (0, 220, 255), 3, cv2.LINE_AA)
                cv2.putText(board, slot["name"], (cx - len(slot["name"]) * 3, cy + 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

    @staticmethod
    def _icon_button(board, cx, cy, label, base_color, is_hover):
        r = 20 if not is_hover else 23
        cv2.circle(board, (cx, cy), r, base_color, -1, cv2.LINE_AA)
        cv2.circle(board, (cx, cy), r, (110, 110, 110), 1, cv2.LINE_AA)
        cv2.putText(board, label[0], (cx - 5, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Gesture -> mode classification
    # ------------------------------------------------------------------
    @staticmethod
    def _classify(fingers):
        thumb, index, middle, ring, pinky = fingers
        if index == 1 and middle == 1 and ring == 0 and pinky == 0:
            return "SELECT"
        if index == 1 and middle == 0 and ring == 0 and pinky == 0:
            return "DRAW"
        return "IDLE"

    def _stable_mode(self, raw_mode):
        if raw_mode == self._pending_mode:
            self._pending_count += 1
        else:
            self._pending_mode = raw_mode
            self._pending_count = 1
        if self._pending_count >= GESTURE_STABLE_FRAMES:
            self.current_mode = raw_mode
        return self.current_mode

    # ------------------------------------------------------------------
    # Undo
    # ------------------------------------------------------------------
    def _push_undo(self):
        self.undo_stack.append(self.canvas.copy())
        if len(self.undo_stack) > MAX_UNDO_STEPS:
            self.undo_stack.pop(0)

    def _undo(self):
        if self.undo_stack:
            self.canvas = self.undo_stack.pop()
            self._mark_dirty()

    # ------------------------------------------------------------------
    # Handwriting -> Text bookkeeping (identical logic to main.py)
    # ------------------------------------------------------------------
    def _mark_dirty(self, clear_text=False):
        self.canvas_dirty = True
        self.last_activity_time = time.time()
        if clear_text:
            self.recognized_text = ""

    def _maybe_trigger_recognition(self):
        if not self.live_text_mode or not self.canvas_dirty:
            return
        if time.time() - self.last_activity_time < RECOGNIZE_DEBOUNCE_SEC:
            return
        if self.recognizer.is_busy():
            return
        self.recognizer.request(self.canvas)
        self.canvas_dirty = False

    def _force_recognition_now(self):
        self.canvas_dirty = True
        self.last_activity_time = 0.0

    def _export_text(self):
        text = self.recognized_text.strip()
        if not text:
            return None
        filename = os.path.join(TEXT_OUTPUT_DIR, f"text_{int(time.time())}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        return filename

    # ------------------------------------------------------------------
    # Drawing engine (identical math to main.py)
    # ------------------------------------------------------------------
    def _current_color(self):
        return WHITE if self.eraser_mode else self.draw_color

    def _current_thickness(self):
        base = ERASER_BRUSH if self.eraser_mode else self.brush_thickness
        ramp = min(1.0, self._stroke_point_count / TAPER_FRAMES)
        fraction = TAPER_MIN_FRACTION + (1 - TAPER_MIN_FRACTION) * ramp
        return max(2, int(base * fraction))

    def _begin_stroke_segment(self, fx, fy):
        if not self._stroke_started:
            self._push_undo()
            self._stroke_started = True
        self.point_history.clear()
        self.point_history.append((fx, fy))
        self._stroke_point_count = 1
        cv2.circle(self.canvas, (int(fx), int(fy)), self._current_thickness() // 2,
                   self._current_color(), -1, cv2.LINE_AA)
        self.prev_x, self.prev_y = fx, fy
        self._mark_dirty()

    def _draw_stroke(self, x, y, timestamp):
        color = self._current_color()
        fx = self.filter_x.filter(x, timestamp)
        fy = self.filter_y.filter(y, timestamp)

        if self.prev_x == 0 and self.prev_y == 0:
            self._begin_stroke_segment(fx, fy)
            return

        jump = math.hypot(fx - self.prev_x, fy - self.prev_y)
        if jump > MAX_JUMP_PX:
            self.filter_x.reset()
            self.filter_y.reset()
            self._begin_stroke_segment(fx, fy)
            return

        self._stroke_point_count += 1
        thickness = self._current_thickness()
        self.point_history.append((fx, fy))

        if len(self.point_history) == 4:
            p0, p1, p2, p3 = self.point_history
            curve_points = [p1] + catmull_rom_segment(p0, p1, p2, p3, CURVE_STEPS)
            for i in range(len(curve_points) - 1):
                a = (int(curve_points[i][0]), int(curve_points[i][1]))
                b = (int(curve_points[i + 1][0]), int(curve_points[i + 1][1]))
                cv2.line(self.canvas, a, b, color, thickness, cv2.LINE_AA)
                cv2.circle(self.canvas, b, thickness // 2, color, -1, cv2.LINE_AA)
        else:
            a = (int(self.prev_x), int(self.prev_y))
            b = (int(fx), int(fy))
            cv2.line(self.canvas, a, b, color, thickness, cv2.LINE_AA)
            cv2.circle(self.canvas, b, thickness // 2, color, -1, cv2.LINE_AA)

        self.prev_x, self.prev_y = fx, fy
        self._mark_dirty()

    def _reset_stroke(self):
        self.prev_x, self.prev_y = 0, 0
        self.filter_x.reset()
        self.filter_y.reset()
        self.point_history.clear()
        self._stroke_point_count = 0
        self._stroke_started = False

    def _apply_selection(self, x, y):
        self._reset_stroke()
        slot = self._tool_at(x, y)
        if not slot:
            self.action_cooldown = max(0, self.action_cooldown - 1)
            return slot

        if slot["value"] == "CLEAR":
            if self.action_cooldown == 0:
                self._push_undo()
                self.canvas[:] = 255
                self._mark_dirty(clear_text=True)
                self.action_cooldown = SELECTION_DEBOUNCE
        elif slot["value"] == "SAVE":
            if self.action_cooldown == 0:
                self._save_canvas()
                self.action_cooldown = SELECTION_DEBOUNCE
        elif slot["value"] == "UNDO":
            if self.action_cooldown == 0:
                self._undo()
                self.action_cooldown = SELECTION_DEBOUNCE
        elif slot["value"] == "TEXT":
            if self.action_cooldown == 0:
                self.text_mode = not self.text_mode
                self.action_cooldown = SELECTION_DEBOUNCE
        elif slot["name"] == "Eraser":
            self.eraser_mode = True
        else:
            self.eraser_mode = False
            self.draw_color = slot["value"]

        if self.action_cooldown > 0:
            self.action_cooldown -= 1
        return slot

    def _apply_voice_commands(self):
        while True:
            command = self.voice.poll()
            if command is None:
                break
            kind, value = command
            if kind == "color":
                self.eraser_mode = False
                self.draw_color = value
            elif kind == "tool":
                self.eraser_mode = (value == "eraser")
            elif kind == "brush":
                delta = 4 * value
                self.brush_thickness = max(2, min(40, self.brush_thickness + delta))
            elif kind == "action":
                if value == "clear":
                    self._push_undo()
                    self.canvas[:] = 255
                    self._mark_dirty(clear_text=True)
                elif value == "save":
                    self._save_canvas()
                elif value == "undo":
                    self._undo()
                elif value == "mute":
                    self.voice.stop()
                elif value == "recognize_text":
                    self.text_mode = True
                    self._force_recognition_now()
                elif value == "show_text":
                    self.text_mode = True
                elif value == "hide_text":
                    self.text_mode = False
                elif value == "export_text":
                    self._export_text()

    def _save_canvas(self):
        filename = os.path.join(OUTPUT_DIR, f"board_{int(time.time())}.png")
        cv2.imwrite(filename, self.canvas)
        self.last_saved_path = filename
        return filename

    def _draw_preview(self, board, tracked_frame, mode):
        if not self.show_preview:
            return
        thumb = cv2.resize(tracked_frame, (PREVIEW_W, PREVIEW_H))
        w = board.shape[1]
        x0 = w - PREVIEW_W - PREVIEW_MARGIN
        y0 = TOOLBAR_HEIGHT + PREVIEW_MARGIN
        board[y0:y0 + PREVIEW_H, x0:x0 + PREVIEW_W] = thumb
        cv2.rectangle(board, (x0, y0), (x0 + PREVIEW_W, y0 + PREVIEW_H), (120, 120, 120), 2)
        cv2.putText(board, f"tracking: {mode}", (x0, y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Handwriting -> Text panel (in-frame, for gesture/keyboard-free parity
    # with main.py; the Streamlit sidebar also shows the same text)
    # ------------------------------------------------------------------
    def text_status(self):
        """Public one-word(ish) status string - used both by the in-frame
        panel and the Streamlit sidebar."""
        if self.recognizer.load_error:
            return "unavailable"
        if not self.recognizer.ready:
            return "loading model..."
        if self.recognizer.is_busy():
            return "reading..."
        return "live" if self.live_text_mode else "paused"

    def _draw_text_panel(self, board):
        if not self.text_mode:
            return
        h, w = board.shape[:2]
        panel_w = int(w * 0.42)
        x0 = w - panel_w
        y0 = TOOLBAR_HEIGHT if self.show_toolbar else 0

        overlay = board.copy()
        cv2.rectangle(overlay, (x0, y0), (w, h), (250, 250, 245), -1)
        cv2.addWeighted(overlay, 0.94, board, 0.06, 0, board)
        cv2.rectangle(board, (x0, y0), (w, h), (180, 180, 170), 2)

        header = f"Recognized Text  [{self.text_status()}]"
        cv2.putText(board, header, (x0 + 14, y0 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.line(board, (x0 + 14, y0 + 38), (w - 14, y0 + 38), (200, 200, 195), 1)

        if self.recognizer.load_error:
            self._draw_wrapped_text(board, self.recognizer.load_error, x0 + 14, y0 + 64,
                                     panel_w - 28, (150, 60, 60))
            return

        text = self.recognized_text or "(start writing - text appears here automatically)"
        self._draw_wrapped_text(board, text, x0 + 14, y0 + 64, panel_w - 28, (35, 35, 35))

    @staticmethod
    def _draw_wrapped_text(board, text, x, y, max_width, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness, line_h = 0.56, 1, 24
        cy = y
        for raw_line in text.split("\n"):
            words = raw_line.split(" ") if raw_line else [""]
            line = ""
            for word in words:
                trial = (line + " " + word).strip()
                (tw, _th), _ = cv2.getTextSize(trial, font, scale, thickness)
                if tw > max_width and line:
                    cv2.putText(board, line, (x, cy), font, scale, color, thickness, cv2.LINE_AA)
                    cy += line_h
                    line = word
                else:
                    line = trial
            cv2.putText(board, line, (x, cy), font, scale, color, thickness, cv2.LINE_AA)
            cy += line_h
            if cy > board.shape[0] - 16:
                break

    # ------------------------------------------------------------------
    # streamlit-webrtc entry point: one call per incoming browser frame
    # ------------------------------------------------------------------
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # mirror for natural interaction
        h, w = img.shape[:2]

        if self.canvas is None:
            self.canvas = np.full((h, w, 3), 255, np.uint8)
            self._build_toolbar_geometry(w)

        with self.lock:
            self._apply_voice_commands()

            # one-shot sidebar-triggered actions
            if self.request_clear:
                self._push_undo()
                self.canvas[:] = 255
                self._mark_dirty(clear_text=True)
                self.request_clear = False
            if self.request_undo:
                self._undo()
                self.request_undo = False
            if self.request_save:
                self._save_canvas()
                self.request_save = False
            if self.request_recognize:
                self._force_recognition_now()
                self.request_recognize = False

            tracked_frame = self.tracker.find_hands(img, draw=True)
            landmarks = self.tracker.find_positions(tracked_frame)
            mode = "IDLE"
            hover_slot = None

            if landmarks:
                fingers = self.tracker.fingers_up()
                raw_mode = self._classify(fingers)
                mode = self._stable_mode(raw_mode)
                x, y = landmarks[8][1], landmarks[8][2]

                if mode == "SELECT":
                    hover_slot = self._apply_selection(x, y)
                elif mode == "DRAW":
                    self._draw_stroke(x, y, time.time())
                else:
                    self._reset_stroke()
            else:
                self._reset_stroke()
                mode = self._stable_mode("IDLE")

            # Handwriting -> Text: fire a background pass once the pen has
            # been idle a moment, and pick up whatever the worker thread
            # most recently finished (never blocks this callback).
            self._maybe_trigger_recognition()
            result = self.recognizer.latest_result()
            if result is not None:
                self.recognized_text = result.text

            board = self.canvas.copy()

            if landmarks:
                x, y = landmarks[8][1], landmarks[8][2]
                if mode == "SELECT":
                    cv2.circle(board, (x, y), 14, (0, 150, 255), 2, cv2.LINE_AA)
                    cv2.circle(board, (x, y), 3, (0, 150, 255), -1, cv2.LINE_AA)
                elif mode == "DRAW":
                    cursor_color = (160, 160, 160) if self.eraser_mode else self.draw_color
                    cv2.circle(board, (x, y), self.brush_thickness // 2 + 3, cursor_color, 2, cv2.LINE_AA)

            if self.show_toolbar:
                self._draw_toolbar(board, hover_slot=hover_slot)
            self._draw_preview(board, tracked_frame, mode)
            self._draw_text_panel(board)

            now = time.time()
            self.status_fps = 1 / (now - self._prev_time) if self._prev_time else 0
            self._prev_time = now
            self.status_mode = mode

            voice_tag = f"Voice: ON ('{self.voice.last_heard}')" if self.voice.enabled else "Voice: OFF"
            status = (f"Mode: {mode} | {'Eraser' if self.eraser_mode else 'Color'} "
                      f"| Brush: {self.brush_thickness}px | FPS: {int(self.status_fps)} | {voice_tag}")
            cv2.putText(board, status, (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(board, format="bgr24")