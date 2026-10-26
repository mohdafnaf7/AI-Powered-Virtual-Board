"""
main.py — AI-Powered Virtual Board (now with live Handwriting -> Text)
------------------------------------------------------------------------
Touchless whiteboard controlled entirely by hand gestures, captured
through a standard webcam. No special hardware required.

The board itself is a clean white canvas — like a real whiteboard — not
a drawing overlaid on your live camera feed. Your webcam only powers hand
tracking; a small preview thumbnail in the corner shows your hand and
skeleton so you can see what the system is tracking, but it never clutters
the writing surface.

Pipeline (matches the project architecture):
  1. Webcam Input        -> cv2.VideoCapture (tracking only, not shown full-screen)
  2. Hand Detection       -> HandTracker (MediaPipe)
  3. Feature Extraction   -> 21 hand landmarks
  4. Gesture Recognition  -> fingers_up() -> mode classification
  5. Drawing Engine       -> maps fingertip coords to strokes on a white canvas
  6. Virtual Board UI     -> color palette / eraser / clear / save / undo / text toolbar
  7. Handwriting -> Text  -> background TrOCR pass turns your ink into real text
  8. Output Display       -> the whiteboard (or the text panel), with a tracking preview

Gestures
  - Index + middle fingers up  -> SELECTION mode (hover the toolbar to
                                   pick a color / eraser / clear / save / undo / text)
  - Index finger only up       -> DRAW mode (paints with selected color)
  - Anything else (fist, open
    palm, etc.)                -> IDLE (pen lifted, nothing drawn)

Keyboard fallback (handy if gestures misfire or no hand is visible):
  1-6  color select   e  eraser   c  clear canvas   s  save canvas (PNG)
  z  undo             +/-  brush size
  v  toggle voice control     p  toggle camera preview
  t  toggle text panel        l  toggle live recognition on/off
  r  re-run recognition now   x  export recognized text (.txt)
  q  quit

Voice commands (hands fully free, no gesture or keyboard needed):
  "red" / "green" / "blue" / "yellow" / "purple"   -> pick a color
  "eraser" / "erase"        "pen" / "draw"          -> switch tool
  "clear" / "clear board"   "save" / "save board"   -> board actions
  "undo"                                             -> undo last stroke
  "bigger" / "thicker"      "smaller" / "thinner"   -> brush size
  "convert to text" / "read my writing"              -> recognize now
  "show text" / "hide text"                          -> toggle text panel
  "export text" / "save text"                        -> write recognized text to disk
  "quit" / "exit"                                    -> close the app
"""

import math
import os
import time
from collections import deque

import cv2
import numpy as np

from hand_tracker import HandTracker
from handwriting_recognizer import HandwritingRecognizer
from voice_control import VoiceController
from smoothing import OneEuroFilter, catmull_rom_segment

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
CAM_WIDTH, CAM_HEIGHT = 1280, 720
TOOLBAR_HEIGHT = 110
DEFAULT_BRUSH = 6
ERASER_BRUSH = 44
SELECTION_DEBOUNCE = 12  # frames a click-style action must not repeat within

SMOOTHING_ALPHA = 0.6      # (legacy, unused - kept only for backwards reference)
ONE_EURO_MIN_CUTOFF = 0.6   # lower = smoother when the hand is nearly still (tightened for cleaner writing)
ONE_EURO_BETA = 0.4         # higher = snappier tracking during fast strokes
ONE_EURO_D_CUTOFF = 1.0

CURVE_STEPS = 10            # interpolated points per Catmull-Rom segment
TAPER_FRAMES = 6            # stroke ramps from TAPER_MIN_FRACTION to full width over this many points
TAPER_MIN_FRACTION = 0.45

SUBPIXEL_SHIFT = 4          # OpenCV fixed-point precision bits (16 sub-pixel steps per pixel) -
                            # draws ink at the fingertip's true fractional position instead of
                            # rounding to the nearest whole pixel, which is what made curves and
                            # diagonal strokes look jagged/pixel-stepped instead of clean.
SUBPIXEL_SCALE = 1 << SUBPIXEL_SHIFT
MAX_JUMP_PX = 120          # a bigger single-frame move than this starts a new stroke
GESTURE_STABLE_FRAMES = 3  # consecutive frames a gesture must hold before it "counts"
MAX_UNDO_STEPS = 20

# Handwriting -> Text
RECOGNIZE_DEBOUNCE_SEC = 0.9   # how long the pen must be idle before a live recognition pass fires
TEXT_PANEL_WIDTH_FRACTION = 0.42

WHITE = (255, 255, 255)
INK_BLACK = (30, 30, 30)
TOOLBAR_BG = (40, 40, 40)
BOARD_BORDER = (210, 210, 210)

PREVIEW_W, PREVIEW_H = 240, 135  # small camera tracking thumbnail, 16:9
PREVIEW_MARGIN = 16

# name, BGR color (None => eraser; a string tag => an action button)
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
    ("Export Txt", "EXPORTTEXT"),
]

OUTPUT_DIR = "saved_boards"
TEXT_OUTPUT_DIR = "recognized_text"


class VirtualBoard:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

        self.tracker = HandTracker(max_hands=1)

        # The whiteboard itself: a plain white canvas that only ever holds
        # ink. Nothing else (cursor, toolbar, camera preview, text panel) is
        # ever drawn into this array, so saved files - and what the
        # handwriting recognizer reads - are always clean, ink-only pixels.
        self.canvas = np.full((CAM_HEIGHT, CAM_WIDTH, 3), 255, np.uint8)

        self.draw_color = TOOLS[0][1]
        self.brush_thickness = DEFAULT_BRUSH
        self.eraser_mode = False
        self.show_preview = True

        self.prev_x, self.prev_y = 0, 0
        self.filter_x = OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
        self.filter_y = OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
        self.point_history = deque(maxlen=4)  # last filtered points, for Catmull-Rom curves
        self._stroke_point_count = 0           # for the pressure-like start taper
        self.action_cooldown = 0  # prevents Clear/Save/Undo from firing every frame
        self._stroke_started = False  # tracks whether we've snapshotted for undo yet

        self._pending_mode = "IDLE"   # gesture debounce state
        self._pending_count = 0
        self.current_mode = "IDLE"

        self.undo_stack = []

        self._build_toolbar_geometry()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(TEXT_OUTPUT_DIR, exist_ok=True)

        # ---- Handwriting -> Text -------------------------------------
        # Runs on its own background thread (model load + every inference
        # call) so the camera/drawing loop never blocks or drops frames
        # waiting on it. See handwriting_recognizer.py for the full design.
        self.recognizer = HandwritingRecognizer()
        self.recognizer.start()
        self.text_mode = False          # is the text panel currently shown?
        self.live_text_mode = True      # auto re-recognize a short moment after you stop writing?
        self.canvas_dirty = False
        self.last_activity_time = 0.0
        self.recognized_text = ""

        self.quit_requested = False
        self.voice = VoiceController()
        voice_ok = self.voice.start()
        if voice_ok:
            print("[voice] listening for commands (say 'red', 'clear', 'save', 'convert to text', ...)")
        else:
            print(f"[voice] disabled - no microphone available ({self.voice.error})")

    # ------------------------------------------------------------------
    # Toolbar layout
    # ------------------------------------------------------------------
    def _build_toolbar_geometry(self):
        n = len(TOOLS)
        slot_w = CAM_WIDTH // n
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
        cv2.rectangle(board, (0, 0), (CAM_WIDTH, TOOLBAR_HEIGHT), TOOLBAR_BG, -1)
        cv2.line(board, (0, TOOLBAR_HEIGHT), (CAM_WIDTH, TOOLBAR_HEIGHT), (15, 15, 15), 2)

        for slot in self.slots:
            cx = (slot["x1"] + slot["x2"]) // 2
            cy = TOOLBAR_HEIGHT // 2 - 8
            is_hover = hover_slot is slot

            if slot["value"] == "CLEAR":
                self._icon_button(board, cx, cy, "Clear", (70, 70, 70), is_hover)
            elif slot["value"] == "SAVE":
                self._icon_button(board, cx, cy, "Save", (70, 70, 70), is_hover)
            elif slot["value"] == "UNDO":
                self._icon_button(board, cx, cy, "Undo", (70, 70, 70), is_hover)
            elif slot["value"] == "TEXT":
                color = (0, 130, 60) if self.text_mode else (70, 70, 70)
                self._icon_button(board, cx, cy, "Text", color, is_hover)
            elif slot["value"] == "EXPORTTEXT":
                self._icon_button(board, cx, cy, "Export", (70, 70, 70), is_hover)
            elif slot["name"] == "Eraser":
                r = 26 if not is_hover else 29
                cv2.circle(board, (cx, cy), r, WHITE, -1, cv2.LINE_AA)
                cv2.circle(board, (cx, cy), r, (90, 90, 90), 2, cv2.LINE_AA)
                cv2.putText(board, "Eraser", (cx - 30, cy + 46),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
                if self.eraser_mode:
                    cv2.circle(board, (cx, cy), r + 5, (0, 220, 255), 3, cv2.LINE_AA)
            else:
                color = slot["value"]
                r = 26 if not is_hover else 29
                cv2.circle(board, (cx, cy), r, color, -1, cv2.LINE_AA)
                cv2.circle(board, (cx, cy), r, (15, 15, 15), 1, cv2.LINE_AA)
                cv2.putText(board, slot["name"], (cx - len(slot["name"]) * 4, cy + 46),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
                if color == self.draw_color and not self.eraser_mode:
                    cv2.circle(board, (cx, cy), r + 5, (0, 220, 255), 3, cv2.LINE_AA)

    @staticmethod
    def _icon_button(board, cx, cy, label, base_color, is_hover):
        r = 26 if not is_hover else 29
        cv2.circle(board, (cx, cy), r, base_color, -1, cv2.LINE_AA)
        cv2.circle(board, (cx, cy), r, (110, 110, 110), 1, cv2.LINE_AA)
        cv2.putText(board, label, (cx - len(label) * 6, cy + 6),
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
        """Only change the active mode once the same gesture has been seen
        for GESTURE_STABLE_FRAMES in a row. This filters out single-frame
        misreads (a finger flickering up/down) that otherwise cause the
        pen to lift and reconnect somewhere else, drawing stray lines."""
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
            print("[undo] reverted last action")
        else:
            print("[undo] nothing to undo")

    # ------------------------------------------------------------------
    # Handwriting -> Text bookkeeping
    # ------------------------------------------------------------------
    def _mark_dirty(self, clear_text=False):
        """Call this any time self.canvas actually changes (stroke, erase,
        clear, undo). Schedules a fresh recognition pass once the pen has
        been idle for RECOGNIZE_DEBOUNCE_SEC, instead of re-running the
        (expensive) model every single frame while you're still writing."""
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
            return  # worker is still finishing the previous pass; try again next frame
        self.recognizer.request(self.canvas)
        self.canvas_dirty = False

    def _force_recognition_now(self):
        self.canvas_dirty = True
        self.last_activity_time = 0.0  # bypass the debounce entirely

    def _export_text(self):
        text = self.recognized_text.strip()
        if not text:
            print("[text] nothing recognized yet - write something first")
            return None
        filename = os.path.join(TEXT_OUTPUT_DIR, f"text_{int(time.time())}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"[text] exported {filename}")
        return filename

    # ------------------------------------------------------------------
    # Drawing engine
    # ------------------------------------------------------------------
    @staticmethod
    def _fixed(x, y):
        """Convert a floating-point canvas coordinate to OpenCV's fixed-point
        sub-pixel format (see SUBPIXEL_SHIFT). Passing these through
        cv2.line/circle with shift=SUBPIXEL_SHIFT draws at the fingertip's
        true fractional position instead of snapping to the nearest whole
        pixel - this is what actually removes the jagged/stair-stepped look
        from curves and diagonal strokes."""
        return (int(round(x * SUBPIXEL_SCALE)), int(round(y * SUBPIXEL_SCALE)))

    def _current_color(self):
        return WHITE if self.eraser_mode else self.draw_color

    def _current_thickness(self):
        """Ramp from a thinner start to full brush width over the first
        few points of a stroke, like pressing a marker down onto the
        board, instead of an abrupt full-width line from frame one."""
        base = ERASER_BRUSH if self.eraser_mode else self.brush_thickness
        ramp = min(1.0, self._stroke_point_count / TAPER_FRAMES)
        fraction = TAPER_MIN_FRACTION + (1 - TAPER_MIN_FRACTION) * ramp
        return max(2, int(base * fraction))

    def _begin_stroke_segment(self, fx, fy):
        """(Re)start stroke tracking at a filtered point - used both for
        the very first point of a stroke and whenever the jump guard fires."""
        if not self._stroke_started:
            self._push_undo()
            self._stroke_started = True
        self.point_history.clear()
        self.point_history.append((fx, fy))
        self._stroke_point_count = 1
        # A single tap should still leave a visible dab.
        cv2.circle(self.canvas, self._fixed(fx, fy), self._current_thickness() // 2 * SUBPIXEL_SCALE,
                   self._current_color(), -1, cv2.LINE_AA, shift=SUBPIXEL_SHIFT)
        self.prev_x, self.prev_y = fx, fy
        self._mark_dirty()

    def _draw_stroke(self, x, y, timestamp):
        color = self._current_color()

        # 1) Adaptive filtering (One Euro Filter) - removes camera/tracking
        #    jitter when the fingertip is nearly still, but loosens up and
        #    tracks closely once it starts moving quickly, unlike a fixed
        #    moving average which can't do both well at once.
        fx = self.filter_x.filter(x, timestamp)
        fy = self.filter_y.filter(y, timestamp)

        # 2) First point of a new stroke - nothing to connect to yet.
        if self.prev_x == 0 and self.prev_y == 0:
            self._begin_stroke_segment(fx, fy)
            return

        # 3) Jump guard - if the point moved further than a real fingertip
        #    could in one frame, tracking likely glitched (lost/re-acquired
        #    the hand, or briefly picked up a different hand/reflection).
        #    Start a fresh stroke instead of drawing a stray line across
        #    the board, and reset the filters so they don't try to smooth
        #    across the discontinuity.
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
            # 4) Smooth curve through the last 4 tracked points (Catmull-Rom)
            #    instead of one straight segment - this is what removes the
            #    faceted/"polygon" look on fast strokes and curves.
            p0, p1, p2, p3 = self.point_history
            curve_points = [p1] + catmull_rom_segment(p0, p1, p2, p3, CURVE_STEPS)
            for i in range(len(curve_points) - 1):
                a = self._fixed(*curve_points[i])
                b = self._fixed(*curve_points[i + 1])
                cv2.line(self.canvas, a, b, color, thickness, cv2.LINE_AA, shift=SUBPIXEL_SHIFT)
                cv2.circle(self.canvas, b, (thickness // 2) * SUBPIXEL_SCALE, color, -1, cv2.LINE_AA, shift=SUBPIXEL_SHIFT)
        else:
            # Not enough history yet for a spline - plain segment as a
            # fallback for the first couple of points in the stroke.
            a = self._fixed(self.prev_x, self.prev_y)
            b = self._fixed(fx, fy)
            cv2.line(self.canvas, a, b, color, thickness, cv2.LINE_AA, shift=SUBPIXEL_SHIFT)
            cv2.circle(self.canvas, b, (thickness // 2) * SUBPIXEL_SCALE, color, -1, cv2.LINE_AA, shift=SUBPIXEL_SHIFT)

        self.prev_x, self.prev_y = fx, fy
        self._mark_dirty()

    def _reset_stroke(self):
        """Lift the pen: clear all stroke-tracking state so the next
        stroke starts clean instead of connecting to or smoothing across
        a stale point."""
        self.prev_x, self.prev_y = 0, 0
        self.filter_x.reset()
        self.filter_y.reset()
        self.point_history.clear()
        self._stroke_point_count = 0
        self._stroke_started = False

    def _apply_selection(self, x, y):
        self._reset_stroke()  # lift pen while selecting
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
        elif slot["value"] == "EXPORTTEXT":
            if self.action_cooldown == 0:
                self._export_text()
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
        """Drain any voice commands queued since the last frame and apply
        them the same way toolbar selection or key presses would."""
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
                elif value == "quit":
                    self.quit_requested = True
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
        print(f"[saved] {filename}")
        return filename

    # ------------------------------------------------------------------
    # Camera tracking preview (small corner thumbnail, not the main board)
    # ------------------------------------------------------------------
    def _draw_preview(self, board, tracked_frame, mode):
        if not self.show_preview:
            return
        thumb = cv2.resize(tracked_frame, (PREVIEW_W, PREVIEW_H))
        x0 = CAM_WIDTH - PREVIEW_W - PREVIEW_MARGIN
        y0 = TOOLBAR_HEIGHT + PREVIEW_MARGIN
        board[y0:y0 + PREVIEW_H, x0:x0 + PREVIEW_W] = thumb
        cv2.rectangle(board, (x0, y0), (x0 + PREVIEW_W, y0 + PREVIEW_H), (120, 120, 120), 2)
        cv2.putText(board, f"tracking: {mode}", (x0, y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Handwriting -> Text panel
    # ------------------------------------------------------------------
    def _text_status(self):
        if self.recognizer.load_error:
            return "unavailable"
        if not self.recognizer.ready:
            return "loading model..."
        if self.recognizer.is_busy():
            return "reading..."
        return "live" if self.live_text_mode else "paused (press l)"

    def _draw_text_panel(self, board):
        if not self.text_mode:
            return
        panel_w = int(CAM_WIDTH * TEXT_PANEL_WIDTH_FRACTION)
        x0 = CAM_WIDTH - panel_w
        y0 = TOOLBAR_HEIGHT

        overlay = board.copy()
        cv2.rectangle(overlay, (x0, y0), (CAM_WIDTH, CAM_HEIGHT), (250, 250, 245), -1)
        cv2.addWeighted(overlay, 0.94, board, 0.06, 0, board)
        cv2.rectangle(board, (x0, y0), (CAM_WIDTH, CAM_HEIGHT), (180, 180, 170), 2)

        header = f"Recognized Text  [{self._text_status()}]"
        cv2.putText(board, header, (x0 + 16, y0 + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.line(board, (x0 + 16, y0 + 44), (CAM_WIDTH - 16, y0 + 44), (200, 200, 195), 1)

        if self.recognizer.load_error:
            self._draw_wrapped_text(board, self.recognizer.load_error, x0 + 16, y0 + 74,
                                     panel_w - 32, (150, 60, 60))
            return

        text = self.recognized_text or "(start writing - text appears here automatically)"
        self._draw_wrapped_text(board, text, x0 + 16, y0 + 74, panel_w - 32, (35, 35, 35))

    @staticmethod
    def _draw_wrapped_text(board, text, x, y, max_width, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness, line_h = 0.62, 1, 27
        cy = y
        for raw_line in text.split("\n"):
            words = raw_line.split(" ") if raw_line else [""]
            line = ""
            for word in words:
                trial = (line + " " + word).strip()
                (w, _h), _ = cv2.getTextSize(trial, font, scale, thickness)
                if w > max_width and line:
                    cv2.putText(board, line, (x, cy), font, scale, color, thickness, cv2.LINE_AA)
                    cy += line_h
                    line = word
                else:
                    line = trial
            cv2.putText(board, line, (x, cy), font, scale, color, thickness, cv2.LINE_AA)
            cy += line_h
            if cy > board.shape[0] - 20:
                break

    # ------------------------------------------------------------------
    # Keyboard fallback controls
    # ------------------------------------------------------------------
    def _handle_key(self, key):
        if key == ord('c'):
            self._push_undo()
            self.canvas[:] = 255
            self._mark_dirty(clear_text=True)
        elif key == ord('s'):
            self._save_canvas()
        elif key == ord('z'):
            self._undo()
        elif key == ord('e'):
            self.eraser_mode = True
        elif key in [ord(str(n)) for n in range(1, 7)]:
            idx = int(chr(key)) - 1
            self.eraser_mode = False
            self.draw_color = TOOLS[idx][1]
        elif key == ord('+') or key == ord('='):
            self.brush_thickness = min(40, self.brush_thickness + 2)
        elif key == ord('-'):
            self.brush_thickness = max(2, self.brush_thickness - 2)
        elif key == ord('v'):
            enabled = self.voice.toggle()
            print(f"[voice] {'listening' if enabled else 'muted'}")
        elif key == ord('p'):
            self.show_preview = not self.show_preview
        elif key == ord('t'):
            self.text_mode = not self.text_mode
        elif key == ord('l'):
            self.live_text_mode = not self.live_text_mode
            print(f"[text] live recognition {'ON' if self.live_text_mode else 'OFF'}")
        elif key == ord('r'):
            self._force_recognition_now()
        elif key == ord('x'):
            self._export_text()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        if not self.cap.isOpened():
            print("ERROR: Could not open webcam. Check camera index / permissions.")
            return

        prev_time = 0
        while True:
            success, raw_frame = self.cap.read()
            if not success:
                break
            raw_frame = cv2.flip(raw_frame, 1)  # mirror for natural interaction

            self._apply_voice_commands()
            if self.quit_requested:
                break

            # Hand tracking runs on the camera frame (used for the small
            # preview thumbnail + gesture coordinates), never shown full-screen.
            tracked_frame = self.tracker.find_hands(raw_frame, draw=True)
            landmarks = self.tracker.find_positions(tracked_frame)
            mode = "IDLE"
            hover_slot = None

            if landmarks:
                fingers = self.tracker.fingers_up()
                raw_mode = self._classify(fingers)
                mode = self._stable_mode(raw_mode)
                x, y = landmarks[8][1], landmarks[8][2]  # index fingertip

                if mode == "SELECT":
                    hover_slot = self._apply_selection(x, y)
                elif mode == "DRAW":
                    self._draw_stroke(x, y, time.time())
                else:
                    self._reset_stroke()
            else:
                self._reset_stroke()
                mode = self._stable_mode("IDLE")

            # Handwriting -> Text: fire a background recognition pass once
            # the pen's been idle a moment, and pick up whatever the worker
            # thread most recently finished (never blocks this loop).
            self._maybe_trigger_recognition()
            result = self.recognizer.latest_result()
            if result is not None:
                self.recognized_text = result.text

            # Build the display frame fresh each loop from the clean ink
            # canvas, so cursor/toolbar/preview/text panel never get baked
            # into saved files or fed back into the recognizer.
            board = self.canvas.copy()

            if landmarks:
                x, y = landmarks[8][1], landmarks[8][2]
                if mode == "SELECT":
                    cv2.circle(board, (x, y), 16, (0, 150, 255), 2, cv2.LINE_AA)
                    cv2.circle(board, (x, y), 4, (0, 150, 255), -1, cv2.LINE_AA)
                elif mode == "DRAW":
                    cursor_color = (160, 160, 160) if self.eraser_mode else self.draw_color
                    cv2.circle(board, (x, y), self.brush_thickness // 2 + 3, cursor_color, 2, cv2.LINE_AA)

            self._draw_toolbar(board, hover_slot=hover_slot)
            self._draw_preview(board, tracked_frame, mode)
            self._draw_text_panel(board)

            # status footer
            now = time.time()
            fps = 1 / (now - prev_time) if prev_time else 0
            prev_time = now
            voice_tag = f"Voice: ON ('{self.voice.last_heard}')" if self.voice.enabled else "Voice: OFF (press v)"
            status = (f"Mode: {mode}  |  Tool: {'Eraser' if self.eraser_mode else 'Color'}  "
                      f"|  Brush: {self.brush_thickness}px  |  FPS: {int(fps)}  |  {voice_tag}"
                      f"  |  Text: {self._text_status() if self.text_mode else 'hidden (press t)'}")
            cv2.putText(board, status, (10, CAM_HEIGHT - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1, cv2.LINE_AA)

            cv2.imshow("AI-Powered Virtual Board", board)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            self._handle_key(key)

        self.voice.stop()
        self.recognizer.stop()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    VirtualBoard(camera_index=0).run()
