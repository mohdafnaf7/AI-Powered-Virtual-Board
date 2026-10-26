"""
hand_tracker.py
----------------
Thin wrapper around MediaPipe's HandLandmarker (Tasks API) that gives the
rest of the app three simple things:

1. find_hands()      -> detect a hand in a BGR frame and (optionally) draw
                         the 21-point skeleton on it
2. find_positions()  -> pixel (x, y) coordinates for every landmark
3. fingers_up()      -> which of the 5 fingers are currently extended

This corresponds to Stage 2 (Hand Detection) and Stage 3 (Feature
Extraction) in the project pipeline.

Note: this uses MediaPipe's newer Tasks API (HandLandmarker), not the
older `mediapipe.solutions.Hands` API. mediapipe >= 0.10.15 removed
`mediapipe.solutions` entirely, so this is the version that actually
works with current pip installs (0.10.30+). The model file
(hand_landmarker.task, ~7-10MB) is downloaded automatically to this
folder the first time HandTracker() is created, then cached for reuse.
"""

import math
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# Standard 21-point MediaPipe hand topology, used to draw the skeleton
# manually since the Tasks API doesn't ship a drawing helper like the old
# `mediapipe.solutions.drawing_utils` did.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]


def _ensure_model():
    """Download the hand landmark model on first run, then reuse it."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        return
    print("[hand_tracker] downloading hand landmark model (one-time, ~7MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[hand_tracker] model saved to {MODEL_PATH}")
    except Exception as exc:
        raise RuntimeError(
            "Could not download the MediaPipe hand landmark model. "
            "Check your internet connection, or manually download it from "
            f"{MODEL_URL} and save it as {MODEL_PATH}."
        ) from exc


class HandTracker:
    # Landmark indices for the tip of each finger
    TIP_IDS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

    def __init__(self, max_hands=1, detection_conf=0.5, tracking_conf=0.5):
        _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        self.landmark_list = []
        self._result = None
        self._start_time = time.time()
        self._last_timestamp_ms = -1

    def find_hands(self, frame, draw=True):
        """Run hand detection on the frame and optionally draw the skeleton."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires strictly increasing timestamps
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        self._result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if draw and self._result.hand_landmarks:
            h, w, _ = frame.shape
            for hand_lms in self._result.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 120, 255), 2)
                for x, y in pts:
                    cv2.circle(frame, (x, y), 4, (0, 200, 255), -1)
        return frame

    def find_positions(self, frame, hand_index=0):
        """Return list of (id, x, y) pixel coords for the requested hand."""
        self.landmark_list = []
        if self._result and self._result.hand_landmarks:
            if hand_index >= len(self._result.hand_landmarks):
                return self.landmark_list
            h, w, _ = frame.shape
            for idx, lm in enumerate(self._result.hand_landmarks[hand_index]):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.landmark_list.append((idx, cx, cy))
        return self.landmark_list

    def fingers_up(self):
        """
        Return [thumb, index, middle, ring, pinky] as 1 (extended) / 0 (folded).
        Assumes a right hand roughly facing the camera, palm forward -
        good enough for a UI gesture system.
        """
        if not self.landmark_list or len(self.landmark_list) < 21:
            return [0, 0, 0, 0, 0]

        lm = self.landmark_list
        fingers = []

        # Thumb: compare tip x-position to the joint two below it
        if lm[self.TIP_IDS[0]][1] > lm[self.TIP_IDS[0] - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other 4 fingers: tip is "up" if it's above (smaller y) the pip joint
        for tip_id in self.TIP_IDS[1:]:
            if lm[tip_id][2] < lm[tip_id - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def distance(self, id1, id2):
        """Euclidean pixel distance + midpoint between two landmarks."""
        if len(self.landmark_list) <= max(id1, id2):
            return None, None
        x1, y1 = self.landmark_list[id1][1], self.landmark_list[id1][2]
        x2, y2 = self.landmark_list[id2][1], self.landmark_list[id2][2]
        length = math.hypot(x2 - x1, y2 - y1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return length, (cx, cy)