"""
handwriting_recognizer.py
--------------------------
Turns the ink already on the whiteboard canvas into real, editable text.
This is the new stage that sits after the Drawing Engine in the pipeline:

    ... Drawing Engine -> Virtual Board UI -> [Handwriting -> Text] -> Output

Why it's built this way (not the "first thing that comes to mind" way):

* Segmentation runs on PIXELS, not on raw stroke polylines. Two strokes
  that legitimately cross (dotting an "i", crossing a "t", a descender
  dipping under the next word) would fool a path-based clusterer. A
  dilate + connected-components pass on the ink mask handles that for
  free, the same way production OCR line/word detectors do it — it's a
  proven, boring technique, which is exactly what you want for something
  this load-bearing.
* Recognition itself is TrOCR — a transformer encoder-decoder pretrained
  specifically on handwriting — rather than a classic engine tuned for
  scanned printed text. TrOCR reads a whole word crop in one shot with no
  fragile per-character segmentation step, which matches what we hand it:
  a single continuous fingertip-driven stroke, not a typeset scan.
* Every word crop is content-hashed and memoized. A "keeps updating as
  you write" live mode therefore only ever pays the (relatively expensive)
  model forward pass for the handful of words that actually changed since
  the last pass, not the whole board, every time.
* Model load + every inference call happen on a dedicated background
  thread behind a single-slot request queue. The camera/drawing loop must
  never block on a multi-hundred-millisecond model call — that would tank
  FPS and make the pen feel laggy while you're mid-stroke.
* Optional light spellcheck cleans up the handful of characters a vision
  model typically gets wrong (e.g. "rn" vs "m") without ever overriding a
  word it already recognizes as valid — so real names/jargon survive.
"""

import hashlib
import queue
import threading
import time

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Tunables — these are the levers to reach for if recognition quality or
# latency needs adjusting for a particular camera/lighting/handwriting style.
# --------------------------------------------------------------------------
MODEL_NAME = "microsoft/trocr-base-handwritten"
# Swap for "microsoft/trocr-large-handwritten" for meaningfully higher
# accuracy at the cost of slower inference / more VRAM — worth it on a GPU
# machine, usually too slow for a smooth CPU-only live experience.

INK_THRESHOLD = 245        # grayscale value below which a pixel counts as "ink" on the white canvas
WORD_DILATE_KERNEL = (18, 14)   # (w, h) structuring element: merges strokes of the *same* word, keeps separate words apart
LINE_BAND_FACTOR = 0.6      # fraction of median word-height used to decide "same text line"
CROP_PADDING = 14           # px of white margin kept around each word crop before it's fed to the model
MIN_INK_PIXELS = 12         # ignore stray specks/taps smaller than this many ink pixels
TARGET_LINE_HEIGHT = 64     # crops are rescaled to roughly this pixel height before OCR
MAX_NEW_TOKENS = 24         # generous for a single handwritten word/short phrase
WORD_CACHE_LIMIT = 4000     # bounded so a long session doesn't leak memory


class RecognitionResult:
    """One completed recognition pass over the board."""

    __slots__ = ("text", "lines", "word_boxes", "elapsed")

    def __init__(self, text, lines, word_boxes, elapsed):
        self.text = text                # full text, "\n" between lines, " " between words
        self.lines = lines              # list[list[str]]  (words per line, reading order)
        self.word_boxes = word_boxes    # list[(x1, y1, x2, y2, text)] for optional overlays/debugging
        self.elapsed = elapsed          # seconds the pass took


class HandwritingRecognizer:
    """
    Background-threaded handwriting -> text engine.

    Usage:
        rec = HandwritingRecognizer()
        rec.start()                 # loads the model on a worker thread; returns immediately
        ...
        rec.request(canvas)          # non-blocking "please re-read the board" (call on a debounce timer)
        result = rec.latest_result() # poll for the freshest completed result, or None if none yet
    """

    def __init__(self, model_name=MODEL_NAME, use_spellcheck=True):
        self.model_name = model_name
        self.use_spellcheck = use_spellcheck

        self._processor = None
        self._model = None
        self._device = "cpu"
        self.ready = False
        self.load_error = None

        self._word_cache = {}   # content-hash -> recognized word text (memoization)
        self._spell = None

        self._request_queue = queue.Queue(maxsize=1)  # only the newest board state matters
        self._result_lock = threading.Lock()
        self._latest_result = None
        self._busy = False
        self._stop_flag = False
        self._worker = None

    # ------------------------------------------------------------------
    # Public, thread-safe API
    # ------------------------------------------------------------------
    def start(self):
        """Kick off model loading + the recognition worker thread. Never
        blocks the caller — safe to call right before the render loop starts,
        the app keeps working (drawing/voice/keyboard) while the model loads."""
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def stop(self):
        self._stop_flag = True

    def is_busy(self):
        return self._busy

    def request(self, canvas):
        """Queue a recognition pass over the current canvas. Non-blocking; if
        a request is already pending, the newer canvas simply replaces it —
        only the most recent board state is ever worth reading."""
        if not self.ready:
            return
        snapshot = canvas.copy()
        try:
            self._request_queue.put_nowait(snapshot)
        except queue.Full:
            try:
                self._request_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._request_queue.put_nowait(snapshot)
            except queue.Full:
                pass

    def latest_result(self):
        with self._result_lock:
            return self._latest_result

    # ------------------------------------------------------------------
    # Model loading (background thread)
    # ------------------------------------------------------------------
    def _load_model(self):
        try:
            import torch  # noqa: F401 (presence check)
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError:
            self.load_error = (
                "transformers/torch not installed — run "
                "`pip install -r requirements.txt` to enable handwriting-to-text."
            )
            return False

        try:
            self._processor = TrOCRProcessor.from_pretrained(self.model_name)
            self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
            self._model.eval()
        except Exception as exc:  # network hiccup, disk space, corrupt cache, etc.
            self.load_error = f"could not load handwriting model ({self.model_name}): {exc}"
            return False

        if self.use_spellcheck:
            try:
                from spellchecker import SpellChecker
                self._spell = SpellChecker()
            except ImportError:
                self._spell = None  # optional dependency — degrade silently, raw OCR still works

        return True

    def _run(self):
        ok = self._load_model()
        self.ready = ok
        if not ok:
            return

        while not self._stop_flag:
            try:
                canvas = self._request_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            self._busy = True
            try:
                result = self._recognize(canvas)
                with self._result_lock:
                    self._latest_result = result
            except Exception as exc:
                with self._result_lock:
                    self._latest_result = RecognitionResult(f"[recognition error: {exc}]", [], [], 0.0)
            finally:
                self._busy = False

    # ------------------------------------------------------------------
    # Segmentation: ink mask -> word boxes -> lines (pure OpenCV, no model)
    # ------------------------------------------------------------------
    @staticmethod
    def _ink_mask(canvas):
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        return (gray < INK_THRESHOLD).astype(np.uint8) * 255

    @staticmethod
    def _word_boxes(mask):
        """Dilate the ink mask so nearby strokes of one word touch, then
        take connected components as word candidates. This is what makes
        crossing/overlapping strokes (dotted i's, crossed t's, cursive
        ligatures) segment correctly without any per-character logic."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, WORD_DILATE_KERNEL)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
        boxes = []
        for i in range(1, n):  # label 0 is background
            x, y, w, h, _area = stats[i]
            ink_pixels = int(np.count_nonzero(mask[y:y + h, x:x + w]))
            if ink_pixels < MIN_INK_PIXELS:
                continue  # stray speck / accidental dot, not a real mark
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
        return boxes

    @staticmethod
    def _group_lines(boxes):
        """Band-based line grouping: sort candidates by vertical center,
        then chain them into the same line while their center stays within
        LINE_BAND_FACTOR * height of that line's running-average center —
        robust to normal handwriting slope without needing full deskewing."""
        if not boxes:
            return []
        heights = [b[3] - b[1] for b in boxes]
        median_h = float(np.median(heights))
        ordered = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2.0)

        lines = [[ordered[0]]]
        centers = [(ordered[0][1] + ordered[0][3]) / 2.0]

        for box in ordered[1:]:
            center = (box[1] + box[3]) / 2.0
            band = max(median_h, box[3] - box[1]) * LINE_BAND_FACTOR
            if abs(center - centers[-1]) <= band:
                lines[-1].append(box)
                centers[-1] = sum((b[1] + b[3]) / 2.0 for b in lines[-1]) / len(lines[-1])
            else:
                lines.append([box])
                centers.append(center)

        lines = [sorted(line, key=lambda b: b[0]) for line in lines]  # left-to-right within a line
        lines.sort(key=lambda line: min(b[1] for b in line))          # top-to-bottom overall
        return lines

    # ------------------------------------------------------------------
    # Per-word crop preprocessing + OCR
    # ------------------------------------------------------------------
    def _prepare_crop(self, canvas, box):
        x1, y1, x2, y2 = box
        h, w = canvas.shape[:2]
        x1 = max(0, x1 - CROP_PADDING)
        y1 = max(0, y1 - CROP_PADDING)
        x2 = min(w, x2 + CROP_PADDING)
        y2 = min(h, y2 + CROP_PADDING)
        crop = canvas[y1:y2, x1:x2]

        # Normalize scale before OCR: TrOCR was trained on a fairly
        # consistent line-height, and a fast, large stroke vs. a small,
        # careful one otherwise hit the model at wildly different pixel
        # scales. Rescaling to a common height noticeably improves accuracy.
        ch, cw = crop.shape[:2]
        if ch > 0 and cw > 0:
            scale = TARGET_LINE_HEIGHT / ch
            new_w = max(1, int(round(cw * scale)))
            crop = cv2.resize(crop, (new_w, TARGET_LINE_HEIGHT), interpolation=cv2.INTER_CUBIC)
        return crop

    @staticmethod
    def _content_hash(crop_bgr):
        """Cheap perceptual-ish hash used purely to memoize OCR calls: if a
        word's pixels haven't meaningfully changed since last time, skip the
        (expensive) model forward pass entirely."""
        small = cv2.resize(crop_bgr, (32, 16), interpolation=cv2.INTER_AREA)
        return hashlib.blake2b(small.tobytes(), digest_size=12).hexdigest()

    def _ocr_word(self, crop_bgr):
        key = self._content_hash(crop_bgr)
        cached = self._word_cache.get(key)
        if cached is not None:
            return cached

        import torch
        from PIL import Image

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pixel_values = self._processor(images=pil_img, return_tensors="pt").pixel_values.to(self._device)
        with torch.no_grad():
            generated_ids = self._model.generate(pixel_values, max_new_tokens=MAX_NEW_TOKENS)
        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        text = self._maybe_correct(text)

        self._word_cache[key] = text
        if len(self._word_cache) > WORD_CACHE_LIMIT:
            self._word_cache.pop(next(iter(self._word_cache)))
        return text

    def _maybe_correct(self, word):
        """Light-touch spellcheck: only steps in for alphabetic tokens the
        dictionary doesn't recognize at all, and only if it has a confident
        replacement — never overrides a word already considered valid, so
        names, jargon, and short/technical tokens pass through untouched."""
        if not self._spell or not word:
            return word
        stripped = word.strip(".,!?;:'\"")
        if not stripped or not stripped.isalpha() or len(stripped) <= 2:
            return word
        if not self._spell.unknown([stripped.lower()]):
            return word  # already a known word — leave it alone
        correction = self._spell.correction(stripped.lower())
        if not correction or correction == stripped.lower():
            return word
        if stripped[0].isupper():
            correction = correction.capitalize()
        return word.replace(stripped, correction)

    # ------------------------------------------------------------------
    def _recognize(self, canvas):
        start = time.time()
        mask = self._ink_mask(canvas)
        if cv2.countNonZero(mask) < MIN_INK_PIXELS:
            return RecognitionResult("", [], [], time.time() - start)

        boxes = self._word_boxes(mask)
        lines = self._group_lines(boxes)

        out_lines, word_boxes_out = [], []
        for line in lines:
            words = []
            for box in line:
                crop = self._prepare_crop(canvas, box)
                text = self._ocr_word(crop)
                if text:
                    words.append(text)
                    word_boxes_out.append((*box, text))
            if words:
                out_lines.append(words)

        full_text = "\n".join(" ".join(w) for w in out_lines)
        return RecognitionResult(full_text, out_lines, word_boxes_out, time.time() - start)
