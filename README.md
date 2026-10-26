# AI-Powered Virtual Board — now with live Handwriting → Text

A touchless, AI-driven digital whiteboard that turns any standard webcam into
a gesture-controlled drawing surface — **and reads what you write back to you
as real, copyable text**, live, as you go. No special hardware required.

Built with **Python, OpenCV, MediaPipe, and TrOCR (Transformers)**, following
the pipeline:

```
Webcam Input → Hand Detection → Feature Extraction (21 landmarks)
   → Gesture Recognition → Drawing Engine → Virtual Board UI
                              ↓                    ↓
                    Voice Commands        Handwriting → Text (TrOCR)
                  (background mic thread)   (background model thread)
                              ↓                    ↓
                         Output Display  +  Recognized Text Panel
```

Gestures, keyboard, and **voice** all drive the same board state — including
the new text panel — so you can mix and match whichever is most convenient
in the moment.

## 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note on hand tracking:** MediaPipe removed the old `mediapipe.solutions.Hands`
> API in versions 0.10.15+. This project's `hand_tracker.py` uses the newer
> **Tasks API** (`HandLandmarker`), which works with current mediapipe
> releases (0.10.30+). The first time you run `main.py`, it automatically
> downloads a small hand-landmark model file (`hand_landmarker.task`,
> ~7MB) into the project folder and reuses it on every run after that —
> you'll need an internet connection the very first time only.
>
> If the automatic download fails (e.g. restrictive network/firewall),
> download it manually from
> `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
> and save it as `hand_landmarker.task` in the project folder.

### Handwriting → Text dependency (Transformers + PyTorch)

The new `handwriting_recognizer.py` module uses **TrOCR**
(`microsoft/trocr-base-handwritten`), a transformer model pretrained
specifically on handwriting, via Hugging Face `transformers` + `torch`.

- First run of the feature downloads the model weights (~1.3GB) and caches
  them under `~/.cache/huggingface`; after that it's fully offline.
- If a CUDA GPU is available, it's used automatically; otherwise it falls
  back to CPU — a bit slower per word, but perfectly usable thanks to
  per-word caching (see "How it works" below).
- If `torch`/`transformers` aren't installed, the app still runs fine —
  every gesture/drawing/voice feature keeps working, and the text panel
  simply shows an "unavailable" message with the install command, exactly
  the same graceful-degradation pattern as the PyAudio/voice fallback.
- Optional: `pyspellchecker` (already in `requirements.txt`) lightly cleans
  up individual misread characters without ever overriding a word it
  already recognizes as valid — remove it from `requirements.txt` if you'd
  rather see the model's raw output.

### Voice control system dependency (PyAudio)

`PyAudio` needs the `portaudio` system library to build. Install it first:

```bash
# macOS
brew install portaudio

# Debian/Ubuntu
sudo apt-get install portaudio19-dev python3-pyaudio

# Windows
pip install pipwin && pipwin install pyaudio
```

Then `pip install -r requirements.txt` as above. If PyAudio still fails to
install, the app still runs fine — it just prints
`[voice] disabled - no microphone available (...)` at startup and falls
back to gestures/keyboard only.

Voice recognition uses Google's free Web Speech API under the hood
(`speech_recognition.recognize_google`), so an internet connection is
required for voice commands specifically; gestures, keyboard shortcuts,
and handwriting recognition all work fully offline (after the one-time
model downloads above).

## 2. Run

**Desktop (native window):**
```bash
python3 main.py
```

**Browser (Streamlit, no display/X server needed):**
```bash
streamlit run streamlit_app.py
```

A window (or browser tab) opens showing a clean **white whiteboard** with a
toolbar across the top. Your camera feed is *not* shown full-screen — a
small tracking preview thumbnail in the top-right corner shows your hand
and its skeleton so you can see what's being tracked, without cluttering
the writing surface.

## 3. Controls

### Hand gestures
| Gesture | Fingers | Mode |
|---|---|---|
| ✌️ Index + middle up | 2 fingers | **Selection** — hover the fingertip over the toolbar to pick a color, the eraser, Undo, Clear, Save, or Text |
| ☝️ Index only up | 1 finger | **Draw** — paints a stroke in the selected color |
| ✊ Fist / other poses | — | **Idle** — pen lifted, nothing drawn |

### Keyboard fallback (works even without a hand in frame)
| Key | Action |
|---|---|
| `1`–`6` | Select Red / Green / Blue / Yellow / Purple / Black |
| `e` | Eraser |
| `z` | Undo last stroke/action |
| `c` | Clear canvas |
| `s` | Save canvas to `saved_boards/` (PNG) |
| `+` / `-` | Increase / decrease brush size |
| `v` | Toggle voice listening on/off |
| `p` | Show/hide the camera tracking preview thumbnail |
| `t` | Show/hide the **Recognized Text** panel |
| `l` | Toggle live (automatic) handwriting recognition on/off |
| `r` | Re-run recognition right now (ignores the idle debounce) |
| `x` | Export recognized text to `recognized_text/` (.txt) |
| `q` | Quit |

### Voice commands (fully hands-free)

The app starts listening on your default microphone automatically (toggle
anytime with `v`). Just speak naturally — extra words around a command are
ignored, e.g. "please clear the board" still triggers Clear.

| Say | Action |
|---|---|
| "red" / "green" / "blue" / "yellow" / "purple" / "black" | Pick that color |
| "eraser" or "erase" | Switch to eraser |
| "pen" or "draw" | Switch back to pen |
| "undo" | Undo the last stroke or action |
| "clear" / "clear board" | Wipe the canvas |
| "save" / "save board" | Export the board to `saved_boards/` (PNG) |
| "bigger" / "thicker" | Increase brush size |
| "smaller" / "thinner" | Decrease brush size |
| "convert to text" / "read my writing" | Show the text panel and recognize right now |
| "show text" / "hide text" | Toggle the recognized-text panel |
| "export text" / "save text" | Write the recognized text to `recognized_text/` (.txt) |
| "quit" / "exit" | Close the app |
| "stop listening" | Mute the microphone |

The bottom status bar shows `Voice: ON ('<last phrase heard>')` so you can
confirm what was picked up, plus a live `Text: <status>` indicator.

## 4. Project structure

```
virtual_board/
├── main.py                    # App loop: capture, gesture mode, drawing engine, UI, output
├── hand_tracker.py             # MediaPipe wrapper: detection, landmarks, finger-state logic
├── handwriting_recognizer.py   # NEW: pixel-based word/line segmentation + TrOCR handwriting model
├── voice_control.py            # Background mic listener: speech -> command queue
├── smoothing.py                 # One Euro Filter + Catmull-Rom spline for marker-like ink
├── board_processor.py           # Streamlit-webrtc VideoProcessor port of the same engine
├── streamlit_app.py             # Browser frontend (sidebar controls + live text panel)
├── requirements.txt
├── saved_boards/               # PNG exports land here (created automatically)
├── recognized_text/            # NEW: .txt exports of recognized handwriting land here
└── README.md
```

## 5. How it works

- **Hand Detection & Feature Extraction** (`hand_tracker.py`) — MediaPipe's
  `HandLandmarker` (Tasks API) locates the hand each frame using a small
  auto-downloaded model file, and returns 21 (x, y) landmarks.
  `fingers_up()` compares fingertip landmarks against their lower joints to
  decide which fingers are extended.
- **Gesture Recognition** (`VirtualBoard._classify` + `_stable_mode`) —
  the finger pattern is mapped to one of three modes (`SELECT`, `DRAW`,
  `IDLE`), and a gesture must hold for a few consecutive frames before the
  mode actually switches, so a single misread finger doesn't interrupt a
  stroke.
- **Drawing Engine** (`VirtualBoard._draw_stroke`, `smoothing.py`) — the
  board itself (`self.canvas`) is a plain white array that only ever holds
  ink; nothing else is drawn into it. Raw fingertip coordinates go through
  an **adaptive One Euro Filter** (steadies fine detail when the hand is
  nearly still, loosens up for low-lag tracking on fast strokes), then a
  **Catmull-Rom spline** is fit through the last 4 filtered points so
  curves render as smooth arcs. Each new stroke ramps up from a thinner
  starting width to the full brush size, like pressing a marker down. A
  "jump guard" starts a fresh stroke instead of connecting across a sudden
  tracking glitch. Every new stroke and every Clear pushes a snapshot onto
  an undo stack.
- **Virtual Board UI** (`VirtualBoard._draw_toolbar`) — a top bar renders
  color swatches, Eraser, Undo, Clear, Save, **Text**, and **Export Txt**;
  the fingertip position during `SELECT` mode is checked against each
  swatch's bounding box, with the hovered button highlighted.
- **Handwriting → Text** (`handwriting_recognizer.py`) — this is the new
  stage. It never touches strokes directly; instead it reads the *pixels*
  already on the canvas, the same clean ink array that gets saved to PNG:
  1. **Segmentation** — an ink mask (anything darker than near-white) is
     dilated just enough that strokes belonging to the same word touch,
     then `cv2.connectedComponentsWithStats` finds each word as a blob.
     This handles crossing strokes (dotted "i"s, crossed "t"s, cursive
     ligatures) correctly with no per-character logic at all.
  2. **Line grouping** — word boxes are chained into lines using a
     running vertical-center band, robust to normal handwriting slope.
  3. **Recognition** — each word crop is rescaled to a consistent
     line-height (TrOCR is sensitive to input scale) and fed through
     `microsoft/trocr-base-handwritten`, a transformer trained
     specifically on handwriting — a meaningfully better fit here than a
     classic printed-text OCR engine, and it reads a whole word in one
     shot instead of needing fragile character segmentation.
  4. **Caching** — every crop is content-hashed and memoized, so a live
     "keeps updating as you write" mode only ever re-runs the model on
     the handful of words that actually changed, not the whole board,
     every pass.
  5. **Light spellcheck** (optional, via `pyspellchecker`) nudges an
     unrecognized token toward the nearest dictionary word — but only if
     the model's raw output isn't already a real word, so names and
     jargon pass through untouched.
  6. All of the above runs on a dedicated background thread with a
     single-slot request queue, so the camera/drawing loop is never
     blocked waiting on a model forward pass. A short debounce
     (`RECOGNIZE_DEBOUNCE_SEC`, default 0.9s) waits for a natural pause
     in writing before triggering a pass, so you're not fighting a
     half-finished word being (mis)read mid-stroke.
  The recognized text shows up in a panel (`t` / "Text" button / "show
  text") and can be exported to a plain `.txt` file at any time (`x` /
  "Export Txt" / "export text").
- **Camera tracking preview** (`VirtualBoard._draw_preview`) — a small
  corner thumbnail shows the live camera feed with the hand skeleton drawn
  on it, purely so you can see what's being tracked. It's cosmetic only —
  drawn onto a fresh copy of the canvas each frame, never onto
  `self.canvas` itself, so saved boards (and recognized text) are always
  based on clean ink.
- **Voice Commands** (`voice_control.py`) — `VoiceController` runs
  `SpeechRecognition`'s `listen_in_background` on its own thread, matches
  recognized phrases against `COMMAND_MAP`, and pushes `(kind, value)`
  tuples onto a queue. Each frame, `VirtualBoard._apply_voice_commands()`
  drains that queue and applies the exact same state changes gestures and
  keyboard shortcuts do — colors, eraser/pen, undo, clear, save, brush
  size, text panel visibility, forced recognition, text export, and quit.

## 6. Tuning handwriting recognition

All the levers live at the top of `handwriting_recognizer.py`:

- `MODEL_NAME` — swap `"microsoft/trocr-base-handwritten"` for
  `"microsoft/trocr-large-handwritten"` for noticeably higher accuracy at
  the cost of slower inference / more VRAM. Worth it on a GPU machine;
  usually too slow for a snappy CPU-only live experience.
- `WORD_DILATE_KERNEL` — how aggressively nearby strokes get merged into
  one "word" blob. Too small and letters within a word split apart; too
  large and separate words merge together. Widen it if you write with a
  lot of natural gaps between letters; narrow it if words are running
  together.
- `LINE_BAND_FACTOR` — how much vertical wobble is tolerated before a word
  is considered "the next line down" instead of the same line. Raise it
  if your lines slope and get mis-split; lower it if separate lines are
  incorrectly merged.
- `INK_THRESHOLD` — how dark a pixel must be to count as ink; adjust if
  you use very light colors as your default pen color.
- `RECOGNIZE_DEBOUNCE_SEC` (in `main.py` / `board_processor.py`) — how
  long the pen must be idle before a live pass fires. Lower for snappier
  updates at the cost of more (mostly cached) background work; raise it
  if you tend to pause mid-word.

## 7. Extending the project (matches the "Future Enhancements" roadmap)

- **Multi-user collaboration** — swap the local `canvas` array for shared
  state (e.g. a WebSocket server or a Redis-backed grid) so multiple clients
  draw on the same board; the recognizer already only reads pixels, so it
  works unmodified against any shared canvas.
- **Cloud storage** — instead of / in addition to local PNG/`.txt` export in
  `_save_canvas()` / `_export_text()`, upload to S3 / GCS and persist a
  session ID.
- **Handwriting → structured data** — pipe `recognized_text` through an
  LLM to turn freeform notes into a to-do list, meeting minutes, or a
  calendar event, rather than just plain text.
- **More voice commands / offline recognition** — swap
  `recognize_google()` in `voice_control.py` for an offline engine (e.g.
  Vosk or `recognize_sphinx`) to remove the internet dependency for voice
  (handwriting recognition is already fully offline after the first
  model download).
- **Video-call integration** — run the composited output through a virtual
  camera driver (e.g. `pyvirtualcam`) so Zoom/Teams can use it as a webcam
  source.

## 8. Troubleshooting

- **`No matching distribution found for mediapipe==0.10.14`** — you're on
  an older copy of this project. Pull the latest `requirements.txt` /
  `hand_tracker.py` (mediapipe is no longer pinned; the tracker now uses
  the Tasks API which works with any current mediapipe release).
- **Black/frozen window, no hand detected** — check webcam permissions and
  that no other app is holding the camera; try `camera_index=1` in
  `main.py` if you have multiple cameras.
- **Gestures feel unreliable** — increase `detection_conf` /
  `tracking_conf` in `HandTracker.__init__`, ensure good, even lighting,
  and keep your palm facing the camera.
- **Text panel says "unavailable"** — `torch`/`transformers` aren't
  installed, or the model failed to download (check your internet
  connection the first time). The message includes the exact `pip
  install` command; every other feature keeps working in the meantime.
- **Text panel says "loading model..." for a long time** — the first run
  downloads ~1.3GB of model weights; subsequent runs load from the local
  Hugging Face cache in a few seconds. A slow/CPU-only load can also just
  take longer — the rest of the app stays fully usable while it loads.
- **Recognized text is wrong or garbled** — see the tuning levers in
  section 6 above; also try writing a bit larger/slower, and prefer the
  `Black` pen for clearest contrast against the white canvas.
- **Words are merged together or split apart** — adjust
  `WORD_DILATE_KERNEL` in `handwriting_recognizer.py` (see section 6).
- **Lines look shaky, choppy, faceted, or jump randomly across the board**
  — the drawing engine (`smoothing.py` + `main.py`) is built specifically
  to avoid this with tunables at the top of `main.py`:
  - `SUBPIXEL_SHIFT` (default `4`, i.e. 16 sub-pixel steps per pixel) —
    ink is drawn at the fingertip's true fractional position (via OpenCV's
    `shift` parameter on `cv2.line`/`cv2.circle`) instead of snapping to
    the nearest whole pixel. This is what removes the jagged, stair-stepped
    look from curves and diagonal strokes.
  - `ONE_EURO_MIN_CUTOFF` (default `0.6`) — how much jitter is removed
    when the fingertip is nearly still (writing detail). Lower it for
    steadier fine strokes; raise it if slow strokes feel laggy.
  - `ONE_EURO_BETA` (default `0.4`) — how quickly the filter "loosens up"
    and tracks closely once the fingertip starts moving fast.
  - `CURVE_STEPS` (default `10`) — how many interpolated points are drawn
    per Catmull-Rom curve segment.
  - `MAX_JUMP_PX` (default `120`) — if the fingertip appears to move
    further than this in one frame, a *new* stroke starts instead of
    drawing a straight line across the jump.
  - `GESTURE_STABLE_FRAMES` (default `3`) — how many consecutive frames a
    gesture must hold before the mode (draw/select/idle) actually switches.
  - `TAPER_FRAMES` / `TAPER_MIN_FRACTION` — control the pressure-like
    ramp-up at the start of each stroke.
- **Low FPS** — lower `CAM_WIDTH` / `CAM_HEIGHT` in `main.py`, reduce
  `max_hands` if you don't need multi-hand support, or turn off `l` (live
  recognition) if you'd rather trigger text conversion manually with `r`.
