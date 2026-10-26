"""
voice_control.py
------------------
Hands-free tool switching via voice, matching the "Voice command support"
future-enhancement item in the project roadmap. Also drives the
handwriting-to-text panel ("convert to text", "show text", "export text", ...)
so reading the board back is just as hands-free as writing on it.

Runs microphone capture on a background thread (via SpeechRecognition's
listen_in_background) so the main OpenCV render loop never blocks. Each
recognized phrase is mapped to a (kind, value) command tuple and pushed
onto a thread-safe queue that main.py polls once per frame.

If no microphone / PyAudio is available, start() fails gracefully and
returns False — the rest of the app keeps working with gestures and
keyboard shortcuts only.
"""

import queue
import speech_recognition as sr


class VoiceController:
    # spoken phrase -> (command kind, command value)
    COMMAND_MAP = {
        "red": ("color", (60, 60, 230)),
        "green": ("color", (60, 170, 60)),
        "blue": ("color", (220, 110, 40)),
        "yellow": ("color", (30, 200, 240)),
        "purple": ("color", (170, 60, 150)),
        "black": ("color", (35, 35, 35)),

        "eraser": ("tool", "eraser"),
        "erase": ("tool", "eraser"),
        "pen": ("tool", "pen"),
        "draw": ("tool", "pen"),

        "clear": ("action", "clear"),
        "clear board": ("action", "clear"),
        "clear canvas": ("action", "clear"),

        "save": ("action", "save"),
        "save board": ("action", "save"),

        "undo": ("action", "undo"),
        "undo that": ("action", "undo"),

        "bigger": ("brush", 1),
        "increase brush": ("brush", 1),
        "thicker": ("brush", 1),
        "smaller": ("brush", -1),
        "decrease brush": ("brush", -1),
        "thinner": ("brush", -1),

        "quit": ("action", "quit"),
        "exit": ("action", "quit"),
        "stop listening": ("action", "mute"),

        # Handwriting -> Text
        "convert to text": ("action", "recognize_text"),
        "read my writing": ("action", "recognize_text"),
        "read the board": ("action", "recognize_text"),
        "recognize text": ("action", "recognize_text"),
        "recognize": ("action", "recognize_text"),

        "show text": ("action", "show_text"),
        "hide text": ("action", "hide_text"),

        "export text": ("action", "export_text"),
        "save text": ("action", "export_text"),
        "save as text": ("action", "export_text"),
    }

    def __init__(self, mic_index=None, energy_threshold=300, pause_threshold=0.6):
        self.command_queue = queue.Queue()
        self.enabled = False
        self.last_heard = ""
        self.error = None

        self._recognizer = sr.Recognizer()
        self._recognizer.energy_threshold = energy_threshold
        self._recognizer.pause_threshold = pause_threshold
        self._stop_listening = None
        self._mic_index = mic_index

    def start(self):
        """Calibrate for ambient noise and start listening in the
        background. Returns True on success, False if no mic is available
        (missing PyAudio, no hardware, permission denied, etc.)."""
        try:
            mic = sr.Microphone(device_index=self._mic_index)
            with mic as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
            self._stop_listening = self._recognizer.listen_in_background(
                mic, self._callback, phrase_time_limit=4
            )
            self.enabled = True
            return True
        except Exception as exc:  # OSError (no mic), ImportError (no PyAudio), etc.
            self.error = str(exc)
            self.enabled = False
            return False

    def stop(self):
        if self._stop_listening:
            self._stop_listening(wait_for_stop=False)
        self.enabled = False

    def toggle(self, mic_index=None):
        """Convenience for a keyboard/gesture mute toggle."""
        if self.enabled:
            self.stop()
        else:
            self.start()
        return self.enabled

    def _callback(self, recognizer, audio):
        """Runs on the background thread each time a phrase is captured."""
        try:
            text = recognizer.recognize_google(audio).lower().strip()
        except sr.UnknownValueError:
            return  # audio captured but not understood - ignore
        except sr.RequestError as exc:
            self.error = f"Speech service unavailable: {exc}"
            return

        self.last_heard = text
        command = self._parse(text)
        if command:
            self.command_queue.put(command)

    def _parse(self, text):
        if text in self.COMMAND_MAP:
            return self.COMMAND_MAP[text]
        # fall back to substring match so e.g. "please clear the board"
        # still triggers "clear board"
        for phrase, command in self.COMMAND_MAP.items():
            if phrase in text:
                return command
        return None

    def poll(self):
        """Return the next queued command as (kind, value), or None."""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None