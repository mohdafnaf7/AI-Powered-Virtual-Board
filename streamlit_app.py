"""
streamlit_app.py — AI-Powered Virtual Board (Streamlit / browser frontend)
---------------------------------------------------------------------------
Runs the exact same pipeline as main.py (Webcam -> HandTracker -> gesture
classification -> One-Euro/Catmull-Rom drawing engine -> toolbar -> voice
commands -> handwriting-to-text) but streams the whiteboard to a browser
tab over WebRTC instead of opening a native OpenCV window, so it works
from `streamlit run` on any machine/browser with a webcam, no X server /
display required.

Three full control paths are wired up, matching the original README:
  1. Hand gestures (index+middle -> select toolbar, index only -> draw)
     directly on the video, exactly like the desktop app.
  2. A Streamlit sidebar that mirrors the keyboard shortcuts (colors,
     eraser, undo, clear, save, brush size, voice toggle, preview toggle)
     for when a gesture misfires or you'd rather click.
  3. A "Handwriting -> Text" sidebar panel: recognized text updates live
     a moment after you stop writing (same debounced background TrOCR
     pass as main.py), with a "Convert now" button and a .txt download.

Voice commands keep working in the background exactly as before, as long
as the machine running `streamlit run` has a microphone.
"""

import time

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from board_processor import TOOLS, BoardProcessor

st.set_page_config(page_title="AI-Powered Virtual Board", layout="wide")

st.title("🖐️ AI-Powered Virtual Board")
st.caption(
    "Touchless whiteboard: draw with your webcam and one finger, select "
    "tools with two fingers, and watch your handwriting turn into real "
    "text in the sidebar — or use voice commands for all of it."
)

RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

COLOR_HEX = {
    "Red": "#E63C3C", "Green": "#3CAA3C", "Blue": "#2878DC",
    "Yellow": "#F0C81E", "Purple": "#AA3C96", "Black": "#232323",
}

with st.sidebar:
    st.header("Controls")
    st.caption("These mirror the keyboard shortcuts from the desktop app.")

    st.subheader("Color")
    color_cols = st.columns(3)
    color_choice = None
    for i, (name, bgr) in enumerate([t for t in TOOLS if t[1] not in (None, "UNDO", "CLEAR", "SAVE", "TEXT")]):
        with color_cols[i % 3]:
            if st.button(name, key=f"color_{name}", use_container_width=True):
                color_choice = bgr

    eraser_choice = st.button("🧹 Eraser", use_container_width=True)

    brush_size = st.slider("Brush size", 2, 40, 6)

    st.subheader("Actions")
    action_cols = st.columns(3)
    undo_clicked = action_cols[0].button("↩️ Undo", use_container_width=True)
    clear_clicked = action_cols[1].button("🗑️ Clear", use_container_width=True)
    save_clicked = action_cols[2].button("💾 Save", use_container_width=True)

    st.subheader("Display")
    show_toolbar = st.checkbox("Show in-video toolbar (for gestures)", value=True)
    show_preview = st.checkbox("Show camera tracking preview", value=True)

    st.divider()
    st.subheader("Gestures")
    st.markdown(
        "- ☝️ **Index only** → Draw\n"
        "- ✌️ **Index + middle** → Select (hover toolbar)\n"
        "- ✊ **Fist / other** → Idle"
    )

    st.divider()
    st.subheader("Handwriting → Text")
    live_text = st.checkbox("Live recognition (auto-updates as you write)", value=True)
    force_recognize = st.button("🔎 Convert now", use_container_width=True)
    text_status_placeholder = st.empty()
    text_area_placeholder = st.empty()
    download_placeholder = st.empty()

    st.divider()
    status_placeholder = st.empty()

ctx = webrtc_streamer(
    key="virtual-board",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=BoardProcessor,
    media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
    async_processing=True,
)

if ctx.video_processor:
    vp = ctx.video_processor
    with vp.lock:
        if color_choice is not None:
            vp.eraser_mode = False
            vp.draw_color = color_choice
        if eraser_choice:
            vp.eraser_mode = True
        vp.brush_thickness = brush_size
        vp.show_toolbar = show_toolbar
        vp.show_preview = show_preview
        vp.live_text_mode = live_text
        if undo_clicked:
            vp.request_undo = True
        if clear_clicked:
            vp.request_clear = True
        if save_clicked:
            vp.request_save = True
        if force_recognize:
            vp.text_mode = True
            vp.request_recognize = True

        recognized_text = vp.recognized_text
        text_status = vp.text_status()
        load_error = vp.recognizer.load_error

    status_placeholder.markdown(
        f"**Mode:** {vp.status_mode}  \n"
        f"**FPS:** {int(vp.status_fps)}  \n"
        f"**Voice:** {'ON' if vp.voice.enabled else 'OFF (' + str(vp.voice.error) + ')'}"
    )

    if load_error:
        text_status_placeholder.error(load_error)
    else:
        text_status_placeholder.caption(f"Status: {text_status}")
    text_area_placeholder.text_area(
        "Recognized text (updates live; feel free to copy/edit here)",
        value=recognized_text, height=200, key="recognized_text_display",
    )
    if recognized_text.strip():
        download_placeholder.download_button(
            "⬇️ Download as .txt", recognized_text, file_name="recognized_text.txt", mime="text/plain",
        )

    voice_col1, voice_col2 = st.sidebar.columns(2)
    if voice_col1.button("🎙️ Toggle voice"):
        vp.voice.toggle()
    if voice_col2.button("📝 Toggle text panel"):
        with vp.lock:
            vp.text_mode = not vp.text_mode
    if vp.last_saved_path:
        st.sidebar.success(f"Saved: {vp.last_saved_path}")
        try:
            with open(vp.last_saved_path, "rb") as f:
                st.sidebar.download_button(
                    "⬇️ Download last save", f, file_name=vp.last_saved_path.split("/")[-1],
                    mime="image/png",
                )
        except FileNotFoundError:
            pass
else:
    st.info("Click **START** above to enable your webcam and begin drawing.")

st.divider()
with st.expander("How it works / troubleshooting"):
    st.markdown(
        """
- **Pipeline:** Webcam (browser, via WebRTC) → MediaPipe hand detection →
  21-landmark feature extraction → gesture classification → One-Euro
  filter + Catmull-Rom drawing engine → toolbar/preview overlay → video
  streamed back to the browser.
- The whiteboard canvas itself only ever holds ink — the toolbar, cursor,
  and camera preview thumbnail are drawn on a *copy* each frame, so saved
  PNGs (in `saved_boards/`) are always clean.
- If the video shows "Mode: IDLE" even with your hand up, check lighting
  and make sure your palm faces the camera; the browser will also ask for
  camera permission the first time.
- Voice commands require a microphone on the machine running
  `streamlit run streamlit_app.py` — they don't come from the browser's
  mic, since that would need extra audio-streaming plumbing. If none is
  found, the sidebar shows why voice is off and gestures/sidebar still
  work fully.
"""
    )