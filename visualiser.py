"""
Displays three scrolling plots:
  - Pitch (F0) via Parselmouth/Praat
  - Intensity (RMS dB)
  - Onset strength (librosa)

Architecture: sounddevice callback -> ring buffer -> processing thread -> matplotlib FuncAnimation
"""

import threading
import collections
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

_paused = threading.Event()   # set = paused, clear = running

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.widgets as widgets
import sounddevice as sd
import librosa
import parselmouth
from parselmouth.praat import call

# ── Audio config ────────────────────────────────────────────────────────────── (Alter based on hardware used)
SAMPLE_RATE   = 22050      # Hz
BLOCK_SIZE    = 512        # samples per sounddevice callback (~23 ms)
DISPLAY_SECS  = 6          # seconds of history to show
PROC_HOP_SECS = 0.05       # feature extraction every ~50 ms
PITCH_FMIN    = 75         # Hz  (below typical speech F0)
PITCH_FMAX    = 400        # Hz  (above typical speech F0)

SILENCE_DB    = -66        # frames quieter than this are treated as unvoiced

# Parselmouth pitch settings (Praat autocorrelation — robust for speech)
PRAAT_PITCH_WIN   = 0.05   # analysis window (s)
PRAAT_PITCH_STEP  = 0.01   # step (s)

# ── Ring buffer ───────────────────────────────────────────────────────────────
_buf_len  = int(SAMPLE_RATE * DISPLAY_SECS * 2)   # keep 2x display window
_audio_buf: collections.deque = collections.deque(maxlen=_buf_len)
_buf_lock  = threading.Lock()

# ── Feature time series (deque = automatic scroll) ───────────────────────────
N_FRAMES = int(DISPLAY_SECS / PROC_HOP_SECS)

_times_d    = collections.deque(maxlen=N_FRAMES)
_pitch_d    = collections.deque(maxlen=N_FRAMES)
_rms_db_d   = collections.deque(maxlen=N_FRAMES)
_onset_d    = collections.deque(maxlen=N_FRAMES)

_feat_lock  = threading.Lock()

# ── Sounddevice callback ──────────────────────────────────────────────────────
def _audio_callback(indata, frames, time_info, status):
    mono = indata[:, 0].copy()
    with _buf_lock:
        _audio_buf.extend(mono)

# ── Processing thread ─────────────────────────────────────────────────────────
_proc_win_secs = 0.25   # 250 ms analysis window per feature frame
_proc_win_samp = int(SAMPLE_RATE * _proc_win_secs)
_start_time    = time.perf_counter()
_running       = True


def _process_loop():
    while _running:
        time.sleep(PROC_HOP_SECS)

        if _paused.is_set():
            continue

        with _buf_lock:
            if len(_audio_buf) < _proc_win_samp:
                continue
            frame = np.array(list(_audio_buf)[-_proc_win_samp:], dtype=np.float32)

        t_now = time.perf_counter() - _start_time

        # ── Intensity (RMS → dB) ──────────────────────────────────────────
        rms   = float(np.sqrt(np.mean(frame ** 2)))
        rms_db = 20 * np.log10(rms + 1e-9)

        # ── Onset strength (librosa) ──────────────────────────────────────
        onset_env = librosa.onset.onset_strength(
            y=frame, sr=SAMPLE_RATE,
            hop_length=BLOCK_SIZE,
            center=True,
        )
        onset_val = float(onset_env.mean())

        # ── Pitch via Parselmouth (Praat SHS / autocorrelation) ──────────
        if rms_db < SILENCE_DB:
            f0 = float("nan")
        else:
            snd       = parselmouth.Sound(frame, sampling_frequency=SAMPLE_RATE)
            pitch_obj = call(snd, "To Pitch", PRAAT_PITCH_STEP, PITCH_FMIN, PITCH_FMAX)
            n_frames  = call(pitch_obj, "Get number of frames")
            f0_vals   = []
            for i in range(1, n_frames + 1):
                v = call(pitch_obj, "Get value in frame", i, "Hertz")
                if v == v and v > 0:   # nan check + unvoiced filter
                    f0_vals.append(v)
            f0 = float(np.median(f0_vals)) if f0_vals else float("nan")

        with _feat_lock:
            _times_d.append(t_now)
            _pitch_d.append(f0)
            _rms_db_d.append(rms_db)
            _onset_d.append(onset_val)


# ── Matplotlib figure ─────────────────────────────────────────────────────────
plt.style.use("dark_background")
fig, (ax_pitch, ax_rms, ax_onset) = plt.subplots(
    3, 1, figsize=(12, 7), sharex=False
)
fig.suptitle("Real-time Speech Feature Visualiser", fontsize=14, color="white", y=0.98)
fig.subplots_adjust(hspace=0.45, left=0.10, right=0.97, top=0.93, bottom=0.12)

# Pitch
ax_pitch.set_title("Pitch  F0  (Hz)", loc="left", fontsize=10)
ax_pitch.set_ylim(PITCH_FMIN, PITCH_FMAX)
ax_pitch.set_ylabel("Hz")
ax_pitch.yaxis.label.set_color("cyan")
ax_pitch.tick_params(colors="grey")
for spine in ax_pitch.spines.values():
    spine.set_edgecolor("#333")
line_pitch, = ax_pitch.plot([], [], color="cyan", lw=1.5)
dot_pitch,  = ax_pitch.plot([], [], "o", color="cyan", ms=5)
text_pitch  = ax_pitch.text(0.98, 0.88, "", transform=ax_pitch.transAxes,
                             ha="right", va="top", color="cyan", fontsize=11, fontweight="bold")

# RMS
ax_rms.set_title("Intensity  (dBFS)", loc="left", fontsize=10)
ax_rms.set_ylim(-80, 0)
ax_rms.set_ylabel("dBFS")
ax_rms.yaxis.label.set_color("lime")
ax_rms.tick_params(colors="grey")
for spine in ax_rms.spines.values():
    spine.set_edgecolor("#333")
line_rms, = ax_rms.plot([], [], color="lime", lw=1.5)
text_rms  = ax_rms.text(0.98, 0.88, "", transform=ax_rms.transAxes,
                         ha="right", va="top", color="lime", fontsize=11, fontweight="bold")

# Onset
ax_onset.set_title("Onset Strength  (spectral flux)", loc="left", fontsize=10)
ax_onset.set_ylim(0, None)
ax_onset.set_ylabel("strength")
ax_onset.yaxis.label.set_color("orange")
ax_onset.tick_params(colors="grey")
for spine in ax_onset.spines.values():
    spine.set_edgecolor("#333")
line_onset, = ax_onset.plot([], [], color="orange", lw=1.5)
# Onset fill uses a PolyCollection updated via set_verts each frame (no remove/recreate)
_fill_verts  = ax_onset.fill_between([0], [0], alpha=0.25, color="orange")
text_onset  = ax_onset.text(0.98, 0.88, "", transform=ax_onset.transAxes,
                             ha="right", va="top", color="orange", fontsize=11, fontweight="bold")


# ── Pause / resume controls ───────────────────────────────────────────────────
ax_btn = fig.add_axes([0.44, 0.02, 0.12, 0.05])
btn_pause = widgets.Button(ax_btn, "Pause", color="#333333", hovercolor="#555555")
btn_pause.label.set_color("white")
btn_pause.label.set_fontsize(11)

# "PAUSED" overlay on the pitch panel — hidden by default
paused_text = ax_pitch.text(
    0.5, 0.5, "PAUSED", transform=ax_pitch.transAxes,
    ha="center", va="center", fontsize=22, fontweight="bold",
    color="white", alpha=0.75, visible=False,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#111111", edgecolor="white", alpha=0.6),
)


def _toggle_pause(event=None):
    if _paused.is_set():
        _paused.clear()
        btn_pause.label.set_text("Pause")
        paused_text.set_visible(False)
    else:
        _paused.set()
        btn_pause.label.set_text("Resume")
        paused_text.set_visible(True)
    fig.canvas.draw_idle()


btn_pause.on_clicked(_toggle_pause)


def _on_key(event):
    if event.key == " ":
        _toggle_pause()


fig.canvas.mpl_connect("key_press_event", _on_key)


def _make_fill_verts(x, y):
    """Build the vertex array that fill_between uses internally."""
    # polygon: walk along top (x, y) then back along bottom (x, 0)
    verts = np.column_stack([
        np.concatenate([x, x[::-1]]),
        np.concatenate([y, np.zeros_like(y)]),
    ])
    return [verts]


def _update(frame_idx):
    with _feat_lock:
        if len(_times_d) < 2:
            return line_pitch, line_rms, line_onset

        times  = np.array(_times_d)
        pitch  = np.array(_pitch_d)
        rms_db = np.array(_rms_db_d)
        onset  = np.array(_onset_d)

    t_max = times[-1]
    t_min = t_max - DISPLAY_SECS

    # Pitch — only voiced frames
    voiced_mask = ~np.isnan(pitch)
    if voiced_mask.any():
        line_pitch.set_data(times[voiced_mask], pitch[voiced_mask])
        last_voiced = pitch[voiced_mask][-1]
        text_pitch.set_text(f"{last_voiced:.0f} Hz")
        dot_pitch.set_data([times[voiced_mask][-1]], [last_voiced])
    else:
        line_pitch.set_data([], [])
        dot_pitch.set_data([], [])
        text_pitch.set_text("unvoiced")
    ax_pitch.set_xlim(t_min, t_max)

    # RMS
    line_rms.set_data(times, np.clip(rms_db, -80, 0))
    text_rms.set_text(f"{rms_db[-1]:.1f} dBFS")
    ax_rms.set_xlim(t_min, t_max)

    # Onset — dynamic y-scale + fill (reuse same PolyCollection)
    line_onset.set_data(times, onset)
    peak = float(onset.max()) if onset.size else 1.0
    ax_onset.set_ylim(0, max(peak * 1.2, 0.5))
    text_onset.set_text(f"{onset[-1]:.3f}")
    ax_onset.set_xlim(t_min, t_max)
    _fill_verts.set_verts(_make_fill_verts(times, onset))

    return line_pitch, dot_pitch, line_rms, line_onset


def main():
    global _running

    # List available input devices so the user can pick
    print("\nAvailable input devices:")
    print(sd.query_devices())
    print(f"\nUsing default input device. Sample rate: {SAMPLE_RATE} Hz\n")
    print("Press Ctrl+C or close the window to stop.\n")

    # Start processing thread
    proc_thread = threading.Thread(target=_process_loop, daemon=True)
    proc_thread.start()

    # Start audio stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,
        dtype="float32",
        callback=_audio_callback,
    )
    stream.start()

    # Animate at ~20 fps (interval=50 ms)
    ani = animation.FuncAnimation(
        fig, _update, interval=50, blit=False, cache_frame_data=False
    )

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        _running = False
        stream.stop()
        stream.close()
        print("\nStopped.")


if __name__ == "__main__":
    main()
