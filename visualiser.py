"""
Real-time speech feature visualiser.
Implemented RMS of speech  F0 and Onset Strength 
"""
"""

import collections
import threading
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import sounddevice as sd
import librosa
import parselmouth
from parselmouth.praat import call

# ── Audio config ──────────────────────────────────────────────────────────────
SAMPLE_RATE = 22050
BLOCK_SIZE  = 512
BUFFER_SECS = 12

# ── Feature extraction config ─────────────────────────────────────────────────
PROC_HOP_SECS     = 0.05
PROC_WIN_SECS     = 0.25
SILENCE_DB        = -66
PITCH_FMIN        = 75     # Hz — below typical speech F0
PITCH_FMAX        = 400    # Hz — above typical speech F0
PRAAT_PITCH_STEP  = 0.01   # Praat internal frame step (s)

# ── Ring buffer ───────────────────────────────────────────────────────────────
_buf_len   = int(SAMPLE_RATE * BUFFER_SECS)
_audio_buf = collections.deque(maxlen=_buf_len)
_buf_lock  = threading.Lock()

# ── Processing state ──────────────────────────────────────────────────────────
_proc_win_samp = int(SAMPLE_RATE * PROC_WIN_SECS)
_running       = True


def _audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio status: {status}")
    mono = indata[:, 0].copy()
    with _buf_lock:
        _audio_buf.extend(mono)


def extract_intensity(frame):
    """Compute RMS energy of audio frame and convert to dBFS."""
    rms    = float(np.sqrt(np.mean(frame ** 2)))
    rms_db = 20 * np.log10(rms + 1e-9)
    return rms_db


def extract_onset(frame):
    """
    Compute onset strength using librosa spectral flux.
    Peaks at syllable boundaries and stressed words in speech.
    Maps to CPG coupling weight alpha in the prosody-to-CPG layer.
    """
    onset_env = librosa.onset.onset_strength(
        y=frame,
        sr=SAMPLE_RATE,
        hop_length=BLOCK_SIZE,
        center=True,
    )
    return float(onset_env.mean())


def extract_pitch(frame, rms_db):
    """
    Extract fundamental frequency (F0) using Parselmouth/Praat autocorrelation.
    Returns NaN for silent or unvoiced frames.
    Praat autocorrelation is more robust than pYIN for short 250ms windows.
    Maps to CPG oscillator frequency omega in the prosody-to-CPG layer.
    """
    if rms_db < SILENCE_DB:
        return float("nan")

    snd       = parselmouth.Sound(frame, sampling_frequency=SAMPLE_RATE)
    pitch_obj = call(snd, "To Pitch", PRAAT_PITCH_STEP, PITCH_FMIN, PITCH_FMAX)
    n_frames  = call(pitch_obj, "Get number of frames")

    f0_vals = []
    for i in range(1, n_frames + 1):
        v = call(pitch_obj, "Get value in frame", i, "Hertz")
        if v == v and v > 0:   # nan check and unvoiced filter
            f0_vals.append(v)

    return float(np.median(f0_vals)) if f0_vals else float("nan")


def _process_loop():
    """Processing thread: extracts all three prosodic features every 50ms."""
    while _running:
        time.sleep(PROC_HOP_SECS)

        with _buf_lock:
            if len(_audio_buf) < _proc_win_samp:
                continue
            frame = np.array(list(_audio_buf)[-_proc_win_samp:], dtype=np.float32)

        rms_db = extract_intensity(frame)
        onset  = extract_onset(frame)
        f0     = extract_pitch(frame, rms_db)

        f0_str = f"{f0:.1f} Hz" if f0 == f0 else "unvoiced"
        state  = "SILENT" if rms_db < SILENCE_DB else "VOICED"

        print(
            f"F0: {f0_str:12s}  |  "
            f"Intensity: {rms_db:6.1f} dBFS  |  "
            f"Onset: {onset:.3f}  |  {state}",
            end="\r"
        )


def main():
    global _running

    print("\nAvailable input devices:")
    print(sd.query_devices())
    print(f"\nSample rate:      {SAMPLE_RATE} Hz")
    print(f"Analysis window:  {PROC_WIN_SECS*1000:.0f} ms")
    print(f"Extraction rate:  {1/PROC_HOP_SECS:.0f} Hz")
    print(f"Silence threshold:{SILENCE_DB} dBFS")
    print(f"Pitch range:      {PITCH_FMIN}-{PITCH_FMAX} Hz")
    print("\nStarting. Press Ctrl+C to stop.\n")

    proc_thread = threading.Thread(target=_process_loop, daemon=True)
    proc_thread.start()

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,
        dtype="float32",
        callback=_audio_callback,
    )

    with stream:
        try:
            while True:
                sd.sleep(500)
        except KeyboardInterrupt:
            _running = False
            print("\nStopped.")


if __name__ == "__main__":
    main()
