"""
Real-time speech feature visualiser.
Implemented RMS of speech. TODO: Implement F0 and Onset Strength 
"""

import collections
import threading
import time
import numpy as np
import sounddevice as sd

# ── Audio config ──────────────────────────────────────────────────────────────
SAMPLE_RATE = 22050
BLOCK_SIZE  = 512
BUFFER_SECS = 12

# ── Feature extraction config ─────────────────────────────────────────────────
PROC_HOP_SECS  = 0.05    # extract features every 50ms
PROC_WIN_SECS  = 0.25    # use a 250ms analysis window each time
SILENCE_DB     = -66     # dBFS threshold below which frame is treated as silent

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


def _process_loop():
    """Processing thread: extracts features from ring buffer every 50ms."""
    while _running:
        time.sleep(PROC_HOP_SECS)

        with _buf_lock:
            if len(_audio_buf) < _proc_win_samp:
                continue
            frame = np.array(list(_audio_buf)[-_proc_win_samp:], dtype=np.float32)

        rms_db = extract_intensity(frame)

        # Determine voiced/silent state
        state = "SILENT" if rms_db < SILENCE_DB else "VOICED"
        print(f"Intensity: {rms_db:6.1f} dBFS  |  {state}", end="\r")


def main():
    global _running

    print("\nAvailable input devices:")
    print(sd.query_devices())
    print(f"\nSample rate:      {SAMPLE_RATE} Hz")
    print(f"Analysis window:  {PROC_WIN_SECS*1000:.0f} ms")
    print(f"Extraction rate:  {1/PROC_HOP_SECS:.0f} Hz")
    print(f"Silence threshold:{SILENCE_DB} dBFS")
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
