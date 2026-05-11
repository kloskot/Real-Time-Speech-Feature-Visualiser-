"""
Real-time speech feature visualiser.
Day 1: Audio capture skeleton using sounddevice.
"""

import collections
import threading
import numpy as np
import sounddevice as sd

# ── Audio config ──────────────────────────────────────────────────────────────
SAMPLE_RATE = 22050      # Hz
BLOCK_SIZE  = 512        # samples per callback (~23 ms)
BUFFER_SECS = 12         # seconds of audio to keep in ring buffer

# ── Ring buffer ───────────────────────────────────────────────────────────────
_buf_len   = int(SAMPLE_RATE * BUFFER_SECS)
_audio_buf = collections.deque(maxlen=_buf_len)
_buf_lock  = threading.Lock()


def _audio_callback(indata, frames, time_info, status):
    """sounddevice callback — fires every ~23ms with a new block of samples."""
    if status:
        print(f"Audio status: {status}")
    mono = indata[:, 0].copy()
    with _buf_lock:
        _audio_buf.extend(mono)


def main():
    print("\nAvailable input devices:")
    print(sd.query_devices())
    print(f"\nSample rate: {SAMPLE_RATE} Hz")
    print(f"Block size:  {BLOCK_SIZE} samples (~{1000*BLOCK_SIZE/SAMPLE_RATE:.1f} ms)")
    print(f"Buffer size: {_buf_len} samples ({BUFFER_SECS}s)")
    print("\nStarting audio capture. Press Ctrl+C to stop.\n")

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
                with _buf_lock:
                    n = len(_audio_buf)
                print(f"Buffer: {n} samples ({n/SAMPLE_RATE:.2f}s)", end="\r")
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
