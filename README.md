# Real-Time-Speech-Feature-Visualiser-
A visualisation of pitch, intensity and onset strength of audio gathered and displayed in real time. 

## Motivation

Beat gestures are rhythmic hand movements that are driven by the prosodic features of
speech — pitch, intensity and rhythmic stress. This tool visualises those features in
real-time from microphone input, forming the prosodic extraction pipeline.

## Features

- Real-time pitch (F0) extraction via Parselmouth/Praat autocorrelation
- Real-time intensity (RMS dBFS) extraction
- Real-time onset strength extraction via librosa spectral flux
- Scrolling 6-second display window at 20Hz update rate
- Pause/resume via button or spacebar

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python visualiser.py
```

## Project Context

This visualiser is Phase 1 of a larger system that maps prosodic features to CPG
oscillator parameters to drive anthropomorphic beat gestures on a Pepper robot in
real-time. TODO: See the technical documentation in `docs/visualiser_documentation.md`
for full implementation detail
