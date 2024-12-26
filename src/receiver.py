#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sounddevice as sd
from scipy import signal
import wcslib as wcs

def design_chebyshev1_bandpass(order, rp, fs, passband=(4750, 4850)):
    nyquist = fs / 2
    f1, f2 = passband
    Wn = [f1 / nyquist, f2 / nyquist]
    b, a = signal.cheby1(order, rp, Wn, btype='band', analog=False, output='ba')
    return b, a

def design_chebyshev1_lowpass(order, rp, fs, cutoff=50):
    nyquist = fs / 2
    Wn = cutoff / nyquist
    b, a = signal.cheby1(order, rp, Wn, btype='low', analog=False, output='ba')
    return b, a

def main():
    # -------------------------- (A) Parameters -------------------------------
    fs = 48000           # Must match transmitter
    Tb = 0.04            # Must match transmitter
    fc = 4800
    bp_order = 5
    bp_rp = 1.0
    passband = (4750, 4850)
    lp_order = 5
    lp_rp = 1.0
    cutoff = 50

    # Decide how long to record. For example, 2 seconds:
    record_time = 10.0

    # ---------------------- (B) Record Audio ---------------------------------
    print(f"Recording for {record_time} seconds ...")
    # Mono recording => channels=1
    y_rec = sd.rec(int(record_time * fs), samplerate=fs, channels=1, blocking=True)
    print("Done recording.")

    # Uncomment this to test that the microphone is capturing the same played signal
    #sd.play(y_rec, samplerate=fs, blocking=True)

    # Reshape from (N,1) -> (N,)
    y_rec = y_rec.flatten()

    # --------------------- (C) Bandpass Filter (Receiver) --------------------
    b_bp, a_bp = design_chebyshev1_bandpass(bp_order, bp_rp, fs, passband)
    yr_f = signal.lfilter(b_bp, a_bp, y_rec)

    # --------------------- (D) IQ Demodulation -------------------------------
    n_r = np.arange(len(yr_f))
    yI_d = yr_f * np.cos(2.0 * np.pi * fc * n_r / fs)
    yQ_d = -yr_f * np.sin(2.0 * np.pi * fc * n_r / fs)

    # --------------------- (E) Lowpass Filter Each Branch --------------------
    b_lp, a_lp = design_chebyshev1_lowpass(lp_order, lp_rp, fs, cutoff)
    yI_b = signal.lfilter(b_lp, a_lp, yI_d)
    yQ_b = signal.lfilter(b_lp, a_lp, yQ_d)

    # Combine into complex baseband
    yb = yI_b + 1j * yQ_b

    # --------------------- (F) Decode Baseband -------------------------------
    br = wcs.decode_baseband_signal(np.abs(yb), np.angle(yb), Tb, fs)
    data_rx = wcs.decode_string(br)
    print("Received message:", data_rx)


if __name__ == "__main__":
    main()
