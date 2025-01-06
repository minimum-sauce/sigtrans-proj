#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sounddevice as sd
from matplotlib import pyplot as plt
from scipy import signal
import wcslib as wcs
from filters import design_passband_filter, design_lowpass_filter

# def design_chebyshev1_bandpass(order, rp, fs, passband=(4750, 4850)):
#     nyquist = fs / 2
#     f1, f2 = passband
#     Wn = [f1 / nyquist, f2 / nyquist]
#     sos = signal.cheby1(order, rp, Wn, btype='band', analog=False, output='sos')
#     return sos
#
# def design_chebyshev1_lowpass(order, rp, fs, cutoff=50):
#     nyquist = fs / 2
#     Wn = cutoff / nyquist
#     sos = signal.cheby1(order, rp, Wn, btype='low', analog=False, output='sos')
#     return sos

def main():
    # -------------------------- (A) Parameters -------------------------------
    fs = 48000           # Must match transmitter
    Tb = 0.04            # Must match transmitter
    fc = 4800
    #bp_order = 5
    #bp_rp = 0.2
    #passband = (4750, 4850)
    #lp_order = 5
    #lp_rp = 0.2
    wp = np.array([4750.0, 4850.0])
    ws = np.array([4700.0, 4900.0])
    wp = np.array([4300.0, 4500.0])
    ws = np.array([4250.0, 4550.0])
    gpass = 0.3
    gstop = 60

    # Decide how long to record. For example, 2 seconds:
    record_time = 18.0

    # ---------------------- (B) Record Audio ---------------------------------
    print(f"Recording for {record_time} seconds ...")
    # Mono recording => channels=1
    y_rec = sd.rec(int(record_time * fs), samplerate=fs, channels=1, blocking=True)
    print("Done recording.")

    # Uncomment this to test that the microphone is capturing the same played signal
    # sd.play(y_rec, samplerate=fs, blocking=True)

    # Reshape from (N,1) -> (N,)
    y_rec = y_rec.flatten()

    # --------------------- (C) Bandpass Filter (Receiver) --------------------

    sos_bp = design_passband_filter(wp, ws, gpass, gstop, fs)
    #sos_bp = design_passband_filter(bp_order, bp_rp, fs, passband)
    yr_f = signal.sosfilt(sos_bp, y_rec)

    # --------------------- (D) IQ Demodulation -------------------------------
    n_r = np.arange(len(yr_f))
    yI_d = yr_f * np.sin(2.0 * np.pi * fc * n_r / fs)
    yQ_d = -1 * yr_f * np.cos(2.0 * np.pi * fc * n_r / fs)

    # --------------------- (E) Lowpass Filter Each Branch --------------------
    passband = 50     # pass up to ~50 Hz in baseband
    stopband = 75      # stopband starts at ~70 Hz

    sos_lp = design_lowpass_filter(passband, stopband, gpass, gstop, fs)
    yI_b = np.array(signal.sosfilt(sos_lp, yI_d)) # In-phase signal
    yQ_b = np.array(signal.sosfilt(sos_lp, yQ_d)) # Quadrature-phase signal

    # Combine into complex baseband
    yb = yI_b + 1j * yQ_b

    br = wcs.decode_baseband_signal(np.abs(yb), np.angle(yb), Tb, fs)
    data_rx = wcs.decode_string(br)
    print("Received message:", data_rx)

    # Generate time array corresponding to n_r
    t = n_r / fs  # Convert sample indices to time

    # Plot filtered I and Q signals
    plt.figure(figsize=(10, 6))
    plt.plot(t, yI_b, label='Filtered In-phase (I)', color='blue')
    plt.plot(t, yQ_b, label='Filtered Quadrature (Q)', color='orange')
    plt.title('Filtered In-phase and Quadrature Signals')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()
    # --------------------- (F) Decode Baseband -------------------------------


if __name__ == "__main__":
    main()
