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
    # wp = np.array([4300.0, 4500.0])
    # ws = np.array([4250.0, 4550.0])
    gpass = 0.3
    gstop = 60

    # Decide how long to record. For example, 2 seconds:
    record_time = 7.0

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
    stopband = 100      # stopband starts at ~70 Hz

    sos_lp = design_lowpass_filter(passband, stopband, gpass, gstop, fs)
    yI_b = np.array(signal.sosfilt(sos_lp, yI_d)) # In-phase signal
    yQ_b = np.array(signal.sosfilt(sos_lp, yQ_d)) # Quadrature-phase signal

    # Combine into complex baseband
    yb = yI_b + 1j * yQ_b

    br = wcs.decode_baseband_signal(np.abs(yb), np.angle(yb), Tb, fs)
    data_rx = wcs.decode_string(br)
    print("Received message:", data_rx)


    # Create a figure with 2 rows and 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # 2 rows, 2 columns

    # Plot y_rec (raw received signal)
    t = np.arange(len(y_rec))
    axs[0, 0].plot(t, y_rec, label='$y_{rec}$ (Raw Signal)', color='blue')
    axs[0, 0].set_title('Raw Signal')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].legend()
    axs[0, 0].grid()

    # Plot yr_f (filtered signal)
    t = n_r / fs  # Convert sample indices to time
    axs[0, 1].plot(t, yr_f, label='$y_{r_f}$ (Filtered Signal)', color='orange')
    axs[0, 1].set_title('Filtered Signal')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Amplitude')
    axs[0, 1].legend()
    axs[0, 1].grid()

    # Plot the angle of the complex baseband signal yb
    axs[1, 0].plot(t, np.angle(yb), label='Angle of $y_b$', color='green')
    axs[1, 0].set_title('Angle of the Complex Baseband Signal ($y_b$)')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Phase Angle (radians)')
    axs[1, 0].grid()
    axs[1, 0].legend()

    # Plot filtered In-phase and Quadrature-phase signals
    axs[1, 1].plot(t, yI_b, label='Filtered In-phase (I)', color='blue')
    axs[1, 1].plot(t, yQ_b, label='Filtered Quadrature (Q)', color='orange')
    axs[1, 1].set_title('Filtered In-phase and Quadrature Signals')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Amplitude')
    axs[1, 1].grid()
    axs[1, 1].legend()

    # Adjust layout to avoid overlap between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
    # --------------------- (F) Decode Baseband -------------------------------


if __name__ == "__main__":
    main()
