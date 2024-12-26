#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
from scipy import signal
import sounddevice as sd
import wcslib as wcs

###############################################################################
# 1. Filter-Design Functions
###############################################################################
def design_chebyshev1_bandpass(order, rp, fs, passband=(4750, 4850)):
    nyquist = fs / 2
    f1, f2 = passband
    Wn = [f1 / nyquist, f2 / nyquist]
    b, a = signal.cheby1(order, rp, Wn, btype='band', analog=False, output='ba')
    return b, a

###############################################################################
# 2. Main Transmitter Program
###############################################################################
def main():
    # -------------------------- (A) Parameters -------------------------------
    fs = 48000           # Audio sampling rate in Hz
    Tb = 0.04            # Symbol width (seconds)
    fc = 4800            # Carrier frequency in Hz
    bp_order = 5         # Bandpass filter order
    bp_rp = 1.0          # Passband ripple (dB)
    passband = (4750, 4850)

    # ---------------------- (B) Get Input from Command Line ------------------
    # E.g., run: python transmitter.py "Hello World!"
    # or       : python transmitter.py -b 0100100001101
    string_data = True
    args = sys.argv[1:]
    if len(args) == 1:
        data = str(args[0])
    elif len(args) == 2 and args[0] == '-b':
        string_data = False
        data = str(args[1])
    else:
        print('No valid input provided, defaulting to "Hello World!"')
        data = "Hello World!"

    # -------------------- (C) Encode into Baseband --------------------------
    # 1. Convert string -> bits
    if string_data:
        bs = wcs.encode_string(data)
    else:
        bs = np.array([int(bit) for bit in data])

    # 2. Convert bits -> pulses (+1/-1) at baseband
    xb = wcs.encode_baseband_signal(bs, Tb, fs)

    # -------------------- (D) Up-Convert & Bandpass Filter -------------------
    n = np.arange(len(xb))
    x_mod = xb * np.cos(2.0 * np.pi * fc * n / fs)   # Real up-conversion
    b_bp, a_bp = design_chebyshev1_bandpass(bp_order, bp_rp, fs, passband)
    x_tx = signal.lfilter(b_bp, a_bp, x_mod)         # Final transmit signal

    # ---------------------- (E) Play Out via Sounddevice ---------------------
    print("Transmitting your signal ...")
    sd.play(x_tx, samplerate=fs, blocking=True)
    print("Done transmitting!")

if __name__ == "__main__":
    main()
