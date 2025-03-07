#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
from scipy import signal
import sounddevice as sd
import wcslib as wcs
from filters import design_passband_filter, design_lowpass_filter
###############################################################################
# 1. Filter-Design Functions
###############################################################################
def design_chebyshev1_bandpass(order, rp, fs, passband=(4750, 4850)):
    nyquist = fs / 2
    f1, f2 = passband
    Wn = [f1 / nyquist, f2 / nyquist]
    sos = signal.cheby1(order, rp, Wn, btype='band', analog=False, output='sos')
    return sos

###############################################################################
# 2. Main Transmitter Program
###############################################################################
def main():
    # -------------------------- (A) Parameters -------------------------------
    fs = 48000           # Audio sampling rate in Hz
    Tb = 0.04            # Symbol width (seconds)
    fc = 4800            # Carrier frequency in Hz

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
        data = "Hello world@"#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"# "Hello World, for gibb"#"Here is a longer sentence to send over"# "Hello World!"
        print('No valid input provided, defaulting to "{}"'.format(data))

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
    sos = design_passband_filter(gpass=0.2, gstop=60)
    x_tx = signal.sosfilt(sos, x_mod)         # Final transmit signal

    # ---------------------- (E) Play Out via Sounddevice ---------------------
    print("Transmitting your signal ...")
    sd.play(x_tx, samplerate=fs, blocking=True)
    print("Done transmitting!")

if __name__ == "__main__":
    main()
