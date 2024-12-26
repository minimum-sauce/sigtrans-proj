#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation template for the wireless communication system project in Signals 
and Transforms.

For plain text inputs, run:
$ python3 simulation.py "Hello World!"

For binary inputs, run:
$ python3 simulation.py -b 010010000110100100100001

2020-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
"""

import sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import wcslib as wcs
from plotfilter import design_lowpass_filter, design_passband_filter, design_chebyshev1_bandpass, design_chebyshev1_lowpass

def main():
    # -------------------------------------------------------------------------
    # (A) Define your main system parameters
    # -------------------------------------------------------------------------
    fs = 48000          # Sampling rate (e.g., 48 kHz)
    Tb = 0.04           # Symbol width in seconds (example)
    fc = 4800           # Carrier frequency in Hz
    channel_id = 19     # Example channel ID; adjust if needed.

    # Filter specs
    bp_order = 5
    bp_rp    = 1.0
    lp_order = 5
    lp_rp    = 1.0
    cutoff   = 50

    # Detect input or set defaults
    string_data = True
    args = sys.argv[1:]
    if len(args) == 1:
        data = str(args[0])
    elif len(args) == 2 and args[0] == '-b':
        string_data = False
        data = str(args[1])
    else:
        print('Warning: No valid input, using defaults.', file=sys.stderr)
        data = "Hello World!"

    # -------------------------------------------------------------------------
    # (B) Convert the input string -> bits -> baseband signal
    # -------------------------------------------------------------------------
    if string_data:
        bs = wcs.encode_string(data)
    else:
        bs = np.array([int(bit) for bit in data])

    xb = wcs.encode_baseband_signal(bs, Tb, fs)

    # -------------------------------------------------------------------------
    # (B1) Baseband-only decode test (NO channel, NO filters):
    #      Convert xb to complex so decode_baseband_signal sees phase flips.
    # -------------------------------------------------------------------------
    yr_c = xb.astype(np.complex128)  # Make it complex: +1 -> phase=0, -1 -> phase=π

    # Decode baseband
    br = wcs.decode_baseband_signal(np.abs(yr_c), np.angle(yr_c), Tb, fs)
    data_rx = wcs.decode_string(br)

    # -------------------------------------------------------------------------
    # (C) TRANSMITTER: Modulate & Band-limit
    # -------------------------------------------------------------------------
    # 1) IQ up-conversion at carrier = 4.8 kHz
    n = np.arange(len(xb))
    x_mod = xb * np.cos(2.0 * np.pi * fc * n / fs)

    # 2) Bandpass filter
    b_bp, a_bp = design_chebyshev1_bandpass(bp_order, bp_rp, fs, passband=(4750, 4850))
    #b_bp, a_bp = design_passband_filter(fs=fs)
    xt = signal.lfilter(b_bp, a_bp, x_mod)  # Transmitted signal

    # -------------------------------------------------------------------------
    # (D) CHANNEL SIMULATION
    # -------------------------------------------------------------------------
    # For now, bypass or keep the channel
    yr = wcs.simulate_channel(xt, fs, channel_id)

    # -------------------------------------------------------------------------
    # (E) RECEIVER: Band-limiting -> IQ demod -> Lowpass -> Decode
    # -------------------------------------------------------------------------
    yr_f = signal.lfilter(b_bp, a_bp, yr)

    # IQ demod
    n_r = np.arange(len(yr_f))
    yI_d = yr_f * np.cos(2.0 * np.pi * fc * n_r / fs)
    yQ_d = -yr_f * np.sin(2.0 * np.pi * fc * n_r / fs)

    # Lowpass each branch
    b_lp, a_lp = design_chebyshev1_lowpass(order=lp_order, rp=lp_rp, fs=fs, cutoff=cutoff)
    yI_b = signal.lfilter(b_lp, a_lp, yI_d)
    yQ_b = signal.lfilter(b_lp, a_lp, yQ_d)

    # Form complex baseband
    yb = yI_b + 1j * yQ_b

    # Decode baseband -> bits -> text
    br = wcs.decode_baseband_signal(np.abs(yb), np.angle(yb), Tb, fs)
    data_rx = wcs.decode_string(br)
    print('Received: ' + data_rx)


if __name__ == "__main__":
    main()
