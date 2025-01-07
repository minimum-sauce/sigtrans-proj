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
import filters
from plotfilter import plot_filter_response

def simulation(data):
    # -------------------------------------------------------------------------
    # (A) Define your main system parameters
    # -------------------------------------------------------------------------
    fs = 38400          # Sampling rate (e.g., 48 kHz)
    Tb = 0.04           # Symbol width in seconds (example)
    fc = 4800           # Carrier frequency in Hz
    channel_id = 19     # Example channel ID; adjust if needed.

    # Filter specs
    bp_order = 5
    bp_rp    = 1.0
    lp_order = 5
    lp_rp    = 1.0
    cutoff   = 50
    string_data = True

    # # Detect input or set defaults
    # string_data = True
    # args = sys.argv[1:]
    # if len(args) == 1:
    #     data = str(args[0])
    # elif len(args) == 2 and args[0] == '-b':
    #     string_data = False
    #     data = str(args[1])
    # else:
    #print('Warning: No valid input, using defaults.', file=sys.stderr)
    #data = "this Is  A long sentence with some different type of letters. There is no Turning back! Are you alright?" #"Hello World!"

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
    # yr_c = xb#.astype(np.complex128)  # Make it complex: +1 -> phase=0, -1 -> phase=π
    # # Decode baseband
    # br = wcs.decode_baseband_signal(np.abs(yr_c), np.angle(yr_c), Tb, fs)
    # data_rx = wcs.decode_string(br)

    # -------------------------------------------------------------------------
    # (C) TRANSMITTER: Modulate & Band-limit
    # -------------------------------------------------------------------------
    # 1) IQ up-conversion at carrier = 4.8 kHz
    n = np.arange(len(xb))
    x_mod = xb * np.sin(2.0 * np.pi * fc * n / fs)

    # 2) Bandpass filter
    wp_bp = [4750.0, 4850.0]
    ws_bp = [4700.0, 4900.0]
    gpass = 0.2
    gstop = 60

    sos_bp = filters.design_passband_filter(wp_bp, ws_bp, gpass, gstop, fs)
    #plot_filter_response(b_bp, a_bp, fs, passband=wp_bp, stopband=ws_bp, gpass=gpass, gstop=gstop)
    
    #b_bp, a_bp = filters.design_chebyshev1_bandpass(bp_order, bp_rp, fs, passband=(4750, 4850))
    #b_bp, a_bp = design_passband_filter(fs=fs)
    #xt = signal.lfilter(b_bp, a_bp, x_mod)  # Transmitted signal
    xt = signal.sosfilt(sos_bp, x_mod)

    # -------------------------------------------------------------------------
    # (D) CHANNEL SIMULATION
    # -------------------------------------------------------------------------
    # For now, bypass or keep the channel
    yr = wcs.simulate_channel(xt, fs, channel_id)


    # -------------------------------------------------------------------------
    # (E) RECEIVER: Band-limiting -> IQ demod -> Lowpass -> Decode
    # -------------------------------------------------------------------------
    #yr_f = signal.soslfilter(b_bp, a_bp, yr)
    yr_f = signal.sosfilt(sos_bp, yr)

    # IQ demod
    n_r = np.arange(len(yr_f))
    yI_d = np.array(yr_f) * np.sin(2.0 * np.pi * fc * n_r / fs)
    yQ_d = -np.array(yr_f) * np.cos(2.0 * np.pi * fc * n_r / fs)

    # Lowpass each branch
    wp_lp = 30
    ws_lp = 80
    sos_lp = filters.design_lowpass_filter(wp_lp, ws_lp, gpass, gstop, fs)
    #b_lp, a_lp = filters.design_chebyshev1_lowpass(order=lp_order, rp=lp_rp, fs=fs, cutoff=cutoff)
    yI_b = signal.sosfilt(sos_lp, yI_d)
    yQ_b = signal.sosfilt(sos_lp, yQ_d)

    # Form complex baseband
    yb = np.array(yI_b) + 1j * np.array(yQ_b)


    # Decode baseband -> bits -> text
    br = wcs.decode_baseband_signal(np.abs(yb), np.angle(yb), Tb, fs)

    data_rx = wcs.decode_string(br)
    # Code for calculating bit error rate
    # char_errors = 0
    # min_len = min(len(data), len(data_rx))
    #
    # # Compare up to the shorter length
    # for i in range(min_len):
    #     if data[i] != data_rx[i]:
    #         char_errors += 1
    #
    # # If there's a length mismatch, count the extra characters as errors too
    # char_errors += abs(len(data) - len(data_rx))
    #
    # # Compute a "character error rate" by dividing by the length of the original data
    # total_chars = len(data)
    # char_error_rate = char_errors / total_chars if total_chars > 0 else 0.0
    # Code for calculating bit error rate
    bit_errors = 0

    # Convert data and data_rx to bit sequences
    data_bits = ''.join(format(ord(c), '08b') for c in data)   # Convert each character to 8-bit binary
    data_rx_bits = ''.join(format(ord(c), '08b') for c in data_rx)

    # Find the minimum length of the bit sequences
    min_len = min(len(data_bits), len(data_rx_bits))

    # Compare up to the shorter length
    for i in range(min_len):
        if data_bits[i] != data_rx_bits[i]:
            bit_errors += 1

    # Count extra bits as errors if the lengths are different
    bit_errors += abs(len(data_bits) - len(data_rx_bits))

    # Compute a "bit error rate" by dividing by the total number of bits in the original data
    total_bits = len(data_bits)
    bit_error_rate = bit_errors / total_bits if total_bits > 0 else 0.0

    # print(f"bit errors: {bit_errors}, total bit length: {total_bits}, BER: {bit_error_rate:.6f}")
    # print('Received: ' + data_rx)
    return (bit_errors, total_bits, bit_error_rate)


if __name__ == "__main__":
    data_strings = [
        "Hello World!",
        "if correct: Felix == Happy",
        "some shorter sentance!)(#/)",
        "Transverign 10923 Kronor into your bankaccount is banger! #$%",
        "this Is  A long sentence with some different type of letters. There is no Turning back! Are you alright?"
    ]

    iterations = 50
    grand_total_bit_errors = 0
    grand_total_bits = 0
    grand_total_correct = 0
    grand_total_transmissions = 0

    print("Starting simulation...\n")

    for data in data_strings:
        print(f"Testing string: \"{data[:50]}{'...' if len(data) > 50 else ''}\"")
        total_bit_errors = 0
        total_bits = 0
        max_ber = 0
        correct = 0
        min_ber = float('inf')
        for _ in range(iterations):
            b_err, b_len, ber = simulation(data)
            if b_err == 0:
                correct += 1
            total_bit_errors += b_err
            total_bits += b_len
            max_ber = max(max_ber, ber)
            min_ber = min(min_ber, ber)
        avg_ber = total_bit_errors / total_bits if total_bits > 0 else 0.0

        # Accumulate global totals
        grand_total_bit_errors += total_bit_errors
        grand_total_bits += total_bits
        grand_total_correct += correct 
        grand_total_transmissions += iterations

        # Display results for this string
        print(f"total nr of correct transmissions: {correct}/{iterations}")
        print(f"Total Bit Errors: {total_bit_errors}")
        print(f"Total Bits Transmitted: {total_bits}")
        print(f"Average BER: {avg_ber*100:.6f}%")
        print(f"Max BER: {max_ber*100:.6f}%")
        print(f"Min BER: {min_ber*100:.6f}%")
        print("-" * 50)

    # Compute and display the overall BER
    grand_total_ber = grand_total_bit_errors / grand_total_bits if grand_total_bits > 0 else 0.0
    print("\nSimulation completed.")
    print(f"Overall Total Bit Errors: {grand_total_bit_errors}")
    print(f"Overall Total Bits Transmitted: {grand_total_bits}")
    print(f"Overall Total BER: {grand_total_ber*100:.6f}%")
    print(f"total nr of correct transmissions: {grand_total_correct}/{grand_total_transmissions}")


