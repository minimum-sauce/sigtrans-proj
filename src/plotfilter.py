from warnings import filters
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sosfreqz

from filters import *

# font = {'family' : 'normal',
#   #      'weight' : 'bold',
#         'size'   : 18}
# plt.rc('font', **font)
#

def plot_lowpass_filter_response_logscale(sos, fs=10000.0, passband=50, stopband=70, rp=0.3, As=60):
    """
    Plots the frequency response of the filter on a *logarithmic* frequency scale.
    passband and stopband are just used for reference lines.
    """
    w, h = sosfreqz(sos, worN=2048)
    freqs = w * fs / (2.0 * np.pi)  # Convert rad/sample to Hz

    plt.figure(figsize=(8, 5))
    # Plot in dB versus a log frequency axis
    plt.semilogx(freqs, 20 * np.log10(np.abs(h)), label='Magnitude Response')
    # Draw vertical lines for passband/stopband edges
    plt.axvline(passband, color='green', linestyle='--', label='Passband Edge')
    plt.axvline(stopband, color='red', linestyle='--', label='Stopband Edge')
    
    # Example horizontal lines for passband ripple & stopband attenuation
    plt.axhline(-rp, color='orange', linestyle='--', label='Passband Ripple Limit')
    plt.axhline(-As, color='blue', linestyle='--', label='Stopband ~40 dB')

    plt.title("Chebyshev I Lowpass Filter Response (Log Scale)")
    plt.xlabel("Frequency (Hz) [Log Scale]")
    plt.ylabel("Magnitude (dB)")

    # Limit the frequency range to avoid log(0):
    plt.xlim([0, 150])    # from 1 Hz up to Nyquist (22050 Hz if fs=44100)
    plt.ylim([-80, 5])     # amplitude range
    plt.grid(True, which='both')  # grid on both major & minor ticks
    plt.legend()
    plt.show()

"""
    Plots the frequency response of the filter on a linear frequency scale.
    Also draws vertical lines for the passband edges.
"""
def plot_filter_response(sos, fs=10000.0, passband=(4750, 4850), stopband=(4700, 4900), title="Filter Response", gpass = 0.3, gstop = 60.0):
    w, h = sosfreqz(sos, worN=2048)  # Compute frequency response
    freqs = w * fs / (2.0 * np.pi)  # Convert rad/sample to Hz
    plt.figure(figsize=(8, 5))
    plt.semilogx(freqs, 20 * np.log10(np.abs(h)), label="Magnitude Response")

    # Highlight passband and stopband edges
    plt.axvline(passband[0], color='green', linestyle='--', label='Passband Edges')
    plt.axvline(passband[1], color='green', linestyle='--')
    plt.axvline(stopband[0], color='red', linestyle='--', label='Stopband Edges')
    plt.axvline(stopband[1], color='red', linestyle='--')

    # Horizontal lines for attenuation
    plt.axhline(-gpass, color='orange', linestyle='--', label='Passband Ripple')
    plt.axhline(-gstop, color='blue', linestyle='--', label='Stopband Attenuation')

    plt.title(title)
    plt.title("Chebyshev I Bandpass Filter Response")
    plt.ylabel("Magnitude (dB)")
    plt.xlim([4600, 5000])    # from 1 Hz up to Nyquist (22050 Hz if fs=44100)
    plt.ylim([-80, 5])     # amplitude range
    # plt.ylim([-60, 5])
    plt.grid(True, which='both')  # grid on both major & minor ticks
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # User-defined parameters
    # fs = 10000.0
    # passband = (4750, 4850)
    # stopband = (4700, 4900)
    # order = 6      # Try changing this to see how it affects steepness & ripple
    # rp = 1.0       # Passband ripple in dB

    # Design the filter
    fs = 10000.0
    wp = [4750.0, 4850.0]
    ws = [4700.0, 4900.0]
    gpass = 0.2
    gstop = 60

    sos = design_passband_filter(wp, ws, gpass, gstop, fs)
    # Plot its response
    #plot_filter_response(b, a, fs, passband, stopband, title=f'Chebyshev I Bandpass (Order={order}, rp={rp} dB)')
    plot_filter_response(sos, fs, passband=wp, stopband=ws)


    #---------------------------------------------------------------------------
    # Lowpass filter example
    #---------------------------------------------------------------------------
    passband = 50     # pass up to ~50 Hz in baseband
    stopband = 100      # stopband starts at ~70 Hz

    sos = design_lowpass_filter(passband, stopband, gpass, gstop, fs)
    plot_lowpass_filter_response_logscale(sos, fs, passband=passband, stopband=stopband, rp=gpass, As=gstop)
