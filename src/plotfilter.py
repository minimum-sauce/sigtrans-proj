import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cheby1, freqz

def design_chebyshev1_bandpass(
    order,
    rp,
    fs=44100,
    passband=(4750, 4850),
):
    """
    Designs a Chebyshev Type I bandpass filter of fixed order.

    Parameters
    ----------
    order : int
        The order of the Chebyshev filter.
    rp : float
        The maximum passband ripple in dB (Chebyshev ripple).
    fs : float
        Sampling frequency in Hz.
    passband : tuple
        Passband frequencies in Hz (f_low, f_high).

    Returns
    -------
    b, a : ndarray
        The IIR filter coefficients.
    """
    # Compute normalized passband edges (0 to 1) with respect to Nyquist frequency
    nyquist = fs / 2
    f1, f2 = passband
    Wn = [f1 / nyquist, f2 / nyquist]  # band edges in [0..1]

    # Design the filter with cheby1
    b, a = cheby1(order, rp, Wn, btype='band', analog=False, output='ba')
    return b, a

def plot_filter_response(b, a, fs=44100, passband=(4750, 4850), title='Fixed-Order Chebyshev Type I Bandpass'):
    """
    Plots the frequency response of the filter on a linear frequency scale.
    Also draws vertical lines for the passband edges.
    """
    w, h = freqz(b, a, worN=2048)
    freqs = w * fs / (2 * np.pi)  # Convert from rad/sample to Hz

    plt.figure(figsize=(8, 5))
    plt.plot(freqs, 20 * np.log10(np.abs(h)), label='Magnitude Response')

    # Draw dashed lines for passband edges (green)
    plt.axvline(x=passband[0], color='green', linestyle='--', label='Passband Edges')
    plt.axvline(x=passband[1], color='green', linestyle='--')

    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.ylim([-80, 5])
    plt.xlim([4600, 5000])  # Just to see the region around the band
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # User-defined parameters
    fs = 44100
    passband = (4750, 4850)
    order = 6      # Try changing this to see how it affects steepness & ripple
    rp = 1.0       # Passband ripple in dB

    # Design the filter
    b, a = design_chebyshev1_bandpass(order, rp, fs, passband)

    # Plot its response
    plot_filter_response(b, a, fs, passband, title=f'Chebyshev I Bandpass (Order={order}, rp={rp} dB)')
