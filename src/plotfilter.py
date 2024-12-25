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

def plot_filter_response(b, a, fs=44100, passband=(4750, 4850), stopband=(4700, 4900), title='Fixed-Order Chebyshev Type I Bandpass'):
    """
    Plots the frequency response of the filter on a linear frequency scale.
    Also draws vertical lines for the passband edges.
    """
    w, h = freqz(b, a, worN=4096)
    freqs = w * fs / (2 * np.pi)  # Convert from rad/sample to Hz

    plt.figure(figsize=(8, 5))
    plt.plot(freqs, 20 * np.log10(np.abs(h)), label='Magnitude Response')

    # Draw dashed lines for passband edges (green)
    plt.axvline(x=passband[0], color='green', linestyle='--', label='Passband Edges')
    plt.axvline(x=passband[1], color='green', linestyle='--')
    # Draw stopband frequencies
    plt.axvline(x=stopband[0], color='red', linestyle='--', label='Stopband Edges')
    plt.axvline(x=stopband[1], color='red', linestyle='--')

    # Draw horisontal lines for passband and stopband attenuation
    plt.axhline(y=-2, color='orange', linestyle='--', label='Passband Attenuation')
    plt.axhline(y=-60, color='blue', linestyle='--', label='Stopband Attenuation')

    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.ylim([-80, 5])
    plt.xlim([4600, 5000])  # Just to see the region around the band
    plt.grid(True)
    plt.legend()
    plt.show()


def design_chebyshev1_lowpass(
    order,
    rp,
    fs=44100,
    cutoff=50,
):
    """
    Designs a Chebyshev Type I lowpass filter of given order.

    Parameters
    ----------
    order : int
        The filter order.
    rp : float
        Passband ripple in dB.
    fs : float
        Sampling frequency in Hz.
    cutoff : float
        Lowpass cutoff frequency in Hz (passband edge).

    Returns
    -------
    b, a : ndarray
        IIR filter coefficients.
    """
    nyquist = fs / 2
    Wn = cutoff / nyquist  # normalized cutoff in [0..1]
    
    b, a = cheby1(order, rp, Wn, btype='low', analog=False, output='ba')
    return b, a

def plot_lowpass_filter_response_logscale(b, a, fs=44100, passband=50, stopband=70, rp=1.0):
    """
    Plots the frequency response of the filter on a *logarithmic* frequency scale.
    passband and stopband are just used for reference lines.
    """
    w, h = freqz(b, a, worN=2048)
    freqs = w * fs / (2.0 * np.pi)  # Convert rad/sample to Hz

    plt.figure(figsize=(8, 5))
    # Plot in dB versus a log frequency axis
    plt.semilogx(freqs, 20 * np.log10(np.abs(h)), label='Magnitude Response')
    
    # Draw vertical lines for passband/stopband edges
    plt.axvline(passband, color='green', linestyle='--', label='Passband Edge')
    plt.axvline(stopband, color='red', linestyle='--', label='Stopband Edge')
    
    # Example horizontal lines for passband ripple & stopband attenuation
    plt.axhline(-rp, color='orange', linestyle='--', label='Passband Ripple Limit')
    plt.axhline(-40, color='blue', linestyle='--', label='Stopband ~40 dB')

    plt.title("Chebyshev I Lowpass Filter Response (Log Scale)")
    plt.xlabel("Frequency (Hz) [Log Scale]")
    plt.ylabel("Magnitude (dB)")

    # Limit the frequency range to avoid log(0):
    plt.xlim([1, fs/2])    # from 1 Hz up to Nyquist (22050 Hz if fs=44100)
    plt.ylim([-80, 5])     # amplitude range
    plt.grid(True, which='both')  # grid on both major & minor ticks
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # User-defined parameters
    fs = 48000
    passband = (4750, 4850)
    order = 6      # Try changing this to see how it affects steepness & ripple
    rp = 1.0       # Passband ripple in dB

    # Design the filter
    b, a = design_chebyshev1_bandpass(order, rp, fs, passband)

    # Plot its response
    plot_filter_response(b, a, fs, passband,title=f'Chebyshev I Bandpass (Order={order}, rp={rp} dB)')


    #---------------------------------------------------------------------------
    # Lowpass filter example
    #---------------------------------------------------------------------------
    order = 4      # Start with 5, can tweak
    rp = 1.0        # 1 dB ripple
    cutoff = 50     # pass up to ~50 Hz in baseband
    stopb = 70      # stopband starts at ~70 Hz

    b, a = design_chebyshev1_lowpass(order, rp, fs, cutoff)
    plot_lowpass_filter_response_logscale(b, a, fs, passband=cutoff, stopband=stopb)
