import numpy as np
from scipy.signal import iirdesign, cheby1
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
def design_chebyshev1_bandpass(
    order,
    rp,
    fs=44100,
    passband=(4750, 4850),
):
    # Compute normalized passband edges (0 to 1) with respect to Nyquist frequency
    nyquist = fs / 2
    f1, f2 = passband
    Wn = [f1 / nyquist, f2 / nyquist]  # band edges in [0..1]

    # Design the filter with cheby1
    b, a = cheby1(order, rp, Wn, btype='band', analog=False, output='ba')
    return b, a

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

def design_passband_filter(wp=[4750.0, 4850.0], ws=[4700.0, 4900.0], gpass=0.5, gstop=40.0, fs=48000.0):
    nyquist = fs / 2.0
    wp_n = np.array(wp) / nyquist
    ws_n = np.array(ws) / nyquist
    
    # Design filter
    sos = iirdesign(
        wp_n,
        ws_n,
        gpass,
        gstop,
        analog=False,
        ftype="cheby1",
        output="sos"
    )
    return sos


def design_lowpass_filter(wp = 50, ws = 75, gpass = 0.5, gstop = 40, fs = 48000.0):
    nyquist = fs / 2.0
    wp_n = wp/nyquist
    ws_n = ws/nyquist
    sos = iirdesign(wp_n,
                     ws_n,
                     gpass,
                     gstop,
                     analog=False,
                     ftype="cheby1",
                     output="sos",
                     )
    return sos
