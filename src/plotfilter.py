import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirdesign, freqs, cheby1

def design_chebyshev_bandpass_analog(
    passband=(4750, 4850),
    stopband=(4700, 4900),
    gpass=1,      # Passband ripple in dB
    gstop=60,     # Stopband attenuation in dB
):
    """
    Designs an analog Chebyshev Type II bandpass filter with the given passband/stopband edges (in Hz).
    Returns the filter's analog transfer function coefficients b (numerator) and a (denominator).
    """
    # Convert from Hz to rad/s: ω (rad/s) = 2πf
    wp = [2 * np.pi * f for f in passband]
    ws = [2 * np.pi * f for f in stopband]
    
    # Design an analog Chebyshev Type II filter:
    b, a = iirdesign(
        wp=wp,         # passband edges in rad/s
        ws=ws,         # stopband edges in rad/s
        gpass=gpass,   # passband ripple (dB)
        gstop=gstop,   # stopband attenuation (dB)
        ftype='cheby1',
        analog=True
    )
    return b, a

def plot_filter_response_analog(b, a, passband=(4750, 4850), stopband=(4700, 4900), title='Analog Chebyshev Type II'):
    """
    Plots the frequency response (magnitude in dB) of the analog filter
    on a linear frequency scale from 4400 Hz to 5200 Hz.
    Also draws dashed lines for passband and stopband edges.
    """
    # Create a linearly spaced range of angular frequencies (rad/s)
    # from 4400 Hz to 5200 Hz (in terms of rad/s).
    w = np.linspace(2 * np.pi * 4400, 2 * np.pi * 5200, 2000)
    
    # Evaluate the filter's analog frequency response
    w, h = freqs(b, a, w)
    
    # Convert rad/s back to Hz for plotting
    freqs_hz = w / (2 * np.pi)
    
    plt.figure(figsize=(8, 5))
    
    # Plot filter magnitude (in dB)
    plt.plot(freqs_hz, 20 * np.log10(np.abs(h)), label='Magnitude Response')
    
    # Draw dashed lines for passband edges (in green)
    plt.axvline(x=passband[0], color='green', linestyle='--', linewidth=1, label='Passband Edges')
    plt.axvline(x=passband[1], color='green', linestyle='--', linewidth=1)
    
    # Draw dashed lines for stopband edges (in red)
    plt.axvline(x=stopband[0], color='red', linestyle='--', linewidth=1, label='Stopband Edges')
    plt.axvline(x=stopband[1], color='red', linestyle='--', linewidth=1)
    
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    
    # Set the y-limits to see enough attenuation
    plt.ylim([-100, 5])
    # Restrict x-axis to [4400, 5200] on a linear scale
    plt.xlim([4400, 5200])
    
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Define passband and stopband in Hz
    passband = (4750, 4850)
    stopband = (4700, 4900)

    # Design the analog filter
    b, a = design_chebyshev_bandpass_analog(
        passband=passband,
        stopband=stopband,
        gpass=1,   # 1 dB ripple in passband
        gstop=60   # 60 dB attenuation in stopband
    )

    # Plot its frequency response from 4400 Hz to 5200 Hz
    plot_filter_response_analog(b, a, passband=passband, stopband=stopband,
                                title='Analog Chebyshev Type I (4750-4850 Hz)')
