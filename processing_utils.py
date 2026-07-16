import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy.signal import butter, lfilter, hilbert
from scipy import signal
import time


def stft(data, window, nfft, shift):
    """
    Compute the Short-Time Fourier Transform (STFT) of the input data.
    Args:
        data (np.ndarray): Input 1D signal data.
        window (int): Length of each segment/window.
        nfft (int): Number of FFT points.
        shift (int): Number of samples to shift for the next segment.
    Returns:
        np.ndarray: STFT of the input data.
    """
    n = (len(data) - window - 1) // shift
    out = np.zeros((nfft, n), dtype=complex)
    win = np.hanning(window)
    for i in range(n):
        segment = data[i * shift : i * shift + window] * win
        out[:, i] = np.fft.fft(segment, n=nfft)
    return out


def plot_spectrogram(spect, duration, prf, max_vel, savename=None):
    """
    Plot and optionally save the spectrogram.
    Args:
        spect (np.ndarray): Spectrogram data to plot.
        duration (float): Duration of the signal in seconds.
        prf (float): Pulse repetition frequency in Hz.
        max_vel (float): Maximum velocity in m/s.
        savename (str, optional): If provided, the path to save the figure. Defaults to None.
    """
    fig = plt.figure(frameon=True)
    ax = plt.Axes(fig, [0., 0., 1., 1.])

    maxval = np.max(spect)
    norm = colors.Normalize(vmin=-20, vmax=None, clip=True)

    # gcf (with axes)
    im = plt.imshow(20 * np.log10((abs(spect) / maxval)), cmap='jet', norm=norm, aspect="auto",
    #               extent=[0, duration, -prf/2, prf/2],
                    extent=[0, duration, -max_vel, max_vel]
                    )
    plt.xlabel('Time (sec)')
    #plt.ylabel('Frequency (Hz)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Micro-Doppler Spectrogram')
    # plt.show()

    if savename is not None:
        print(f"Saving spectrogram to {savename}")
        plt.axis('off')
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off',
                        labelright='off', labelbottom='off')
        im.get_figure().gca().set_title("")
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(savename, bbox_inches='tight', transparent=True, pad_inches=0)
        plt.close()


def spectrogram(data, duration, prf, max_vel, mti=False, is_save=None, savename=None):
    """
    Generate and plot the spectrogram from radar data.
    Args:
        data (np.ndarray): Raw radar data.
        duration (float): Duration of the signal in seconds.
        prf (float): Pulse repetition frequency in Hz.
        max_vel (float): Maximum velocity in m/s.
        mti (bool, optional): Whether to apply Moving Target Indication (MTI) filtering. Defaults to False.
        is_save (bool, optional): Whether to save the spectrogram. Defaults to None.
        savename (str, optional): If provided, the path to save the figure. Defaults to None.
    Usage:
        spectrogram(data, duration=duration, prf=prf, max_vel=max_vel, mti=True, is_save=True, savename="microdoppler_spectrogram.png")
    """
    start_time = time.time()

    data = data[:, 0, :, :]
    data = np.transpose(data, (2, 1, 0))
    num_samples = data.shape[0]
    num_chirps = data.shape[1]*data.shape[2]
    data = data.reshape((num_samples, num_chirps), order='F').astype(np.float64)

    data = hilbert(data, axis=0)

    range_fft = np.fft.fft(data, 2*num_samples, axis=0)[:num_samples] / num_samples
    range_fft -= np.mean(range_fft, axis=1, keepdims=True)

    if mti:
        b, a = butter(1, 0.01, 'high', output='ba')
        rngpro = np.zeros_like(range_fft)
        for r in range(rngpro.shape[0]):
            rngpro[r, :] = lfilter(b, a, range_fft[r, :].real) + 1j * lfilter(b, a, range_fft[r, :].imag)
    else:
        rngpro = range_fft

    energy = np.sum(np.abs(rngpro) ** 2, axis=1)
    peak = int(np.argmax(energy))
    range_bin_offset = 3 #3*2 bins around the peak
    r0 = max(1, peak - range_bin_offset)
    r1 = min(rngpro.shape[0], peak + range_bin_offset + 1)
    rBin = np.arange(r0, r1)

    #rBin = np.arange(1, num_samples//2 + 1) # 1...128
    #rBin = np.arange(20, 31) # 20...30
    
    nfft = 2 ** 10
    window = 256
    noverlap = 200
    shift = window - noverlap

    vec = np.sum(rngpro[rBin, :], 0)

    spect = stft(vec, window, nfft, shift)
    spect = np.abs(np.fft.fftshift(spect, 0))

    print(f"Generated spectrogram in {time.time() - start_time:.2f} seconds")

    plot_spectrogram(spect, duration, prf, max_vel, savename=savename)
