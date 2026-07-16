#!/usr/bin/env python3
"""
Recorded Doppler spectrogram — Infineon SDK style
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

CLIPPING_VALUE = 1e-6
SPECT_THRESHOLD = 1e-6
SMOOTH_WINDOW = 7


def compute(
    input_path: Path,
    *,
    antenna: int,
    frame_rate_hz: float,
    max_speed_m_s: float,
    jet_vmin: float,
):
    """Load .npy, build global-range Doppler spectrogram, plot jet (time x, velocity y)."""
    data = np.load(input_path)
    if data.ndim != 4:
        raise ValueError(f"Expected (n_frame, n_ant, n_chirp, n_sample), got {data.shape}")

    n_frame, n_ant, n_chirp, n_sample = data.shape
    if antenna >= n_ant:
        raise ValueError(f"antenna {antenna} out of range (n_ant={n_ant})")

    range_fft_size = n_sample * 4
    doppler_fft_size = n_chirp * 4
    n_range_bins = range_fft_size // 2

    try:
        range_window = signal.blackmanharris(n_sample)
        doppler_window = signal.chebwin(n_chirp, at=100.0)
    except AttributeError:
        range_window = signal.windows.blackmanharris(n_sample)
        doppler_window = signal.windows.chebwin(n_chirp, at=100.0)
    doppler_window = doppler_window / np.sum(doppler_window)

    clip_db = 20.0 * np.log10(CLIPPING_VALUE)
    rdm_cube = np.zeros((n_frame, n_range_bins, doppler_fft_size), dtype=np.float64)

    for frame_idx in range(n_frame):
        frame = data[frame_idx, antenna].astype(np.float64, copy=False)
        rdm_complex = np.zeros((n_range_bins, n_chirp), dtype=np.complex128)

        for chirp_idx in range(n_chirp):
            chirp = frame[chirp_idx]
            x = chirp - np.mean(chirp)
            x = x * range_window
            buf = np.zeros(range_fft_size, dtype=np.complex128)
            buf[:n_sample] = x
            spectrum = np.fft.fft(buf)
            rdm_complex[:, chirp_idx] = spectrum[:n_range_bins]

        for range_idx in range(n_range_bins):
            slow_time = rdm_complex[range_idx] - np.mean(rdm_complex[range_idx])
            slow_time = slow_time * doppler_window
            buf = np.zeros(doppler_fft_size, dtype=np.complex128)
            buf[:n_chirp] = slow_time
            shifted = np.fft.fftshift(np.fft.fft(buf))
            power = np.abs(shifted) ** 2
            db = np.empty(doppler_fft_size, dtype=np.float64)
            above = power >= SPECT_THRESHOLD ** 2
            db[above] = 10.0 * np.log10(power[above])
            db[~above] = clip_db
            rdm_cube[frame_idx, range_idx] = db

    per_frame_energy = np.argmax(rdm_cube.sum(axis=2), axis=1).astype(np.int32)
    window = max(1, SMOOTH_WINDOW | 1)
    smoothed = median_filter(per_frame_energy.astype(np.float64), size=window)
    range_bin = np.clip(np.round(smoothed), 0, n_range_bins - 1).astype(np.int32)

    spectrogram = np.zeros((n_frame, doppler_fft_size), dtype=np.float64)
    for i in range(n_frame):
        spectrogram[i] = rdm_cube[i, range_bin[i], :]
    plot_data = spectrogram.T

    vel_min, vel_max = -max_speed_m_s, max_speed_m_s
    # duration_s = n_frame / frame_rate_hz
    duration_s = 3.0

    # Color limits: vmax from data; keep jet_vmin only if it is strictly below vmax.
    # Otherwise Normalize(vmin > vmax) raises ValueError on draw.
    vmax = float(np.nanmax(plot_data))
    vmin = jet_vmin if jet_vmin < vmax else vmax - 40.0

    fig = plt.figure(frameon=True)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])

    im = plt.imshow(
        plot_data,
        cmap="jet",
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True),
        aspect="auto",
        extent=[0, duration_s, vel_min, vel_max],
        # origin="lower",
    )
    
    plt.xlabel("time (s)")
    plt.ylabel("velocity (m/s)")
    plt.title(f"Doppler Spectrogram")
    plt.show()

    print(f"Input: {input_path}")
    print(f"Shape: {data.shape}")
    print(f"Duration: {duration_s:.3f} s, velocity: [{vel_min:.3f}, {vel_max:.3f}] m/s")

    return spectrogram, fig


def main() -> None:
    params = {
        "input_path": Path(r"C:\Infineon\Tools\Radar-Development-Kit\3.6.5\doppler_spectrogram\mix\3.npy"),
        "antenna": 0,
        "frame_rate_hz": 10.0,
        "max_speed_m_s": 6.19405905,
        "jet_vmin": -20.0,
    }

    _, fig = compute(**params)
    plt.show(block=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
