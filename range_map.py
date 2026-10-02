#!/usr/bin/env python3
"""
Recorded range map — same style as the Doppler spectrogram script.

Range FFT matches that script (mean removal, Blackman–Harris, 4× zero-pad).
mti=True sums Doppler power except the zero-velocity bin. mti=False averages
the range-FFT magnitude and keeps stationary reflections.
Plot is jet, time on x, range on y.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

CLIPPING_VALUE = 1e-6
SPECT_THRESHOLD = 1e-6


def compute(
    input_path: Path,
    *,
    antenna: int,
    frame_rate_hz: float,
    max_range_m: float,
    jet_vmin: float,
    mti: bool,
):
    """Load .npy, build a range–time map, plot jet (time x, range y)."""
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
    range_map_db = np.full((n_frame, n_range_bins), clip_db, dtype=np.float64)
    dc = doppler_fft_size // 2

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

        if mti:
            range_power = np.zeros(n_range_bins, dtype=np.float64)
            for range_idx in range(n_range_bins):
                slow_time = rdm_complex[range_idx] - np.mean(rdm_complex[range_idx])
                slow_time = slow_time * doppler_window
                buf = np.zeros(doppler_fft_size, dtype=np.complex128)
                buf[:n_chirp] = slow_time
                shifted = np.fft.fftshift(np.fft.fft(buf))
                power = np.abs(shifted) ** 2
                range_power[range_idx] = np.sum(power) - power[dc]
            above = range_power >= SPECT_THRESHOLD ** 2
            range_map_db[frame_idx, above] = 10.0 * np.log10(range_power[above])
        else:
            magnitude = np.mean(np.abs(rdm_complex), axis=1)
            above = magnitude >= CLIPPING_VALUE
            range_map_db[frame_idx, above] = 20.0 * np.log10(magnitude[above])
    plot_data = range_map_db.T

    # duration_s = n_frame / frame_rate_hz
    duration_s = 3.0

    vmax = float(np.nanmax(plot_data))
    vmin = jet_vmin if jet_vmin < vmax else vmax - 40.0

    fig = plt.figure(frameon=True)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])

    im = plt.imshow(
        plot_data,
        cmap="jet",
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True),
        aspect="auto",
        origin="lower",
        extent=[0, duration_s, 0.0, max_range_m],
    )

    plt.xlabel("time (s)")
    plt.ylabel("range (m)")
    mti_note = "MTI on" if mti else "MTI off"
    plt.title(f"Range Map  |  {mti_note}")
    plt.show()

    print(f"Input: {input_path}")
    print(f"Shape: {data.shape}")
    print(f"MTI: {mti_note}")
    print(f"Duration: {duration_s:.3f} s, range: [0, {max_range_m:.3f}] m")

    return range_map_db, fig


def main() -> None:
    params = {
        "input_path": Path(r".\doppler_spectrogram\mix\3.npy"),
        "antenna": 0,
        "frame_rate_hz": 10.0,
        "max_range_m": 3.48,
        "jet_vmin": -20.0,
        "mti": True,
    }

    _, fig = compute(**params)
    plt.show(block=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
