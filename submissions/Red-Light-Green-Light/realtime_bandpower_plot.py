import time
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

from brainflow_stream import BrainFlowBoardSetup
import brainflow


def remove_dc_offset(eeg_data: np.ndarray) -> np.ndarray:
    return eeg_data - np.mean(eeg_data, axis=1, keepdims=True)


def compute_band_powers(eeg_data: np.ndarray, sfreq: float, bands=None, nperseg: int = 256):
    if bands is None:
        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 12),
            "beta": (12, 30),
            "gamma": (30, 100),
        }

    n_channels = eeg_data.shape[0]
    band_powers = {band: np.zeros(n_channels) for band in bands}

    for ch in range(n_channels):
        seg = min(nperseg, eeg_data.shape[1])
        seg = max(seg, 64)
        freqs, psd = welch(eeg_data[ch], sfreq, nperseg=seg)
        for band, (low, high) in bands.items():
            idx = (freqs >= low) & (freqs <= high)
            if np.any(idx):
                band_powers[band][ch] = np.trapezoid(psd[idx], freqs[idx])
            else:
                band_powers[band][ch] = 0.0

    return band_powers


def main(serial_port: str = None, window_seconds: int = 2, refresh_hz: float = 5.0):
    board_id = brainflow.BoardIds.CYTON_BOARD.value
    setup = BrainFlowBoardSetup(board_id=board_id, serial_port=serial_port, name="Cyton")
    setup.setup()

    sfreq = setup.get_sampling_rate()
    if not sfreq:
        print("Failed to get sampling rate; exiting.")
        return

    eeg_chs = getattr(setup, "eeg_channels", []) or []
    if not eeg_chs:
        # Fallback for Cyton typical layout if descriptor lookup failed
        eeg_chs = list(range(1, 9))

    # Matplotlib setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    bands_order = ["delta", "theta", "alpha", "beta", "gamma"]
    x = np.arange(len(bands_order))
    bars = ax.bar(x, np.zeros(len(bands_order)))
    ax.set_xticks(x)
    ax.set_xticklabels(bands_order)
    ax.set_ylabel("Average Band Power (a.u.)")
    ax.set_title("Real-time EEG Band Powers (Averaged across 8 channels)")
    ax.set_ylim(0, 1)

    samples_needed = max(int(window_seconds * sfreq), 64)
    dt = 1.0 / refresh_hz

    try:
        # Give the stream a moment to buffer
        time.sleep(max(0.5, window_seconds))
        while True:
            data = setup.get_current_board_data(num_samples=samples_needed)
            if data is None or data.size == 0 or data.shape[1] < 8:
                time.sleep(dt)
                continue

            if max(eeg_chs) >= data.shape[0]:
                # Data shape not as expected yet; wait for buffer to fill
                time.sleep(dt)
                continue

            eeg = data[eeg_chs, :]
            eeg = remove_dc_offset(eeg)
            bands = compute_band_powers(eeg, sfreq)
            # Average across channels
            avg_values = np.array([bands[b].mean() for b in bands_order])

            # Update plot scaling smoothly
            cur_max = max(avg_values.max(), 1e-6)
            prev_top = ax.get_ylim()[1]
            new_top = max(prev_top * 0.9 + cur_max * 0.2, cur_max * 1.2)
            ax.set_ylim(0, new_top)

            for i, bar in enumerate(bars):
                bar.set_height(avg_values[i])

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        setup.stop()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-time band power visualization for OpenBCI Cyton")
    parser.add_argument("--port", type=str, default=None, help="Serial port like \\ \\.\\COM3 (Windows) or /dev/ttyUSB0 (Linux)")
    parser.add_argument("--window", type=float, default=2.0, help="Window length in seconds for PSD")
    parser.add_argument("--fps", type=float, default=5.0, help="Refresh rate (updates per second)")
    args = parser.parse_args()

    main(serial_port=args.port, window_seconds=int(args.window), refresh_hz=args.fps)
