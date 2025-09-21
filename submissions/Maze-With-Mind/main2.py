import scipy
import numpy as np
import matplotlib.pyplot as plt
import time
import mne

from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

from scipy.signal import butter, filtfilt

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BrainFlowError, BoardIds

from brainflow_stream import BrainFlowBoardSetup


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a Butterworth bandpass filter to multichannel data.

    Parameters:
        data (np.ndarray): Shape (n_channels, n_samples)
        lowcut (float): Low cutoff frequency (Hz)
        highcut (float): High cutoff frequency (Hz)
        fs (float): Sampling rate (Hz)
        order (int): Filter order

    Returns:
        np.ndarray: Filtered data, same shape as input
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=1)


board_id = BoardIds.CYTON_BOARD.value # Set the board_id to match the Cyton board

# Lets quickly take a look at the specifications of the Cyton board
for item1, item2 in BoardShim.get_board_descr(board_id).items():
    print(f"{item1}: {item2}")

cyton_board = BrainFlowBoardSetup(
                                board_id = board_id,
                                name = 'Board_1',   # Optional name for the board. This is useful if you have multiple boards connected and want to distinguish between them.
                                serial_port = None  # If the serial port is not specified, it will try to auto-detect the board. If this fails, you will have to assign the correct serial port. See https://docs.openbci.com/GettingStarted/Boards/CytonGS/ 
                                ) 

cyton_board.setup() # This will establish a connection to the board and start streaming data.

board_info = cyton_board.get_board_info() # Retrieves the EEG channel and sampling rate of the board.
print(f"Board info: {board_info}")

board_srate = cyton_board.get_sampling_rate() # Retrieves the sampling rate of the board.
print(f"Board sampling rate: {board_srate}")

time.sleep(5) # Wait for 5 seconds to allow the board to build up some samples into the buffer

raw_data_500 = cyton_board.get_current_board_data(num_samples = 1000) # Get the latest 1000 samples from the buffer
print(f"raw_data_1000 shape: {raw_data_500.shape}")

raw_data_all = cyton_board.get_board_data() 
print(f"raw_data_all shape: {raw_data_all.shape}")

eeg_data = raw_data_500[1:9, :] # Get the EEG data from the first 8 channels
print(f"eeg_data shape: {eeg_data.shape}")

# To do this we can subtract the mean of the data from the data itself. This will center the data around zero.
eeg_data_dc_removed = eeg_data - np.mean(eeg_data, axis=1, keepdims=True)
bandpass_filtered_data = bandpass_filter(eeg_data_dc_removed, lowcut=5.0, highcut=30.0, fs=board_srate, order=4)

data_to_use = eeg_data_dc_removed
print(f"bandpass_filtered_data shape: {bandpass_filtered_data.shape}")
print(f"eeg_data_dc_removed shape: {eeg_data_dc_removed.shape}")

# While we're at it, lets make a small function that performs all of our minimal processing this since we'll have to do it every time we pull data from the board.
def remove_dc_offset(data):
    return data[1:9, :] - np.mean(data[1:9, :], axis=1, keepdims=True)

# Now let's plot the results
num_channels = data_to_use.shape[0]
num_samples = data_to_use.shape[1]

fig, axes = plt.subplots(num_channels, 1, figsize=(10, 2 * num_channels), sharex=True, sharey=True)

for i in range(num_channels):
    axes[i].plot(data_to_use[i, :])
    axes[i].set_title(f'Channel {i+1}')
    axes[i].set_ylabel('Amplitude (µV)')

axes[-1].set_xlabel('Samples')

#plt.show()

freqs = [5, 10, 15, 20] # Move Top, Right, Bottom, Left
sfreq = board_srate
# CCA Sampling
def basic_cca(eeg_data, sfreq, freqs):
    n_channels, n_samples = eeg_data.shape 
    if n_samples < 10: 
        raise ValueError("Not enough samples for CCA.") 
    t = np.arange(n_samples) / sfreq 
 
    # X: samples × channels 
    X = eeg_data.T 
    Xs = StandardScaler().fit_transform(X) 
 
    scores = {} 
    for f in freqs: 
        # Reference signals: sine/cosine at f and 2f 
        ref = np.column_stack([ 
            np.sin(2*np.pi*f*t), 
            np.cos(2*np.pi*f*t), 
            np.sin(2*np.pi*2*f*t), 
            np.cos(2*np.pi*2*f*t) 
        ]) 
        Rs = StandardScaler().fit_transform(ref) 
 
        cca = CCA(n_components=1) 
        U, V = cca.fit_transform(Xs, Rs) 
        corr = np.corrcoef(U[:, 0], V[:, 0])[0, 1] 
        scores[f] = float(np.abs(corr))  # abs is common in SSVEP CCA 
 
    return scores 

scores = basic_cca(data_to_use[[0, 4, 7], :], sfreq, freqs)

#The best frequency will trigger a command
best_freq = max(scores, key=scores.get)
print(f"Best frequency: {best_freq} Hz with score {scores[best_freq]:.4f}")
print("CCA scores:", scores)