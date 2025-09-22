import serial.tools.list_ports
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BrainFlowError, BoardIds


class BrainFlowBoard:
    """
    A class to manage the setup, configuration, and control of a BrainFlow board.
    This class provides methods for initializing, configuring, and streaming data from the board.
    It also enables access to all attributes and methods from the BoardShim instance (even if not explicitly defined in this class).
    
    Attributes:
        name (str): A user-friendly name or identifier for the board setup instance.
        board_id (int): The ID of the BrainFlow board to use.
        serial_port (str): The serial port to which the BrainFlow board is connected.
        master_board (int): The ID of the master board (if using playback or synthetic boards).
        params (BrainFlowInputParams): Instance of BrainFlowInputParams representing the board's input parameters.
        board (BoardShim): Instance of BoardShim representing the active board.
        session_prepared (bool): Flag indicating if the session has been prepared.
        streaming (bool): Flag indicating if the board is actively streaming data.
        eeg_channels (list): List of EEG channel indices for the board (empty if not applicable).
        sampling_rate (int): Sampling rate of the board.
    """

    _id_counter = 0  # Class-level variable to assign default IDs

    def __init__(self, board_id, serial_port=None, master_board=None, name=None, **kwargs):
        """
        Initializes the BrainFlowBoardSetup class with the given board ID, serial port, master board, and additional parameters.

        Args:
            board_id (int): The ID of the BrainFlow board to be used.
            serial_port (str, optional): The serial port to which the BrainFlow board is connected.
            master_board (int, optional): The master board ID, used for playback or synthetic boards.
            name (str, optional): A user-friendly name or identifier for this instance. Defaults to 'Board X'.
            **kwargs: Additional keyword arguments to be set as attributes on the BrainFlowInputParams instance.
        """
        self.instance_id = BrainFlowBoard._id_counter  # Unique identifier for each instance
        BrainFlowBoard._id_counter += 1

        self.board_id = board_id
        self.serial_port = serial_port
        self.master_board = master_board

        # Assign default name if not provided, based on the class-level ID counter
        self.name = name or f"Board {BrainFlowBoard._id_counter}"
        BrainFlowBoard._id_counter += 1

        # Initialize BrainFlow input parameters
        self.params = BrainFlowInputParams()
        self.params.serial_port = self.serial_port
        self.params.other_info = f"instance_id_{self.instance_id}"  # Add unique instance ID to 'other_info' -> allows for multiple instances of essentially the same board
        self.params.serial_port = self.serial_port
        if self.master_board is not None:
            self.params.master_board = self.master_board  # Set master board if provided

        # Retrieve EEG channels and sampling rate based on the provided board or master board
        try:
            self.eeg_channels, self.sampling_rate = self.get_board_info()
        except BrainFlowError as e:
            print(f"Error getting board info for board {self.board_id}: {e}")
            self.eeg_channels = []
            self.sampling_rate = None

        # Apply additional parameters
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
            else:
                print(f"Warning: {key} is not a valid parameter for BrainFlowInputParams")

        # Initialize board and state flags
        self.board = None
        self.session_prepared = False
        self.streaming = False

    def __getattr__(self, name):
        """
        Delegates attribute access to the BoardShim instance if the attribute is not found in the current instance.
        This allows access to BoardShim-specific attributes that may not be directly defined in the BrainFlowBoardSetup class.

        Args:
            name (str): The name of the attribute to be accessed.

        Returns:
            The attribute from the BoardShim instance if it exists.

        Raises:
            AttributeError: If the attribute is not found in the current instance or the BoardShim instance.
        """
        if self.board is not None and hasattr(self.board, name):
            return getattr(self.board, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def get_board_info(self):
        """
        Retrieves the EEG channels and sampling rate for the board. Uses the master board if provided.

        Returns:
            tuple: A tuple containing EEG channels (list) and sampling rate (int).

        Raises:
            ValueError: If a master_board is provided for a board that doesn't support it.
        """
        if self.board_id not in [BoardIds.PLAYBACK_FILE_BOARD.value,
                                 BoardIds.SYNTHETIC_BOARD.value] and self.master_board:
            raise ValueError(
                f"Master board is only used for PLAYBACK_FILE_BOARD (-3) and SYNTHETIC_BOARD (-1). But {self.board_id} was provided.")

        board_to_use = self.master_board if self.master_board is not None else self.board_id
        board_descr = BoardShim.get_board_descr(board_to_use)

        eeg_channels = board_descr.get("eeg_channels", [])
        sampling_rate = BoardShim.get_sampling_rate(board_to_use)

        return eeg_channels, sampling_rate

    def find_device_ports(self):
        """
        Finds all compatible BrainFlow devices by checking the available serial ports.

        This method iterates over available serial ports on the computer and attempts
        to detect and verify BrainFlow-compatible devices by initializing a session.

        Returns:
            list: A list of dictionaries containing 'port', 'serial_number', and 'description' for each compatible device.
                    Returns an empty list if no devices are found.
        """
        BoardShim.disable_board_logger()
        ports = serial.tools.list_ports.comports()
        compatible_ports = []

        for port in ports:
            try:
                self.params.serial_port = port.device
                board = BoardShim(self.board_id, self.params)
                board.prepare_session()
                board.release_session()

                device_info = {'port': port.device, 'serial_number': port.serial_number,
                    'description': port.description}
                print(f"Compatible device found: Serial Number: {port.serial_number}, Description: {port.description}")
                compatible_ports.append(device_info)
            except BrainFlowError:
                continue

        if not compatible_ports:
            print(f"No compatible BrainFlow devices found.")

        BoardShim.enable_board_logger()
        return compatible_ports

    def setup(self):
        """
        Prepares the session and starts the data stream from the BrainFlow board.

        If no serial port is provided during initialization, this method attempts to auto-detect
        a compatible device. Once the board is detected or provided, it prepares the session and starts streaming.

        Raises:
            BrainFlowError: If the board fails to prepare the session or start streaming.
        """
        if self.serial_port is None and self.master_board is None:
            print("No serial port provided, attempting to auto-detect...")
            ports_info = self.find_device_ports()
            self.serial_port = ports_info[0]['port'] if ports_info else None
            if not self.serial_port:
                print("No compatible device found. Setup failed.")
                return
        elif self.serial_port is None and self.master_board is not None:
            self.serial_port = ''

        self.params.serial_port = self.serial_port
        self.board = BoardShim(self.board_id, self.params)
        try:
            self.board.prepare_session()
            self.session_prepared = True
            self.board.start_stream(450000)
            self.streaming = True
            print(f"[{self.name}, {self.serial_port}] Board setup and streaming started successfully.")
        except BrainFlowError as e:
            print(f"[{self.name}, {self.serial_port}] Error setting up board: {e}")
            self.board = None

    def show_params(self):
        """
        Prints the current parameters of the BrainFlowInputParams instance.
        This method provides a simple way to inspect the current input parameters
        being used to configure the BrainFlow board.
        """
        print(f"[{self.name}] Current BrainFlow Input Parameters:")
        for key, value in vars(self.params).items():
            print(f"{key}: {value}")

    def get_sampling_rate(self):
        """
        Retrieves the sampling rate of the BrainFlow board.

        Returns:
            int: The sampling rate of the BrainFlow board.
        """
        return self.sampling_rate

    def is_streaming(self):
        """
        Checks if the BrainFlow board is currently streaming data.

        Returns:
            bool: True if the board is streaming, False otherwise.
        """
        return self.streaming

    def get_board_name(self):
        """
        Retrieves the name of the BrainFlow board.

        Returns:
            str: The name of the board, useful for logging or display purposes.
        """
        return self.name

    def get_board_data(self):
        """
        Retrieves all accumulated data from the BrainFlow board and clears it from the buffer.

        Returns:
            numpy.ndarray: The current data from the BrainFlow board if the board is set up.
            None: If the board is not set up.
        """
        if self.board is not None:
            return self.board.get_board_data()
        else:
            print("Board is not set up.")
            return None

    def get_current_board_data(self, num_samples):
        """
        Retrieves the most recent num_samples data from the BrainFlow board without clearing it from the buffer.

        Args:
            num_samples (int): Number of recent samples to fetch.

        Returns:
            numpy.ndarray: The latest num_samples data from the BrainFlow board if the board is set up.
            None: If the board is not set up.
        """
        if self.board is not None:
            return self.board.get_current_board_data(num_samples)
        else:
            print("Board is not set up.")
            return None

    def insert_marker(self, marker, verbose=True):
        """
        Inserts a marker into the data stream at the current time. Useful for tagging events in the data stream.

        Args:
            marker (float): The marker value to be inserted.
            verbose (bool): Whether to print a confirmation message. Default is True.
        """
        if self.board is not None and self.streaming:
            try:
                self.board.insert_marker(marker)
                if verbose:
                    print(f"[{self.name}] Marker {marker} inserted successfully.")
            except BrainFlowError as e:
                print(f"[{self.name}] Error inserting marker: {e}")
        else:
            print("Board is not streaming, cannot insert marker.")

    def stop(self):
        """
        Stops the data stream and releases the session of the BrainFlow board.

        This method safely stops the data stream and releases any resources used by the BrainFlow board.
        It also resets the streaming and session flags.
        """
        try:
            if hasattr(self, 'board') and self.board is not None:
                if self.streaming:
                    self.board.stop_stream()
                    self.streaming = False
                    print(f"[{self.name}, {self.serial_port}] Streaming stopped.")
                if self.session_prepared:
                    self.board.release_session()
                    self.session_prepared = False
                    print(f"[{self.name}, {self.serial_port}] Session released.")
        except BrainFlowError as e:
            if "BOARD_NOT_CREATED_ERROR:15" not in str(e):
                print(f"[{self.name}, {self.serial_port}] Error stopping board: {e}")

    def __del__(self):
        """
        Ensures that the data stream is stopped and the session is released when the object is deleted.
        This method ensures that the session is properly released and resources are freed when the object is garbage collected.
        """
        self.stop()


# def compute_band_powers(eeg_data, sfreq, bands=None, nperseg=1024):
#     """
#     Compute band powers from EEG data using Welch's method.
#
#     Parameters
#     ----------
#     eeg_data : ndarray
#         EEG signal, shape (n_channels, n_samples).
#     sfreq : float
#         Sampling frequency in Hz.
#     bands : dict, optional
#         Frequency bands as {"band": (low, high)} in Hz.
#         Defaults to standard EEG bands.
#     nperseg : int, optional
#         Length of each segment for Welch PSD.
#
#     Returns
#     -------
#     band_powers : dict
#         Dictionary with keys = band names, values = array of shape (n_channels,)
#         containing average band power for each channel.
#     """
#     if bands is None:
#         bands = {
#             "delta": (1, 4),
#             "theta": (4, 8),
#             "alpha": (8, 12),
#             "beta": (12, 30),
#             "gamma": (30, 100),
#         }
#
#     n_channels = eeg_data.shape[0]
#     band_powers = {band: 0.0 for band in bands}
#
#     # Compute PSD per channel
#     for ch in range(n_channels):
#         freqs, psd = welch(eeg_data[ch], sfreq, nperseg=nperseg)
#
#         for band, (low, high) in bands.items():
#             idx = np.logical_and(freqs >= low, freqs <= high)
#             # print(f"{band=}, {idx=}")
#             res = np.trapezoid(psd[idx], freqs[idx])  # integrate power
#             band_powers[band] = res.mean()
#
#     return band_powers

import numpy as np
from scipy.signal import welch
from typing import Dict, Tuple, Sequence, Optional


def compute_band_powers(data: np.ndarray, sf: float, bands: Dict[str, Tuple[float, float]] = None,
        window_sec: Optional[float] = None, overlap: float = 0.5, detrend: str = "constant", relative: bool = False,
        return_log: bool = False, total_band: Optional[Tuple[float, float]] = None, ) -> Tuple[
    np.ndarray, Sequence[str]]:
    """
    Compute bandpowers for data of shape (n_channels, n_samples).

    Parameters
    ----------
    data : np.ndarray
        EEG/MEG/LFP data, shape (n_channels, n_samples).
    sf : float
        Sampling frequency in Hz.
    bands : dict[str, (float, float)], optional
        Mapping of band name to (low, high) in Hz.
        Default: common EEG bands.
    window_sec : float, optional
        Welch window length in seconds. If None, picks a good default:
        min(4 s, n_samples/sf), but at least 1 s.
    overlap : float
        Fractional overlap between segments in Welch. In [0, 1). Default 0.5.
    detrend : {"constant", "linear", None}
        Detrending applied before Welch.
    relative : bool
        If True, divide each bandpower by total power in `total_band`
        (or by the full [0, sf/2) range if total_band is None).
    return_log : bool
        If True, return 10*log10(power) in dB (applied after relative if used).
    total_band : (float, float) | None
        Frequency range for the denominator when `relative=True`.
        If None and `relative=True`, uses [0, sf/2).

    Returns
    -------
    bp : np.ndarray
        Bandpowers, shape (n_channels, n_bands), in V^2 if absolute or unitless if relative.
        If return_log=True, units are dB.
    labels : list[str]
        Band labels in the same order as columns of `bp`.

    Notes
    -----
    - Welch PSD is computed along the last axis (samples).
    - Integration uses the trapezoidal rule over the PSD.
    - If a band has no frequency bins (e.g., too narrow vs. resolution),
      the power for that band is set to 0.
    """
    # if data.ndim != 2:
    #     raise ValueError("`data` must be 2D with shape (n_channels, n_samples).")
    n_channels, n_samples = data.shape

    if bands is None:
        bands = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0),
            "gamma": (30.0, 80.0), }

    # Welch parameters
    if window_sec is None:
        # choose up to 4 seconds, but no more than data length, and at least 1 second
        window_sec = max(1.0, min(4.0, n_samples / sf))
    nperseg = int(round(window_sec * sf))
    nperseg = max(8, min(nperseg, n_samples))  # guardrails
    noverlap = int(round(nperseg * overlap)) if 0 <= overlap < 1 else 0

    # Compute PSD: Pxx units are V^2/Hz if input is in volts
    freqs, Pxx = welch(data, fs=sf, nperseg=nperseg, noverlap=noverlap, detrend=detrend, axis=-1, return_onesided=True,
        scaling="density", average="mean", )
    # Pxx shape: (n_channels, n_freqs)
    # Integrate PSD over each band
    labels = list(bands.keys())
    bp = np.zeros((n_channels, len(labels)), dtype=float)

    for j, (lo, hi) in enumerate(bands.values()):
        if hi <= lo:
            raise ValueError(f"Invalid band {labels[j]} with lo>=hi: {(lo, hi)}")
        idx = np.logical_and(freqs >= lo, freqs < hi)
        if not np.any(idx):
            bp[:, j] = 0.0
        else:
            bp[:, j] = np.trapz(Pxx[:, idx], freqs[idx], axis=-1)

    # Relative power normalization if requested
    if relative:
        if total_band is None:
            lo_tot, hi_tot = (0.0, sf / 2.0 - 1e-12)  # upper open interval
        else:
            lo_tot, hi_tot = total_band
            if hi_tot <= lo_tot:
                raise ValueError(f"Invalid total_band with lo>=hi: {total_band}")
        idx_tot = np.logical_and(freqs >= lo_tot, freqs < hi_tot)
        if not np.any(idx_tot):
            raise ValueError("total_band does not include any frequency bins; increase window length or adjust band.")
        total_power = np.trapz(Pxx[:, idx_tot], freqs[idx_tot], axis=-1)  # (n_channels,)
        # Avoid divide by zero
        total_power = np.where(total_power <= 0, np.nan, total_power)
        bp = bp / total_power[:, None]

    if return_log:
        # 10*log10(power). Guard against log(0).
        bp = 10.0 * np.log10(np.maximum(bp, np.finfo(float).tiny))

    return bp.mean(axis=0), labels

#######
# Example streaming from a single board
######
# if __name__ == "__main__":
#     import time
#
#     board_id_cyton = BoardIds.CYTON_BOARD.value
#
#     brainflow_board = BrainFlowBoard(board_id=board_id_cyton)
#
#     brainflow_board.setup()
#
#     # Stream from both boards for 5 seconds
#     time.sleep(5)
#
#     # Retrieve and print data from the second board
#     data = brainflow_board.get_board_data()
#     print(f"Data from brainflow_board: {data}")
#
#     # Stop the streaming and release the sessions for both boards
#     brainflow_board.stop()

############
# Example streaming from two boards simultaneously
###########
## Method 1 - automatically finding multiple devices and starting them
# if __name__ == "__main__":
#     import time

#     board_id_cyton = BoardIds.CYTON_BOARD.value

#     # Instantiate BrainFlowBoardSetup for the first board with verbose=False to suppress logs
#     brainflow_setup_1 = BrainFlowBoardSetup(board_id=board_id_cyton)

#     # Instantiate BrainFlowBoardSetup for the second board with verbose=True for detailed output
#     brainflow_setup_2 = BrainFlowBoardSetup(board_id=board_id_cyton)

#     # Set up the first board and start streaming
#     brainflow_setup_1.setup()

#     # Set up the second board and start streaming (with detailed logs)
#     brainflow_setup_2.setup()

#     # Stream for 5 seconds
#     time.sleep(5)

#     # Retrieve and print data from the first board
#     data_1 = brainflow_setup_1.get_board_data()
#     print(f"Data from board 1: {data_1}")

#     # Retrieve and print data from the second board
#     data_2 = brainflow_setup_2.get_board_data()
#     print(f"Data from board 2: {data_2}")

#     # Stop streaming and release both boards
#     brainflow_setup_1.stop()
#     brainflow_setup_2.stop()

## Method 2 - Detecting compatible devices and assigning them consistently -> ideal for ensuring the same board is assigned to the same port on each run
# if __name__ == "__main__":
#     import time

#     board_id_cyton = BoardIds.CYTON_BOARD.value

#     # Instantiate BrainFlowBoardSetup with verbose enabled for detailed output
#     brainflow_setup_detector = BrainFlowBoardSetup(board_id=board_id_cyton)

#     # Find all compatible devices
#     compatible_ports = brainflow_setup_detector.find_device_ports()

#     # Check if at least two devices are found
#     if len(compatible_ports) >= 2:
#         serial_port_1 = compatible_ports[0]['port']
#         serial_port_2 = compatible_ports[1]['port']

#         # Instantiate BrainFlowBoardSetup for the first board
#         brainflow_setup_1 = BrainFlowBoardSetup(board_id=board_id_cyton, serial_port=serial_port_1)

#         # Instantiate BrainFlowBoardSetup for the second board
#         brainflow_setup_2 = BrainFlowBoardSetup(board_id=board_id_cyton, serial_port=serial_port_2)

#         # Set up the first board and start streaming
#         brainflow_setup_1.setup()

#         # Set up the second board and start streaming
#         brainflow_setup_2.setup()

#         # Stream from both boards for 5 seconds
#         time.sleep(5)

#         # Retrieve and print data from the first board
#         data_1 = brainflow_setup_1.get_board_data()
#         print(f"Data from board 1: {data_1}")

#         # Retrieve and print data from the second board
#         data_2 = brainflow_setup_2.get_board_data()
#         print(f"Data from board 2: {data_2}")

#         # Stop the streaming and release the sessions for both boards
#         brainflow_setup_1.stop()
#         brainflow_setup_2.stop()

#     else:
#         print("Not enough compatible devices found.")
