import time

import numpy as np
from brainflow import LogLevels, DataFilter
from brainflow.board_shim import BoardIds, BoardShim

from bci_control.brainflow_stream import BrainFlowBoard, compute_band_powers
from game.game import Game


def main():
    increasing = False
    max_ratio = 0.0
    board_id = BoardIds.CYTON_BOARD.value
    board = BrainFlowBoard(board_id=board_id, serial_port="/dev/cu.usbserial-DM01IK21")
    board.setup()
    board_descr = BoardShim.get_board_descr(board_id)
    sampling_rate = int(board_descr['sampling_rate'])

    while True:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
        time.sleep(2)
        nfft = DataFilter.get_nearest_power_of_two(sampling_rate)
        data = board.get_current_board_data(num_samples=500)

        eeg_data = data

        def remove_dc_offset(data):
            return data[1:4, :] - np.mean(data[1:4, :], axis=1, keepdims=True)

        eeg_data = remove_dc_offset(eeg_data)
        band_powers = compute_band_powers(eeg_data, sampling_rate, relative=True)
        powers, _ = band_powers

        ratio = powers[3] / powers[2]
        if ratio > max_ratio:
            max_ratio = ratio
            increasing = True
        else:
            increasing = False

    # board.stop_stream()  # board.release_session()


if __name__ == '__main__':
    game = Game()
    game.run()
