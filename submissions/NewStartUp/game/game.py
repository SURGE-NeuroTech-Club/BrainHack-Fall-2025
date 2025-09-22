import sys
import threading
import time

import numpy as np
import pygame

from BLEDataCollector.BLEDataCollector import BLEDataCollector, run_ble_collector
from bci_control.brainflow_stream import compute_band_powers
from game.ball import START_SPEED, Ball
from game.brick import Brick
from game.constants import *
from game.paddle import Paddle


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.Font(None, 36)
        self.large_font = pygame.font.Font(None, 74)

        self.state = 'calibrating'

        self.start_time = pygame.time.get_ticks()
        self.final_time = 0
        self.best_time = self._load_best_time()

        self.calibration_end_time = self.start_time + 15000
        self.calibration_data_a = []
        self.calibration_data_b = []
        self.calibrated_ratio = 0.0

        self.game_ratio = 0.0
        self.previous_game_ratio = 0.0

        # --- Threading variables ---
        self.collector = None
        self.bci_thread = None
        self.bci_thread_stop_event = threading.Event()

        self.setup_objects()

    def setup_objects(self):
        self.paddle = Paddle(SCREEN_HEIGHT - 40, SCREEN_WIDTH, PADDLE_HEIGHT, WHITE)
        self.ball = Ball(SCREEN_WIDTH // 2, self.paddle.rect.top - 10, 10, WHITE, speed=START_SPEED)
        self.bricks = self._create_bricks()

    def _create_bricks(self):
        bricks = []
        rows, cols = len(VIBGYOR), 10
        brick_width, brick_height = 75, 20
        gap = 5
        for row in range(rows):
            for col in range(cols):
                intensity = (rows - 1) - row
                color = VIBGYOR[row]
                brick_x = col * (brick_width + gap) + (gap * 4)
                brick_y = row * (brick_height + gap) + 50
                bricks.append(Brick(brick_x, brick_y, brick_width, brick_height, color, intensity))
        return bricks

    def remove_dc_offset(self, data):
        return np.array(data) - np.mean(data)

    def run(self):
        self.running = True

        self.collector = BLEDataCollector()
        ble_thread = threading.Thread(target=run_ble_collector, args=(self.collector, "ORBIT_61"), daemon=True)
        ble_thread.start()
        time.sleep(2)

        while self.running:
            self._handle_events()

            if self.state == 'calibrating':
                self._run_calibration_step()

            elif self.state == 'playing':
                # BCI logic now runs in the background. The main loop only updates and draws.
                self._update()
                self._draw()

            elif self.state in ('game_over', 'win'):
                self._draw_end_screen()

        # --- Graceful shutdown of all threads ---
        if self.bci_thread:
            self.bci_thread_stop_event.set()
            self.bci_thread.join(timeout=2)

        self.collector.stop()
        ble_thread.join(timeout=2)
        self.collector.close()
        self._cleanup()

    def _run_calibration_step(self):
        """Handles logic for the 30-second calibration phase."""
        data = self.collector.get_current_data(num_samples=500)
        if data and data[0] and data[1]:
            self.calibration_data_a.extend(data[0])
            self.calibration_data_b.extend(data[1])

        self._draw_calibration_screen()

        if pygame.time.get_ticks() >= self.calibration_end_time:
            self._compute_calibration_results()
            self.state = 'playing'
            self.start_time = pygame.time.get_ticks()

            # --- Start the asynchronous BCI processing thread ---
            self.bci_thread_stop_event.clear()
            self.bci_thread = threading.Thread(target=self._bci_processing_loop, daemon=True)
            self.bci_thread.start()

    def _bci_processing_loop(self):
        """
        Runs in a separate thread. This is the logic from the old _run_game_step,
        now running asynchronously every 5 seconds.
        """
        print("BCI processing thread started.")
        while not self.bci_thread_stop_event.is_set():
            sampling_rate = 256
            data = self.collector.get_current_data(num_samples=3200)

            if data and data[0]:
                try:
                    # NOTE: Assuming you want to process the first channel (data[0])
                    eeg_data = self.remove_dc_offset(data)
                    band_powers = compute_band_powers(eeg_data, sampling_rate, relative=True)
                    powers, _ = band_powers

                    if powers[3] > 0:
                        self.game_ratio = powers[3] / powers[2]
                except Exception as e:
                    print(f"Error computing in-game ratio: {e}")

            # Adjust speed based on the newly computed ratio
            if self.game_ratio > self.calibrated_ratio and self.game_ratio != self.previous_game_ratio:
                self.ball.increase_speed()
                print(f"Ratio {self.game_ratio:.2f} > {self.calibrated_ratio:.2f} -> Speed Increased")

            if self.game_ratio < self.calibrated_ratio and self.game_ratio != self.previous_game_ratio:
                self.ball.decrease_speed()
                print(f"Ratio {self.game_ratio:.2f} < {self.calibrated_ratio:.2f} -> Speed Decreased")

            self.previous_game_ratio = self.game_ratio

            # Wait for 5 seconds before the next computation
            time.sleep(5)

        print("BCI processing thread stopped.")

    def _compute_calibration_results(self):
        """Processes all collected calibration data to find the baseline ratio."""
        print("Calibration finished. Computing baseline...")
        if not self.calibration_data_a:
            print("Warning: No data collected during calibration. Using a default ratio.")
            self.calibrated_ratio = 0.4
            return

        try:
            sampling_rate = 256
            # NOTE: Using both channels for calibration as in original code
            eeg_data = self.remove_dc_offset([self.calibration_data_a[-3200:], self.calibration_data_b[-3200:]])
            band_powers = compute_band_powers(eeg_data, sampling_rate, relative=True)
            powers, _ = band_powers

            if powers[3] > 0:
                self.calibrated_ratio = powers[3] / powers[2]
            else:
                self.calibrated_ratio = 1.0

            print(f"Calibration successful. Baseline Alpha/Beta Ratio: {self.calibrated_ratio:.4f}")
        except Exception as e:
            print(f"Error computing calibration results: {e}. Using a default ratio.")
            self.calibrated_ratio = 1.0

    def _draw_calibration_screen(self):
        # This method is unchanged
        self.screen.fill(BLACK)
        cal_text = self.large_font.render("Calibrating...", True, WHITE)
        cal_rect = cal_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 40))
        self.screen.blit(cal_text, cal_rect)
        remaining_time = (self.calibration_end_time - pygame.time.get_ticks()) / 1000
        timer_text = self.font.render(f"Please relax. Time remaining: {max(0, int(remaining_time))}s", True, WHITE)
        timer_rect = timer_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 30))
        self.screen.blit(timer_text, timer_rect)
        pygame.display.flip()

    def _handle_events(self):
        # This method is unchanged
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if self.state == 'playing':
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.ball.light_force()

    def _update(self):
        # This method is unchanged
        self.ball.update()
        self._handle_collisions()

    def _handle_collisions(self):
        # This method is unchanged
        if self.ball.rect.colliderect(self.paddle.rect):
            self.ball.bounce()
            self.ball.rect.bottom = self.paddle.rect.top
        hit_brick = next((b for b in self.bricks if self.ball.rect.colliderect(b.rect)), None)
        if hit_brick:
            self.ball.bounce()
            if hit_brick.hit(factor=self.ball.speed % len(VIBGYOR)):
                self.bricks.remove(hit_brick)
        if self.ball.rect.bottom >= SCREEN_HEIGHT:
            if self.state == 'playing':
                self.state = 'game_over'
                self._handle_game_end()
        if not self.bricks:
            if self.state == 'playing':
                self.state = 'win'
                self._handle_game_end()

    def _draw(self):
        # This method is unchanged
        self.screen.fill(BLACK)
        self.paddle.draw(self.screen)
        self.ball.draw(self.screen)
        for brick in self.bricks:
            brick.draw(self.screen)
        self._draw_timer()
        pygame.display.flip()
        self.clock.tick(FPS)

    def _draw_timer(self):
        # This method is unchanged
        elapsed_time = (pygame.time.get_ticks() - self.start_time) // 1000
        minute, sec = divmod(elapsed_time, 60)
        timer_text = self.font.render(f"Time: {minute:02d}:{sec:02d}", True, WHITE)
        self.screen.blit(timer_text, timer_text.get_rect(topright=(SCREEN_WIDTH - 10, 10)))

    def _handle_game_end(self):
        # Signal the BCI thread to stop as soon as the game ends
        self.bci_thread_stop_event.set()

        self.final_time = (pygame.time.get_ticks() - self.start_time) // 1000
        if self.best_time == 0 or (self.final_time < self.best_time and self.state == 'win'):
            self.best_time = self.final_time
            self._save_best_time(self.best_time)

    def _draw_end_screen(self):
        # This method is unchanged
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        message = "Game Over" if self.state == 'game_over' else "You Win!"
        text_surf = self.large_font.render(message, True, WHITE)
        text_rect = text_surf.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 50))
        self.screen.blit(text_surf, text_rect)
        time_surf = self.font.render(f"Your Time: {self.final_time // 60:02d}:{self.final_time % 60:02d}", True, WHITE)
        time_rect = time_surf.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 20))
        self.screen.blit(time_surf, time_rect)
        if self.best_time > 0:
            best_time_surf = self.font.render(f"Best Time: {self.best_time // 60:02d}:{self.best_time % 60:02d}", True,
                                              WHITE)
            best_time_rect = best_time_surf.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 60))
            self.screen.blit(best_time_surf, best_time_rect)
        pygame.display.flip()

    def _load_best_time(self):
        # This method is unchanged
        try:
            with open("best_time.txt", "r") as f:
                return int(f.read())
        except (FileNotFoundError, ValueError):
            return 0

    def _save_best_time(self, time):
        # This method is unchanged
        with open("best_time.txt", "w") as f:
            f.write(str(time))

    def _cleanup(self):
        # This method is unchanged
        pygame.quit()
        sys.exit()
