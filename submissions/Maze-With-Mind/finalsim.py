import pygame
import random
import sys
import math
import numpy as np
import time
import threading
from queue import Queue
import mne
from collections import deque

from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800
BACKGROUND_COLOR = (128, 128, 128)  # Gray background for border area
WALL_COLOR = (0, 0, 0)  # Black walls
PATH_COLOR = (255, 255, 255)  # White paths
PLAYER_COLOR = (255, 0, 0)
CELL_SIZE = 40
WALL_THICKNESS = 2
FPS = 60
BORDER_SIZE = 100  # Extra space around maze

# BCI Constants
CCA_WINDOW_SIZE = 1000  # Number of samples for CCA analysis
CCA_THRESHOLD = 0.2  # Minimum correlation threshold for movement
UPDATE_INTERVAL = 1.0  # Seconds between CCA updates
FIF_FILE_PATH = "eeg_data.fif"  # Path to your .fif file

# Shape colors and frequencies
SHAPE_COLORS = [
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 165, 0),  # Orange
]

# Frequency mapping: freq -> direction
FREQ_TO_DIRECTION = {
    15: (0, -1),   # Up - Top shape (Circle)
    10: (1, 0),   # Right - Right shape (Square)
    5: (0, 1),   # Down - Bottom shape (Triangle)
    20: (-1, 0),  # Left - Left shape (Diamond)
}

# Screen setup
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('BCI Maze Game (FIF File Mode)')
clock = pygame.time.Clock()


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply a Butterworth bandpass filter to multichannel data."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=1)


def basic_cca(eeg_data, sfreq, freqs):
    """Perform CCA analysis to detect SSVEP responses."""
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
        scores[f] = float(np.abs(corr))
 
    return scores


class FIFDataController:
    def __init__(self, fif_file_path):
        self.fif_file_path = fif_file_path
        self.raw = None
        self.sfreq = None
        self.movement_queue = Queue()
        self.running = False
        self.freqs = [5, 10, 15, 20]  # Top, Right, Bottom, Left
        self.current_position = 0  # Current time position in the data
        self.data_array = None
        
        # For UI display
        self.metadata = {}
        self.latest_scores = {f: 0.0 for f in self.freqs}
        self.current_movement = None
        self.movement_history = deque(maxlen=10)  # Keep last 10 movements
        
    def load_fif_file(self):
        """Load and preprocess the .fif file."""
        try:
            # Load the .fif file
            self.raw = mne.io.read_raw_fif(self.fif_file_path, preload=True, verbose=False)
            self.sfreq = self.raw.info['sfreq']
            
            # Get EEG data (assuming standard 10-20 system channels)
            eeg_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
                          'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
            
            # Pick available EEG channels
            available_channels = [ch for ch in eeg_channels if ch in self.raw.ch_names]
            if not available_channels:
                # If no standard names, just use first 8 channels
                available_channels = self.raw.ch_names[:8]
            
            self.raw.pick_channels(available_channels[:8])  # Use up to 8 channels
            
            # Apply basic preprocessing
            self.raw.filter(l_freq=1.0, h_freq=50.0, verbose=False)  # Basic filtering
            
            # Get data as numpy array (channels x samples)
            self.data_array = self.raw.get_data()
            
            # Store metadata
            self.metadata = {
                'File': self.fif_file_path.split('/')[-1],
                'Channels': len(self.raw.ch_names),
                'Sample Rate': f"{self.sfreq} Hz",
                'Duration': f"{self.data_array.shape[1] / self.sfreq:.1f}s",
                'Ch Names': ', '.join(self.raw.ch_names[:4]) + ('...' if len(self.raw.ch_names) > 4 else '')
            }
            
            print(f"Loaded .fif file: {self.fif_file_path}")
            print(f"Sampling rate: {self.sfreq} Hz")
            print(f"Channels: {self.raw.ch_names}")
            print(f"Data shape: {self.data_array.shape}")
            print(f"Duration: {self.data_array.shape[1] / self.sfreq:.1f} seconds")
            
            return True
            
        except Exception as e:
            print(f"Error loading .fif file: {e}")
            print("Generating synthetic SSVEP data for demonstration...")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic SSVEP data for demonstration."""
        try:
            duration = 300  # 5 minutes of synthetic data
            self.sfreq = 250  # 250 Hz sampling rate
            n_samples = int(duration * self.sfreq)
            n_channels = 8
            
            # Generate synthetic EEG with embedded SSVEP responses
            np.random.seed(42)  # For reproducible results
            
            # Base EEG noise
            self.data_array = np.random.randn(n_channels, n_samples) * 10  # 10 μV noise
            
            # Add synthetic SSVEP responses with varying strengths over time
            t = np.arange(n_samples) / self.sfreq
            
            for i, freq in enumerate(self.freqs):
                # Create periods where each frequency is "attended"
                period_duration = duration / len(self.freqs)
                start_time = i * period_duration
                end_time = (i + 1) * period_duration
                
                # Find time indices for this period
                time_mask = (t >= start_time) & (t < end_time)
                
                if np.any(time_mask):
                    # Add SSVEP signal to posterior channels (simulating visual cortex)
                    ssvep_signal = 3 * np.sin(2 * np.pi * freq * t[time_mask])  # 3 μV amplitude
                    
                    # Add to channels 6 and 7 (simulating O1, O2 - occipital electrodes)
                    if n_channels > 6:
                        self.data_array[6, time_mask] += ssvep_signal
                    if n_channels > 7:
                        self.data_array[7, time_mask] += ssvep_signal
            
            # Store metadata for synthetic data
            self.metadata = {
                'File': 'Synthetic SSVEP Data',
                'Channels': n_channels,
                'Sample Rate': f"{self.sfreq} Hz",
                'Duration': f"{duration}s",
                'Ch Names': 'Synthetic EEG channels'
            }
            
            print(f"Generated synthetic SSVEP data:")
            print(f"Sampling rate: {self.sfreq} Hz")
            print(f"Channels: 8 (synthetic)")
            print(f"Data shape: {self.data_array.shape}")
            print(f"Duration: {duration} seconds")
            print(f"SSVEP frequencies embedded: {self.freqs} Hz")
            
            return True
            
        except Exception as e:
            print(f"Error generating synthetic data: {e}")
            return False
    
    def start_monitoring(self):
        """Start the data processing thread."""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._process_data_stream)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop the data processing."""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
            
    def _process_data_stream(self):
        """Process the .fif data in chunks to simulate real-time processing."""
        samples_per_update = int(self.sfreq * UPDATE_INTERVAL)
        
        while self.running:
            try:
                # Check if we have enough data remaining
                if self.current_position + CCA_WINDOW_SIZE >= self.data_array.shape[1]:
                    print("Reached end of data, looping back to beginning...")
                    self.current_position = 0
                
                # Extract data window
                start_idx = self.current_position
                end_idx = self.current_position + CCA_WINDOW_SIZE
                data_window = self.data_array[:, start_idx:end_idx]
                
                # Remove DC offset
                data_window = data_window - np.mean(data_window, axis=1, keepdims=True)
                
                # Apply bandpass filter
                filtered_data = bandpass_filter(data_window, lowcut=3.0, highcut=40.0, 
                                              fs=self.sfreq, order=4)
                
                # Use posterior channels for SSVEP detection (simulating occipital electrodes)
                n_channels = filtered_data.shape[0]
                if n_channels >= 8:
                    # Use last 3 channels (simulating posterior electrodes)
                    selected_data = filtered_data[-3:, :]
                elif n_channels >= 3:
                    # Use last 3 available channels
                    selected_data = filtered_data[-3:, :]
                else:
                    # Use all available channels
                    selected_data = filtered_data
                
                # Perform CCA analysis
                scores = basic_cca(selected_data, self.sfreq, self.freqs)
                
                # Update latest scores for UI
                self.latest_scores = scores.copy()
                
                # Find the best frequency
                best_freq = max(scores, key=scores.get)
                best_score = scores[best_freq]
                
                # If score is above threshold, queue movement
                if best_score > CCA_THRESHOLD:
                    direction = FREQ_TO_DIRECTION.get(best_freq)
                    if direction:
                        self.movement_queue.put(direction)
                        
                        # Update current movement and add to history
                        direction_names = {(0, -1): 'UP', (1, 0): 'RIGHT', (0, 1): 'DOWN', (-1, 0): 'LEFT'}
                        move_name = direction_names.get(direction, 'UNKNOWN')
                        self.current_movement = f"{move_name} ({best_freq}Hz, {best_score:.3f})"
                        
                        timestamp = time.strftime("%H:%M:%S")
                        self.movement_history.append(f"{timestamp}: {move_name}")
                        
                        print(f"Movement detected: freq={best_freq}Hz, score={best_score:.3f}, "
                              f"direction={direction}, time={self.current_position/self.sfreq:.1f}s")
                
                # Advance position in data
                self.current_position += samples_per_update
                
                time.sleep(UPDATE_INTERVAL)
                
            except Exception as e:
                print(f"Error in data processing: {e}")
                time.sleep(1)
    
    def get_movement(self):
        """Get the next movement from the queue."""
        if not self.movement_queue.empty():
            return self.movement_queue.get()
        return None


class FlickeringShape:
    def __init__(self, x, y, shape_type, color, frequency):
        self.x = x
        self.y = y
        self.shape_type = shape_type  # 'circle', 'square', 'triangle', 'diamond'
        self.color = color
        self.frequency = frequency
        self.size = 80  # Reduced size to fit in border
        self.visible = True
        self.timer = 0
    
    def update(self, dt):
        self.timer += dt
        # Different frequencies for different shapes
        if self.timer >= 1000 / self.frequency:  # Convert frequency to milliseconds
            self.visible = not self.visible
            self.timer = 0
    
    def draw(self, screen, offset_x, offset_y):
        if not self.visible:
            return
        
        x = offset_x + self.x
        y = offset_y + self.y
        
        if self.shape_type == 'circle':
            pygame.draw.circle(screen, self.color, (x, y), self.size // 2)
        elif self.shape_type == 'square':
            rect = pygame.Rect(x - self.size // 2, y - self.size // 2, self.size, self.size)
            pygame.draw.rect(screen, self.color, rect)
        elif self.shape_type == 'triangle':
            points = [
                (x, y - self.size // 2),
                (x - self.size // 2, y + self.size // 2),
                (x + self.size // 2, y + self.size // 2)
            ]
            pygame.draw.polygon(screen, self.color, points)
        elif self.shape_type == 'diamond':
            points = [
                (x, y - self.size // 2),
                (x + self.size // 2, y),
                (x, y + self.size // 2),
                (x - self.size // 2, y)
            ]
            pygame.draw.polygon(screen, self.color, points)


class Player:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)

    def move(self, dx, dy):
        self.rect.x += dx * CELL_SIZE
        self.rect.y += dy * CELL_SIZE


def create_flickering_shapes():
    """Create four flickering shapes positioned in the border area around maze"""
    shapes = []
    
    maze_center_x = SCREEN_WIDTH // 2
    maze_center_y = SCREEN_HEIGHT // 2
    
    # Top border - Circle (5 Hz) - UP movement
    shapes.append(FlickeringShape(maze_center_x, BORDER_SIZE // 2, 'circle', SHAPE_COLORS[0], 5))
    
    # Right border - Square (10 Hz) - RIGHT movement
    shapes.append(FlickeringShape(SCREEN_WIDTH - BORDER_SIZE // 2, maze_center_y, 'square', SHAPE_COLORS[1], 10))
    
    # Bottom border - Triangle (15 Hz) - DOWN movement
    shapes.append(FlickeringShape(maze_center_x, SCREEN_HEIGHT - BORDER_SIZE // 2, 'triangle', SHAPE_COLORS[2], 15))
    
    # Left border - Diamond (20 Hz) - LEFT movement
    shapes.append(FlickeringShape(BORDER_SIZE // 2, maze_center_y, 'diamond', SHAPE_COLORS[3], 20))
    
    return shapes


def draw_info_panels(screen, data_controller, current_time):
    """Draw information panels in the four corners of the screen."""
    font = pygame.font.Font(None, 20)
    small_font = pygame.font.Font(None, 16)
    
    # Colors
    panel_bg = (40, 40, 40, 180)  # Semi-transparent dark background
    text_color = (255, 255, 255)
    highlight_color = (255, 255, 0)
    
    # Top Left: Metadata
    if data_controller and hasattr(data_controller, 'metadata'):
        metadata_surface = pygame.Surface((250, 120), pygame.SRCALPHA)
        metadata_surface.fill(panel_bg)
        
        y_offset = 5
        title = font.render("DATA INFO", True, highlight_color)
        metadata_surface.blit(title, (5, y_offset))
        y_offset += 25
        
        for key, value in data_controller.metadata.items():
            text = small_font.render(f"{key}: {value}", True, text_color)
            metadata_surface.blit(text, (5, y_offset))
            y_offset += 18
        
        time_text = small_font.render(f"Time: {current_time:.1f}s", True, text_color)
        metadata_surface.blit(time_text, (5, y_offset))
        
        screen.blit(metadata_surface, (10, 10))
    
    # Top Right: Frequency Plot/Scores
    if data_controller and hasattr(data_controller, 'latest_scores'):
        scores_surface = pygame.Surface((250, 120), pygame.SRCALPHA)
        scores_surface.fill(panel_bg)
        
        title = font.render("CCA SCORES", True, highlight_color)
        scores_surface.blit(title, (5, 5))
        
        # Draw frequency bars
        bar_width = 40
        bar_height_max = 60
        x_start = 15
        y_base = 90
        
        freq_colors = {5: SHAPE_COLORS[0], 10: SHAPE_COLORS[1], 15: SHAPE_COLORS[2], 20: SHAPE_COLORS[3]}
        
        for i, freq in enumerate(data_controller.freqs):
            score = data_controller.latest_scores.get(freq, 0)
            normalized_score = min(score / 0.5, 1.0)  # Normalize to 0-1 range
            bar_height = int(normalized_score * bar_height_max)
            
            # Draw bar
            bar_rect = pygame.Rect(x_start + i * (bar_width + 10), y_base - bar_height, bar_width, bar_height)
            pygame.draw.rect(scores_surface, freq_colors[freq], bar_rect)
            
            # Draw frequency label
            freq_text = small_font.render(f"{freq}Hz", True, text_color)
            text_rect = freq_text.get_rect()
            text_rect.centerx = bar_rect.centerx
            scores_surface.blit(freq_text, (text_rect.x, y_base + 5))
            
            # Draw score value
            score_text = small_font.render(f"{score:.3f}", True, text_color)
            score_rect = score_text.get_rect()
            score_rect.centerx = bar_rect.centerx
            scores_surface.blit(score_text, (score_rect.x, y_base + 20))
        
        screen.blit(scores_surface, (SCREEN_WIDTH - 260, 10))
    
    # Bottom Right: Current Movement
    if data_controller and hasattr(data_controller, 'current_movement'):
        movement_surface = pygame.Surface((250, 80), pygame.SRCALPHA)
        movement_surface.fill(panel_bg)
        
        title = font.render("CURRENT MOVE", True, highlight_color)
        movement_surface.blit(title, (5, 5))
        
        if data_controller.current_movement:
            # Split long text into multiple lines
            move_text = data_controller.current_movement
            if len(move_text) > 30:
                words = move_text.split(' ')
                line1 = ' '.join(words[:2])
                line2 = ' '.join(words[2:])
                
                text1 = small_font.render(line1, True, text_color)
                text2 = small_font.render(line2, True, text_color)
                movement_surface.blit(text1, (5, 30))
                movement_surface.blit(text2, (5, 48))
            else:
                text = small_font.render(move_text, True, text_color)
                movement_surface.blit(text, (5, 30))
        else:
            text = small_font.render("No movement detected", True, (150, 150, 150))
            movement_surface.blit(text, (5, 30))
        
        screen.blit(movement_surface, (SCREEN_WIDTH - 260, SCREEN_HEIGHT - 90))
    
    # Bottom Left: Movement History
    if data_controller and hasattr(data_controller, 'movement_history'):
        history_surface = pygame.Surface((250, 150), pygame.SRCALPHA)
        history_surface.fill(panel_bg)
        
        title = font.render("MOVE HISTORY", True, highlight_color)
        history_surface.blit(title, (5, 5))
        
        y_offset = 25
        for move in list(data_controller.movement_history)[-8:]:  # Show last 8 moves
            text = small_font.render(move, True, text_color)
            history_surface.blit(text, (5, y_offset))
            y_offset += 16
        
        if len(data_controller.movement_history) == 0:
            text = small_font.render("No moves yet", True, (150, 150, 150))
            history_surface.blit(text, (5, 25))
        
        screen.blit(history_surface, (10, SCREEN_HEIGHT - 160))
def draw_maze(maze, player, shapes, dt, data_controller, current_time):
    # Calculate maze dimensions and positioning with border
    x_offset = BORDER_SIZE
    y_offset = BORDER_SIZE

    # Draw maze
    for y in range(len(maze)):
        for x in range(len(maze[y])):
            cell = maze[y][x]
            if cell == 'W':
                pygame.draw.rect(screen, WALL_COLOR, (x_offset + x * CELL_SIZE, y_offset + y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            else:
                pygame.draw.rect(screen, PATH_COLOR, (x_offset + x * CELL_SIZE, y_offset + y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    # Draw player
    pygame.draw.rect(screen, PLAYER_COLOR, (x_offset + player.rect.x, y_offset + player.rect.y, CELL_SIZE, CELL_SIZE))
    
    # Update and draw flickering shapes
    for shape in shapes:
        shape.update(dt)
        shape.draw(screen, 0, 0)
    
    # Draw information panels
    draw_info_panels(screen, data_controller, current_time)
    
    # Draw center instructions
    font = pygame.font.Font(None, 24)
    instructions = [
        "BCI Maze Game (.fif file mode)",
        "Look at flickering shapes to move:",
        "Triangle (Cyan, 5Hz) = UP",
        "Square (Magenta, 10Hz) = RIGHT", 
        "Circle (Yellow, 15Hz) = DOWN",
        "Diamond (Orange, 20Hz) = LEFT",
        "Keyboard: Arrow keys work too"
    ]
    
    # Position instructions in center area, avoiding corners
    center_x = SCREEN_WIDTH // 2
    start_y = 200
    
    for i, instruction in enumerate(instructions):
        color = (255, 255, 255) if i == 0 else (200, 200, 200)
        text = font.render(instruction, True, color)
        text_rect = text.get_rect()
        text_rect.centerx = center_x
        text_rect.y = start_y + i * 25
        
        # Add semi-transparent background for readability
        bg_rect = text_rect.inflate(10, 5)
        bg_surface = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        bg_surface.fill((0, 0, 0, 100))
        screen.blit(bg_surface, bg_rect)
        screen.blit(text, text_rect)


def is_path(maze, x, y):
    if 0 <= x < len(maze[0]) and 0 <= y < len(maze) and maze[y][x] == 'P':
        return True
    return False


def find_neighbors(x, y, width, height, maze):
    """Identify and return valid neighboring cells for maze generation."""
    neighbors = []
    if x > 1 and maze[y][x - 2] == 'W':
        neighbors.append((x - 2, y))
    if x < width - 2 and maze[y][x + 2] == 'W':
        neighbors.append((x + 2, y))
    if y > 1 and maze[y - 2][x] == 'W':
        neighbors.append((x, y - 2))
    if y < height - 2 and maze[y + 2][x] == 'W':
        neighbors.append((x, y + 2))
    return neighbors


def connect_cells(maze, cell, next_cell):
    """Connect the current cell to the chosen next cell in the maze."""
    x, y = cell
    nx, ny = next_cell
    if nx == x:
        maze[min(ny, y) + 1][x] = 'P'
    else:
        maze[y][min(nx, x) + 1] = 'P'


def generate_maze(width, height):
    maze = [['W' for _ in range(width)] for _ in range(height)]
    stack = [(1, 1)]

    while stack:
        cell = stack[-1]
        x, y = cell
        maze[y][x] = 'P'
        neighbors = find_neighbors(x, y, width, height, maze)

        if neighbors:
            next_cell = random.choice(neighbors)
            connect_cells(maze, cell, next_cell)
            stack.append(next_cell)
        else:
            stack.pop()

    return maze


def process_events(player, maze, data_controller):
    """Handle all Pygame events and BCI movements."""
    # Handle pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            # Allow keyboard control as fallback
            if process_keydown(event.key, player, maze):
                return False
    
    # Handle BCI movements from .fif data
    if data_controller:
        movement = data_controller.get_movement()
        if movement:
            dx, dy = movement
            new_x = player.rect.x // CELL_SIZE + dx
            new_y = player.rect.y // CELL_SIZE + dy
            if is_path(maze, new_x, new_y):
                player.move(dx, dy)
    
    return True


def process_keydown(key, player, maze):
    """Process keydown events and move the player if the path is valid."""
    directions = {
        pygame.K_LEFT: (-1, 0),
        pygame.K_RIGHT: (1, 0),
        pygame.K_UP: (0, -1),
        pygame.K_DOWN: (0, 1),
    }
    if key in directions:
        dx, dy = directions[key]
        new_x = player.rect.x // CELL_SIZE + dx
        new_y = player.rect.y // CELL_SIZE + dy
        if is_path(maze, new_x, new_y):
            player.move(dx, dy)


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    # Generate maze
    maze_cells_width = (SCREEN_WIDTH - 2 * BORDER_SIZE) // CELL_SIZE
    maze_cells_height = (SCREEN_HEIGHT - 2 * BORDER_SIZE) // CELL_SIZE
    maze = generate_maze(maze_cells_width, maze_cells_height)
    player = Player(1, 1)
    shapes = create_flickering_shapes()
    
    # Setup .fif data controller
    data_controller = FIFDataController(FIF_FILE_PATH)
    
    # Try to load .fif file
    try:
        if data_controller.load_fif_file():
            data_controller.start_monitoring()
            print("Data loaded successfully! BCI processing started.")
        else:
            print("Data loading failed. Using keyboard controls only.")
            data_controller = None
    except Exception as e:
        print(f"Data initialization error: {e}")
        print("Running in keyboard-only mode.")
        data_controller = None

    running = True
    start_time = time.time()
    
    while running:
        dt = clock.tick(FPS)  # Get delta time in milliseconds
        current_time = time.time() - start_time
        
        running = process_events(player, maze, data_controller)

        screen.fill(BACKGROUND_COLOR)
        draw_maze(maze, player, shapes, dt, data_controller, current_time)
        pygame.display.flip()

    # Cleanup
    if data_controller:
        data_controller.stop_monitoring()
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()