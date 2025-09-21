import pygame
import random
import sys
import math
import scipy
import numpy as np
import time
import threading
from queue import Queue

from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BrainFlowError, BoardIds
from brainflow_stream import BrainFlowBoardSetup

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
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
CCA_THRESHOLD = 0.3  # Minimum correlation threshold for movement
UPDATE_INTERVAL = 0.5  # Seconds between CCA updates

# Shape colors and frequencies
SHAPE_COLORS = [
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 165, 0),  # Orange
]

# Frequency mapping: freq -> direction
FREQ_TO_DIRECTION = {
    5: (0, -1),   # Up - Top shape (Circle)
    10: (1, 0),   # Right - Right shape (Square)
    15: (0, 1),   # Down - Bottom shape (Triangle)
    20: (-1, 0),  # Left - Left shape (Diamond)
}

# Screen setup
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('BCI Maze Game')
clock = pygame.time.Clock()


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply a Butterworth bandpass filter to multichannel data."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=1)


def remove_dc_offset(data):
    """Remove DC offset from EEG data."""
    return data[1:9, :] - np.mean(data[1:9, :], axis=1, keepdims=True)


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


class BCIController:
    def __init__(self):
        self.board_id = BoardIds.CYTON_BOARD.value
        self.cyton_board = None
        self.board_srate = None
        self.movement_queue = Queue()
        self.running = False
        self.freqs = [5, 10, 15, 20]  # Top, Right, Bottom, Left
        
    def setup_board(self):
        """Setup and connect to the BCI board."""
        try:
            self.cyton_board = BrainFlowBoardSetup(
                board_id=self.board_id,
                name='Board_1',
                serial_port=None
            )
            self.cyton_board.setup()
            self.board_srate = self.cyton_board.get_sampling_rate()
            print(f"BCI Board connected. Sampling rate: {self.board_srate}")
            return True
        except Exception as e:
            print(f"Failed to setup BCI board: {e}")
            return False
    
    def start_monitoring(self):
        """Start the BCI monitoring thread."""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_brain_signals)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop the BCI monitoring."""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
            
    def _monitor_brain_signals(self):
        """Monitor brain signals and detect movement intentions."""
        while self.running:
            try:
                # Get recent EEG data
                raw_data = self.cyton_board.get_current_board_data(num_samples=CCA_WINDOW_SIZE)
                if raw_data.shape[1] < CCA_WINDOW_SIZE:
                    time.sleep(0.1)
                    continue
                    
                # Process the data
                eeg_data = remove_dc_offset(raw_data)
                filtered_data = bandpass_filter(eeg_data, lowcut=5.0, highcut=30.0, 
                                              fs=self.board_srate, order=4)
                
                # Use selected channels for CCA (you can adjust these)
                selected_data = filtered_data[[0, 4, 7], :]  # Channels 1, 5, 8
                
                # Perform CCA analysis
                scores = basic_cca(selected_data, self.board_srate, self.freqs)
                
                # Find the best frequency
                best_freq = max(scores, key=scores.get)
                best_score = scores[best_freq]
                
                # If score is above threshold, queue movement
                if best_score > CCA_THRESHOLD:
                    direction = FREQ_TO_DIRECTION.get(best_freq)
                    if direction:
                        self.movement_queue.put(direction)
                        print(f"BCI Movement detected: freq={best_freq}Hz, score={best_score:.3f}, direction={direction}")
                
                time.sleep(UPDATE_INTERVAL)
                
            except Exception as e:
                print(f"Error in BCI monitoring: {e}")
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


def draw_maze(maze, player, shapes, dt, bci_status=""):
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
    
    # Draw BCI status
    font = pygame.font.Font(None, 24)
    text = font.render(f"BCI Status: {bci_status}", True, (255, 255, 255))
    screen.blit(text, (10, 10))
    
    # Draw instructions
    instructions = [
        "Look at flickering shapes to move:",
        "Circle (Yellow, 5Hz) = UP",
        "Square (Magenta, 10Hz) = RIGHT", 
        "Triangle (Cyan, 15Hz) = DOWN",
        "Diamond (Orange, 20Hz) = LEFT"
    ]
    
    for i, instruction in enumerate(instructions):
        text = font.render(instruction, True, (255, 255, 255))
        screen.blit(text, (10, 40 + i * 25))


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


def process_events(player, maze, bci_controller):
    """Handle all Pygame events and BCI movements."""
    # Handle pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            # Allow keyboard control as fallback
            if process_keydown(event.key, player, maze):
                return False
    
    # Handle BCI movements
    if bci_controller:
        movement = bci_controller.get_movement()
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
    
    # Setup BCI controller
    bci_controller = BCIController()
    bci_status = "Disconnected"
    
    # Try to setup BCI board
    try:
        if bci_controller.setup_board():
            bci_controller.start_monitoring()
            bci_status = "Connected"
            print("BCI system ready! Look at the flickering shapes to control movement.")
            time.sleep(2)  # Give time for initial data collection
        else:
            print("BCI setup failed. Using keyboard controls only.")
            bci_controller = None
    except Exception as e:
        print(f"BCI initialization error: {e}")
        print("Running in keyboard-only mode.")
        bci_controller = None

    running = True
    while running:
        dt = clock.tick(FPS)  # Get delta time in milliseconds
        running = process_events(player, maze, bci_controller)

        screen.fill(BACKGROUND_COLOR)
        draw_maze(maze, player, shapes, dt, bci_status)
        pygame.display.flip()

    # Cleanup
    if bci_controller:
        bci_controller.stop_monitoring()
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()