import pygame
import random
import sys
import math

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

# Shape colors
SHAPE_COLORS = [
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 165, 0),  # Orange
]

# Screen setup
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Maze Game')
clock = pygame.time.Clock()


class FlickeringShape:
    def __init__(self, x, y, shape_type, color, frequency):
        self.x = x
        self.y = y
        self.shape_type = shape_type  # 'circle', 'square', 'triangle', 'diamond'
        self.color = color
        self.frequency = frequency
        self.size = 200  # Size of the shape
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
    
    # Calculate maze area
    maze_width = (SCREEN_WIDTH - 2 * BORDER_SIZE)
    maze_height = (SCREEN_HEIGHT - 2 * BORDER_SIZE)
    maze_center_x = SCREEN_WIDTH // 2
    maze_center_y = SCREEN_HEIGHT // 2
    
    # Top border - Circle (2 Hz)
    shapes.append(FlickeringShape(maze_center_x, BORDER_SIZE // 2, 'circle', SHAPE_COLORS[0], 5))
    
    # Right border - Square (3 Hz)
    shapes.append(FlickeringShape(SCREEN_WIDTH - BORDER_SIZE // 2, maze_center_y, 'square', SHAPE_COLORS[1], 10))
    
    # Bottom border - Triangle (4 Hz)
    shapes.append(FlickeringShape(maze_center_x, SCREEN_HEIGHT - BORDER_SIZE // 2, 'triangle', SHAPE_COLORS[2], 15))
    
    # Left border - Diamond (5 Hz)
    shapes.append(FlickeringShape(BORDER_SIZE // 2, maze_center_y, 'diamond', SHAPE_COLORS[3], 20))
    
    return shapes


def draw_maze(maze, player, shapes, dt):
    # Calculate maze dimensions and positioning with border
    maze_width = len(maze[0]) * CELL_SIZE
    maze_height = len(maze) * CELL_SIZE

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
        shape.draw(screen, 0, 0)  # No offset needed as shapes are positioned absolutely


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


def process_events(player, maze):
    """Handle all Pygame events and return a boolean status for the game loop."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if process_keydown(event.key, player, maze):
                return False
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

    # Generate smaller maze to account for border
    maze_cells_width = (SCREEN_WIDTH - 2 * BORDER_SIZE) // CELL_SIZE
    maze_cells_height = (SCREEN_HEIGHT - 2 * BORDER_SIZE) // CELL_SIZE
    maze = generate_maze(maze_cells_width, maze_cells_height)
    player = Player(1, 1)
    shapes = create_flickering_shapes()

    running = True
    while running:
        dt = clock.tick(FPS)  # Get delta time in milliseconds
        running = process_events(player, maze)

        screen.fill(BACKGROUND_COLOR)
        draw_maze(maze, player, shapes, dt)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()