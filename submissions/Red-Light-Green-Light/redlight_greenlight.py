import sys
import random
import pygame
import math
import os
import time
import numpy as np
import brainflow_stream

# Optional BrainFlow import (graceful fallback if unavailable)
EEG_AVAILABLE = False
BrainFlowBoardSetup = None
BoardIds = None
try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _parent_dir = os.path.abspath(os.path.join(_this_dir, os.pardir))
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    from brainflow_stream import BrainFlowBoardSetup  # uses your repo helper
    import brainflow
    from brainflow.board_shim import BoardIds  # correct enum import
    EEG_AVAILABLE = True
except Exception:
    BrainFlowBoardSetup = None
    BoardIds = None
    EEG_AVAILABLE = False

WIDTH, HEIGHT = 640, 480
FPS = 60

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 30, 30)
GREEN = (30, 200, 60)
DK_RED = (120, 0, 0)
DK_GREEN = (0, 80, 0)
YELLOW = (255, 215, 0)
DK_YELLOW = (120, 100, 0)
BLUE = (40, 120, 255)
ORANGE = (255, 140, 0)
GRAY = (200, 200, 200)
P1_ACCENT = (90, 150, 255)
P2_ACCENT = (255, 120, 120)

# Perspective scaling controls
SCALE_FAR = 0.35  # size at horizon
SCALE_NEAR = 1.35  # size near bottom
SCALE_GAMMA = 1.15  # >1 exaggerates shrinking near horizon

# Optional PNG chicken data
CHICKEN_IMG = None
_IMG_SCALE_CACHE = {}

# Optional PNG car sprites
CAR_IMG = None  # legacy single image fallback
CAR_IMG_FRONT = None
CAR_IMG_BACK = None
_CAR_IMG_SCALE_CACHE = {}

# ---------------- EEG utilities (restored for real data) ---------------- #

def _remove_dc_offset(eeg_data: np.ndarray) -> np.ndarray:
    return eeg_data - np.mean(eeg_data, axis=1, keepdims=True)

try:
    from scipy.signal import welch
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    welch = None

def _band_power_ratio_fft(eeg_data, sfreq, bands=None, nperseg=1024):
    """
    Compute a single scalar: the alpha:beta power ratio averaged across all channels
    using Welch's method.
    """
    if not SCIPY_AVAILABLE:
        return 1.0, 1.0, 1.0  # fallback if scipy not available (ratio, alpha_avg, beta_avg)
        
    if bands is None:
        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 12),
            "beta": (12, 30),
            "gamma": (30, 100),
        }

    alpha_band = bands.get("alpha", (8, 12))
    beta_band = bands.get("beta", (12, 30))

    eeg = np.asarray(eeg_data)
    if eeg.ndim == 1:
        eeg = eeg[None, :]  # (1, n_samples)

    nperseg = int(min(nperseg, eeg.shape[1])) if eeg.shape[1] > 0 else 0
    if nperseg <= 1 or sfreq <= 0:
        return 1.0, 1.0, 1.0

    freqs, psd = welch(eeg, fs=sfreq, nperseg=nperseg, axis=1)  # psd: (n_channels, n_freqs)

    def band_power(low, high):
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return np.zeros(psd.shape[0], dtype=float)
        return np.trapezoid(psd[:, mask], freqs[mask], axis=1)

    alpha_power = band_power(*alpha_band)
    beta_power = band_power(*beta_band)

    # Calculate averages across channels
    alpha_avg = float(np.mean(alpha_power))
    beta_avg = float(np.mean(beta_power))

    epsilon = 1e-12
    ratios = alpha_power / (beta_power + epsilon)  # per-channel ratio
    ratio_avg = float(np.mean(ratios))
    
    return ratio_avg, alpha_avg, beta_avg

# ---------------- Drawing helpers (copied from v2) ---------------- #

def draw_traffic_light(surface, x, y, state):
    pygame.draw.rect(surface, (60, 60, 60), (x + 24, y + 160, 12, 120), border_radius=6)
    pygame.draw.rect(surface, (30, 30, 30), (x, y, 60, 160), border_radius=12)
    pygame.draw.rect(surface, (80, 80, 80), (x, y, 60, 160), width=2, border_radius=12)
    cx = x + 30
    colors = {
        "red": RED if state == "red" else DK_RED,
        "yellow": YELLOW if state == "yellow" else DK_YELLOW,
        "green": GREEN if state == "green" else DK_GREEN,
    }
    positions = {"red": y + 35, "yellow": y + 80, "green": y + 125}
    pygame.draw.circle(surface, colors["red"], (cx, positions["red"]), 22)
    pygame.draw.circle(surface, colors["yellow"], (cx, positions["yellow"]), 22)
    pygame.draw.circle(surface, colors["green"], (cx, positions["green"]), 22)
    active = state
    if active in ("red", "yellow", "green"):
        glow_color = {
            "red": (255, 80, 80, 70),
            "yellow": (255, 240, 100, 70),
            "green": (120, 255, 140, 70),
        }[active]
        glow = pygame.Surface((140, 140), pygame.SRCALPHA)
        pygame.draw.circle(glow, glow_color, (70, 70), 60)
        gy = positions[active] - 70
        surface.blit(glow, (cx - 70, gy), special_flags=pygame.BLEND_ADD)

def draw_background(surface):
    top = (180, 210, 255)
    mid = (120, 170, 240)
    h1 = int(HEIGHT * 0.35)
    h2 = int(HEIGHT * 0.5)
    pygame.draw.rect(surface, top, (0, 0, WIDTH, h1))
    pygame.draw.rect(surface, mid, (0, h1, WIDTH, h2 - h1))

def draw_cloud(surface, x, y, scale=1.0):
    color = (245, 250, 255)
    r = int(22 * scale)
    c = [
        (x, y),
        (x + r * 2, y - int(r * 0.4)),
        (x + r * 4, y),
        (x + r * 1, y + int(r * 0.6)),
        (x + r * 3, y + int(r * 0.5)),
    ]
    for cx, cy in c:
        pygame.draw.circle(surface, color, (int(cx), int(cy)), r)

def draw_horizon_fog(surface, horizon_y):
    fog_h = int(HEIGHT * 0.35)
    fog = pygame.Surface((WIDTH, fog_h), pygame.SRCALPHA)
    for i in range(fog_h):
        t = i / max(1, fog_h)
        alpha = int(90 * (1 - t))
        pygame.draw.rect(fog, (255, 255, 255, alpha), (0, i, WIDTH, 1))
    surface.blit(fog, (0, horizon_y))

def draw_road(surface, horizon_y, center_tilt_x, scroll):
    bottom_width = int(WIDTH * 0.8)
    top_width = int(WIDTH * 0.2)
    bottom_y = HEIGHT
    color_road = (40, 40, 48)
    center_bottom = WIDTH // 2
    center_top = int(WIDTH * 0.5 + center_tilt_x)
    pts = [
        (center_top - top_width // 2, horizon_y),
        (center_top + top_width // 2, horizon_y),
        (center_bottom + bottom_width // 2, bottom_y),
        (center_bottom - bottom_width // 2, bottom_y),
    ]
    pygame.draw.polygon(surface, color_road, pts)
    line_color = (230, 230, 230)
    segments = 16
    for i in range(segments):
        v = (i / segments + scroll) % 1.0
        y0 = int(horizon_y + (HEIGHT - horizon_y) * v)
        y1 = int(horizon_y + (HEIGHT - horizon_y) * min(1.0, v + 0.05))
        scale = 1 - v
        cx = int(center_top * (1 - v) + center_bottom * v)
        w = max(2, int(6 * scale))
        pygame.draw.line(surface, line_color, (cx, y0), (cx, y1), width=w)

def draw_car(surface, sx, sy, v, lane_halfw, color_body=(180, 30, 30), img=None):
    lane_w = max(8, int(lane_halfw * 1.8))
    lane_w = min(lane_w, int(lane_halfw * 2 - 2))
    if img is not None:
        car_w = max(8, lane_w)
        img_id = id(img)
        cache_key = (img_id, int(car_w))
        scaled = _CAR_IMG_SCALE_CACHE.get(cache_key)
        if scaled is None:
            orig_w, orig_h = img.get_width(), img.get_height()
            car_h = max(8, int(orig_h * (car_w / max(1, orig_w))))
            try:
                scaled = pygame.transform.smoothscale(img, (int(car_w), int(car_h)))
            except Exception:
                scaled = pygame.transform.scale(img, (int(car_w), int(car_h)))
            _CAR_IMG_SCALE_CACHE[cache_key] = scaled
            if len(_CAR_IMG_SCALE_CACHE) > 256:
                _CAR_IMG_SCALE_CACHE.clear()
        surface.blit(scaled, (int(sx - scaled.get_width() // 2), int(sy - scaled.get_height())))
        return
    car_w = lane_w
    car_h = max(10, int(car_w * 1.7))
    rect = pygame.Rect(int(sx - car_w // 2), int(sy - car_h), int(car_w), int(car_h))
    pygame.draw.rect(surface, color_body, rect, border_radius=max(4, int(6 * v)))
    w = rect.inflate(-int(car_w * 0.4), -int(car_h * 0.6))
    if w.height > 0 and w.width > 0:
        pygame.draw.rect(surface, (230, 230, 230), w, border_radius=max(2, int(4 * v)))

def draw_side_scenery(surface, horizon_y, road_tilt, scroll):
    bottom_width = int(WIDTH * 0.8)
    top_width = int(WIDTH * 0.2)
    center_bottom = WIDTH // 2
    center_top = int(WIDTH * 0.5 + road_tilt)
    segments = 24
    left_top = (center_top - top_width // 2, horizon_y)
    left_bottom = (center_bottom - bottom_width // 2, HEIGHT)
    right_top = (center_top + top_width // 2, horizon_y)
    right_bottom = (center_bottom + bottom_width // 2, HEIGHT)
    grass_left = (88, 155, 95)
    grass_right = (76, 135, 85)
    pygame.draw.polygon(surface, grass_left, [(0, horizon_y), left_top, left_bottom, (0, HEIGHT)])
    pygame.draw.polygon(surface, grass_right, [right_top, (WIDTH, horizon_y), (WIDTH, HEIGHT), right_bottom])

    def draw_tree(x, y, sc):
        trunk_h = max(10, int(28 * sc))
        trunk_w = max(3, int(6 * sc))
        pygame.draw.rect(surface, (110, 80, 50), (x - trunk_w // 2, y - trunk_h, trunk_w, trunk_h))
        leaf_r = max(8, int(16 * sc))
        leaf_color = (50, 120, 60)
        pygame.draw.circle(surface, leaf_color, (x, y - trunk_h - int(leaf_r * 0.2)), leaf_r)
        pygame.draw.circle(surface, leaf_color, (x - leaf_r, y - trunk_h), int(leaf_r * 0.8))
        pygame.draw.circle(surface, leaf_color, (x + leaf_r, y - trunk_h), int(leaf_r * 0.8))

    def prand(n: int, salt: int = 0) -> float:
        return (math.sin(n * 127.1 + salt * 311.7) * 43758.5453) % 1.0

    fracs = [0.12, 0.28, 0.45, 0.62, 0.78, 0.92]

    for i in range(segments):
        v = (i / segments + scroll) % 1.0
        y = int(horizon_y + (HEIGHT - horizon_y) * v)
        if y < horizon_y or y > HEIGHT:
            continue
        half_top = top_width * 0.5
        half_bottom = bottom_width * 0.5
        halfw = int(half_top * (1 - v) + half_bottom * v)
        cx = int(center_top * (1 - v) + center_bottom * v)
        left_edge = cx - halfw
        right_edge = cx + halfw
        margin = max(6, int(halfw * 0.10))
        sc = SCALE_FAR + (SCALE_NEAR - SCALE_FAR) * (v ** SCALE_GAMMA)
        row_idx = i % 64
        maskL = int(prand(row_idx, 101) * (1 << len(fracs)))
        maskR = int(prand(row_idx, 202) * (1 << len(fracs)))
        if maskL == 0:
            maskL = 0b001010
        if maskR == 0:
            maskR = 0b010100
        left_pad = 16
        right_pad = 16
        left_min = left_pad
        left_max = max(left_min, left_edge - margin)
        right_min = min(WIDTH - right_pad, right_edge + margin)
        right_max = WIDTH - right_pad
        for k, f in enumerate(fracs):
            jL = int((prand(row_idx * 17 + k, 303) - 0.5) * 12)
            jR = int((prand(row_idx * 19 + k, 404) - 0.5) * 12)
            if (maskL >> k) & 1:
                gx = int(left_min + f * (left_max - left_min)) + jL
                draw_tree(gx, y, sc)
            if (maskR >> k) & 1:
                gx = int(right_min + f * (right_max - right_min)) + jR
                draw_tree(gx, y, sc)

def draw_player(surface, x, y, size, accent_color=None, anim_phase: float = 0.0, moving: bool = False):
    bob = int(size * 0.04 * math.sin(anim_phase * math.tau)) if moving else 0
    if CHICKEN_IMG is not None and size > 0:
        shadow_rect = pygame.Rect(x + int(size * 0.1), (y - bob) + int(size * 0.85), int(size * 0.8), int(size * 0.18))
        pygame.draw.ellipse(surface, (0, 0, 0, 60), shadow_rect)
        img_key = int(size)
        img = _IMG_SCALE_CACHE.get(img_key)
        if img is None:
            try:
                img = pygame.transform.smoothscale(CHICKEN_IMG, (int(size), int(size)))
            except Exception:
                img = pygame.transform.scale(CHICKEN_IMG, (int(size), int(size)))
            _IMG_SCALE_CACHE[img_key] = img
            if len(_IMG_SCALE_CACHE) > 128:
                _IMG_SCALE_CACHE.clear()
        surface.blit(img, (x, y - bob))
        return
    yb = y - bob
    shadow_rect = pygame.Rect(x + int(size * 0.1), yb + int(size * 0.85), int(size * 0.8), int(size * 0.18))
    pygame.draw.ellipse(surface, (0, 0, 0, 60), shadow_rect)
    body_rect = pygame.Rect(x + int(size * 0.08), yb + int(size * 0.28), int(size * 0.84), int(size * 0.62))
    pygame.draw.ellipse(surface, WHITE, body_rect)
    pygame.draw.ellipse(surface, (180, 180, 180), body_rect, width=2)
    head_center = (x + int(size * 0.35), yb + int(size * 0.25))
    head_radius = max(6, int(size * 0.18))
    pygame.draw.circle(surface, WHITE, head_center, head_radius)
    pygame.draw.circle(surface, (180, 180, 180), head_center, head_radius, width=2)
    eye_center = (head_center[0] + int(size * 0.04), head_center[1] - int(size * 0.04))
    pygame.draw.circle(surface, BLACK, eye_center, max(2, int(size * 0.03)))
    beak_len = int(size * 0.18)
    beak_height = int(size * 0.10)
    beak_tip = (head_center[0] + head_radius + beak_len, head_center[1])
    beak_top = (head_center[0] + head_radius, head_center[1] - beak_height)
    beak_bottom = (head_center[0] + head_radius, head_center[1] + beak_height)
    pygame.draw.polygon(surface, ORANGE, [beak_top, beak_tip, beak_bottom])
    comb_base_x = head_center[0] - int(head_radius * 0.6)
    comb_y = head_center[1] - head_radius - int(size * 0.02)
    comb_r = max(3, int(size * 0.05))
    pygame.draw.circle(surface, RED, (comb_base_x, comb_y), comb_r)
    pygame.draw.circle(surface, RED, (comb_base_x + comb_r, comb_y - int(comb_r * 0.6)), comb_r)
    pygame.draw.circle(surface, RED, (comb_base_x + comb_r * 2, comb_y), comb_r)
    leg_y = yb + int(size * 0.82)
    leg_x1 = x + int(size * 0.42)
    leg_x2 = x + int(size * 0.62)
    pygame.draw.line(surface, YELLOW, (leg_x1, leg_y - int(size * 0.10)), (leg_x1, leg_y), width=4)
    pygame.draw.line(surface, YELLOW, (leg_x2, leg_y - int(size * 0.10)), (leg_x2, leg_y), width=4)
    foot_w = int(size * 0.10)
    pygame.draw.line(surface, YELLOW, (leg_x1 - foot_w, leg_y), (leg_x1 + foot_w, leg_y), width=4)
    pygame.draw.line(surface, YELLOW, (leg_x2 - foot_w, leg_y), (leg_x2 + foot_w, leg_y), width=4)
    if accent_color is not None:
        band_rect = pygame.Rect(
            head_center[0] - head_radius,
            head_center[1] + int(head_radius * 0.35),
            head_radius * 2,
            max(3, int(size * 0.08)),
        )
        pygame.draw.rect(surface, accent_color, band_rect, border_radius=6)

def show_center_text(screen, text, color, font):
    msg = font.render(text, True, color)
    rect = msg.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(msg, rect)

def main(serial_port: str = None):
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    info = pygame.display.Info()
    global WIDTH, HEIGHT
    WIDTH, HEIGHT = info.current_w, info.current_h
    pygame.display.set_caption("Red Light Green Light - Real Alpha + Test P2 (Fullscreen)")
    clock = pygame.time.Clock()

    font_big = pygame.font.SysFont(None, 56)
    font_small = pygame.font.SysFont(None, 28)

    global CHICKEN_IMG, CAR_IMG, CAR_IMG_FRONT, CAR_IMG_BACK
    # PNGs
    png_path = os.path.join(os.path.dirname(__file__), "chicken.png")
    if os.path.isfile(png_path):
        try:
            CHICKEN_IMG = pygame.image.load(png_path).convert_alpha()
        except Exception:
            CHICKEN_IMG = None
    base_dir = os.path.dirname(__file__)
    car_front_path = os.path.join(base_dir, "car_front.png")
    car_back_path = os.path.join(base_dir, "car_back.png")
    car_png_path = os.path.join(base_dir, "car.png")
    if os.path.isfile(car_front_path):
        try:
            CAR_IMG_FRONT = pygame.image.load(car_front_path).convert_alpha()
        except Exception:
            CAR_IMG_FRONT = None
    if os.path.isfile(car_back_path):
        try:
            CAR_IMG_BACK = pygame.image.load(car_back_path).convert_alpha()
        except Exception:
            CAR_IMG_BACK = None
    if os.path.isfile(car_png_path):
        try:
            CAR_IMG = pygame.image.load(car_png_path).convert_alpha()
        except Exception:
            CAR_IMG = None

    # World setup
    player_size = max(28, int(min(WIDTH, HEIGHT) * 0.06))
    horizon_y = int(HEIGHT * 0.35)
    road_tilt = -int(WIDTH * 0.08)
    road_bottom_width = int(WIDTH * 0.8)
    road_top_width = int(WIDTH * 0.2)
    player1_world_x = -player_size * 0.35
    player2_world_x = player_size * 0.35
    player1_vx = 0.0
    player2_vx = 0.0
    start_world_y = 0.0
    player1_world_y = start_world_y
    player2_world_y = start_world_y
    camera_y = 0.0
    camera_y_prev = camera_y
    camera_anchor_screen_y = int(HEIGHT * 0.62)
    move_step = max(8, int(HEIGHT * 0.02))
    MOVE_SPEED = max(60, int(move_step * 5))
    LATERAL_ACCEL_BASE = MOVE_SPEED * 1.4 * 6.0
    LATERAL_MAX_BASE = MOVE_SPEED * 1.2 * 6.0
    LATERAL_DRAG = 6.0
    WALL_BOUNCE = 0.25

    # Cars
    road_scroll = 0.0
    cloud_off_x = 0.0
    cloud_off_y = 0.0
    scenery_scroll = 0.0
    car_active = False
    car_type = None
    car_world_x = 0.0
    car_world_y = 0.0
    car_speed = 0.0
    car_color = (180, 30, 30)
    car_spawn_cooldown = 0
    CAR_MIN_COOLDOWN = 3500
    CAR_MAX_COOLDOWN = 7000
    CAR_ONCOMING_SPEED = MOVE_SPEED * 0.55
    CAR_TRAILING_SPEED = MOVE_SPEED * 0.45

    # Traffic light FSM
    state_sequence = ["green", "yellow", "red", "yellow"]
    state_index = 0
    light_state = state_sequence[state_index]
    light_timer_ms = 0
    YELLOW_MS = 800
    def next_interval_for(state: str) -> int:
        if state == "yellow":
            return YELLOW_MS
        elif state == "green":
            return random.randint(4000, 8000)  # Extended green: 4-8 seconds
        else:  # red
            return random.randint(1500, 3000)  # Keep red duration the same
    light_interval_ms = next_interval_for(light_state)

    running = True
    game_over = False
    win = False
    winner_label = ""
    elapsed_ms = 0

    # Speed bars
    current_p1_mult = 0.0
    current_p2_mult = 0.0

    # P1: Real EEG alpha/beta ratio
    alpha_ratio_p1 = 1.0  # fallback value
    last_eeg_update_ms = 0
    eeg_refresh_ms = 200
    eeg_accum_ms = 0
    samples_needed = 0

    # P2: Test random values between 0.1 and 3.0
    alpha_ratio_p2 = 0.0
    alpha_ratio_p2_timer = 0
    alpha_ratio_p2_interval = 1500  # Different interval for P2 to make them independent

    # EEG integration (alpha controls P1)
    eeg_ready = False
    eeg_setup = None
    sfreq = 0
    eeg_chs = []

    # BrainFlow setup for P1
    if EEG_AVAILABLE:
        try:
            board_id = brainflow.BoardIds.CYTON_BOARD.value
            eeg_setup = BrainFlowBoardSetup(board_id=board_id, serial_port=serial_port, name="Cyton")
            eeg_setup.setup()
            sfreq = eeg_setup.get_sampling_rate() or 0
            if sfreq > 0:
                eeg_chs = getattr(eeg_setup, "eeg_channels", []) or list(range(1, 9))
                samples_needed = max(int(2.0 * sfreq), 64)
                eeg_ready = True
                print(f"EEG ready: {sfreq} Hz, channels: {eeg_chs}")
                print("Alpha/Beta ratio monitoring started...")
        except Exception as e:
            print("EEG init failed:", e)
            eeg_setup = None
            eeg_ready = False
    else:
        print("BrainFlow not available - using fallback mode")

    def reset_game():
        nonlocal player1_world_y, player2_world_y, camera_y, game_over, win, winner_label, elapsed_ms
        nonlocal state_index, light_state, light_timer_ms, light_interval_ms
        nonlocal current_p1_mult, current_p2_mult
        nonlocal camera_y_prev, road_scroll, cloud_off_x, cloud_off_y
        nonlocal player1_vx, player2_vx
        nonlocal car_active, car_spawn_cooldown
        nonlocal alpha_ratio_p1, alpha_ratio_p2, alpha_ratio_p2_timer
        player1_world_y = start_world_y
        player2_world_y = start_world_y
        player1_vx = 0.0
        player2_vx = 0.0
        camera_y = 0.0
        camera_y_prev = camera_y
        game_over = False
        win = False
        winner_label = ""
        elapsed_ms = 0
        state_index = 0
        light_state = state_sequence[state_index]
        light_timer_ms = 0
        light_interval_ms = next_interval_for(light_state)
        current_p1_mult = 0.0
        current_p2_mult = 0.0
        road_scroll = 0.0
        scenery_scroll = 0.0
        cloud_off_x = 0.0
        cloud_off_y = 0.0
        _CAR_IMG_SCALE_CACHE.clear()
        car_active = False
        car_spawn_cooldown = 0
        alpha_ratio_p1 = 1.0  # fallback
        alpha_ratio_p2 = random.uniform(0.1, 3.0)  # Initialize with random value
        alpha_ratio_p2_timer = 0

    # Initialize alpha ratios for first time
    alpha_ratio_p1 = 1.0  # fallback
    alpha_ratio_p2 = random.uniform(0.1, 3.0)

    while running:
        dt = clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_x:
                    reset_game()

        if not game_over:
            keys = pygame.key.get_pressed()
            dt_sec = dt / 1000.0

            # Update P1 EEG alpha/beta ratio
            if eeg_ready:
                eeg_accum_ms += dt
                if eeg_accum_ms >= eeg_refresh_ms:
                    eeg_accum_ms = 0
                    data = eeg_setup.get_current_board_data(num_samples=samples_needed)
                    if data is not None and data.size > 0:
                        eeg = data[eeg_chs, :]
                        eeg = _remove_dc_offset(eeg)
                        ratio, alpha_power, beta_power = _band_power_ratio_fft(eeg, sfreq)  # alpha:beta ratio + individual powers
                        
                        # Fallback for zero/very low alpha ratio - simulate reasonable values
                        if ratio <= 0.05 or alpha_power <= 0.01:  # Very low or zero ratio/alpha power
                            alpha_ratio_p1 = random.uniform(0.5, 2.0)  # Simulate reasonable alpha/beta ratio
                            print(f"P1 EEG | FALLBACK MODE - Original ratio too low ({ratio:.3f}), using simulated: {alpha_ratio_p1:.3f}")
                        else:
                            alpha_ratio_p1 = ratio
                            
                        last_eeg_update_ms = elapsed_ms
                        
                        # Print alpha/beta ratio and individual band powers for debugging
                        print(f"P1 EEG | Alpha Power: {alpha_power:.3f} | Beta Power: {beta_power:.3f} | Ratio (α/β): {alpha_ratio_p1:.3f} | Speed Mult: {max(0.0, min(1.6, alpha_ratio_p1 * 0.5)):.3f}")

            # Update P2 test alpha ratio periodically
            alpha_ratio_p2_timer += dt
            if alpha_ratio_p2_timer >= alpha_ratio_p2_interval:
                alpha_ratio_p2 = random.uniform(0.1, 3.0)
                alpha_ratio_p2_timer = 0
                
                # Print P2 test ratio for comparison
                print(f"P2 Test Alpha Ratio: {alpha_ratio_p2:.3f} | Speed Mult: {max(0.0, min(1.6, alpha_ratio_p2 * 0.5)):.3f}")

            # Direct translation of alpha ratios to player multipliers (clamped to reasonable range)
            max_mult = 1.6
            current_p1_mult = max(0.0, min(max_mult, alpha_ratio_p1 * 0.5))  # Scale by 0.5 to keep in reasonable range
            current_p2_mult = max(0.0, min(max_mult, alpha_ratio_p2 * 0.5))  # Scale by 0.5 to keep in reasonable range

            # Apply forward/back movement with traffic light for P1
            if current_p1_mult > 0.0:
                if light_state in ("green", "yellow"):
                    player1_world_y -= (MOVE_SPEED * current_p1_mult) * dt_sec
                elif light_state == "red":
                    player1_world_y = min(player1_world_y + (MOVE_SPEED * current_p1_mult) * dt_sec, start_world_y)

            # Apply forward/back movement with traffic light for P2 (using test alpha ratio)
            if current_p2_mult > 0.0:
                if light_state in ("green", "yellow"):
                    player2_world_y -= (MOVE_SPEED * current_p2_mult) * dt_sec
                elif light_state == "red":
                    player2_world_y = min(player2_world_y + (MOVE_SPEED * current_p2_mult) * dt_sec, start_world_y)

            # Lateral input: P1 (A/D) and P2 (Left/Right arrows); scaled with 10% baseline
            p1_scale = max(0.1, current_p1_mult)
            p1_accel = LATERAL_ACCEL_BASE * p1_scale
            p1_max = LATERAL_MAX_BASE * p1_scale
            p1_ax = 0.0
            if keys[pygame.K_a]:
                p1_ax -= p1_accel
            if keys[pygame.K_d]:
                p1_ax += p1_accel
            player1_vx += p1_ax * dt_sec
            player1_vx -= player1_vx * LATERAL_DRAG * dt_sec
            player1_vx = max(-p1_max, min(p1_max, player1_vx))
            player1_world_x += player1_vx * dt_sec

            p2_scale = max(0.1, current_p2_mult)
            p2_accel = LATERAL_ACCEL_BASE * p2_scale
            p2_max = LATERAL_MAX_BASE * p2_scale
            p2_ax = 0.0
            if keys[pygame.K_LEFT]:
                p2_ax -= p2_accel
            if keys[pygame.K_RIGHT]:
                p2_ax += p2_accel
            player2_vx += p2_ax * dt_sec
            player2_vx -= player2_vx * LATERAL_DRAG * dt_sec
            player2_vx = max(-p2_max, min(p2_max, player2_vx))
            player2_world_x += player2_vx * dt_sec

            # Clamp to road and collisions (as in v2)
            def compute_edges_and_scale(wx: float, wy: float):
                v = 0.0 if HEIGHT == horizon_y else max(0.0001, min(1.0, (wy - camera_y) / (HEIGHT - horizon_y)))
                center_top = (WIDTH * 0.5 + road_tilt * (1 - v))
                halftop = road_top_width * 0.5
                halfbot = road_bottom_width * 0.5
                halfw = halftop * (1 - v) + halfbot * v
                cx = center_top
                sc = SCALE_FAR + (SCALE_NEAR - SCALE_FAR) * (v ** SCALE_GAMMA)
                px_half = (player_size * sc * 0.35)
                world_half_x = px_half / v
                left_edge = ((cx - halfw) + px_half - cx) / v
                right_edge = ((cx + halfw) - px_half - cx) / v
                h_world = player_size * sc * 0.65
                return v, sc, left_edge, right_edge, world_half_x, h_world

            p1_v, p1_sc, p1_wxmin, p1_wxmax, p1_hbx, p1_hh = compute_edges_and_scale(player1_world_x, player1_world_y)
            p2_v, p2_sc, p2_wxmin, p2_wxmax, p2_hbx, p2_hh = compute_edges_and_scale(player2_world_x, player2_world_y)

            if player1_world_x < p1_wxmin:
                player1_world_x = p1_wxmin
                player1_vx = abs(player1_vx) * WALL_BOUNCE
            elif player1_world_x > p1_wxmax:
                player1_world_x = p1_wxmax
                player1_vx = -abs(player1_vx) * WALL_BOUNCE
            if player2_world_x < p2_wxmin:
                player2_world_x = p2_wxmin
                player2_vx = abs(player2_vx) * WALL_BOUNCE
            elif player2_world_x > p2_wxmax:
                player2_world_x = p2_wxmax
                player2_vx = -abs(player2_vx) * WALL_BOUNCE

            p1_left = player1_world_x - p1_hbx
            p1_right = player1_world_x + p1_hbx
            p1_top = player1_world_y - p1_hh
            p1_bottom = player1_world_y
            p2_left = player2_world_x - p2_hbx
            p2_right = player2_world_x + p2_hbx
            p2_top = player2_world_y - p2_hh
            p2_bottom = player2_world_y
            x_overlap = min(p1_right, p2_right) - max(p1_left, p2_left)
            y_overlap = min(p1_bottom, p2_bottom) - max(p1_top, p2_top)
            if x_overlap > 0 and y_overlap > 0:
                push = x_overlap * 0.5
                if player1_world_x <= player2_world_x:
                    player1_world_x -= push
                    player2_world_x += push
                else:
                    player1_world_x += push
                    player2_world_x -= push
                v1, v2 = player1_vx, player2_vx
                player1_vx = v2 * 0.5
                player2_vx = v1 * 0.5

            p1_v, p1_sc, p1_wxmin, p1_wxmax, p1_hbx, p1_hh = compute_edges_and_scale(player1_world_x, player1_world_y)
            p2_v, p2_sc, p2_wxmin, p2_wxmax, p2_hbx, p2_hh = compute_edges_and_scale(player2_world_x, player2_world_y)
            player1_world_x = max(p1_wxmin, min(p1_wxmax, player1_world_x))
            player2_world_x = max(p2_wxmin, min(p2_wxmax, player2_world_x))

            # Camera and animations
            leader_world_y = min(player1_world_y, player2_world_y)
            camera_y = leader_world_y - (camera_anchor_screen_y - horizon_y)
            cam_dy = camera_y - camera_y_prev
            road_scroll = (road_scroll + (-cam_dy) * 0.004) % 1.0
            scenery_scroll = (scenery_scroll + (-cam_dy) * 0.003) % 1.0
            cloud_off_x += (-cam_dy) * 0.01
            cloud_off_y += (-cam_dy) * 0.02
            camera_y_prev = camera_y

            # Cars spawn/update
            if car_spawn_cooldown > 0:
                car_spawn_cooldown = max(0, car_spawn_cooldown - dt)
            if (not car_active) and car_spawn_cooldown == 0:
                if random.random() < 0.01:
                    car_type = 'oncoming' if random.random() < 0.6 else 'trailing'
                    if car_type == 'oncoming':
                        spawn_v = 0.08
                        center_top = (WIDTH * 0.5 + road_tilt * (1 - spawn_v))
                        halftop = road_top_width * 0.5
                        halfbot = road_bottom_width * 0.5
                        halfw = halftop * (1 - spawn_v) + halfbot * spawn_v
                        lane_center_offset = -halfw * 0.33
                        car_world_x = (center_top + lane_center_offset - center_top) / max(0.001, spawn_v)
                        car_world_y = camera_y + ((horizon_y + (HEIGHT - horizon_y) * spawn_v) - horizon_y)
                        car_speed = -CAR_ONCOMING_SPEED
                        car_color = (40, 40, 160)
                    else:
                        spawn_v = 0.95
                        center_top = (WIDTH * 0.5 + road_tilt * (1 - spawn_v))
                        halftop = road_top_width * 0.5
                        halfbot = road_bottom_width * 0.5
                        halfw = halftop * (1 - spawn_v) + halfbot * spawn_v
                        lane_center_offset = halfw * 0.33
                        car_world_x = (center_top + lane_center_offset - center_top) / max(0.001, spawn_v)
                        car_world_y = camera_y + ((horizon_y + (HEIGHT - horizon_y) * spawn_v) - horizon_y)
                        car_speed = CAR_TRAILING_SPEED
                        car_color = (160, 80, 40)
                    car_active = True
                    car_spawn_cooldown = random.randint(CAR_MIN_COOLDOWN, CAR_MAX_COOLDOWN)

            if car_active:
                car_world_y += (-car_speed) * dt_sec
                car_sy = horizon_y + (car_world_y - camera_y)
                v = 0.0 if HEIGHT == horizon_y else max(0.0001, min(1.0, (car_sy - horizon_y) / (HEIGHT - horizon_y)))
                center_top = (WIDTH * 0.5 + road_tilt * (1 - v))
                halftop = road_top_width * 0.5
                halfbot = road_bottom_width * 0.5
                halfw = halftop * (1 - v) + halfbot * v
                lane_center_offset = (-halfw * 0.33) if car_type == 'oncoming' else (halfw * 0.33)
                car_world_x = (center_top + lane_center_offset - center_top) / max(0.001, v)
                if car_sy < horizon_y - 40 or car_sy > HEIGHT + 80:
                    car_active = False
                lane_w = halfw * 0.5
                car_px_half = max(6, lane_w * 0.35)
                car_world_half = car_px_half / v
                car_left = car_world_x - car_world_half
                car_right = car_world_x + car_world_half
                car_top = car_world_y - (player_size * (SCALE_FAR + (SCALE_NEAR - SCALE_FAR) * (v ** SCALE_GAMMA)) * 0.9)
                car_bottom = car_world_y
                def player_hitbox(wx, wy):
                    pv = 0.0 if HEIGHT == horizon_y else max(0.0001, min(1.0, ((horizon_y + (wy - camera_y)) - horizon_y) / (HEIGHT - horizon_y)))
                    psc = SCALE_FAR + (SCALE_NEAR - SCALE_FAR) * (pv ** SCALE_GAMMA)
                    p_half = (player_size * psc * 0.35) / pv
                    p_hh = (player_size * psc * 0.65)
                    return wx - p_half, wx + p_half, wy - p_hh, wy
                p1_l, p1_r, p1_t, p1_b = player_hitbox(player1_world_x, player1_world_y)
                p2_l, p2_r, p2_t, p2_b = player_hitbox(player2_world_x, player2_world_y)
                def overlap(ax1, ax2, ay1, ay2, bx1, bx2, by1, by2):
                    return (min(ax2, bx2) - max(ax1, bx1) > 0) and (min(ay2, by2) - max(ay1, by1) > 0)
                if overlap(car_left, car_right, car_top, car_bottom, p1_l, p1_r, p1_t, p1_b):
                    game_over = True
                    win = True
                    winner_label = "Player 2"
                elif overlap(car_left, car_right, car_top, car_bottom, p2_l, p2_r, p2_t, p2_b):
                    game_over = True
                    win = True
                    winner_label = "Player 1"

            # Trailing off-screen win
            sy1 = horizon_y + (player1_world_y - camera_y)
            sy2 = horizon_y + (player2_world_y - camera_y)
            off_margin = 40
            if sy1 > HEIGHT + off_margin:
                game_over = True
                win = True
                winner_label = "Player 2"
            elif sy2 > HEIGHT + off_margin:
                game_over = True
                win = True
                winner_label = "Player 1"

        if not game_over:
            elapsed_ms += dt
            light_timer_ms += dt
            if light_timer_ms >= light_interval_ms:
                state_index = (state_index + 1) % len(state_sequence)
                light_state = state_sequence[state_index]
                light_timer_ms = 0
                light_interval_ms = next_interval_for(light_state)

        # Render
        draw_background(screen)
        cloud_dx = int(cloud_off_x)
        cloud_dy = int(cloud_off_y)
        draw_cloud(screen, int(WIDTH * 0.15) + cloud_dx, int(HEIGHT * 0.16) + int(cloud_dy * 0.6), 1.2)
        draw_cloud(screen, int(WIDTH * 0.65) - int(cloud_dx * 0.5), int(HEIGHT * 0.12) + int(cloud_dy * 0.4), 1.4)
        draw_cloud(screen, int(WIDTH * 0.42) + int(cloud_dx * 0.3), int(HEIGHT * 0.20) + int(cloud_dy * 0.8), 1.0)
        draw_side_scenery(screen, horizon_y, road_tilt, scenery_scroll)
        draw_road(screen, horizon_y, road_tilt, road_scroll)
        if car_active:
            car_sy = horizon_y + (car_world_y - camera_y)
            v = 0.0 if HEIGHT == horizon_y else max(0.0001, min(1.0, (car_sy - horizon_y) / (HEIGHT - horizon_y)))
            center_top = (WIDTH * 0.5 + road_tilt * (1 - v))
            sx = int(center_top + car_world_x * v)
            halftop = road_top_width * 0.5
            halfbot = road_bottom_width * 0.5
            halfw = halftop * (1 - v) + halfbot * v
            lane_halfw = halfw * 0.5
            img = None
            if car_type == 'oncoming' and CAR_IMG_FRONT is not None:
                img = CAR_IMG_FRONT
            elif car_type == 'trailing' and CAR_IMG_BACK is not None:
                img = CAR_IMG_BACK
            elif CAR_IMG is not None:
                img = CAR_IMG
            draw_car(screen, sx, int(car_sy), v, lane_halfw, (180, 30, 30), img)
        draw_horizon_fog(screen, horizon_y)
        draw_traffic_light(screen, WIDTH // 2 - 30, int(HEIGHT * 0.02), light_state)

        def world_to_screen(wx, wy):
            sy = horizon_y + (wy - camera_y)
            v = 0.0 if HEIGHT == horizon_y else max(0.0, min(1.0, (sy - horizon_y) / (HEIGHT - horizon_y)))
            cx = int((WIDTH * 0.5 + road_tilt * (1 - v)))
            scale = SCALE_FAR + (SCALE_NEAR - SCALE_FAR) * (v ** SCALE_GAMMA)
            sx = int(cx + wx * v)
            return sx, int(sy), scale, v

        p1_sx, p1_sy, p1_scale, p1_v = world_to_screen(player1_world_x, player1_world_y)
        p2_sx, p2_sy, p2_scale, p2_v = world_to_screen(player2_world_x, player2_world_y)
        p1_sx += int(-WIDTH * 0.02 * p1_v)
        p2_sx += int(WIDTH * 0.02 * p2_v)
        screen.blit(font_small.render("P1", True, P1_ACCENT), (p1_sx - 10, p1_sy - int(player_size * p1_scale)))
        screen.blit(font_small.render("P2", True, P2_ACCENT), (p2_sx - 10, p2_sy - int(player_size * p2_scale)))
        p1_moving = (current_p1_mult > 0 and light_state in ("green", "yellow"))
        p2_moving = (current_p2_mult > 0 and light_state in ("green", "yellow"))
        players = [
            (p1_v, p1_sx, p1_sy, p1_scale, P1_ACCENT, (elapsed_ms / 1000.0) * (2.0 + 2.0 * current_p1_mult), p1_moving),
            (p2_v, p2_sx, p2_sy, p2_scale, P2_ACCENT, (elapsed_ms / 1000.0) * (2.0 + 2.0 * current_p2_mult), p2_moving),
        ]
        players.sort(key=lambda t: t[0])
        for v, sx, sy, sc, accent, phase, moving in players:
            draw_player(screen, sx - int(player_size * sc * 0.5), sy - int(player_size * sc * 0.8), int(player_size * sc), accent_color=accent, anim_phase=phase, moving=moving)

        if light_state == "green":
            state_text = "GREEN - P1: Alpha/Beta EEG  |  P2: Alpha (TEST)"
            state_color = GREEN
        elif light_state == "red":
            state_text = "RED - Do NOT move"
            state_color = RED
        else:
            state_text = "YELLOW - You may move"
            state_color = YELLOW
        screen.blit(font_small.render(state_text, True, state_color), (10, HEIGHT - 34))
        controls_text = "X: Restart   ESC: Quit   P1 A/D, P2 Left/Right"
        screen.blit(font_small.render(controls_text, True, BLACK), (WIDTH - 380, HEIGHT - 34))

        mins = int(elapsed_ms // 60000)
        secs = int((elapsed_ms // 1000) % 60)
        tenths = int((elapsed_ms % 1000) // 100)
        time_text = f"Time: {mins:02}:{secs:02}.{tenths}"
        screen.blit(font_small.render(time_text, True, BLACK), (10, 10))

        # Speed bars
        bar_w = int(WIDTH * 0.25)
        bar_h = 10
        bar_x = 10
        p1_bar_y = 40
        p2_bar_y = 60
        max_mult = 1.6
        pygame.draw.rect(screen, (220, 220, 220), (bar_x, p1_bar_y, bar_w, bar_h), border_radius=4)
        pygame.draw.rect(screen, (220, 220, 220), (bar_x, p2_bar_y, bar_w, bar_h), border_radius=4)
        p1_fill = int(bar_w * min(1.0, current_p1_mult / max_mult))
        p2_fill = int(bar_w * min(1.0, current_p2_mult / max_mult))
        pygame.draw.rect(screen, P1_ACCENT, (bar_x, p1_bar_y, p1_fill, bar_h), border_radius=4)
        pygame.draw.rect(screen, P2_ACCENT, (bar_x, p2_bar_y, p2_fill, bar_h), border_radius=4)
        
        # Show EEG status and values
        eeg_status = "EEG: Connected" if eeg_ready else "EEG: Fallback"
        eeg_color = P1_ACCENT if eeg_ready else (255, 100, 100)
        screen.blit(font_small.render(f"P1 Speed (α/β: {alpha_ratio_p1:.2f}) - {eeg_status}", True, eeg_color), (bar_x + bar_w + 10, p1_bar_y - 6))
        screen.blit(font_small.render(f"P2 Speed (α: {alpha_ratio_p2:.2f}) - TEST", True, P2_ACCENT), (bar_x + bar_w + 10, p2_bar_y - 6))

        # Mode indicator
        mode_text = "REAL EEG P1 + TEST P2" if eeg_ready else "FALLBACK MODE - Both Test"
        mode_color = (100, 255, 100) if eeg_ready else (255, 100, 100)
        screen.blit(font_small.render(mode_text, True, mode_color), (10, 80))

        if game_over:
            if win:
                show_center_text(screen, f"{winner_label} Wins!", GREEN, font_big)
                screen.blit(font_small.render("Press X to restart", True, BLACK), (10, 40))

        pygame.display.flip()

    # EEG cleanup
    if EEG_AVAILABLE and eeg_setup is not None:
        try:
            eeg_setup.stop()
        except Exception:
            pass
    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Red Light Green Light with real EEG alpha/beta control for P1")
    parser.add_argument("--port", type=str, default=None, help="Serial port like \\\\.\\COM3 (Windows) or /dev/ttyUSB0 (Linux)")
    args = parser.parse_args()
    
    # Try environment variable if no CLI arg
    serial_port = args.port or os.environ.get("BRAIN_PORT")
    main(serial_port=serial_port)