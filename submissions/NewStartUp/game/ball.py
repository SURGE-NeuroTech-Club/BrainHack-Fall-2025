import math
import random

import pygame

from game.constants import SCREEN_WIDTH
from game.game_object import GameObject

MAX_BALL_SPEED = 25
START_SPEED = 14
MIN_BALL_SPEED = 10


class Ball(GameObject):

    def __init__(self, x, y, radius, color, speed):
        super().__init__(x - radius, y - radius, radius * 2, radius * 2, color)
        self.radius = radius
        self.speed = speed
        self.direction = random.uniform(-math.pi * 0.75, -math.pi * 0.25)

    def update(self):
        self.rect.x += self.speed * math.cos(self.direction)
        self.rect.y += self.speed * math.sin(self.direction)

        if self.rect.left <= 0 or self.rect.right >= SCREEN_WIDTH:
            self.direction = math.pi - self.direction
        if self.rect.top <= 0:
            self.direction = -self.direction


    def bounce(self):
        reflection = -self.direction
        random_factor = random.uniform(-0.2, 0.2)
        self.direction = reflection + random_factor
        if math.sin(self.direction) == 0:
            self.direction += 0.1

    def draw(self, screen):
        pygame.draw.ellipse(screen, self.color, self.rect)

    def increase_speed(self):
        if self.speed <= MAX_BALL_SPEED:
            self.speed += 1
            print(f"Speed increased to {self.speed}")
    def decrease_speed(self):
        if self.speed >= MIN_BALL_SPEED:
            self.speed -= 1
            print(f"Speed decreased to {self.speed}")

    def light_force(self):
        self.direction *= -1
