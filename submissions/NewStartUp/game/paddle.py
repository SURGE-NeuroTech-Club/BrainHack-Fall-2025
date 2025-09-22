import pygame

from game.game_object import GameObject


class Paddle(GameObject):

    def __init__(self, y_position, width, height, color):
        super().__init__(0, y_position, width, height, color)

        self.points = self._calculate_v_points()

    def _calculate_v_points(self):
        top_y = self.rect.top
        bottom_y = self.rect.bottom
        center_x = self.rect.centerx

        dip_y = self.rect.top + self.rect.height

        points = [(self.rect.left, top_y), (self.rect.right, top_y), (self.rect.right, bottom_y),
                  (self.rect.left, bottom_y), (center_x, dip_y)]

        points = [(self.rect.left, top_y), (self.rect.right, top_y), (self.rect.right, dip_y - self.rect.height / 2),
                  (center_x, dip_y), (self.rect.left, dip_y - self.rect.height / 2)]

        points = [(self.rect.left, top_y), (self.rect.right, top_y), (self.rect.right, bottom_y),
                  (self.rect.left, bottom_y), (center_x, top_y)]

        points = [(self.rect.left, top_y), (center_x, dip_y), (self.rect.right, top_y), (self.rect.right, bottom_y),
                  (self.rect.left, bottom_y)]
        return points

    def draw(self, screen):
        pygame.draw.polygon(screen, self.color, self.points)
