from game.game_object import GameObject


class Brick(GameObject):

    def __init__(self, x, y, width, height, color, intensity):
        super().__init__(x, y, width, height, color)
        self.intensity = intensity
        self.max_intensity = intensity if intensity > 0 else 1
        self.original_color = color

    def hit(self, factor:int = 1):
        self.intensity -= factor
        if self.intensity <= 0:
            return True
        else:
            factor = 0.5 + 0.5 * (self.intensity / self.max_intensity)
            r = int(self.original_color[0] * factor)
            g = int(self.original_color[1] * factor)
            b = int(self.original_color[2] * factor)
            self.color = (r, g, b)
            return False
