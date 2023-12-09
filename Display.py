 
import pygame
from pygame.locals import DOUBLEBUF

class FrameDisplay:
    def __init__(self, width, height):
        pygame.init()
        self.display_screen = pygame.display.set_mode((width, height), DOUBLEBUF)
        self.display_surface = pygame.Surface(self.display_screen.get_size()).convert()

    def show2D(self, image):
        pygame.surfarray.blit_array(self.display_surface, image.swapaxes(0, 1)[:, :, [0, 1, 2]])
        self.display_screen.blit(self.display_surface, (0, 0))
        pygame.display.flip()
