import pygame
import time

FPS = 30
SCREEN_SIZE = (500, 500)


class Sim(object):
    def __init__(self):
        self.run_sim = True
    def process_events(self, screen):
        if self.run_sim is True:
            for event in pygame.event.get():
                if event.type is pygame.KEYDOWN:
                    if event.key is pygame.K_x:
                        pygame.display.quit()
                        pygame.quit()
    def display_frame(self, screen):
        screen.fill((0,0,0))
        if self.run_sim is True:
            pass


def main():
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption('~~wow~~')
    clock = pygame.time.Clock()
    simulation = Sim()
    while simulation.run_sim is True:
        simulation.process_events(screen)
        simulation.display_frame(screen)
        clock.tick(FPS)
    pygame.quit()

main()
