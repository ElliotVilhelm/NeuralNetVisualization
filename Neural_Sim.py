import pygame
import time

FPS = 30
SCREEN_WIDTH = 700
SCREEN_HEIGHT = 500
SCREEN_SIZE = (SCREEN_WIDTH, SCREEN_HEIGHT)
NEURON_SIZE = 50
RED = (200, 20, 0)
GREEN = (0, 220, 10)

class Sim(object):
    def __init__(self):
        self.run_sim = True
        self.input_layer = []
        for i in range(2):
            self.input_layer.append(Neuron(SCREEN_WIDTH/2, (i+1)*100))
    def process_events(self, screen):
        if self.run_sim is True:
            self.display_frame(screen)
            for event in pygame.event.get():
                if event.type is pygame.KEYDOWN:
                    if event.key is pygame.K_x:
                        # pygame.display.quit()
                        pygame.quit()
        pygame.display.flip()
    def display_frame(self, screen):
        screen.fill(RED)
        my_connection = Connection()
        if self.run_sim is True:
            for n in self.input_layer:
                n.draw_Neuron(screen)
                my_connection.draw_Connection(screen, (10, 10), (200, 200), GREEN)
class Neuron(object):
    def __init__(self, x_pos, y_pos):
        self.x = x_pos
        self.y = y_pos
    def draw_Neuron(self, screen):
        pygame.draw.circle(screen, (12,22,100), (self.x,self.y), NEURON_SIZE, 10)
class Connection(object):
    def __init__(self):
        pass
    def draw_Connection(self, screen, (x1,y1), (x2,y2), color=GREEN):
        pygame.draw.line(screen, (color), (x1,y1), (x2, y2))

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
