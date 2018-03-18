#############################  SIMULATION  ################################
###########################################################################
import pygame
import numpy as np

FPS = 30
SCREEN_SIZE = (1000, 800)
GREEN = (0, 205, 102)
RED = (220, 20, 60)
BLUE = (0, 245, 255)
PURPLE = (224, 102, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GOLD = (255, 215, 0)
GRAY = (110, 110, 110)


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
        pygame.display.flip()

    def display_frame(self, screen, sizes, weights):
        screen.fill(GRAY)
        self.draw_neurons(screen, sizes, weights)

    def draw_neurons(self, screen, sizes, weights):
        initial_Y = 350
        buffer = 90
        x_buffer = 150
        coordinates = []
        column = []
        for x in range(len(sizes)):
            if sizes[x] % 2 is 0:
                offset = -(sizes[x]) // 2 * buffer
            else:
                offset = -(sizes[x] - 1) // 2 * buffer - buffer // 2
            for y in range(sizes[x]):

                x_pos = (x + 1) * x_buffer
                y_pos = (y + 1) * buffer + initial_Y + offset
                if x is 0:
                    color = BLUE
                elif x is len(sizes) - 1:
                    color = GOLD
                else:
                    color = PURPLE
                pygame.draw.circle(screen, color, (x_pos, y_pos), buffer // 3, buffer // 3)
                column.append((x_pos, y_pos))
            coordinates.append(list(column))
            column.clear()

        for i in range(len(sizes) - 1):
            current_weights = weights[i]
            for j in range(len(coordinates[i])):
                for k in range(len(coordinates[i + 1])):
                    x_final = coordinates[i + 1][k][0]
                    y_final = coordinates[i + 1][k][1]
                    thickness = int(current_weights[j][k] / 5)
                    if thickness is 0:
                        thickness = 1
                    if thickness < 0:
                        thickness = abs(thickness)
                        color = RED
                    else:
                        color = GREEN
                    pygame.draw.line(screen, color, coordinates[i][j], (x_final, y_final), thickness)


def train_with_sim(nn, X, y, epooch):
    import time
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption('~~wow~~')
    simulation = Sim()
    time.sleep(20)

    for i in range(epooch):
        predictions = nn.forward(X)
        error = y - predictions
        nn.back_prop(error, X)
        if (i % 10000) == 0:
            print("Training Accuracy: ", (100 * (1 - np.round(np.mean(np.abs(error)), 4))), " %")
            simulation.process_events(screen)
            simulation.display_frame(screen, nn.all_Layers, nn.weights)
    pygame.quit()
