import pygame
import time
import math
import numpy as np
FPS = 60
SCREEN_WIDTH = 1100
SCREEN_HEIGHT = 700
SCREEN_SIZE = (SCREEN_WIDTH, SCREEN_HEIGHT)
NEURON_SIZE = 50
RED = (200, 20, 0)
GREEN = (0, 220, 10)

# yes its hard coded what did you think >.<
C_1_1_1 = [[236,251],[495,202],[754,254]]
C_1_1_2 = [[235,254],[496,203],[751,356]]
C_1_2_1 = [[235,255],[496,304],[754,358]]
C_1_2_2 = [[237,254],[494,306],[750,359]]
C_1_3_1 = [[236,256],[495,411],[753,255]]
C_1_3_2 = [[234,256],[495,410],[752,358]]
C_1_4_1 = [[236,253],[495,514],[754,254]]
C_1_4_2 = [[234,255],[493,513],[754,357]]
C_1_5_1 = [[235,252],[496,617],[753,254]]
C_1_5_2 = [[234,255],[493,617],[754,357]]

C_2_1_1 = [[236,357],[495,202],[754,254]]
C_2_1_2 = [[235,357],[496,203],[751,356]]
C_2_2_1 = [[235,357],[496,304],[754,358]]
C_2_2_2 = [[237,357],[494,306],[750,359]]
C_2_3_1 = [[236,357],[495,411],[753,255]]
C_2_3_2 = [[234,357],[495,410],[752,358]]
C_2_4_1 = [[236,357],[495,514],[754,254]]
C_2_4_2 = [[234,357],[493,513],[754,357]]
C_2_5_1 = [[235,357],[496,617],[753,254]]
C_2_5_2 = [[234,357],[493,617],[754,357]]


class Sim(object):
    def __init__(self, screen):
        self.run_sim = True
        self.input_layer = []
        # self.initialize_display(screen)

        self.BackGround = Background('nn.png', [0,0])
        self.Blobs =[Blob(C_1_1_1), Blob(C_1_1_2)]
        self.Blobs.extend((Blob(C_1_2_1),Blob(C_1_2_2)))
        self.Blobs.extend((Blob(C_1_3_1),Blob(C_1_3_2)))
        self.Blobs.extend((Blob(C_1_4_1),Blob(C_1_4_2)))
        self.Blobs.extend((Blob(C_1_5_1),Blob(C_1_5_2)))
        self.Blobs.extend((Blob(C_2_1_1),Blob(C_2_1_2)))
        self.Blobs.extend((Blob(C_2_2_1),Blob(C_2_2_2)))
        self.Blobs.extend((Blob(C_2_3_1),Blob(C_2_3_2)))
        self.Blobs.extend((Blob(C_2_4_1),Blob(C_2_4_2)))
        self.Blobs.extend((Blob(C_2_5_1),Blob(C_2_5_2)))
    def process_events(self, screen):
        if self.run_sim is True:

            for blob in enumerate(self.Blobs):
                blob[1].move()

            self.display_frame(screen)
            for event in pygame.event.get():
                if event.type is pygame.KEYDOWN:
                    if event.key is pygame.K_x:
                        # pygame.display.quit()
                        pygame.quit()
        pygame.display.flip()
    def display_frame(self, screen):
        screen.fill([255, 255, 255])
        screen.blit(self.BackGround.image, self.BackGround.rect)
        if self.run_sim is True:
            for blob in enumerate(self.Blobs):
                blob[1].draw(screen)
class Blob(object):
    def __init__(self, cordinates, color=None):
        if color is None:
            self.color = GREEN
        else:
            self.color = color

        # cordinates : [[0,0],[100,100],[200, 100]]
        self.cordinates = list(cordinates)
        self.current_pos = list(self.cordinates[0])
        self.next_pos = 1

    def move(self):

        delta_x = self.cordinates[self.next_pos][0]-self.current_pos[0]
        delta_y = self.cordinates[self.next_pos][1]-self.current_pos[1]
        displacement_vector = np.array([delta_x, delta_y])
        unit_vector = displacement_vector/np.sqrt(np.sum(displacement_vector**2))*10
        dx = int(round(unit_vector[0]))
        dy = int(round(unit_vector[1]))

        self.current_pos[0] += dx
        self.current_pos[1] += dy
        if abs(self.current_pos[0]-self.cordinates[self.next_pos][0]) < 10:
            if abs(self.current_pos[1]-self.cordinates[self.next_pos][1]) < 10:
                if self.next_pos is len(self.cordinates)-1:
                    self.current_pos = list(self.cordinates[0])
                    self.next_pos = 1
                else:
                    self.next_pos += 1
    def draw(self,screen):
        if self.next_pos > 1:
            self.color = RED
        else:
            self.color = GREEN
        pygame.draw.circle(screen, self.color, self.current_pos, 5, 5)

class Background(pygame.sprite.Sprite):
    def __init__(self, image_file, location):
        pygame.sprite.Sprite.__init__(self)  #call Sprite initializer
        self.image = pygame.image.load(image_file)
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location

class Neural_Network(object):
    def __init__(self):
        self.input_layer_size = 2
        self.hidden_layer_size = 5
        self.output_layer_size = 2
    def sigmoid(self):
        pass

def main():
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption('~~Artificial Neural Network Visualization~~')
    clock = pygame.time.Clock()
    simulation = Sim(screen)
    while simulation.run_sim is True:
        simulation.process_events(screen)
        simulation.display_frame(screen)
        clock.tick(FPS)
    pygame.quit()

main()

