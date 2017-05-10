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

class Sim(object):
    def __init__(self, screen):
        self.run_sim = True
        self.input_layer = []
        # self.initialize_display(screen)
        cartesian_cords = [[236, 251],[495,202],[754, 254]]
        screen_cords = []
        for i in range(len(cartesian_cords)):
            screenX = cartesian_cords[i][0] + SCREEN_WIDTH/2
            screenY = SCREEN_HEIGHT/2 -cartesian_cords[i][1]
            screen_cords.append([screenX, screenY])

        self.BackGround = Background('nn.png', [0,0])
        self.Blob = Blob(cartesian_cords)
    def initialize_display(self, screen):
        for i in range(3):
            self.input_layer.append(Neuron(SCREEN_WIDTH/2, (i+1)*100))
    def process_events(self, screen):
        if self.run_sim is True:
            self.display_frame(screen)
            self.Blob.move()
            for event in pygame.event.get():
                if event.type is pygame.KEYDOWN:
                    if event.key is pygame.K_x:
                        # pygame.display.quit()
                        pygame.quit()
        pygame.display.flip()
    def display_frame(self, screen):
        screen.fill([255, 255, 255])
        screen.blit(self.BackGround.image, self.BackGround.rect)
        my_connection = Connection()
        if self.run_sim is True:
            for n in self.input_layer:
                n.draw_Neuron(screen)
                my_connection.draw_Connection(screen, (10, 10), (200, 200), GREEN)
            self.Blob.draw(screen)
class Blob(object):
    def __init__(self, cordinates):
        # cordinates : [[0,0],[100,100],[200, 100]]
        self.cordinates = cordinates
        self.current_pos = cordinates[0]
        self.next_pos = 1
    def move(self):

        delta_x = (self.cordinates[self.next_pos][0]-self.current_pos[0])
        delta_y = (self.cordinates[self.next_pos][1]-self.current_pos[1])
        displacement_vector = np.array([delta_x, delta_y])
        unit_vector = displacement_vector/np.sqrt(np.sum(displacement_vector**2))*10
        # print("unit vecteor: ", unit_vector)
        #unit_vector = np.rint(unit_vector)
        #unit_vector = np.ceil(unit_vector)
        #dx = int(unit_vector[0] + (0.5 if unit_vector[0] > 0 else -0.5))
        #dy = int(unit_vector[1] + (0.5 if unit_vector[1] > 0 else -0.5))i
        dx = int(round(unit_vector[0]))
        dy = int(round(unit_vector[1]))
        

        # dx = int(unit_vector[0])
        # dy = int(unit_vector[1])
        #print("dy: ", dy)
        self.current_pos[0] += dx
        self.current_pos[1] += dy
        # print(self.next_pos)
        #print("current pos: ", self.current_pos)
        #print(abs(self.current_pos[1]-self.cordinates[self.next_pos][1]))
        print("next: ", self.next_pos)
        if abs(self.current_pos[0]-self.cordinates[self.next_pos][0]) < 10:
            if abs(self.current_pos[1]-self.cordinates[self.next_pos][1]) < 10:
                if self.next_pos is 2:
                    self.next_pos = 1
                    self.current_pos = self.cordinates[0]
                else:
                    #self.current_pos = self.cordinates[self.next_pos]
                    self.next_pos += 1

               # print("next: ", self.next_pos)
                # if self.next_pos is 3:
            #    print("end")
                  #  print("next is 3,:", self.next_pos)
                    # self.current_pos = [0, 0] #self.cordinates[0]
                    # self.next_pos = 1

        
    def draw(self,screen):
        pygame.draw.circle(screen, (12,22,122), self.current_pos, 5, 5)

class Background(pygame.sprite.Sprite):
    def __init__(self, image_file, location):
        pygame.sprite.Sprite.__init__(self)  #call Sprite initializer
        self.image = pygame.image.load(image_file)
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location

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
    simulation = Sim(screen)
    while simulation.run_sim is True:
        simulation.process_events(screen)
        simulation.display_frame(screen)
        clock.tick(FPS)
    pygame.quit()

main()
