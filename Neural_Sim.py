import pygame
import time
import math
import numpy as np
import NN_Coordinates as cd
FPS = 30
SCREEN_WIDTH = 1100
SCREEN_HEIGHT = 700
SCREEN_SIZE = (SCREEN_WIDTH, SCREEN_HEIGHT)
NEURON_SIZE = 50
RED = (200, 20, 0)
GREEN = (0, 220, 10)
BLUE = (0, 0, 200)
BLACK = (0,0,0)

##########      DATA     ##################
X = np.round(np.random.rand(20,2),2)


##########  SIMULATION   ##################
class Sim(object):
    def __init__(self, screen):
        self.run_sim = True
        self.show_Weights = True
        self.clear = True
        self.f_stage = 0
        self.input_pos = 0
        self.initialize_display(screen)

        self.NN = Neural_Network()

    def initialize_display(self, sceen):
        self.small_font = pygame.font.SysFont(None, 30)
        self.BackGround = Background('nn.png', [0,0])

        self.Blobs =[Blob(cd.C_1_1_1), Blob(cd.C_1_1_2)]
        self.Blobs.extend((Blob(cd.C_1_2_1),Blob(cd.C_1_2_2)))
        self.Blobs.extend((Blob(cd.C_1_3_1),Blob(cd.C_1_3_2)))
        self.Blobs.extend((Blob(cd.C_1_4_1),Blob(cd.C_1_4_2)))
        self.Blobs.extend((Blob(cd.C_1_5_1),Blob(cd.C_1_5_2)))
        self.Blobs.extend((Blob(cd.C_2_1_1),Blob(cd.C_2_1_2)))
        self.Blobs.extend((Blob(cd.C_2_2_1),Blob(cd.C_2_2_2)))
        self.Blobs.extend((Blob(cd.C_2_3_1),Blob(cd.C_2_3_2)))
        self.Blobs.extend((Blob(cd.C_2_4_1),Blob(cd.C_2_4_2)))
        self.Blobs.extend((Blob(cd.C_2_5_1),Blob(cd.C_2_5_2)))

    def process_events(self, screen):
        if self.run_sim is True:
            self.display_frame(screen)
            for blob in enumerate(self.Blobs):
                blob[1].move()
            for event in pygame.event.get():
                if event.type is pygame.KEYDOWN:
                    if event.key is pygame.K_x:
                        # pygame.display.quit()
                        pygame.quit()
                    # if event.key is pygame.K_g:
                    #     if self.clear is True:
                    #         if self.f_stage is 0 or self.f_stage is 2:
                    #             for blob in enumerate(self.Blobs):
                    #                 blob[1].step = True
                    #                 blob[1].move()
                    #         self.f_stage += 1
                    #         self.NN.forward_step(self.f_stage, X[self.input_pos,:])
                    #         print("f_stage: ", self.f_stage)
                    #         if self.f_stage is 5:
                    #             for blob in enumerate(self.Blobs):
                    #                 blob[1].Reset = True
                    #             self.f_stage = 0
                    #             self.input_pos += 1

            if self.clear is True:

                if self.f_stage is 0 or self.f_stage is 2:
                    #time.sleep(1)
                    for blob in enumerate(self.Blobs):
                        blob[1].step = True
                        blob[1].move()


                self.f_stage += 1
                print("f_stage: ", self.f_stage)
                if self.f_stage is 5:
                    for blob in enumerate(self.Blobs):
                        blob[1].Reset = True
                    self.f_stage = 0
                    self.input_pos += 1

            self.clear = True
            for blob in enumerate(self.Blobs):
                if blob[1].step is True:
                    self.clear = False

        self.NN.forward_step(self.f_stage, X[self.input_pos, :])

        pygame.display.flip()

    #######################################

    def display_frame(self, screen):
        screen.fill([255, 255, 255])
        screen.blit(self.BackGround.image, self.BackGround.rect)
        if self.run_sim is True:
            for blob in enumerate(self.Blobs):
                blob[1].draw(screen)

        if self.show_Weights is True:
            # draw inputs 
            screen.blit(self.small_font.render(str(X[self.input_pos,0]), True, RED), [160,232])
            screen.blit(self.small_font.render(str(X[self.input_pos,1]), True, BLUE), [160,336])
            for i in range(self.NN.input_layer_size):
                for j in range(self.NN.hidden_layer_size):
                    # draw input weights
                    screen.blit(self.small_font.render(str(round(self.NN.W1[0,j], 2)), True, RED), cd.W1_C_1[j])
                    screen.blit(self.small_font.render(str(round(self.NN.W1[1,j], 2)), True, BLUE), cd.W1_C_2[j])
                    screen.blit(self.small_font.render(str(round(self.NN.W2[j,0], 2)), True, RED), cd.W2_C_1[j])
                    screen.blit(self.small_font.render(str(round(self.NN.W2[j,1], 2)), True, BLUE), cd.W2_C_2[j])

            # Show Hidden Layer Values
            for i in range(len(self.NN.z2)):
                screen.blit(self.small_font.render(str(np.round(self.NN.z2[i], 2)), True, BLACK), cd.HIDDEN_C[i])

            for i in range(len(self.NN.z3)):
                screen.blit(self.small_font.render(str(np.round(self.NN.z3[i], 2)), True, BLACK), cd.OUT_C[i])

##############################################################################
##############################################################################


##############################################################################
##############################################################################

class Blob(object):
    def __init__(self, cordinates, color=None):
        self.Reset = False
        if color is None:
            self.color = GREEN
        else:
            self.color = color
        # cordinates : [[0,0],[100,100],[200, 100]]
        self.cordinates = list(cordinates)
        self.current_pos = list(self.cordinates[0])
        self.next_pos = 1
        self.step = False
        # step means I am nid cordinates waiting for a move stage

    def move(self):
        if self.step is True:
            delta_x = self.cordinates[self.next_pos][0]-self.current_pos[0]
            delta_y = self.cordinates[self.next_pos][1]-self.current_pos[1]
            displacement_vector = np.array([delta_x, delta_y])
            unit_vector = displacement_vector/np.sqrt(np.sum(displacement_vector**2))*10
            dx = int(round(unit_vector[0]))
            dy = int(round(unit_vector[1]))


            self.current_pos[0] += dx*3
            self.current_pos[1] += dy*3
        if abs(self.current_pos[0]-self.cordinates[self.next_pos][0]) < 15:
            if abs(self.current_pos[1]-self.cordinates[self.next_pos][1]) < 15:
                self.step = False
                if self.next_pos is len(self.cordinates)-1:
                    # STOP
                    if self.Reset is True:
                    
                        self.current_pos = list(self.cordinates[0])
                        self.next_pos = 1
                        self.Reset = False # I have been reset
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








##############################################################################
##############################################################################
##############################################################################


##############################################################################
class Neural_Network(object):
    def __init__(self):
        self.input_layer_size = 2
        self.hidden_layer_size = 5
        self.output_layer_size = 2

        self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size, self.output_layer_size)
        self.z2 = np.zeros(5)
        self.z3 = np.zeros(2)

        self.input_number = 0
    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat
    def forward_step(self, stage, X):
        if stage is 1:
            print("stage 1 dot input w/ weight 1")
            self.z2 = np.dot(X, self.W1)
            print(self.z2)
            return self.z2
        if stage is 2:
            # let z2 be our a2
            self.z2 = self.sigmoid(self.z2)
        if stage is 3:
            # z2 has had the activation function applied thus z2 = a2
            self.z3 = np.dot(self.z2, self.W2)
        if stage is 4:
            # let z3 represent our y hat these substitutions are done to make the simulation easier
            self.z3 = self.sigmoid(self.z3)


    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

##############################################################################
##############################################################################
##############################################################################





##############################################################################
##############################################################################

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
    X = np.array([[1,2],[2,4],[3,3]])
    NN = Neural_Network()
    yhat = NN.forward(X)
    print(yhat)
    print(NN.W1)

main()
