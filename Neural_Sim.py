import pygame
import time
import math
import numpy as np
FPS = 30
SCREEN_WIDTH = 1100
SCREEN_HEIGHT = 700
SCREEN_SIZE = (SCREEN_WIDTH, SCREEN_HEIGHT)
NEURON_SIZE = 50
RED = (200, 20, 0)
GREEN = (0, 220, 10)
BLUE = (0, 0, 200)
BLACK = (0,0,0)
# yes its hard coded what did you think >.<
C_1_1_1 = [[235,255],[495,202],[754,254]]
C_1_1_2 = [[235,255],[495,202],[754,358]]
C_1_2_1 = [[235,255],[495,306],[754,358]]
C_1_2_2 = [[235,255],[495,306],[754,358]]
C_1_3_1 = [[235,255],[495,410],[754,255]]
C_1_3_2 = [[235,255],[495,410],[754,358]]
C_1_4_1 = [[235,255],[495,514],[754,254]]
C_1_4_2 = [[235,255],[495,514],[754,358]]
C_1_5_1 = [[235,255],[495,617],[754,254]]
C_1_5_2 = [[235,255],[495,617],[754,358]]

C_2_1_1 = [[235,357],[495,202],[754,254]]
C_2_1_2 = [[235,357],[495,202],[754,358]]
C_2_2_1 = [[235,357],[495,306],[754,358]]
C_2_2_2 = [[235,357],[495,306],[754,358]]
C_2_3_1 = [[235,357],[495,410],[754,254]]
C_2_3_2 = [[235,357],[495,410],[754,358]]
C_2_4_1 = [[235,357],[495,514],[754,254]]
C_2_4_2 = [[235,357],[495,514],[754,358]]
C_2_5_1 = [[235,357],[495,617],[754,254]]
C_2_5_2 = [[235,357],[495,617],[754,358]]

X = np.array([[69691,222],[2,2],[1.1,2.2]])
print("X: ", X)

W1_C_1 = [[415,190],[421,273],[420,354],[430,440],[420,511]]
W1_C_2 = [[378,251],[372,305],[364,367],[371,428],[343,479]]

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
        self.clear = True

        self.NN = Neural_Network()
        self.show_Weights = True
        self.small_font = pygame.font.SysFont(None, 30)
        



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
                    if event.key is pygame.K_g:

                        if self.clear is True:
                            for blob in enumerate(self.Blobs):
                                blob[1].step = True
                                blob[1].move()
                            self.NN.forward_W1(X[0,:])


            for blob in enumerate(self.Blobs):
                if blob[1].step is True:
                    self.clear = False
                else:
                    self.clear = True


        pygame.display.flip()
    def display_frame(self, screen):
        screen.fill([255, 255, 255])
        screen.blit(self.BackGround.image, self.BackGround.rect)
        if self.run_sim is True:
            for blob in enumerate(self.Blobs):
                blob[1].draw(screen)

        if self.show_Weights is True:
            # draw inputs 
            screen.blit(self.small_font.render(str(X[self.NN.input_number,0]), True, RED), [160,232])
            screen.blit(self.small_font.render(str(X[self.NN.input_number,1]), True, BLUE), [160,336])
            for i in range(self.NN.input_layer_size):
                for j in range(self.NN.hidden_layer_size):
                    # draw first input weights
                    screen.blit(self.small_font.render(str(round(self.NN.W1[0,j], 2)), True, RED), W1_C_1[j])
                    # draw second input weights 
                    screen.blit(self.small_font.render(str(round(self.NN.W1[1,j], 2)), True, BLUE), W1_C_2[j])
           # draw outputs ** INSERT TRUE OUTPUTS
            screen.blit(self.small_font.render(str(X[self.NN.input_number,0]), True, RED), [789,232])
            screen.blit(self.small_font.render(str(X[self.NN.input_number,1]), True, BLUE), [791,3364])
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
        self.step = False

    def move(self):

        delta_x = self.cordinates[self.next_pos][0]-self.current_pos[0]
        delta_y = self.cordinates[self.next_pos][1]-self.current_pos[1]
        displacement_vector = np.array([delta_x, delta_y])
        unit_vector = displacement_vector/np.sqrt(np.sum(displacement_vector**2))*10
        dx = int(round(unit_vector[0]))
        dy = int(round(unit_vector[1]))

        if self.step is True:

            self.current_pos[0] += dx*2
            self.current_pos[1] += dy*2
        if abs(self.current_pos[0]-self.cordinates[self.next_pos][0]) < 15:
            if abs(self.current_pos[1]-self.cordinates[self.next_pos][1]) < 15:
                self.step = False
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

        self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size, self.output_layer_size)

        self.input_number = 0
    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat
    def forward_W1(self, X):
        self.z2 = np.dot(X, self.W1)
        return self.z2

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

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
