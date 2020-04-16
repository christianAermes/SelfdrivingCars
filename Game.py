import sys
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import matplotlib.pyplot as plt
from constants import *
from Track import *
from Car import Car
from Genetics import *

class Game(object):
    """docstring for Car"""
    def __init__(self):
        super(Game, self).__init__()
    def start(self):
        pygame.init()
        pygame.display.set_caption('Smart Cars')
        print("Start by drawing a track for training. Press 'Enter' when you are finished.")
        self.mouse = pygame.mouse
        self.running = True

        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        self.canvas = self.window.copy() # for drawing of the track
        self.window.fill(BLACK)
        self.canvas.fill(BLACK)

        self.generation = 1
        self.generationRecord = []
        self.maxFitRecord     = []
        self.minFitRecord     = []
        self.avgFitRecord     = []
        self.bestColor        = []

        self.pxarray = []
        self.checkpoints = []
        
        self.t = 0
        
        self.states = ['drawTrainTrack', 'training', 'drawTestTrack', 'testing']
        self.stateIdx = 0
    
    def initializeCars(self):
        self.generation = 1
        self.generationRecord = []
        self.maxFitRecord     = []
        self.minFitRecord     = []
        self.avgFitRecord     = []
        self.bestColor        = []
        self.cars = []
        for ps in range(PopulationSize):
            firstPoint = self.checkpoints[-1]
            p0 = float(firstPoint.pos[0]), float(firstPoint.pos[1])
            self.cars.append(Car(p0))
        

    def run(self):
        while self.running:
            # define mouse clicks:
            left_pressed, middle_pressed, right_pressed = self.mouse.get_pressed()
            
            for event in pygame.event.get():
                # Quit the game?
                if event.type == pygame.QUIT:
                    pygame.quit()
                    print("\n\nGood bye.")
                    if len(self.generationRecord)>0: self.plotResults()
                    sys.exit()
                # check for mouse clicks:
                elif left_pressed and (self.states[self.stateIdx] == "drawTrainTrack" or self.states[self.stateIdx] == "drawTestTrack"):
                    pygame.draw.circle(self.canvas, WHITE, (self.mouse.get_pos()), TRACKWIDTH)

                
                elif event.type == pygame.KEYDOWN and event.key == 13: # if enter is pressed
                    # Finished drawing and ready to start training?
                    # if self.drawing and not self.training and not self.testing:
                    if self.states[self.stateIdx] == 'drawTrainTrack':
                        # if in the drawing state, start the training of the cars
                        # generate track and checkpoints
                        # create population of cars
                        self.stateIdx += 1
                        
                        self.checkpoints, track = generateTrack(self.window)
                        print("Start training...\n")
                        print("Performance Stats:\nGen\tMax\tMin\tAvg\tLaps")
                        self.window.blit(track, (0,0))
                        self.canvas = self.window.copy()
                        self.pxarray = pygame.PixelArray(self.canvas)
                        
                        self.initializeCars()

                    elif self.states[self.stateIdx] == 'training':
                        self.stateIdx += 1
                        print("Draw a new track. Press 'Enter' when you are finished.")
                        self.window.fill(BLACK)
                        self.canvas.fill(BLACK)
                        self.window.blit(self.canvas, (0,0))
                    
                    elif self.states[self.stateIdx] == 'drawTestTrack':
                        print('Start Testing...')
                        self.stateIdx += 1

                        self.checkpoints, track = generateTrack(self.window)
                        self.window.blit(track, (0,0))
                        self.canvas = self.window.copy()
                        self.pxarray = pygame.PixelArray(self.canvas)

                        for car in self.cars:
                            firstPoint = self.checkpoints[-1]
                            p0 = float(firstPoint.pos[0]), float(firstPoint.pos[1])
                            car.pos = p0



            if self.states[self.stateIdx] == 'drawTrainTrack':
                # draw track for cars to train on
                self.window.blit(self.canvas, (0,0))
            elif self.states[self.stateIdx] == 'drawTestTrack':
                # draw track for cars to train on
                self.window.blit(self.canvas, (0,0))

            elif self.states[self.stateIdx] == 'training':
                self.t += 1
                self.window.blit(track, (0,0))
                self.canvas = self.window.copy()

                for cp in self.checkpoints:
                    cp.show(self.window)
                for car in self.cars:
                    car.see(self.pxarray, self.canvas)
                    car.show(self.window)
                    car.checkFitness(self.checkpoints)
                    car.move()
                
                # Training time over or all cars crashed into the walls
                if self.t==T_MAX or all([car.dead for car in self.cars]):
                    self.t = 0
                    
                    # calculate indvidual fitness scores
                    maxF, minF, avgF = self.evaluate()
                    
                    # best car must complete at least 10% of the track
                    # else start over with a new rndomly generated generation of cars
                    if maxF > 0.1:
                        self.cars = newGen(self.cars, self.checkpoints)
                        self.generation += 1
                    else:
                        print("Starting new...\n")
                        print("Performance Stats:\nGen\tMax\tMin\tAvg\tLaps")
                        self.initializeCars()
                        
            elif self.states[self.stateIdx] == 'testing':
                self.window.blit(track, (0,0))
                self.canvas = self.window.copy()

                for cp in self.checkpoints:
                    cp.show(self.window)
                for car in self.cars:
                    car.see(self.pxarray, self.canvas)
                    car.show(self.window)
                    car.checkFitness(self.checkpoints)
                    car.move()

            
            pygame.display.update()
    
    def evaluate(self):
        fitnessScores = []
        maxLaps = max([car.lapCounter for car in self.cars])
        for car in self.cars:
            car.fitness += car.lapCounter*len(self.checkpoints)
            car.fitness -= 1
            car.fitness /= ((maxLaps+1)*len(self.checkpoints)-1)
            fitnessScores.append(car.fitness)
        results = '{gen}\t{max:.4f}\t{min:.4f}\t{avg:.4f}\t{laps}'.format(gen=self.generation, max=max(fitnessScores), min=min(fitnessScores), avg=np.mean(fitnessScores), laps=maxLaps)
        print(results)

        bestCar = getBestCar(self.cars, [0,0])
        
        def clamp(x): 
            return int(max(0, min(x, 255)))

        bestCarColor = "#{0:02x}{1:02x}{2:02x}".format(clamp(bestCar.color[0]), clamp(bestCar.color[1]), clamp(bestCar.color[2]))
       
        self.bestColor.append(bestCarColor)
        self.generationRecord.append(self.generation)
        self.maxFitRecord.append(max(fitnessScores))
        self.minFitRecord.append(min(fitnessScores))
        self.avgFitRecord.append(np.mean(fitnessScores))

        return max(fitnessScores), min(fitnessScores), np.mean(fitnessScores)

    def plotResults(self):
        fig, ax = plt.subplots(1,3)
        # ax.plot(self.generationRecord, self.maxFitRecord, label="Max Fitness")
        # ax.plot(self.generationRecord, self.minFitRecord, label="Min Fitness")
        # ax.plot(self.generationRecord, self.avgFitRecord, label="Avg Fitness")
        ax[0].scatter(self.generationRecord, self.maxFitRecord, c=self.bestColor)
        ax[1].scatter(self.generationRecord, self.minFitRecord, c=self.bestColor)
        ax[2].scatter(self.generationRecord, self.avgFitRecord, c=self.bestColor)

        for a in ax:
            a.set_xlabel("Generation", fontsize=16)
            a.tick_params(axis="both", labelsize=12)
            a.set_ylim(-0.05, 1.05)
        ax[0].set_ylabel("Max", fontsize=16)
        ax[1].set_ylabel("Min", fontsize=16)
        ax[2].set_ylabel("Avg", fontsize=16)
        # ax[2].set_xlabel("Generation", fontsize=16)

        fig.suptitle("Fitness Scores", fontsize=20)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()