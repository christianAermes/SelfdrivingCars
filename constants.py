import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

N_Sensors = 7
TRACKWIDTH = 20
T_MAX = 400 
PopulationSize = 50

WIDTH, HEIGHT = 400, 400
BLACK = pygame.Color(0,0,0)
WHITE = pygame.Color(255,255,255)

mutationRate = 0.1 #0.05
changeRate = 0.1 #0.1

maxFOV = 210
keepBestCar = False