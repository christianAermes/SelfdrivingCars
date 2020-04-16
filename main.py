import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from Car import Car
from constants import *

from Game import Game
import numpy as np
np.random.seed(156865)

car = Car([0,0])

game = Game()

game.start()
game.run()