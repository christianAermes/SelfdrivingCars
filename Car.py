import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from constants import *
class Car(object):
    """docstring for Car"""
    def __init__(self, pos, color=None, startOrientation=None, fov=None, vStart=None, maxSteering=None, weightsS=None, biasesS=None, weightsV=None, biasesV=None):
        super(Car, self).__init__()
        self.pos = pos
        self.size = 8
        self.STEERING_OPTIONS = [
                            "", 
                            "LEFT", 
                            "RIGHT",
                            ]
        self.VELOCITY_OPTIONS = [
                            "", 
                            "ACC", 
                            "BREAK",
                            ]

        self.color = 255*np.random.random(3) 						if color is None else color
        self.startOrientation = np.random.random()*np.pi*2 			if startOrientation is None else startOrientation
        # self.fov = np.pi/180 * 75 * np.random.random()				if fov is None else fov
        self.fov = np.pi/180 * maxFOV * np.random.random()				if fov is None else fov

        # self.weightsV = np.random.normal(size=(len(self.VELOCITY_OPTIONS),N_Sensors+1))	if weightsV is None else weightsV
        # self.biasesV = np.random.normal(size=(1,len(self.VELOCITY_OPTIONS)))	        if biasesV is None else biasesV
        # self.weightsS = np.random.normal(size=(len(self.STEERING_OPTIONS),N_Sensors+1))	if weightsS is None else weightsS
        # self.biasesS = np.random.normal(size=(1,len(self.STEERING_OPTIONS)))	        if biasesS is None else biasesS

        self.weightsV = np.random.uniform(low=-1, high=1, size=(len(self.VELOCITY_OPTIONS),N_Sensors+1))	if weightsV is None else weightsV
        self.biasesV = np.random.uniform(low=-1, high=1, size=(1,len(self.VELOCITY_OPTIONS)))	            if biasesV is None else biasesV
        self.weightsS = np.random.uniform(low=-1, high=1, size=(len(self.STEERING_OPTIONS),N_Sensors+1))	if weightsS is None else weightsS
        self.biasesS = np.random.uniform(low=-1, high=1, size=(1,len(self.STEERING_OPTIONS)))	            if biasesS is None else biasesS


        self.vStart = 0.25 + 0.5*np.random.random()					if vStart is None else vStart
        self.maxSteering = (1 + 2*np.random.random())*np.pi/180 	if maxSteering is None else maxSteering

        self.v = self.vStart
        self.heading = self.startOrientation
        self.sensors = []
        for i in range(N_Sensors):
            dA = (i-N_Sensors/2)*self.fov/(N_Sensors-1)
            dx = np.cos(self.heading + dA)
            dy = np.sin(self.heading + dA)
            self.sensors.append([dx, dy])
       
        self.contactPoints = np.asarray([(0,0) for i in range(N_Sensors)])
        self.dead = False
        self.fitness = 0
        self.lapCounter = 0

        self.acc = 0.1
        self.vel = self.v*np.asarray([np.cos(self.heading), np.sin(self.heading)])


    def show(self, screen):
        # x2 = self.pos[0] + 100*np.cos(self.heading)
        # y2 = self.pos[1] + 100*np.sin(self.heading)
        # pygame.draw.line(screen, self.color, (int(self.pos[0]), int(self.pos[1])), (int(x2), int(y2)))

        pygame.draw.circle(screen, self.color, (int(self.pos[0]), int(self.pos[1])), int(self.size/2))
        if not self.dead:
            for point in self.contactPoints:
                pygame.draw.line(screen, self.color, (int(self.pos[0]), int(self.pos[1])), (int(point[0]), int(point[1])))

    def see(self, pxarray, canvas, borderColor=(0,0,0)):
        self.contactPoints = []
        for sensor in self.sensors:
            heading = np.arctan2(sensor[1], sensor[0])
            madeContact = False
            contactX = self.pos[1]
            contactY = self.pos[0]
            counter = 0
            while madeContact == False:
                contactX += 2*np.sin(heading)
                contactY += 2*np.cos(heading)
                d = np.linalg.norm(self.pos-np.asarray([contactY, contactX]))
                # if d > TRACKWIDTH:
                # 	madeContact = True
                # else:
                try:
                    if pxarray[int(round(contactY)), int(round(contactX))] == canvas.map_rgb(borderColor):
                        madeContact = True
                except IndexError:
                    madeContact = True
            
            self.contactPoints.append((int(round(contactY)), int(round(contactX))))

    def checkFitness(self, checkpoints):
        if not self.dead:
            for checkpoint in checkpoints:
                if checkpoint.id == self.fitness:
                    r = checkpoint.pos - self.pos
                    d = np.linalg.norm(r)
                    if d < TRACKWIDTH: self.fitness += 1
                    if self.fitness == len(checkpoints):
                        self.fitness = 0
                        self.lapCounter += 1

    def steer(self, d):
        # STEERING_OPTIONS = ["RIGHT", "", "LEFT", "ACC", "BREAK"]
        ###
        # d = [di if di <= TRACKWIDTH*3 else np.inf for di in d]
        ###
        d = np.asarray(d)
        # d = 1/d

        # def sigmoid(z):
        #     return 1/(1+np.exp(-z))

        inputs = [di for di in d]
        inputs.append(self.v)
        inputs = np.asarray(inputs)
        # inputs = sigmoid(inputs)
        
        rS = np.dot(self.weightsS, inputs) + self.biasesS

        rS = rS[0]
        indexS = 0
        maxVal = -np.inf
        for i in range(len(rS)):
            if rS[i] > maxVal:
                maxVal = rS[i]
                indexS = i

        rV = np.dot(self.weightsV, inputs) + self.biasesV
        rV = rV[0]
        indexV = 0
        maxVal = -np.inf
        for i in range(len(rV)):
            if rV[i] > maxVal:
                maxVal = rV[i]
                indexV = i


        moveTo = self.STEERING_OPTIONS[indexS]
        if moveTo == "LEFT":
            self.heading += self.maxSteering
        elif moveTo == "RIGHT":
            self.heading -= self.maxSteering
        elif moveTo == "":
            pass

        changeSpeed = self.VELOCITY_OPTIONS[indexV]
        if changeSpeed == "ACC":
            self.v += self.acc
        elif changeSpeed == "BREAK":
            if self.v-self.acc > 0:
                self.v -= self.acc
        elif changeSpeed == "":
            pass

        self.vel = self.v*np.asarray([np.cos(self.heading), np.sin(self.heading)])
        self.pos += self.vel
        
        #### rotate sensors to new direction
        # self.sensors = [[np.cos(self.heading + self.fov/2 *(i-1)), np.sin(self.heading + self.fov/2 *(i-1))] for i in range(3)]
        self.sensors = []
        for i in range(N_Sensors):
            dA = (i-N_Sensors/2)*self.fov/(N_Sensors-1)
            dx = np.cos(self.heading + dA)
            dy = np.sin(self.heading + dA)
            self.sensors.append([dx, dy])


    def move(self):
        dist = []
        for cp in self.contactPoints:
            r = np.asarray(self.pos) - np.asarray(cp)
            dist.append(np.linalg.norm(r))
        if any(d<self.size*0.5 for d in dist):
            self.dead = True
        if not self.dead:
            self.steer(dist)