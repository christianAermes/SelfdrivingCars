import numpy as np
from Car import Car
from constants import N_Sensors, PopulationSize, mutationRate, changeRate, keepBestCar
from random import shuffle

def geneIdx(parents):
    fitness0 = parents[0].fitness
    fitness1 = parents[1].fitness

    if fitness0 > fitness1: better = 0
    else: better = 1
    
    rnd = np.random.random()

    return 0 if rnd < fitness0 else 1
    # if better==0 and rnd<fitness1: return 1
    # elif better==1 and rnd<fitness0: return 0
    # else: return np.random.randint(0,2)


def breed(parents, p0):
    mutated = False
    props = {}
    color = parents[np.random.randint(0,2)].color
    color = parents[0].color if parents[0].fitness > parents[1].fitness else parents[1].color
    p = 0 if parents[0].fitness > parents[1].fitness else 1
    color = [0,0,0]
    for i in range(len(color)):
        color[i] = parents[p].color[i]
    
    n_steering = len(parents[0].STEERING_OPTIONS)
    n_velocity = len(parents[0].VELOCITY_OPTIONS)

    # change = (-1)**np.random.randint(0,2) * 0.05 * np.random.random()
    # changeRate = 0.05

    startOrientation = 0
    # startOrientation += parents[np.random.randint(0,2)].startOrientation
    startOrientation += parents[geneIdx(parents)].startOrientation
    if np.random.random() < mutationRate:
        startOrientation *= np.random.uniform(.9, 1.1)
        mutated = True

    fov = 0
    # fov += parents[np.random.randint(0,2)].fov
    fov += parents[geneIdx(parents)].fov
    if np.random.random() < mutationRate:
        fov *= np.random.uniform(.9, 1.1)
        mutated = True

    vStart = 0
    # vStart += parents[np.random.randint(0,2)].vStart
    vStart += parents[geneIdx(parents)].vStart
    if np.random.random() < mutationRate:
        vStart *= np.random.uniform(.9, 1.1)
        mutated = True

    maxSteering = 0
    # maxSteering += parents[np.random.randint(0,2)].maxSteering
    maxSteering += parents[geneIdx(parents)].maxSteering
    if np.random.random() < mutationRate:
        maxSteering *= np.random.uniform(.9, 1.1)
        mutated = True

    weightsS = np.zeros((n_steering, N_Sensors+1))
    for i in range(len(weightsS)):
        for j in range(len(weightsS[i])):
            # idx = np.random.randint(0,2)
            idx = geneIdx(parents)
            weightsS[i,j] = parents[idx].weightsS[i,j]
            if np.random.random() < mutationRate:
                if np.random.random()<0.5:
                    weightsS[i,j] *= np.random.uniform(.9, 1.1)
                else:
                    weightsS[i,j] += (-1)**np.random.randint(0,2) * changeRate * np.random.random()
                mutated = True

    weightsV = np.zeros((n_velocity, N_Sensors+1))
    for i in range(len(weightsV)):
        for j in range(len(weightsV[i])):
            # idx = np.random.randint(0,2)
            idx = geneIdx(parents)
            weightsV[i,j] = parents[idx].weightsV[i,j]
            if np.random.random() < mutationRate:
                if np.random.random()<0.5:
                    weightsV[i,j] *= np.random.uniform(.9, 1.1)
                else:
                    weightsV[i,j] += (-1)**np.random.randint(0,2) * changeRate * np.random.random()
                mutated = True

    biasesS = np.zeros((1,n_steering))
    for i in range(len(biasesS)):
        # idx = np.random.randint(0,2)
        idx = geneIdx(parents)
        biasesS[i] = parents[idx].biasesS[i]
        if np.random.random() < mutationRate:
            if np.random.random()<0.5:
                biasesS[i] *= np.random.uniform(.9, 1.1)
            else:
                biasesS[i] += (-1)**np.random.randint(0,2) * changeRate * np.random.random()
            mutated = True

    biasesV = np.zeros((1,n_velocity))
    for i in range(len(biasesV)):
        # idx = np.random.randint(0,2)
        idx = geneIdx(parents)
        biasesV[i] = parents[idx].biasesV[i]
        if np.random.random() < mutationRate:
            if np.random.random()<0.5:
                biasesV[i] *= np.random.uniform(.9, 1.1)
            else:
                biasesV[i] += (-1)**np.random.randint(0,2) * changeRate * np.random.random()
            mutated = True

    if mutated:
        color[np.random.randint(0,3)] = np.random.random()*255
        # color = np.random.random(3)*255

    props = { "startOrientation": startOrientation,
            "fov": fov,
            "weightsS": weightsS,
            "weightsV": weightsV,
            "biasesS":biasesS ,
            "biasesV":biasesV ,
            "color": color,
            "vStart": vStart,
            "maxSteering": maxSteering,
    }

    baby = Car(p0, **props)
    return baby


def getBestCar(cars, p0):
    maxFit = max([car.fitness for car in cars])
    for car in cars:
        if car.fitness == maxFit:
            bestCar = car
            break
    
    color = [0,0,0]
    for i in range(len(color)):
        color[i] = bestCar.color[i]

    n_steering = len(bestCar.STEERING_OPTIONS)
    n_velocity = len(bestCar.VELOCITY_OPTIONS)

    startOrientation = 0
    startOrientation += bestCar.startOrientation

    fov = 0
    fov += bestCar.fov

    vStart = 0
    vStart += bestCar.vStart

    maxSteering = 0
    maxSteering += bestCar.maxSteering

    weightsS = np.zeros((n_steering, N_Sensors+1))
    for i in range(len(weightsS)):
        for j in range(len(weightsS[i])):
            idx = np.random.randint(0,2)
            weightsS[i,j] = bestCar.weightsS[i,j]

    weightsV = np.zeros((n_velocity, N_Sensors+1))
    for i in range(len(weightsV)):
        for j in range(len(weightsV[i])):
            idx = np.random.randint(0,2)
            weightsV[i,j] = bestCar.weightsV[i,j]

    biasesS = np.zeros((1,n_steering))
    for i in range(len(biasesS)):
        idx = np.random.randint(0,2)
        biasesS[i] = bestCar.biasesS[i]

    biasesV = np.zeros((1,n_velocity))
    for i in range(len(biasesV)):
        idx = np.random.randint(0,2)
        biasesV[i] = bestCar.biasesV[i]

    props = { "startOrientation": startOrientation,
            "fov": fov,
            "weightsS": weightsS,
            "weightsV": weightsV,
            "biasesS":biasesS ,
            "biasesV":biasesV ,
            "color": color,
            "vStart": vStart,
            "maxSteering": maxSteering,
    }

    newBestCar = Car(p0, **props)
    return newBestCar


def selectParents(cars):
    parents = []
    while len(parents)<1:
        shuffle(cars)
        idx1 = np.random.randint(0,len(cars))
        p1 = cars[idx1]
        if np.random.random() < p1.fitness:
            parents.append(p1)
    while len(parents)<2:
        shuffle(cars)
        idx2 = np.random.randint(0,len(cars))
        p2 = cars[idx2]
        if np.random.random() < p2.fitness and parents[0]!=p2:
            parents.append(p2)

    return parents


def newGen(cars, checkpoints):
    nextGen = []
    
    if keepBestCar:
        firstPoint = checkpoints[-1]
        p0 = float(firstPoint.pos[0]), float(firstPoint.pos[1])
        previousBestCar = getBestCar(cars, p0)
        nextGen.append(previousBestCar)
    
    while len(nextGen) < PopulationSize:
        parents = selectParents(cars)
        firstPoint = checkpoints[-1]
        p0 = float(firstPoint.pos[0]), float(firstPoint.pos[1])
        baby = breed(parents, p0)
        # baby.mutate(mutationRate=0.01)
        nextGen.append(baby)
    return nextGen