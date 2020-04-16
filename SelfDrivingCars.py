import pygame, sys
import numpy as np
from skimage import io
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize, skeletonize_3d
import matplotlib.pyplot as plt
# np.random.seed(42)
np.random.seed(156865)
WIDTH, HEIGHT = 400, 400
BLACK = pygame.Color(0,0,0)
WHITE = pygame.Color(255,255,255)
TRACKWIDTH = 20

N_Sensors = 5

STEERING_OPTIONS = [
					"", 
					"LEFT", 
					"RIGHT",
					]
VELOCITY_OPTIONS = [
					"", 
					"ACC", 
					"BREAK",
					]
# N_CHECKPOINTS = 30
T_MAX = 1200 

PopulationSize = 15

pygame.init()
mouse = pygame.mouse
running = True

window = pygame.display.set_mode((WIDTH, HEIGHT))
canvas = window.copy() # for drawing of the track
window.fill(BLACK)
canvas.fill(BLACK)

drawing =  True #
### Car class:
class Car(object):
	"""docstring for Car"""
	def __init__(self, pos, color=None, startOrientation=None, fov=None, vStart=None, maxSteering=None, weightsS=None, biasesS=None, weightsV=None, biasesV=None):
		super(Car, self).__init__()
		self.pos = pos
		self.size = 8

		self.color = 255*np.random.random(3) 						if color is None else color
		self.startOrientation = np.random.random()*np.pi*2 			if startOrientation is None else startOrientation
		# self.fov = np.pi/180 * 75 * np.random.random()				if fov is None else fov
		self.fov = np.pi/180 * 175 * np.random.random()				if fov is None else fov

		
		self.weightsV = np.random.normal(size=(len(VELOCITY_OPTIONS),N_Sensors+1))	if weightsV is None else weightsV
		self.biasesV = np.random.normal(size=(1,len(VELOCITY_OPTIONS)))	if biasesV is None else biasesV
		self.weightsS = np.random.normal(size=(len(STEERING_OPTIONS),N_Sensors+1))	if weightsS is None else weightsS
		self.biasesS = np.random.normal(size=(1,len(STEERING_OPTIONS)))	if biasesS is None else biasesS

		self.vStart = 0.25 + 0.5*np.random.random()					if vStart is None else vStart
		self.maxSteering = (1 + 2*np.random.random())*np.pi/180 	if maxSteering is None else maxSteering

		self.v = self.vStart
		self.heading = self.startOrientation
		# self.sensors = [[np.cos(self.heading + self.fov/2 *(i-1)), np.sin(self.heading + self.fov/2 *(i-1))] for i in range(3)]
		self.sensors = []
		for i in range(N_Sensors):
			dA = (i-N_Sensors/2)*self.fov/(N_Sensors-1)
			dx = np.cos(self.heading + dA)
			dy = np.sin(self.heading + dA)
			self.sensors.append([dx, dy])
		# self.contactPoints = np.asarray([(0,0) for i in range(3)])
		self.contactPoints = np.asarray([(0,0) for i in range(N_Sensors)])
		self.dead = False
		self.fitness = 0

		self.acc = 0.1
		self.vel = self.v*np.asarray([np.cos(self.heading), np.sin(self.heading)])

		# print self.weightsS
		# print "\n\n\n"

	def show(self, screen):
		# x2 = self.pos[0] + 100*np.cos(self.heading)
		# y2 = self.pos[1] + 100*np.sin(self.heading)
		# pygame.draw.line(screen, self.color, (int(self.pos[0]), int(self.pos[1])), (int(x2), int(y2)))

		pygame.draw.circle(screen, self.color, (int(self.pos[0]), int(self.pos[1])), self.size/2)
		if not self.dead:
			for point in self.contactPoints:
				pygame.draw.line(screen, self.color, (int(self.pos[0]), int(self.pos[1])), (int(point[0]), int(point[1])))

	def see(self, pxarray, borderColor=(0,0,0)):
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

	def steer(self, d):
		# STEERING_OPTIONS = ["RIGHT", "", "LEFT", "ACC", "BREAK"]
		###
		# d = [di if di <= TRACKWIDTH*3 else np.inf for di in d]
		###
		d = np.asarray(d)
		# d = 1/d
		inputs = [di for di in d]
		inputs.append(self.v)
		inputs = np.asarray(inputs)
		
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


		moveTo = STEERING_OPTIONS[indexS]
		if moveTo == "LEFT":
			self.heading += self.maxSteering
		elif moveTo == "RIGHT":
			self.heading -= self.maxSteering
		elif moveTo == "":
			pass

		changeSpeed = VELOCITY_OPTIONS[indexV]
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


### Checkpoint class:
class Checkpoint(object):
	"""docstring for Checkpoint"""
	def __init__(self, pos, n, rel):
		super(Checkpoint, self).__init__()
		self.pos = np.asarray(pos)
		self.id = n
		self.rel = rel
	def show(self, screen):
		red = int(self.rel*255)
		pygame.draw.circle(screen, (red,0,0), self.pos, 2)

###
def dist(p1, p2):
	p1 = np.asarray(p1)
	p2 = np.asarray(p2)
	return np.sqrt(sum((p1-p2)**2))

def generateTrack(screen, imgFile="screenshot.jpg", dcp=20):
	"""
	
	"""
	print "Building a track..."
	pygame.image.save(screen, imgFile)
	# imgFile = "screenshotTrack.jpg"
	# imgFile = "newTrack.jpg"
	# imgFile = "whyUnoWorking.jpg"
	imgFile = "screenshot - Kopie.jpg"
	img = io.imread(imgFile, as_gray=True)
	thresh = threshold_otsu(img)
	binary = img > thresh
	skeleton = skeletonize(binary)
	# skeleton = skeletonize_3d(binary)
	
	centerpoints = []
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if skeleton[i,j]:
				centerpoints.append((j,i))

	points = [centerpoints[0]]
	old = len(points)
	new = len(points)+1

	while old != new:
	    old = len(points)
	    last = points[-1]
	    for p in centerpoints:
	        d = round(dist(last, p))
	        if d-1<=dcp<=d+1 and all(dist(p, cp)>=dcp for cp in points):
	            points.append(p)
	            break
	    new = len(points)
	for p in centerpoints:
	    d = round(dist(last, p))
	    if d-1<=dcp<=d+1 and all(dist(p, cp)>=dcp for cp in points[1:]):
	        points.append(p)
	        break
	checkpoints = []
	N = len(points)
	for i in range(N):
		n = N-1-i
		# n = i
		rel = float(n)/N
		checkpoints.append(Checkpoint(points[i], n, rel))
	
	
	track = pygame.image.load(imgFile)
	return checkpoints, track

### Genetics
def breed(parents, p0, mutationRate=0.01):
	mutated = False
	props = {}
	color = parents[np.random.randint(0,2)].color
	color = parents[0].color if parents[0].fitness > parents[1].fitness else parents[1].color
	p = 0 if parents[0].fitness > parents[1].fitness else 1
	color = [0,0,0]
	for i in range(len(color)):
		color[i] = parents[p].color[i]
	# if np.random.random() < mutationRate:
	# 	color = np.random.random(3)*255
	# 	mutated = True
	
	startOrientation = 0
	startOrientation += parents[np.random.randint(0,2)].startOrientation
	if np.random.random() < mutationRate:
		startOrientation *= np.random.uniform(.9, 1.1)
		mutated = True

	fov = 0
	fov += parents[np.random.randint(0,2)].fov
	if np.random.random() < mutationRate:
		fov *= np.random.uniform(.9, 1.1)
		mutated = True

	vStart = 0
	vStart += parents[np.random.randint(0,2)].vStart
	if np.random.random() < mutationRate:
		vStart *= np.random.uniform(.9, 1.1)
		mutated = True

	maxSteering = 0
	maxSteering += parents[np.random.randint(0,2)].maxSteering
	if np.random.random() < mutationRate:
		maxSteering *= np.random.uniform(.9, 1.1)
		mutated = True

	weightsS = np.zeros((len(STEERING_OPTIONS), N_Sensors+1))
	for i in range(len(weightsS)):
		for j in range(len(weightsS[i])):
			idx = np.random.randint(0,2)
			weightsS[i,j] = parents[idx].weightsS[i,j]
			if np.random.random() < mutationRate:
				weightsS[i,j] *= np.random.uniform(.9, 1.1)
				mutated = True

	weightsV = np.zeros((len(STEERING_OPTIONS), N_Sensors+1))
	for i in range(len(weightsV)):
		for j in range(len(weightsV[i])):
			idx = np.random.randint(0,2)
			weightsV[i,j] = parents[idx].weightsV[i,j]
			if np.random.random() < mutationRate:
				weightsV[i,j] *= np.random.uniform(.9, 1.1)
				mutated = True

	biasesS = np.zeros((1,len(STEERING_OPTIONS)))
	for i in range(len(biasesS)):
		idx = np.random.randint(0,2)
		biasesS[i] = parents[idx].biasesS[i]
		if np.random.random() < mutationRate:
			biasesS[i] *= np.random.uniform(.9, 1.1)
			mutated = True

	biasesV = np.zeros((1,len(STEERING_OPTIONS)))
	for i in range(len(biasesV)):
		idx = np.random.randint(0,2)
		biasesV[i] = parents[idx].biasesV[i]
		if np.random.random() < mutationRate:
			biasesV[i] *= np.random.uniform(.9, 1.1)
			mutated = True

	if mutated:
		color = np.random.random(3)*255

	props = { "startOrientation": startOrientation,
			"fov": fov,
			# "weights": weights,
			"weightsS": weightsS,
			"weightsV": weightsV,
			# "biases":biases ,
			"biasesS":biasesS ,
			"biasesV":biasesV ,
			"color": color,
			"vStart": vStart,
			"maxSteering": maxSteering,
	}
	
	baby = Car(p0, **props)
	return baby


def selectParents(cars):
	parents = []
	while len(parents)<1:
		idx1 = np.random.randint(0,len(cars))
		p1 = cars[idx1]
		if np.random.random() < p1.fitness:
			parents.append(p1)
	while len(parents)<2:
		idx2 = np.random.randint(0,len(cars))
		p2 = cars[idx2]
		if np.random.random() < p2.fitness and parents[0]!=p2:
			parents.append(p2)
	
	return parents

def newGen(cars, checkpoints):
	nextGen = []
	while len(nextGen) < PopulationSize:
		parents = selectParents(cars)
		p0 = float(checkpoints[0].pos[0]), float(checkpoints[0].pos[1])
		baby = breed(parents, p0)
		# baby.mutate(mutationRate=0.01)
		nextGen.append(baby)
	return nextGen


###


# display loop
t = 0

if not drawing: 
	checkpoints, track = generateTrack(window)
	cars = []
	for ps in range(PopulationSize):
		p0 = float(checkpoints[0].pos[0]), float(checkpoints[0].pos[1])
		cars.append(Car(p0))
while running:
	left_pressed, middle_pressed, right_pressed = mouse.get_pressed()
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			sys.exit()
		elif left_pressed:
			pygame.draw.circle(canvas, WHITE, (mouse.get_pos()), TRACKWIDTH)
		elif event.type == pygame.KEYDOWN and event.key == 13: # if enter is pressed
			if drawing: 
				checkpoints, track = generateTrack(window)
				cars = []
				for ps in range(PopulationSize):
					p0 = float(checkpoints[0].pos[0]), float(checkpoints[0].pos[1])
					cars.append(Car(p0))
			else:
				pass
			drawing = not drawing

	if drawing:
		window.blit(canvas, (0,0))
	else:
		t += 1
		window.blit(track, (0,0))
		canvas = window.copy()

		pxarray = pygame.PixelArray(canvas)
		x,y = cars[0].pos+np.asarray([200,0])
		
		for cp in checkpoints:
			cp.show(window)
			
		for car in cars:
			car.see(pxarray)
			car.move()
			car.checkFitness(checkpoints)
			car.show(window)

		# deadCars = sum([1 for car in cars if car.dead])
		if t == T_MAX or all([car.dead for car in cars]):
			t = 0
			for car in cars:
				car.fitness /= float(len(checkpoints))
			fitnesses = [car.fitness for car in cars]
			print max(fitnesses), min(fitnesses), np.mean(fitnesses)
			if max(fitnesses) > 0.1:
				cars = newGen(cars, checkpoints)
			else:
				print "Starting new..."
				cars = []
				for ps in range(PopulationSize):
					p0 = float(checkpoints[0].pos[0]), float(checkpoints[0].pos[1])
					cars.append(Car(p0))
		# maxFitness = max([car.fitness for car in cars])
		# print maxFitness


	pygame.display.update()
