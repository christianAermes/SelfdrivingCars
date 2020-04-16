import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from skimage import io
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize, skeletonize_3d


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

def dist(p1, p2):
	p1 = np.asarray(p1)
	p2 = np.asarray(p2)
	return np.sqrt(sum((p1-p2)**2))

def generateTrack(screen, imgFile="screenshot.jpg", dcp=20):
	"""
	
	"""
	print("Building a track...")
	pygame.image.save(screen, imgFile)
	# imgFile = "circle.jpg"
	# imgFile = "circle2.jpg"
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
		rel = float(n)/N
		checkpoints.append(Checkpoint(points[i], n, rel))
	
	
	track = pygame.image.load(imgFile)
	return checkpoints, track