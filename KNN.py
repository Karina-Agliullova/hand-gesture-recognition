import numpy as np
from interpolate import *
from metrics import euclidean

class KnearestNeighbors:
	def __init__(self, k=3, distance_metric=euclidean):
		# initializing values
		self.k = k
		self.distance = distance_metric
		self.data = None
	
	def train(self, X, Y):
		# will raise an error if x and y are not compatible
		if len(X) != len(Y) or type(X) != type(Y):
			raise ValueError("X and y are incompetible")
		
		# converting ndarrays to lists
		if type(X) == np.ndarray:
			X, Y = X.tolist(), Y.tolist()
		# setting data attribute containing instances and labels
		self.data = [X[i]+[Y[i]] for i in range(len(X))]
		
	def predict(self, a):
		neighbors = []
		# creating mapping from distance to instance
		distances = {self.distance(x[:-1], a): x for x in self.data}
		# collect classes of k instances with shortest distance
		for key in sorted(distances.keys())[:self.k]:
			neighbors.append(distances[key][-1])
		# returning most common vote
		return max(set(neighbors), key=neighbors.count)
