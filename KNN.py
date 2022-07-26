import numpy as np
from interpolate import X, Y
from metrics import euclidean

class KnearestNeighbors:
	def __init__(self, k=3, distance_metric=euclidean):
		self.k = k
		self.distance = distance_metric
		self.data = None
	
	def train(self, X, Y):
		if len(X) != len(Y) or type(X) != type(Y):
			raise ValueError("X and y are incompetible")
		
		if type(X) == np.ndarray:
			X, Y = X.tolist(), Y.tolist()
		self.data = [X[i]+[Y[i]] for i in range(len(X))]
		
	def predict(self, a):
		neighbors = []
		distances = {self.distance(x[:-1], a): x for x in self.data}
		for key in sorted(distances.keys())[:self.k]:
			neighbors.append(distances[key][-1])
		return max(set(neighbors), key=neighbors.count)
