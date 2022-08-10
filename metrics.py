from numpy.linalg import norm

def euclidean(a, b):
	# returning the euclidean distance between a and b
	return norm(a-b)
