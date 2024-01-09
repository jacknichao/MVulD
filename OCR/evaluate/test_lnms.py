import numpy as np
from shapely.geometry import Polygon


def intersection(g, p):
	# Take the geometry information in g and p to form a polygon
	g = Polygon(g[:8].reshape((4, 2)))
	p = Polygon(p[:8].reshape((4, 2)))

	# whether g and p are valid
	if not g.is_valid or not p.is_valid:
		return 0

	# Take the intersection and union of two geometry
	inter = Polygon(g).intersection(Polygon(p)).area
	union = g.area + p.area - inter
	if union == 0:
		return 0
	else:
		return inter / union


def weighted_merge(g, p):
	# Take the weight of g and p geometry (the weight is calculated according to the corresponding detection score)
	g[:8] = (g[8] * g[:8] + p[8] * p[:8]) / (g[8] + p[8])

	# The score of the combined geometry is the sum of the scores of the two geometries
	g[8] = (g[8] + p[8])
	return g


def standard_nms(S, thres):
	#  standard NMS
	order = np.argsort(S[:, 8])[::-1]
	keep = []
	while order.size > 0:
		i = order[0]
		keep.append(i)
		ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])
		inds = np.where(ovr <= thres)[0]
		order = order[inds + 1]

	return S[keep]


def nms_locality(polys, thres=0.3):
	'''
	locality aware nms of EAST
	:param polys: a N*9 numpy array. first 8 coordinates, then prob # 感觉像是输出的
	:return: boxes after nms
	'''
	S = []  
	p = None  
	for g in polys:
		if p is not None and intersection(g, p) > thres: 
			p = weighted_merge(g, p)
		else:  
			if p is not None:
				S.append(p)
			p = g
	if p is not None:
		S.append(p)
	if len(S) == 0:
		return np.array([])
	return standard_nms(np.array(S), thres)


if __name__ == '__main__':
	# 343,350,448,135,474,143,369,359
	print(Polygon(np.array([[343, 350], [448, 135],
							[474, 143], [369, 359]])).area)