import skimage.filters as sf
from skimage.io import imsave, imread
from skimage.feature import canny
from scipy.ndimage.filters import sobel, gaussian_filter
from scipy.sparse import dok_matrix, csr_matrix
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from scipy.sparse.csgraph import connected_components
from skimage.measure import regionprops

eps = 10e-6 #for division of zero stuff
MAX_ANGLE = np.pi/4

#REMINDER: dy = rows, first in numpy indexing, dx = columns, second in numpy indexing

def plot_img(image):
	plt.imshow(image, cmap=plt.cm.gray); plt.show()

def plot_regions(regions, image):
	fig, ax = plt.subplots(ncols=1, nrows=1)
	ax.imshow(image, cmap=plt.cm.gray)

	for region in regions:
	    minr, minc, maxr, maxc = region.bbox
	    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
	                              fill=False, edgecolor='red', linewidth=1)
	    ax.add_patch(rect)

	plt.show()

def check_edge_grad(source_grads, target_grads):
	#check if gradient is in the opposite direction
	sdy, sdx = source_grads; tdy, tdx = target_grads
	sg = np.linalg.norm(np.array(source_grads)); tg = np.linalg.norm(np.array(target_grads))
	angle = np.arccos(np.clip(np.dot(sg, tg), -1.0, 1.0)) 
	if angle - np.pi < MAX_ANGLE: #angle is pi if theyre completely opposite
		return True

	return False

def create_adjacency_graph(swt_image):
	#check pixel to the right, left, up and down. If their ratio is between some determined ratio, add as edge
	#TODO: connected components tuntuu toimivan, mutta se ottaa kaikki yksinäiset nollat omana grouppina.
	#täytyy tehdä niin että se jotenkin ignoraa ne.
	
	y, x = swt_image.shape
	edges = dok_matrix((y*x, y*x), dtype=np.int8)
	idx_to_coords = {}

	for row in range(y):
		for col in range(x):
			idx = row*x + col
			idx_to_coords[idx] = (row, col)
			if swt_image[row, col] == np.inf:
				continue

			if col+1 < x and swt_image[row,col+1] != np.inf:
				ratio = swt_image[row,col] / (swt_image[row,col+1] + eps)
				if ratio > 0.33 and ratio < 3.00:
					edgeidx = row*x + col + 1
					edges[idx, edgeidx] = 1

			if col-1 > 0 and swt_image[row,col-1] != np.inf:
				ratio = swt_image[row,col] / (swt_image[row,col-1] + eps)
				if ratio > 0.33 and ratio < 3.00:
					edgeidx = row*x + col - 1
					edges[idx, edgeidx] = 1

			if row+1 < y and swt_image[row+1,col] != np.inf:
				ratio = swt_image[row,col] / (swt_image[row+1,col] + eps)
				if ratio > 0.33 and ratio < 3.00:
					edgeidx = (row+1)*x + col
					edges[idx, edgeidx] = 1

			if row-1 > 0 and swt_image[row-1,col] != np.inf:
				ratio = swt_image[row,col] / (swt_image[row-1,col] + eps)
				if ratio > 0.33 and ratio < 3.00:
					edgeidx = (row-1)*x + col
					edges[idx, edgeidx] = 1

	return edges

def get_swts(edges, dxs, dys):
	posy, posx = np.where(edges)
	edgeset = set(zip(posy, posx)) #for checking if weve hit another edgepixel in the line
	stroke_widths = np.zeros_like(edges).astype(np.float64)
	stroke_widths[:,:] = np.inf
	maxy,maxx = edges.shape
	nondiscarded_rays = []

	for i in range(len(posx)): #go through all the edge pixels
		this_x = posx[i]; this_y = posy[i]
		dy = dys[this_y, this_x]; dx = dxs[this_y, this_x]
		source_grad = (dy, dx)
		xs = [this_x]; ys = [this_y]

		for step in range(1, 15):
			incrx = int(dx*step); incry = int(dy*step)
			newx = this_x + incrx; newy = this_y + incry
			if newx >= maxx or newy >= maxy or newy < 0 or newx < 0: break

			xs.append(newx); ys.append(newy) #do we include edge pixels or not?
			if (newy, newx) in edgeset and len(xs) > 0: #we hit another edge
				target_grad = (dys[newy, newx], dxs[newy, newx])
				if check_edge_grad(source_grad, target_grad):
					assign_pixels = (np.array(ys), np.array(xs))
					nondiscarded_rays.append(assign_pixels)
					width = np.hypot(newy - this_y, newx - this_x)
					stroke_widths[assign_pixels] = np.minimum(stroke_widths[assign_pixels], width)

				break

			#xs.append(newx); ys.append(newy) #need these to possibly assign new values to these pixels (ray pixels)

	for ray in nondiscarded_rays:
		median = np.median(stroke_widths[ray])
		stroke_widths[ray] = np.minimum(stroke_widths[ray], median)


	return stroke_widths

def get_and_filter_regions(labels, swts):

	#first reject ccs with too high SWT variance
	var_threshold = 5.0
	
	for label in uniq[uniq_counts > 1]:
		component = swts[labels==label]
		cc_variance = np.var(component)
		cc_mean = np.mean(component)
		if cc_variance > var_threshold:
			labels[labels==label] = 0

	regions = regionprops(labels)
	print("After SWT variance filtering: ", len(regions))

	#filter by height
	regions = [r for r in regions if r.bbox[2] - r.bbox[0] > 8 and r.bbox[2] - r.bbox[0] < 300]
	print("After height filtering: ", len(regions))
	#filter by aspect ratio
	def region_aspect_check(r):
		height = r.bbox[2] - r.bbox[0]; width = r.bbox[3] - r.bbox[1]
		if width / height > 0.1 and width / height < 10.0:
			return True

		return False

	regions = [r for r in regions if region_aspect_check(r)]
	print("After aspect ratio filtering: ", len(regions))

	return regions
	
img = imread("../images/testsmall.png", as_grey=True).astype(np.float64)
img /= np.max(img) #normalizing so every image is similar, otherwise the edge detection messes up
dx = sobel(img, 1).astype(np.float64) * -1 #maybe multiply with -1, easier to think that way (normally white = 1, black = 0)
dy = sobel(img, 0).astype(np.float64) * -1
maxes = np.max(np.abs(np.stack([dx, dy])), axis=0)
maxes[maxes==0] = 1.0 #if both are zero max is zero, fix divide by zero (number doesnt matter here as zero will be divided)
magnitudes = np.hypot(dx, dy)
dx = dx/maxes #for normalizing gradients so the bigger is 1, easier to step in the future
dy = dy/maxes
edges = canny(img, low_threshold=0.3, high_threshold=0.9, sigma=0.5).astype(np.int32) #still not as good as it should be..

swts = get_swts(edges, dx, dy)
gg = create_adjacency_graph(swts)
n_component, labels = connected_components(gg)
uniq, uniq_idx, uniq_counts = np.unique(labels, return_inverse=True, return_counts=True)
non_unique_mask = np.in1d(labels, uniq[uniq_counts > 1])
labels[np.logical_not(non_unique_mask)] = 0
labels = labels.reshape(swts.shape)

#then get regionprops from those that are left?
regions = get_and_filter_regions(labels, swts)

#components = np.reshape(non_unique_mask, swts.shape)

#plot_img(swts)