test_img = np.ones((12, 12)).astype(np.float64)
test_img[3:10, 3:8] = 0.0
edges = canny(test_img).astype(np.int32)
dx = sobel(test_img, 1).astype(np.float64) *-1
dy = sobel(test_img, 0	).astype(np.float64) *-1
maxes = np.max(np.abs(np.stack([dx, dy])), axis=0) #for normalizing gradients so the bigger is 1, easier to step
maxes[maxes==0] = 1.0 #if both are zero max is zero, fix divide by zero (number doesnt matter here as zero will be divided)
dx = dx/maxes
dy = dy/maxes

swts = get_steps(edges, dx, dy)