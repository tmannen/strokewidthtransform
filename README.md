# strokewidthtransform
Implementing the Stroke Width Transform algorithm for detecting text from: http://www.math.tau.ac.il/~turkel/imagepapers/text_detection.pdf in numpy/scipy.

Before SWT (from: http://www.cs.cornell.edu/courses/cs4670/2010fa/projects/final/results/group_of_arp86_sk2357/Writeup.pdf)

![Before SWT](test2.png)

After SWT:

![After SWT](figure1.png)

The results aren't as smooth as in the paper. This might be due to the edge detection, as it's not as smooth as it should be. I'm not sure how to set the canny parameters to get it smoother.
