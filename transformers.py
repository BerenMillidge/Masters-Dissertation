#Author: Beren Millidge
#Msc Dissertation
#Summer 2017

import numpy as np
import scipy
from tensorflow.examples.tutorials.mnist import input_data


from scipy import misc
import matplotlib.pyplot as plt
from scipy import ndimage
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#data_mnist_train = mnist.train.images
test = mnist.test.images
print type(test)
print test.shape

def image_translate(img, shift):
	trans = scipy.ndimage.interpolation.shift(img, shift)
	return trans

def image_rotate(img, angle):
	rot = scipy.ndimage.interpolation.rotate(img, angle, reshape = False)
	return rot

#img = test[5]
#img = np.reshape(img, [28,28])
#trans = scipy.ndimage.interpolation.shift(img, [0,5]) 

def plot_mnist_digit_transformations(img, angle, shift, show = False):
	plt.imshow(img, cmap=plt.cm.gray)
	if show:
		plt.show()
	imgshift = image_translate(img, shift)
	plt.imshow(imgshift, cmap = plt.cm.gray)
	if show:
		plt.show()
	imgrot = image_rotate(img, angle)
	plt.imshow(imgrot, cmap = plt.cm.gray)
	if show:
		plt.show()
	imgrot2 = image_rotate(img, angle* -1)
	plt.imshow(imgrot, cmap = plt.cm.gray)
	if show:
		plt.show()
	return [img, imgshift, imgrot, imgrot2]

	fig = plt.figure()
	a=fig.add_subplot(1,2,1)
	imgplot = plt.imshow(imgshift)
	a.set_title('Before')
	plt.colorbar(ticks=[0.1,0.3,0.5,0.7], orientation ='horizontal')
	a=fig.add_subplot(1,2,2)
	imgplot = plt.imshow(imgrot)
	imgplot.set_clim(0.0,0.7)
	a.set_title('After')
	plt.colorbar(ticks=[0.1,0.3,0.5,0.7], orientation='horizontal')
	a = fig.add_subplot(2,2,2)
	imgplot = plt.imshow(img)
	imgplot.set_clim(0.0,0.7)
	a.set_title('After')
	plt.colorbar(ticks=[0.1,0.3,0.5,0.7], orientation='horizontal')
	
	plt.show()
	print "WORKING"
	return img, imgshift, imgrot

#images = plot_mnist_digit_transformations(img, -20, [0,5])


#show_images(img, imgshift, imgrot)


def example_plot(ax, img, title,labels = None):
	ax.imshow(img, cmap = plt.cm.gray)
	if labels is not None:
		ax.set_xlabel(labels[0])
		ax.set_ylabel(labels[1])
	ax.set_title(title)


#fig, ax = plt.subplots()
#example_plot(ax)
#plt.tight_layout()

def plot_four_digits(digits, titles, save = True, savepath = None):
	assert len(digits) == len(titles) == 4, 'all must be length four here'

	if savepath is None and save == True:
		savepath = 'res/graphs/mnist_shifted_digits.png'

	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
	example_plot(ax1, digits[0], titles[0])
	example_plot(ax2, digits[1], titles[1])
	example_plot(ax3, digits[2], titles[2])
	example_plot(ax4, digits[3], titles[3])
	plt.tight_layout()
	plt.show()

	if save:
		fig.savefig(savepath)
		plt.close(fig)
	return fig
	
#titles = ['Original Digit', 'Forward translation', 'Forward Rotation', 'Backwards Rotation']
#fig = plot_four_digits(images, titles)

	

def gen_image_shift(test, shift):
	testlist = []
	for i in xrange(len(test)):
		img = test[i]
		img = np.reshape(img, [28,28])
		alt = image_translate(img, shift)
		alt = np.reshape(alt, [784])
		testlist.append(alt)
	testlist = np.array(testlist)
	return testlist

def gen_image_rotate(test, angle):
	testlist = []
	for i in xrange(len(test)):
		img = test[i]
		img = np.reshape(img, [28,28])
		alt = image_rotate(img, angle)
		alt = np.reshape(alt, [784])
		testlist.append(alt)
	testlist = np.array(testlist)
	return testlist


def gen_image_random_shift(test, shift_range_vert, shift_range_horiz):
	testlist = []
	for i in xrange(len(test)):
		img = test[i]
		img = np.reshape(img, [28,28])
		rand1 = np.random.uniform(shift_range_vert[0], shift_range_vert[1])
		rand2 = np.random.uniform(shift_range_horiz[0], shift_range_horiz[1])
		alt = image_translate(img, [rand1, rand2])
		alt = np.reshape(alt, [784])
		testlist.append(alt)
	testlist = np.array(testlist)
	return testlist

def gen_image_random_rotate(test, angle_range):
	testlist = []
	for i in xrange(len(test)):
		img = test[i]
		img = np.reshape(img, [28,28])
		rand = np.random.uniform(angle_range[0], angle_range[1])
		alt = image_rotate(img, rand)
		alt = np.reshape(alt, [784])
		testlist.append(alt)
	testlist = np.array(testlist)
	return testlist

#trans = image_rotate(img, 5)

#plt.imshow(img, cmap=plt.cm.gray)
#plt.show()
#plt.imshow(trans,cmap=plt.cm.gray)
#plt.show()


#shifted = gen_image_random_shift(test, [-1,3],[0,0])
#print type(shifted)
#print shifted.shape
#sh = shifted[4]
#sh = np.reshape(sh, [28,28])
#plt.imshow(sh)
#plt.show()


