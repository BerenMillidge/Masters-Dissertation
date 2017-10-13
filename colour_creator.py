#Author: Beren Millidge
#Msc Dissertation
#Summer 2017

from __future__ import division
import numpy as np
from PIL import Image
import pickle
from tensorflow.examples.tutorials.mnist import input_data
from collections import defaultdict


def generate_uniform_colour(shape, r,g,b):
	assert type(shape) == tuple and all(i >0 and type(i) == int for i in shape), 'Inadmissable shape parameter. It must be a tuple of all positive integers'
	assert type(r) == int and r >=0, 'r must be an integer positive or 0'
	assert type(g) == int and g >=0, 'g must be an integer positive or 0'
	assert type(b) == int and b >=0, 'b must be an integer positive or 0'

	w,h = shape
	#initialise our data structure
	data = np.zeros((h,w,3), dtype=np.uint8)
	for i in xrange(w):
		for j in xrange(h):
			data[i,j] = [r,g,b]
	return data

def generate_gaussian_random_colour(shape,r,g,b, variance):	
	assert type(shape) == tuple and all(i >0 and type(i) == int for i in shape), 'Inadmissable shape parameter. It must be a tuple of all positive integers'
	assert type(r) == int and r >=0, 'r must be an integer positive or 0'
	assert type(g) == int and g >=0, 'g must be an integer positive or 0'
	assert type(b) == int and b >=0, 'b must be an integer positive or 0'
	assert type(variance) == int and variance >=0, 'The variance must be an integer positive or 0'
	w,h = shape
	#initialise our data structure
	data = np.zeros((h,w,3),dtype=np.uint8)
	for i in xrange(w):
		for j in xrange(h):
			# apply the random gaussian peturbations
			r_rand = r + np.random.normal(0,variance)
			g_rand = g + np.random.normal(0,variance)
			b_rand = b + np.random.normal(0,variance)
			#set each element of the image
			data[i,j] = [r_rand,g_rand,b_rand]
	return data


def show_image(data):
	img = Image.fromarray(data,'RGB')
	img.show()
	return img
	

#img = generate_uniform_colour((32,32), 200,0,0)
#show_image(img)
#print type(img)
#print img.shape
#print img

def colour_labeler(img):
	#create empty labels
	label = np.zeros([9])
	cols = img[0][0]
	r = cols[0]
	g = cols[1]
	b = cols[2]
	#print r,g,b
	#red:
	if (r > 100 and g < 50 and b<50) or (r > 200 and g <=150 and b<=150):
		label[0] = 1
	#orange:
	if (r > 100 and g > 50 and b <50) or (r > 200 and g>150 and b <120):
		label[1] = 1
	#yellow:
	if (r > 150 and g > 150 and b<50) or (r>250 and g > 250 and b <200):
		label[2] =1
	#green: 
	if (g > 100 and r <89 and b<10) or (g>100 and r<10 and b<89) or (g>200 and r <200 and b <200):
		label[3] =1
	#cyan:
	if (g>100 and b>100 and r <50) or (g>200 and b> 200 and r <160):
		label[4]=1 
	#blue
	if (b >100 and r <50 and g< 50) or (b> 200 and r< 170 and b<170):
		label[5] =1
	#purple
	if (b-(2*r) <30 and b-(2*r) >-30 and g <50) or (b > 200 and r >150 and g<160):
		label[6]=1
	#pink:
	if (b-r >-30 and b-r <30 and g <50) or (b>200 and r >200 and g<150):
		label[7] =1
	#indigo:
 
	return label


def colour_printer(label):
	#this function just prints the colour associated with the label for debugging use and testing and such
	testval = 0
	if label[0] == 1:
		print "RED"
	if label[1] == 1:
		print "ORANGE"
	if label[2] == 1:
		print "YELLOW"
	if label[3] ==1:
		print "GREEN"
	if label[4] ==1:
		print "CYAN"
	if label[5] ==1:
		print "BLUE"
	if label[6] ==1:
		print "PURPLE"
	if label[7] == 1:
		print "PINK"
	if label[8] == 1:
		print "INDIGO"
	#check if the array is all zeros, so we failed to generate a colour label correctly!
	if not np.any(label):
		print "NO COLOURS DETECTED --ERROR"
		testval = 1
	if np.count_nonzero(label) > 1:
		print "COLOUR CLASH! -- ERROR"
		testval = 1
	return testval

def colour_check(label):
	#this is the same as colour_printer without the printing, for standard use
	testval = 0
	if not np.any(label):
		testval = 1
	if np.count_nonzero(label)>1:
		testval = 1
	return testval

def generate_test_colours(N, shape):
	i = 0
	imgs = []
	labels = []
	while i < N:
		r_rand = np.random.randint(0,255)
		g_rand = np.random.randint(0,255)
		b_rand = np.random.randint(0,255)
		img = generate_uniform_colour(shape, r_rand,g_rand,b_rand)
		label = colour_labeler(img)
		check = colour_check(label)
		#show_image(img)	# this is for debugging and interest use
		assert check ==0 or check ==1, 'something went wrong here!' 
		if check ==0:
			i +=1 
			imgs.append(img)
			labels.append(label)
	imgs = np.array(imgs)
	labels = np.array(labels)
	return imgs, labels

def save_colours(colours,name):
	assert type(name) == str, 'name needs to be a string!'
	assert type(colours) == tuple and len(colours) == 2, 'colours should be a tuple consisting of images and labels'
	pickle.dump(colours, open(name, 'wb'))
	print "Saved!"

def generate_red(shape):
	label = np.zeros([9])
	frand = np.random.uniform(0,1)
	if frand >=0.5:
		r_rand = np.random.randint(100,255)
		g_rand = np.random.randint(0,50)
		b_rand = np.random.randint(0,50)
		img = generate_uniform_colour(shape, r_rand,g_rand, b_rand)
	if frand <0.5:
		r_rand = np.random.randint(0,200)
		g_rand = np.random.randint(0, 150)
		b_rand = np.random.randint(0, 150)
		img = generate_uniform_colour(shape, r_rand,g_rand,b_rand)
	label[0] = 1
	return img, label

def generate_orange(shape):
	label = np.zeros([9])
	frand = np.random.uniform(0,1)
	if frand >=0.5:
		r_rand = np.random.randint(100,255)
		g_rand = np.random.randint(50,255)
		b_rand = np.random.randint(0,50)
		img = generate_uniform_colour(shape, r_rand,g_rand, b_rand)
	if frand <0.5:
		r_rand = np.random.randint(200,255)
		g_rand = np.random.randint(150,255)
		b_rand = np.random.randint(0, 120)
		img = generate_uniform_colour(shape, r_rand,g_rand,b_rand)
	label[1] = 1
	return img, label

def generate_yellow(shape):
	label = np.zeros([9])
	frand = np.random.uniform(0,1)
	if frand >=0.66:
		r_rand = np.random.randint(150,255)
		g_rand = np.random.randint(150,255)
		b_rand = np.random.randint(0,50)
		img = generate_uniform_colour(shape, r_rand,g_rand, b_rand)
	if frand <0.66 and frand >=0.33:
		r_rand = np.random.randint(0,100)
		g_rand = np.random.randint(100,255)
		b_rand = np.random.randint(0, 89)
		img = generate_uniform_colour(shape, r_rand,g_rand,b_rand)
	if frand <0.33:
		r_rand = np.random.randint(0,200)
		g_rand = np.random.randint(200,255)
		b_rand = np.random.randint(0, 200)
		img = generate_uniform_colour(shape, r_rand,g_rand,b_rand)
	label[2] = 1
	return img, label

def generate_green(shape):
	label = np.zeros([9])
	frand = np.random.uniform(0,1)
	if frand >=0.66:
		r_rand = np.random.randint(0,89)
		g_rand = np.random.randint(100,255)
		b_rand = np.random.randint(0,10)
		img = generate_uniform_colour(shape, r_rand,g_rand, b_rand)
	if frand <0.66 and frand >=0.33:
		r_rand = np.random.randint(0,10)
		g_rand = np.random.randint(100,255)
		b_rand = np.random.randint(0, 89)
		img = generate_uniform_colour(shape, r_rand,g_rand,b_rand)
	if frand <0.33:
		r_rand = np.random.randint(0,200)
		g_rand = np.random.randint(200,255)
		b_rand = np.random.randint(0, 200)
		img = generate_uniform_colour(shape, r_rand,g_rand,b_rand)
	label[3] = 1
	return img, label

def generate_cyan(shape):
	label = np.zeros([9])
	frand = np.random.uniform(0,1)
	if frand >=0.5:
		r_rand = np.random.randint(0,50)
		g_rand = np.random.randint(100,255)
		b_rand = np.random.randint(100,255)
		img = generate_uniform_colour(shape, r_rand,g_rand, b_rand)
	if frand <0.5:
		r_rand = np.random.randint(200,255)
		g_rand = np.random.randint(150,255)
		b_rand = np.random.randint(0, 160)
		img = generate_uniform_colour(shape, r_rand,g_rand,b_rand)
	label[4] = 1
	return img, label

def generate_blue(shape):
	label = np.zeros([9])
	frand = np.random.uniform(0,1)
	if frand >=0.5:
		r_rand = np.random.randint(0,50)
		g_rand = np.random.randint(0,50)
		b_rand = np.random.randint(100,255)
		img = generate_uniform_colour(shape, r_rand,g_rand, b_rand)
	if frand <0.5:
		r_rand = np.random.randint(150,255)
		g_rand = np.random.randint(0,160)
		b_rand = np.random.randint(200, 255)
		img = generate_uniform_colour(shape, r_rand,g_rand,b_rand)
	label[5] = 1
	return img, label

def generate_purple(shape):
	label = np.zeros([9])
	frand = np.random.uniform(0,1)
	if frand >=0.5:
		r_rand = np.random.randint(15,125)
		g_rand = np.random.randint(0,50)
		b_rand = np.random.randint((2*r_rand-30),(2*r_rand)+30)
		img = generate_uniform_colour(shape, r_rand,g_rand, b_rand)
	if frand <0.5:
		r_rand = np.random.randint(0,150)
		g_rand = np.random.randint(0,160)
		b_rand = np.random.randint(200, 255)
		img = generate_uniform_colour(shape, r_rand,g_rand,b_rand)
	label[6] = 1
	return img, label

def generate_pink(shape):
	label = np.zeros([9])
	frand = np.random.uniform(0,1)
	if frand >=0.5:
		r_rand = np.random.randint(30,220)
		g_rand = np.random.randint(100,255)
		b_rand = np.random.randint(r_rand-30,r_rand + 30)
		img = generate_uniform_colour(shape, r_rand,g_rand, b_rand)
	if frand <0.5:
		r_rand = np.random.randint(200,255)
		g_rand = np.random.randint(0,150)
		b_rand = np.random.randint(200, 255)
		img = generate_uniform_colour(shape, r_rand,g_rand,b_rand)
	label[7] = 1
	return img, label


def generate_indigo(shape):
	pass
	#not implemented yet for some reason - idk. I should really do that

def split_mnist(save = False, path = None):

	if save:
		assert type(path) == str, ' you need a path if you are going to save'

	#set colour shape
	colour_shape = (28,28)

	#download the mnist
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	#data_mnist_train = mnist.train.images
	#labels_mnist_train = mnist.train.labels

	data_mnist_valid = mnist.validation.images
	labels_mnist_valid = mnist.validation.labels

	#data_mnist_test = mnist.test.images
	#labels_mnist_test = mnist.test.labels

	print type(data_mnist_valid)
	print data_mnist_valid.shape
	print labels_mnist_valid.shape
	len_mnist = labels_mnist_valid.shape[1]
	
	assert len(data_mnist_valid) == len(labels_mnist_valid), 'if this isnt true then it will not work'

	#This first sorts and splits the mnist data according to which images it is. i.e. it creates a dictionary where the key "3" will return a list of all mnist digits of a 3
	sorted_dict = defaultdict(list)
	for i in xrange(len(labels_mnist_valid)):
		for j in xrange(len_mnist):
			if labels_mnist_valid[i][j] ==1:
				sorted_dict[j].append(data_mnist_valid[i])

	fn_list = [generate_red, generate_orange, generate_yellow, generate_green, generate_cyan, generate_blue, generate_purple, generate_pink]
	len_cols = len(fn_list)	
	
	#the meat of the function, this will split the split dict again on coloru and assign an equal one and generate our ending dict of dict of 4-tuple structure
	validation_dict = {}
	for i in xrange(len_mnist):
		images = sorted_dict[i]	#this gives us a list of all 1 digits, for isntance
		split_num = len(images)//len_cols	#this is integer division. We just discard the remainder
		colour_dict = defaultdict(list)
		for j in xrange(len_cols):
			col_list = []
			tuple_dict = defaultdict(list)
			for k in xrange(j*split_num, (j+1)*split_num):			
				img = images[k]	#get image
				label = np.zeros(len_mnist)
				label[i] = 1	#get label
				col_img, col_label = fn_list[j](colour_shape)
				tuple_dict['img'].append(img)
				tuple_dict['label'].append(label)
				tuple_dict['col_img'].append(col_img)
				tuple_dict['col_label'].append(col_label)	
				#tup = np.array([img, label, col_img, col_label]) # this had better be a tuple
				#col_list.append(tup)
			#col_list = np.array(col_list)
			#print col_list.shape
			#colour_dict[j] = col_list	
			colour_dict[j] = tuple_dict	
		validation_dict[i] = colour_dict
	
	#now for the saving functionality
	if save:
		with open(path, 'wb') as f:
			pickle.dump(validation_dict, f)
    		f.close()
		print "SAVED!"	


	return validation_dict


def main():
	#to split the dataset mnist dataset for validation purposes
	path = 'synaesthesia/val_test'
	split_mnist(save = True, path = path)


	#to generate the colour input datasets. This will give you runtime warning about ubyte overflows. 
	#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	#images_train = mnist.train.images
	##N = len(images_train)
	#shape = (28,28)
	#path = 'synaesthesia/colours_test'
	#cols = generate_test_colours(N, shape)
	#save_colours(cols, path)

if __name__ == '__main__':
	main()

