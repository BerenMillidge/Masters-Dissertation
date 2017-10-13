#Author: Beren Millidge
#Msc Dissertation
#Summer 2017


from __future__ import division
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
from tensorflow.examples.tutorials.mnist import input_data
from data_providers import *
from nn_funcs import *

path = 'synaesthesia/test_mnist_colours2_0white'

def colourise_image(img,r,g,b):
	col_img = np.zeros((784,3),dtype = np.uint8)
	for i in xrange(len(img)):
		if img[i] == 0:
			col_img[i] = [0,0,0]
		if not img[i] == 0:
			col_img[i] = [r,g,b]
	return col_img


def colour_labeler(r,g,b):
	#create empty labels
	label = np.zeros([9])
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

def colour_check(label):
	testval = 0
	if not np.any(label):
		testval = 1
	if np.count_nonzero(label)>1:
		testval = 1
	return testval


def colourise_mnist(images, labels):
	N = len(images)
	assert N == len(labels), 'label and image lengths do not match!'
	new_imgs = []
	new_labels = []
	i = 0
	while i < N:
		r_rand = np.random.randint(0,255)
		g_rand = np.random.randint(0,255)
		b_rand = np.random.randint(0,255)
		label = colour_labeler(r_rand,g_rand,b_rand)
		check = colour_check(label)
		assert check==0 or check==1, 'check out of bounds!'
		if check == 0:
			colimg = colourise_image(images[i], r_rand, g_rand, b_rand)
			new_imgs.append(colimg)
			concatenated_label = np.zeros(len(labels[i]) + len(label))
			concatenated_label[0:len(labels[i])] = labels[i]
			concatenated_label[len(labels[i]):] = label
			new_labels.append(concatenated_label)
			i+=1
	new_imgs = np.array(new_imgs)
	new_labels = np.array(new_labels)
	return new_imgs, new_labels


def save_colours(colours,name):
	assert type(name) == str, 'name needs to be a string!'
	assert type(colours) == tuple and len(colours) == 2, 'colours should be a tuple consisting of images and labels'
	pickle.dump(colours, open(name, 'wb'))
	print "Saved!"


def make_colour_mnist(path):
	#get our MNIST data:
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	images_train = mnist.train.images
	labels_train = mnist.train.labels

	images_valid = mnist.validation.images
	labels_valid = mnist.validation.labels

	images_test = mnist.test.images
	labels_test = mnist.test.labels


	img = images_train[0]
	print len(img)
	#colourise it
	cols = colourise_mnist(images_train, labels_train)
	#and save
	save_colours(cols, path)
	

#make_colour_mnist(path)


def train_combined_network(path, batch_size, lrate, epochs, hidden_dim, log_dir, cost_func = cross_entropy):

	# This is our pure colour data

	colours = pickle.load(open(path,'rb'))
	data = colours[0]
	labels = colours[1]

	print type(data)
	print data.shape

	#convert our data to tensorflow types
	#data = tf.convert_to_tensor(data, dtype=tf.float32)
	#labels = tf.convert_to_tensor(labels, dtype=tf.float32)

	# we'll define the shape variables here if they are useful
	shape = data.shape
	number = shape[0]
	dim = shape[1]
	channels = shape[2]

	num_labels = labels.shape[1]

	#stuff our data into data provider
	colours_provider = DataProvider(data,labels,batch_size)
	#start session
	sess = tf.InteractiveSession()

	# get our placeholders for input data and labels
	x = tf.placeholder(tf.float32, shape = [None,dim,channels])
	y_ = tf.placeholder(tf.float32, shape=[None,num_labels])

	total = dim*channels
	flat = tf.reshape(x, [-1, total])
	print type(flat)

	#first hidden layer
	hfc1 = fc_layer(flat, total, hidden_dim, 'hidden_1', tf.tanh)
	#second hidden layer
	hfc2 = fc_layer(hfc1, hidden_dim, hidden_dim, 'hidden_2', tf.sigmoid)
	#third hidden layer
	hfc3 = fc_layer(hfc2, hidden_dim, hidden_dim, 'hidden_3', tf.sigmoid)
	#output layer
	y = fc_layer(hfc3, hidden_dim, num_labels, 'output', tf.identity)

	#define our cost function
	#with tf.name_scope('cross_entropy'):
		#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))
		#tf.summary.scalar('cross_entropy', cross_entropy)
	# let's define cost functions in the nn_func as well, as this could be a cool way to get additional customisability!
	#train_step = tf.train.GradientDescentOptimizer(lrate).minimize(cross_entropy)

	#least_squares = tf.reduce_mean(tf.reduce_sum(tf.square(y-y_)))
	#train_step = tf.train.GradientDescentOptimizer(lrate).minimize(least_squares)

	with tf.name_scope('cost'):
		cost = cost_func(y_, y)
		tf.summary.scalar('cost', cost) # I'm not sure if this bit is requited!

	with tf.name_scope('train'):	
		train_step = tf.train.GradientDescentOptimizer(lrate).minimize(cost)	

	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', accuracy)

	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(log_dir, sess.graph)
	tf.global_variables_initializer().run()

	
	def feed_dict():
		xs, ys =colours_provider.next()
		return {x:xs, y_: ys}

	for i in xrange(epochs):
		if i %10 ==0: # we save every 10 epochs
			run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
			run_metadata = tf.RunMetadata()
			summary, _ = sess.run([merged, train_step], feed_dict = feed_dict(), options = run_options, run_metadata = run_metadata)
			train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
			train_writer.add_summary(summary,i)
			print('Adding run metadata for', i)

	train_writer.close()
	print "ACCURACY:"
	print(accuracy.eval(feed_dict={x: data, y_:labels}))

	#sess.run(tf.global_variables_initializer())
	#for i in xrange(epochs):
		#batch = colours_provider.next()
		#data_batch = tf.convert_to_tensor(batch[0],dtype = tf.float32)
		#labels_batch = tf.convert_to_tensor(batch[1],dtype = tf.float32)
		#actually run the train step!
		#train_step.run(feed_dict = {x:batch[0], y_:batch[1]})

	#print our results.
	#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	#accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	#print "ACCURACY:"
	#print(accuracy.eval(feed_dict={x: data, y_:labels}))
	return accuracy



#batch_size = 50
#lrate = 0.01
#epochs = 1000
#hidden_dim = 2000
#log_dir = 'tmp/tensorflow/mnist/logs/syn_combined'
#train_combined_network(path, batch_size, lrate,epochs,hidden_dim, log_dir)

