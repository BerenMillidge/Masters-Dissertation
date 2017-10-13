#Author: Beren Millidge
#Msc Dissertation
#Summer 2017

from __future__ import division
import numpy as np
import tensorflow as tf
from nn_funcs import *
from data_providers import *
from tensorflow.examples.tutorials.mnist import input_data
import pickle
import matplotlib.pyplot as plt

EPOCHS = 10000
LRATE = 0.0001
BATCH_SIZE = 50
LOGDIR = 'tmp/tensorflow/logs/synseparate'
#COLOUR_PATH = 'synaesthesia/test_mnist_colours2_0white'
COLOUR_PATH = 'synaesthesia/colours_test'
MNIST_PATH = None
HIDDEN_DIM = 1000
VALID = True
VALID_PATH = 'synaesthesia/val_test'
SAVE_PATH  = 'synaesthesia/synseparate_results_multi_2'
CHANCE = 0.5

def train_double_network(epochs, lrate,batch_size,logdir, colour_path, mnist_path, hidden_dim, valid = False, valid_path = None, dropout = True, drop_rate = 0.5, logging = True, costlist = False, variable_weights = False, autoencoding = False):
	#Initialise useful variables here so they can be easily changed instead of scattered about through the script
	comb_dim =2000
	act_func = tf.tanh
	comb_func = tf.add
	mnist_weight =1
	colour_weight = 1

	if variable_weights:
		mnist_weight = variable_variable(1)
		colour_weight = variable_variable(1)



	#assert statements re valid
	assert valid == True or valid == False,' valid must be a boolean variable'
	if valid == True:
		assert type(valid_path) == str and len(valid_path) > 0, 'valid path must be valid'

	# first we'll get our data. 
	#COLOURS
	colours = pickle.load(open(colour_path,'rb'))
	data_colours = colours[0]
	labels_colours = colours[1]
	shape = data_colours.shape
	assert len(shape) == 3 or len(shape) ==4, 'otherwise it is not an image format we recognise'
	number = shape[0]
	dim_cols = shape[1]
	if len(shape) == 3:
		channels = shape[2]
	if len(shape) == 4:
		channels = shape[3]
	dim_labels_cols = labels_colours.shape[1]
	print shape

	#MNIST:
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	data_mnist_train = mnist.train.images
	labels_mnist_train = mnist.train.labels

	#data_mnist_valid = mnist.validation.images
	#labels_mnist_valid = mnist.validation.labels

	#data_mnist_test = mnist.test.images
	#labels_mnist_test = mnist.test.labels

	dim_labels_mnist = labels_mnist_train.shape[1]
	dim_mnist = data_mnist_train.shape[1]


	#Dataproviders
	colours_provider = DataProvider(data_colours,labels_colours,batch_size)
	mnist_provider = DataProvider(data_mnist_train, labels_mnist_train, batch_size)

	#start session
	sess = tf.InteractiveSession()

	#for mnist:
	x_mnist = tf.placeholder(tf.float32, shape=[None, dim_mnist])
	y_mnist = tf.placeholder(tf.float32, shape=[None, dim_labels_mnist])
	if autoencoding:
		y_mnist = tf.placeholder(tf.float32, shape = [None, dim_mnist])
	#for colours
	if len(shape) == 3:
		x_colours = tf.placeholder(tf.float32, shape=[None, dim_cols, channels]) # dimensionality will be tricky
	if len(shape) == 4:
		x_colours = tf.placeholder(tf.float32, shape=[None, dim_cols,dim_cols, channels])
	y_colours = tf.placeholder(tf.float32, shape=[None, dim_labels_cols])
	if autoencoding:
		y_colours = tf.placeholder(tf.float32, shape = [None, dim_cols, channels])

	#MNIST early network@
	if dropout == False:
		h1_mnist = fc_layer(x_mnist, dim_mnist, hidden_dim, 'h1_mnist', act_func)
		h2_mnist = fc_layer(h1_mnist, hidden_dim, hidden_dim, 'h2_mnist', act_func)

		#COLS early network:
		#flatten or something xcol = ???
		if len(shape)==3:
			total = dim_cols*channels
		if len(shape) == 4:
			total = dim_cols*dim_cols*channels
		flat = tf.reshape(x_colours, [-1, total])
		h1_col = fc_layer(flat, total, hidden_dim, 'h1_cols', act_func)
		h2_col = fc_layer(h1_col, hidden_dim, hidden_dim, 'h2_cols',act_func)

	#combination layer using our own special combination layer function 
		comb = combination_layer([h2_mnist, h2_col], [hidden_dim, hidden_dim], comb_dim, 'combination_layer', act_func, comb_func) 

		output_mnist = fc_layer(comb, comb_dim, dim_labels_mnist, 'mnist_output', tf.identity)
		output_colours = fc_layer(comb, comb_dim, dim_labels_cols, 'colours_output', tf.identity)

	if dropout:
		h1_mnist = fc_layer(x_mnist, dim_mnist, hidden_dim, 'h1_mnist', act_func)
		drop1 = dropout_layer(h1_mnist, drop_rate, drop_bool = True, name = 'dropout_h1_mnist')
		h2_mnist = fc_layer(drop1, hidden_dim, hidden_dim, 'h2_mnist', act_func)
		drop2 = dropout_layer(h2_mnist, drop_rate, drop_bool = True, name = 'dropout_h2_mnist')

		#COLS early network:
		if len(shape)==3:
			total = dim_cols*channels
		if len(shape) == 4:
			total = dim_cols*dim_cols*channels
		flat = tf.reshape(x_colours, [-1, total])
		h1_col = fc_layer(flat, total, hidden_dim, 'h1_cols', act_func)
		drop3 = dropout_layer(h1_col, drop_rate, drop_bool = True, name = 'dropout_h1_colours')
		h2_col = fc_layer(drop3, hidden_dim, hidden_dim, 'h2_cols',act_func)
		drop4 = dropout_layer(h2_col, drop_rate, drop_bool = True, name = 'dropout_h2_colours')


		comb = combination_layer([drop2, drop4], [hidden_dim, hidden_dim], comb_dim, 'combination_layer', act_func, comb_func) 
		drop_comb = dropout_layer(comb, drop_rate, drop_bool = True, name = 'dropout_h1')

	
		output_mnist = fc_layer(comb, comb_dim, dim_labels_mnist, 'mnist_output', tf.identity)
		output_colours = fc_layer(comb, comb_dim, dim_labels_cols, 'colours_output', tf.identity)



	with tf.name_scope('cost'):
		cost_func = mnist_weight *tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_mnist, logits = output_mnist)) + colour_weight * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_colours, logits = output_colours))
		tf.summary.scalar('cost', cost_func)
		if variable_weights:
			tf.summary.scalar('mnist_weight', mnist_weight)
			tf.summary.scalar('colour_weight', colour_weight)

	#we now do the train function as normal!
	with tf.name_scope('train'):	
		train_step = tf.train.GradientDescentOptimizer(lrate).minimize(cost_func)

	#we define our accuracies
	with tf.name_scope('accuracy'):
		with tf.name_scope('accuracy_mnist'):
			correct_prediction_mnist = tf.equal(tf.argmax(output_mnist, 1), tf.argmax(y_mnist, 1))
      			accuracy_mnist = tf.reduce_mean(tf.cast(correct_prediction_mnist, tf.float32))
			tf.summary.scalar('accuracy_mnist', accuracy_mnist)
	
		with tf.name_scope('accuracy_colours'):
			correct_prediction_colours = tf.equal(tf.argmax(output_colours, 1), tf.argmax(y_colours, 1))
      			accuracy_colours = tf.reduce_mean(tf.cast(correct_prediction_colours, tf.float32))
			tf.summary.scalar('accuracy_colours', accuracy_colours)
	with tf.name_scope('total_accuracy'):
		total_accuracy = (accuracy_mnist + accuracy_colours)/2
		tf.summary.scalar('total_accuracy', total_accuracy)

	#we merge all our ops and start the filewriters for the tensorboard
	if logging:
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(logdir, sess.graph)
	
	tf.global_variables_initializer().run()

	# let's try using the NNfunct feeddict repository which might be fun!
	feeddict = FeedDict([colours_provider, mnist_provider], [x_colours, y_colours, x_mnist, y_mnist])
	if autoencoding:
		feeddict = FeedDict([colours_provider, mnist_provider], [x_colours, x_colours, x_mnist, x_mnist])
	print type(feeddict[x_mnist])
	print feeddict[x_colours].shape
	acclist = []
	costlist = []
	if variable_weights:
		var_list_mnist = []
		var_list_col = []	
	for i in xrange(epochs):
		if i %10 ==0: # we save every 10 epochs
			if logging:
				run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
				run_metadata = tf.RunMetadata()
				summary, _ , acc, cost = sess.run([merged, train_step, total_accuracy,cost_func], feed_dict = feeddict, options = run_options, run_metadata = run_metadata)
				if variable_weights:
					mn_val = mnist_weight.eval(sess)
					col_val = colour_weight.eval(sess)
					var_list_mnist.append(mn_val)
					var_list_col.append(col_val)
				train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
				train_writer.add_summary(summary,i)
				acclist.append(acc)
				costlist.append(cost)
				print('Adding run metadata for', i)
			if logging ==False:
				_, acc, cost  = sess.run([train_step, total_accuracy, cost_func], feed_dict = feeddict)
				acclist.append(acc)
				costlist.append(cost)
				if variable_weights:
					mn_val = mnist_weight.eval(sess)
					col_val = colour_weight.eval(sess)
					var_list_mnist.append(mn_val)
					var_list_col.append(col_val)

	#acclist = np.array(acclist)
			

	def unpacker(valid_dict):

		digits = valid_dict.keys()
		acc_list = []
		for i in xrange(len(digits)):
			digit_list = []
			col_dict = valid_dict[digits[i]]
			col_list = col_dict.keys()
			for j in xrange(len(col_list)):
				tuple_dict = col_dict[j]
				x_mn = tuple_dict['img']
				y_mn = tuple_dict['label']
				x_col = tuple_dict['col_img']
				y_col = tuple_dict['col_label']
				
				
				x_mn = np.array(x_mn)
				#print x_mn.shape
				y_mn = np.array(y_mn)
				#print y_mn.shape
				x_col = np.array(x_col)
				y_col = np.array(y_col)
				#awesome it worked!
				#print x_mn.shape
				#print y_mn.shape
				#print x_col.shape	
				#print y_col.shape
			
				feeddict = {x_mnist: x_mn, y_mnist: y_mn, x_colours: x_col, y_colours:y_col}
				run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
				run_metadata = tf.RunMetadata()
				acc_mnist, acc_col = sess.run([accuracy_mnist, accuracy_colours], feed_dict = feeddict,options = run_options, run_metadata = run_metadata)
				colour = colour_checker(j)
				print "validation accuracies for: MNIST DIGIT " + str(i) + " COLOUR: " + colour
				print str(acc_mnist) + "  " + str(acc_col)
				print "  "
				digit_list.append([acc_mnist, acc_col])
			acc_list.append(digit_list)
		return acc_list

	if valid == True:
		#first we'll just download our file
		valid_dict = pickle.load(open(valid_path, 'rb'))
		#then we unpack it and this deals with everything, which is nice
		valid_list = unpacker(valid_dict)
		



	if logging:
		train_writer.close()

	print "TRAINING ACCURACY:"
	total_accuracy = total_accuracy.eval(feed_dict=feeddict)
	print total_accuracy
	
	if variable_weights:
		return var_list_mnist, var_list_col

	
	if valid == True and logging == True:
		print type(total_accuracy)
		print type(valid_list)
		if costlist:
			return total_accuracy, valid_list, acclist, costlist

		return total_accuracy, valid_list, acclist, costlist

	if valid == True:
		print type(total_accuracy)
		print type(valid_list)
		if costlist:
			return total_accuracy, valid_list, acclist, costlist

		return total_accuracy, valid_list, acclist, costlist
	else:
		return total_accuracy





def train_lower_comb_network(epochs, lrate,batch_size,logdir, colour_path, mnist_path, hidden_dim, valid = False, valid_path = None, logging = False):
	#Initialise useful variables here so they can be easily changed instead of scattered about through the script
	comb_dim =2000
	act_func = tf.tanh
	comb_func = tf.add
	mnist_weight =1
	colour_weight = 1



	#assert statements re valid
	assert valid == True or valid == False,' valid must be a boolean variable'
	if valid == True:
		assert type(valid_path) == str and len(valid_path) > 0, 'valid path must be valid'

	assert logging == True or logging == False,'must be boolean value'
	# first we'll get our data. 
	#COLOURS
	colours = pickle.load(open(colour_path,'rb'))
	data_colours = colours[0]
	labels_colours = colours[1]
	shape = data_colours.shape
	assert len(shape) == 3 or len(shape) == 4, 'otherwise not recognised image format'
	number = shape[0]
	dim_cols = shape[1]
	if len(shape) ==3:
		channels = shape[2]
	if len(shape) == 4:
		channels = shape[3]
	dim_labels_cols = labels_colours.shape[1]
	print shape

	#MNIST:
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	data_mnist_train = mnist.train.images
	labels_mnist_train = mnist.train.labels

	#data_mnist_valid = mnist.validation.images
	#labels_mnist_valid = mnist.validation.labels

	#data_mnist_test = mnist.test.images
	#labels_mnist_test = mnist.test.labels

	dim_labels_mnist = labels_mnist_train.shape[1]
	dim_mnist = data_mnist_train.shape[1]


	#Dataproviders
	colours_provider = DataProvider(data_colours,labels_colours,batch_size)
	mnist_provider = DataProvider(data_mnist_train, labels_mnist_train, batch_size)

	#start session
	sess = tf.InteractiveSession()

	#for mnist:
	x_mnist = tf.placeholder(tf.float32, shape=[None, dim_mnist])
	y_mnist = tf.placeholder(tf.float32, shape=[None, dim_labels_mnist])
	#for colours
	if len(shape) == 3:
		x_colours = tf.placeholder(tf.float32, shape=[None, dim_cols, channels]) # dimensionality will be tricky
	if len(shape) ==4:
		x_colours = tf.placeholder(tf.float32, shape=[None, dim_cols,dim_cols, channels]) # dimensionality will be tricky
	y_colours = tf.placeholder(tf.float32, shape=[None, dim_labels_cols])

	#EARLY NETWORK
	#mnist
	h1_mnist = fc_layer(x_mnist, dim_mnist, hidden_dim, 'h1_mnist', act_func)
	#colours
	if len(shape) == 3:
		total = dim_cols*channels
	if len(shape) ==4:
		total = dim_cols*dim_cols*channels
	flat = tf.reshape(x_colours, [-1, total])
	h1_col = fc_layer(flat, total, hidden_dim, 'h1_cols', act_func)

	#COMBINED NETWORK
	#mnist comb layer
	h2_mnist = combination_layer([h1_mnist,h1_col], [hidden_dim,hidden_dim], hidden_dim, 'h2_mnist_combined', act_func, comb_func)
	#colour comb_layer
	h2_col = combination_layer([h1_col,h1_mnist], [hidden_dim,hidden_dim], hidden_dim, 'h2_cols_combined',act_func, comb_func)

	#multi comb layer
	comb = combination_layer([h2_mnist, h2_col], [hidden_dim, hidden_dim], comb_dim, 'combination_layer', act_func, comb_func) 

	#output layers
	output_mnist = fc_layer(comb, comb_dim, dim_labels_mnist, 'mnist_output', tf.identity)
	output_colours = fc_layer(comb, comb_dim, dim_labels_cols, 'colours_output', tf.identity)

	#we define our cost function which is just the weighted sum of the two output errors
	with tf.name_scope('cost'):
		cost = mnist_weight *tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_mnist, logits = output_mnist)) + colour_weight * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_colours, logits = output_colours))
		tf.summary.scalar('cost', cost)

	#we now do the train function as normal!
	with tf.name_scope('train'):	
		train_step = tf.train.GradientDescentOptimizer(lrate).minimize(cost)

	#we define our accuracies
	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction_mnist'):
			correct_prediction_mnist = tf.equal(tf.argmax(output_mnist, 1), tf.argmax(y_mnist, 1))
      			accuracy_mnist = tf.reduce_mean(tf.cast(correct_prediction_mnist, tf.float32))
			tf.summary.scalar('accuracy_mnist', accuracy_mnist)
	
		with tf.name_scope('correct_prediction_colours'):
			correct_prediction_colours = tf.equal(tf.argmax(output_colours, 1), tf.argmax(y_colours, 1))
      			accuracy_colours = tf.reduce_mean(tf.cast(correct_prediction_colours, tf.float32))
			tf.summary.scalar('accuracy_colours', accuracy_colours)
	with tf.name_scope('total_accuracy'):
		total_accuracy = (accuracy_mnist + accuracy_colours)/2
		tf.summary.scalar('total_accuracy', total_accuracy)

	#we merge all our ops and start the filewriters for the tensorboard
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(logdir, sess.graph)
	tf.global_variables_initializer().run()
	
	# let's try using the NNfunct feeddict repository which might be fun!
	feeddict = FeedDict([colours_provider, mnist_provider], [x_colours, y_colours, x_mnist, y_mnist])
	for i in xrange(epochs):
		if i %10 ==0: # we save every 10 epochs
			run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
			run_metadata = tf.RunMetadata()
			summary, _ = sess.run([merged, train_step], feed_dict = feeddict, options = run_options, run_metadata = run_metadata)
			train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
			train_writer.add_summary(summary,i)
			print('Adding run metadata for', i)

	def unpacker(valid_dict):
		digits = valid_dict.keys()
		acc_list = []
		for i in xrange(len(digits)):
			digit_list = []
			col_dict = valid_dict[digits[i]]
			col_list = col_dict.keys()
			for j in xrange(len(col_list)):
				tuple_dict = col_dict[j]
				x_mn = tuple_dict['img']
				y_mn = tuple_dict['label']
				x_col = tuple_dict['col_img']
				y_col = tuple_dict['col_label']
				
				x_mn = np.array(x_mn)
				#print x_mn.shape
				y_mn = np.array(y_mn)
				#print y_mn.shape
				x_col = np.array(x_col)
				y_col = np.array(y_col)
				#awesome it worked!
				#print x_mn.shape
				#print y_mn.shape
				#print x_col.shape	
				#print y_col.shape
			
				feeddict = {x_mnist: x_mn, y_mnist: y_mn, x_colours: x_col, y_colours:y_col}
				run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
				run_metadata = tf.RunMetadata()
				acc_mnist, acc_col = sess.run([accuracy_mnist, accuracy_colours], feed_dict = feeddict,options = run_options, run_metadata = run_metadata)
				colour = colour_checker(j)
				print "validation accuracies for: MNIST DIGIT " + str(i) + " COLOUR: " + colour
				print str(acc_mnist) + "  " + str(acc_col)
				print "  "
				digit_list.append([acc_mnist, acc_col])
			acc_list.append(digit_list)
		return acc_list



	if valid == True:
		#first we'll just download our file
		valid_dict = pickle.load(open(valid_path, 'rb'))
		#then we unpack it and this deals with everything, which is nice
		valid_list = unpacker(valid_dict)
		



	train_writer.close()
	print "TRAINING ACCURACY:"
	total_accuracy = total_accuracy.eval(feed_dict=feeddict)
	print total_accuracy
	

	if valid == True:
		#print type(total_accuracy)
		#print type(valid_list)
		return total_accuracy, valid_list
	else:
		return total_accuracy



def train_masked_lower_comb_network(epochs, lrate,batch_size,logdir, colour_path, mnist_path, hidden_dim, valid = False, valid_path = None, mask_list = None, mask_flag_list = None, chance = None, mask_func = None, logging = False, comb_dim = 2000, act_func = tf.tanh, comb_func = tf.add, mnist_weight = 1, colour_weight = 1):


	#if no custom mask lists and no custom function, we define our default mask function
	if mask_list == None and mask_func == None:
		mask_func = generate_random_mask


	# this is our flag for no mask_flag_list
	default_flag =0
	if mask_flag_list:
		default_flat = 1
		assert type(mask_flag_list) == list and len(mask_flag_list) == 2, 'else params fail'

	if mask_list:
		assert type(mask_list) == list and len(mask_list) ==2, ' incorrect mask_list params'
		assert mask_func == None,' you cant have both mask list and mask func, mask list must override'
	if chance:
		assert chance >=0 and chance <=1, 'chance must be a probability'

	#do our checking of valid
	assert valid == True or valid == False,' valid must be a boolean variable'
	if valid == True:
		assert type(valid_path) == str and len(valid_path) > 0, 'valid path must be valid'

	assert logging == True or logging == False, 'do our logging must be boolean'
	
	# first we'll get our data. 
	#COLOURS
	colours = pickle.load(open(colour_path,'rb'))
	data_colours = colours[0]
	labels_colours = colours[1]
	shape = data_colours.shape
	assert len(shape) == 3 or len(shape) ==4, 'otherwise it is not an image format we recognise'
	number = shape[0]
	dim_cols = shape[1]
	if len(shape) == 3:
		channels = shape[2]
	if len(shape) == 4:
		channels = shape[3]
	dim_labels_cols = labels_colours.shape[1]
	print shape

	#MNIST:
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	data_mnist_train = mnist.train.images
	labels_mnist_train = mnist.train.labels

	#data_mnist_valid = mnist.validation.images
	#labels_mnist_valid = mnist.validation.labels

	#data_mnist_test = mnist.test.images
	#labels_mnist_test = mnist.test.labels

	dim_labels_mnist = labels_mnist_train.shape[1]
	dim_mnist = data_mnist_train.shape[1]


	#Dataproviders
	colours_provider = DataProvider(data_colours,labels_colours,batch_size)
	mnist_provider = DataProvider(data_mnist_train, labels_mnist_train, batch_size)

	#start session
	sess = tf.InteractiveSession()

	#for mnist:
	x_mnist = tf.placeholder(tf.float32, shape=[None, dim_mnist])
	y_mnist = tf.placeholder(tf.float32, shape=[None, dim_labels_mnist])
	#for colours
	if len(shape) == 3:
		x_colours = tf.placeholder(tf.float32, shape=[None, dim_cols, channels]) # dimensionality will be tricky
	if len(shape) == 4:
		x_colours = tf.placeholder(tf.float32, shape=[None, dim_cols,dim_cols, channels])
	y_colours = tf.placeholder(tf.float32, shape=[None, dim_labels_cols])

	#MNIST early network@
	h1_mnist = fc_layer(x_mnist, dim_mnist, hidden_dim, 'h1_mnist', act_func)
	#h2_mnist = fc_layer(h1_mnist, hidden_dim, hidden_dim, 'h2_mnist', act_func)

	if len(shape)==3:
		total = dim_cols*channels
	if len(shape) == 4:
		total = dim_cols*dim_cols*channels
	flat = tf.reshape(x_colours, [-1, total])
	h1_col = fc_layer(flat, total, hidden_dim, 'h1_cols', act_func)
	#h2_col = fc_layer(h1_col, hidden_dim, hidden_dim, 'h2_cols',act_func)


	#COMBINED NETWORK
	#mnist masked comb layer
	if default_flag ==0:
		mask_flags = [0,1]
	if default_flag ==1:
		mask_flags = mask_flag_list[0]
	h2_mask_mnist = masked_comb_layer([h1_mnist, h1_col], [hidden_dim, hidden_dim], hidden_dim, 'h2_mnist_mask_combined', mask_flags = mask_flags, mask_list = mask_list, mask_func = mask_func, chance = chance, act_func = act_func, comb_func = comb_func)

	#colour masked comb layer
	if default_flag ==0:
		mask_flags = [0,1]
	if default_flag ==1:
		mask_flags = mask_flag_list[1]
	h2_mask_col = masked_comb_layer([h1_col, h1_mnist],[hidden_dim, hidden_dim],hidden_dim, 'h2_col_mask_combined', mask_flags = mask_flags, mask_list = mask_list, mask_func = mask_func, chance = chance, act_func = act_func, comb_func = comb_func)
	
	#multi comb layer
	#comb = combination_layer([h2_mnist, h2_col], [hidden_dim, hidden_dim], comb_dim, 'combination_layer', act_func, comb_func) 
	comb = combination_layer([h2_mask_mnist, h2_mask_col], [hidden_dim, hidden_dim], comb_dim, 'combination_layer', act_func, comb_func) 

	#output layers
	output_mnist = fc_layer(comb, comb_dim, dim_labels_mnist, 'mnist_output', tf.identity)
	output_colours = fc_layer(comb, comb_dim, dim_labels_cols, 'colours_output', tf.identity)


	with tf.name_scope('cost'):
		cost = mnist_weight *tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_mnist, logits = output_mnist)) + colour_weight * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_colours, logits = output_colours))
		tf.summary.scalar('cost', cost)

	#we now do the train function as normal!
	with tf.name_scope('train'):	
		train_step = tf.train.GradientDescentOptimizer(lrate).minimize(cost)

	#we define our accuracies
	with tf.name_scope('accuracy'):
		with tf.name_scope('accuracy_mnist'):
			correct_prediction_mnist = tf.equal(tf.argmax(output_mnist, 1), tf.argmax(y_mnist, 1))
      			accuracy_mnist = tf.reduce_mean(tf.cast(correct_prediction_mnist, tf.float32))
			tf.summary.scalar('accuracy_mnist', accuracy_mnist)
	
		with tf.name_scope('accuracy_colours'):
			correct_prediction_colours = tf.equal(tf.argmax(output_colours, 1), tf.argmax(y_colours, 1))
      			accuracy_colours = tf.reduce_mean(tf.cast(correct_prediction_colours, tf.float32))
			tf.summary.scalar('accuracy_colours', accuracy_colours)
	with tf.name_scope('total_accuracy'):
		total_accuracy = (accuracy_mnist + accuracy_colours)/2
		tf.summary.scalar('total_accuracy', total_accuracy)

	#we merge all our ops and start the filewriters for the tensorboard
	merged = tf.summary.merge_all()
	if logging == True:
		train_writer = tf.summary.FileWriter(logdir, sess.graph)
	tf.global_variables_initializer().run()

	feeddict = FeedDict([colours_provider, mnist_provider], [x_colours, y_colours, x_mnist, y_mnist])
	#print type(feeddict[x_mnist])
	#print feeddict[x_mnist].shape
	#print feeddict[x_colours].shape
	for i in xrange(epochs):
		if i %10 ==0: # we save every 10 epochs
			if logging == False:
				#run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
				#run_metadata = tf.RunMetadata()
				_ = sess.run(train_step, feed_dict = feeddict)
				#train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
				#train_writer.add_summary(summary,i)
				#print('Adding run metadata for', i)
			if logging == True:
				run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
				run_metadata = tf.RunMetadata()
				summary, _ = sess.run([merged, train_step], feed_dict = feeddict, options = run_options, run_metadata = run_metadata)
				train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
				train_writer.add_summary(summary,i)
				print('Adding run metadata for', i)

	def unpacker(valid_dict):
		digits = valid_dict.keys()
		acc_list = []
		for i in xrange(len(digits)):
			digit_list = []
			col_dict = valid_dict[digits[i]]
			col_list = col_dict.keys()
			for j in xrange(len(col_list)):
				tuple_dict = col_dict[j]
				x_mn = tuple_dict['img']
				y_mn = tuple_dict['label']
				x_col = tuple_dict['col_img']
				y_col = tuple_dict['col_label']
				
				x_mn = np.array(x_mn)
				#print x_mn.shape
				y_mn = np.array(y_mn)
				#print y_mn.shape
				x_col = np.array(x_col)
				y_col = np.array(y_col)
				#awesome it worked!
				#print x_mn.shape
				#print y_mn.shape
				#print x_col.shape	
				#print y_col.shape
			
				feeddict = {x_mnist: x_mn, y_mnist: y_mn, x_colours: x_col, y_colours:y_col}
				#run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
				#run_metadata = tf.RunMetadata()
				acc_mnist, acc_col = sess.run([accuracy_mnist, accuracy_colours], feed_dict = feeddict)
				colour = colour_checker(j)
				if logging == True:
					print "validation accuracies for: MNIST DIGIT " + str(i) + " COLOUR: " + colour
					print str(acc_mnist) + "  " + str(acc_col)
					print "  "
				digit_list.append([acc_mnist, acc_col])
			acc_list.append(digit_list)
		return acc_list

	if valid == True:
		#first we'll just download our file
		valid_dict = pickle.load(open(valid_path, 'rb'))
		#then we unpack it and this deals with everything, which is nice
		valid_list = unpacker(valid_dict)
		


	if logging == True:
		train_writer.close()
	print "TRAINING ACCURACY:"
	total_accuracy = total_accuracy.eval(feed_dict=feeddict)
	print total_accuracy

	def calculate_total_validation_accuracy(acc_list):
		endlist = []
		for i in xrange(len(acc_list)):
			mu = np.mean(acc_list[i])
			endlist.append(mu)
		mean = np.mean(endlist)
		return mean
	

	if valid == True:
		print type(total_accuracy)
		print type(valid_list)
		mean = calculate_total_validation_accuracy(valid_list)
		print mean
		return total_accuracy, mean
	else:
		return total_accuracy




def save_results(result, save_path):
	with open(save_path, 'wb') as f:
		pickle.dump(result,f)
	f.close()
	print "SAVED!"

def save_valids(valid_list, path):
	with open(path,'wb') as f:
		pickle.dump(valid_list, f)
	f.close()


def hyperparam_grid_search(lrates, batch_sizes, hidden_dims, comb_dims, act_funcs, comb_funcs, verbose = True, network_verbose = False, savepath = None):
	train_results = []
	valid_results = {}
	if savepath is not None:
		savepath = "res/synseparate_grid_search"
	#set up default savepath
	

	lratelist = []
	for lrate in lrates:
		train, valid = train_masked_lower_comb_network(EPOCHS, lrate, BATCH_SIZE, LOGDIR, COLOUR_PATH, MNIST_PATH, HIDDEN_DIM, VALID, VALID_PATH, chance = CHANCE, logging = network_verbose)
		if verbose:
			print "Validation Error is: " + str(valid) + "For Lrate: " + str(lrate)
		lratelist.append(valid)
	valid_results['lrates'] = lratelist
	with open(savepath, 'wb') as f:
		pickle.dump(valid_results, f)
	f.close()
	print "SAVED LRATES"
	
	batchlist = []
	for batch_size in batch_sizes:
		train, valid = train_masked_lower_comb_network(EPOCHS, LRATE, batch_size, LOGDIR, COLOUR_PATH, MNIST_PATH, HIDDEN_DIM, VALID, VALID_PATH, chance = CHANCE, logging = network_verbose)
		batchlist.append(valid)
		if verbose:
			print "Validation Error is: " + str(valid) + "For batch size: " + str(batch_size)
	valid_results['batch_sizes'] = batchlist
	with open(savepath, 'wb') as f:
		pickle.dump(valid_results, f)
	f.close()
	print "SAVED batches"

	hdimlist = []
	for hidden_dim in hidden_dims:
		train,valid = train_masked_lower_comb_network(EPOCHS, LRATE, BATCH_SIZE, LOGDIR, COLOUR_PATH, MNIST_PATH, hidden_dim, VALID, VALID_PATH, chance = CHANCE, logging = network_verbose)
		hdimlist.append(valid)
		if verbose:
			print "Validation Error is: " + str(valid) + "For hidden dim: " + str(hidden_dim)
	valid_results['hidden_dims'] = hdimlist
	with open(savepath, 'wb') as f:
		pickle.dump(valid_results, f)
	f.close()
	print "SAVED hidden dims"
	
	
	combdimlist = []
	for combdim in comb_dims:
		train,valid = train_masked_lower_comb_network(EPOCHS, LRATE, BATCH_SIZE, LOGDIR, COLOUR_PATH, MNIST_PATH, HIDDEN_DIM, VALID, VALID_PATH, chance = CHANCE, logging = network_verbose, comb_dim = combdim)
		combdimlist.append(valid)
		if verbose:
			print "Validation Error is: " + str(valid) + "For comb dim: " + str(combdim)
	valid_results['comb_dims'] = combdimlist
	with open(savepath+"2", 'wb') as f:
		pickle.dump(valid_results, f)
	f.close()
	print "SAVED combdims"

	actfunclist = []
	for act_func in act_funcs:
		train,valid = train_masked_lower_comb_network(EPOCHS, LRATE, BATCH_SIZE, LOGDIR, COLOUR_PATH, MNIST_PATH, HIDDEN_DIM, VALID, VALID_PATH, chance = CHANCE, logging = network_verbose, act_func = act_func)
		actfunclist.append(valid)
		if verbose:
			print "Validation Error is: " + str(valid) + "For act func: " + str(act_func)
	valid_results['act_funcs'] = actfunclist
	with open(savepath+"2", 'wb') as f:
		pickle.dump(valid_results, f)
	f.close()
	print "SAVED actfuncs"

	combfunclist = []
	for combfunc in comb_funcs:
		train, valid = train_masked_lower_comb_network(EPOCHS, LRATE, BATCH_SIZE, LOGDIR, COLOUR_PATH, MNIST_PATH, HIDDEN_DIM, VALID, VALID_PATH, chance = CHANCE, logging = network_verbose, comb_func = combfunc)
		combfunclist.append(valid)
		if verbose:
			print "Validation Error is: " + str(valid) + "For combfunc: " + str(combfunc)
	valid_results['com_funcs'] = combfunclist
	with open(savepath+"2", 'wb') as f:
		pickle.dump(valid_results, f)
	f.close()
	print "SAVED COMBFUNCS"

	

	
	#set up our automatic saving functionality, for ease of use

	
	#save
	with open(savepath, 'wb') as f:
		pickle.dump(valid_results, f)
	f.close()
	print "SAVED ALL"
		



def run_multiple_chances(chance_list):

	assert type(chance_list) == list and len(chance_list)>=1, 'list wrong'
	result_list = []
	for i in xrange(len(chance_list)):
		#the logging is going to be annoying here. you shuold prob just none it
		acc = train_masked_lower_comb_network(EPOCHS, LRATE, BATCH_SIZE, LOGDIR, COLOUR_PATH, MNIST_PATH, HIDDEN_DIM, VALID, VALID_PATH, chance = chance_list[i])
		result_list.append(acc)
	return result_list


def double_graph(save = True, savepath = None, validation = False):

	if savepath is None and save == True:
		savepath = 'res/autism_integration_synseparate_hdims'

	#we change this to simulate local overconnectivity in the autistic networks
	hdim_autism = 1000
	hdim_normal = 2000

	acc1,vlist1, acclist_autism, costlist_autism = train_double_network(EPOCHS, LRATE, BATCH_SIZE, LOGDIR, COLOUR_PATH, MNIST_PATH, hdim_autism, VALID, VALID_PATH, dropout = True, drop_rate = 0.01, logging = False, costlist = True)
	acc2,vlist2, acclist_normal, costlist_normal = train_double_network(EPOCHS, LRATE, BATCH_SIZE, LOGDIR, COLOUR_PATH, MNIST_PATH, hdim_normal, VALID, VALID_PATH, dropout = False, drop_rate = 0.1, logging = False, costlist = True)


	# accuracy graph
	fig1 = plt.figure()
	ax = fig1.add_subplot(1,1,1)
	x = range(EPOCHS//10)
	#print x
	#print acclist_autism
	#print len(x)
	#print len(acclist_autism)
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Training Accuracy")
	ax.set_title('Training accuracy of autistic vs neurotypical integrative network')
	
	#plot
	ax.plot(x, acclist_autism, label = "autism")
	ax.plot(x, acclist_normal, label = "neurotypical")	
	plt.legend()
	plt.show()

	if save:
		fig1.savefig(savepath + '_accuracy.png')
		plt.close(fig1)
	
	fig2 = plt.figure()
	ax = fig2.add_subplot(1,1,1)
	x = range(EPOCHS//10)
	#print x
	#print acclist_autism
	#print len(x)
	#print len(acclist_autism)
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Training Error")
	ax.set_title('Training error of autistic vs neurotypical integrative network')
	
	#plot
	ax.plot(x, costlist_autism, label = "autism")
	ax.plot(x, costlist_normal, label = "neurotypical")	
	plt.legend()
	plt.show()

	if save:
		fig2.savefig(savepath + '_error.png')
		plt.close(fig2)

	if validation:
		fig3 = validation_bar_chart(vlist1, vlist2)
		return fig1, fig2, fig3

	return fig1,fig2

def get_validation_mean(valid_list):
	endlist = []
	for i in xrange(len(valid_list)):
		mu = np.mean(valid_list[i])
		endlist.append(mu)
	mean = np.mean(endlist)
	return mean


def validation_bar_chart(valid_autism, valid_normal, save = True, savepath = None):

	if savepath is None and save == True:
		savepath = 'res/autism_integration_validation_bar_chart'

	mu_autism = get_validation_mean(valid_autism)
	mu_normal = get_validation_mean(valid_normal)
	vallist = [mu_autism, mu_normal]
	valpos = ['Autism', 'Neurotypical']
	x = np.arange(len(vallist))

	fig = plt.figure()
	plt.bar(x,vallist, align = 'center')
	plt.xticks(x,valpos)
	plt.xlabel('network type')
	plt.ylabel('validation accuracy')
	plt.title('Validation accuracies for autistic and control networks')
	plt.show()

	if save:
		fig.savefig(savepath)
		plt.close(fig)
	return fig


def plot_precision_variables(save = True, savepath = None, average = True):
	
	if savepath is None and save == True:
		savepath = 'res/graphs/autism_precision_variable_'

	vlist_mnist_autism, vlist_col_autism = train_double_network(EPOCHS, LRATE, BATCH_SIZE, LOGDIR, COLOUR_PATH, MNIST_PATH, HIDDEN_DIM, VALID, VALID_PATH, dropout = True, drop_rate = 0.1, variable_weights = True, logging = False)

	vlist_mnist_normal, vlist_col_normal = train_double_network(EPOCHS, LRATE, BATCH_SIZE, LOGDIR, COLOUR_PATH, MNIST_PATH, HIDDEN_DIM, VALID, VALID_PATH, dropout = False, drop_rate = 0.1, variable_weights = True, logging = False)

	x = range(len(vlist_mnist_autism))
	print len(x)

	if average:
		average_list_autism = []
		average_list_normal = []
		for i in xrange(len(vlist_mnist_autism)):
			avg_autism = (vlist_mnist_autism[i] + vlist_col_autism[i])/2
			avg_normal = (vlist_mnist_normal[i] + vlist_col_normal[i])/2	
			average_list_autism.append(avg_autism)
			average_list_normal.append(avg_normal)
		fig_avg = plt.figure()
		ax = fig_avg.add_subplot(1,1,1)
		ax.plot(x,average_list_autism, label='Precision autism')
		ax.plot(x,average_list_normal, label='Precision neurotypical')
		ax.set_xlabel('Epoch')
		ax.set_ylabel('Precision value')
		ax.set_title('Average Precision for autistic and neurotypical networks')
		plt.legend()
		plt.show()

	
	fig_mnist = plt.figure()
	ax = fig_mnist.add_subplot(1,1,1)
	ax.plot(x,vlist_mnist_autism, label='mnist precision autism')
	ax.plot(x,vlist_mnist_normal, label='mnist precision neurotypical')
	ax.set_xlabel('Epoch')
	ax.set_ylabel('Precision value')
	ax.set_title('MNIST precision value for autistic and neurotypical networks')
	plt.legend()
	plt.show()

	fig_colour = plt.figure()
	ax = fig_colour.add_subplot(1,1,1)
	ax.plot(x,vlist_col_autism, label='colour precision autism')
	ax.plot(x,vlist_col_normal, label='colour precision neurotypical')
	ax.set_xlabel('Epoch')
	ax.set_ylabel('Precision value')
	ax.set_title('Colour precision value for autistic and neurotypical networks')
	plt.legend()
	plt.show()

	if save:
		fig_mnist.savefig(savepath + 'mnist.png')
		plt.close(fig_mnist)
		fig_colour.savefig(savepath + 'colour.png')
		plt.close(fig_colour)
		if average:
			fig_avg.savefig(savepath + 'average.png')
			plt.close(fig_avg)

	if average:
		return fig_mnist, fig_colour, fig_avg
	return fig_mnist, fig_colour
	
	




def main():
	#for the single comb network
	acc = train_double_network(EPOCHS, LRATE, BATCH_SIZE, LOGDIR, COLOUR_PATH, MNIST_PATH, HIDDEN_DIM, VALID, VALID_PATH)

	# for the double comb network
	#acc = train_lower_comb_network(EPOCHS, LRATE, BATCH_SIZE, LOGDIR, COLOUR_PATH, MNIST_PATH, HIDDEN_DIM)

	#for the masked network
	#acc = train_masked_lower_comb_network(EPOCHS, LRATE, BATCH_SIZE, LOGDIR, COLOUR_PATH, MNIST_PATH, HIDDEN_DIM, VALID, VALID_PATH,chance =  CHANCE, logging = True)
	#save
	#save_results(acc, SAVE_PATH)

	#chancelist =[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
	#chancelist = [0, 0.1,0.3,0.5,0.7,0.9]
	#res_list = run_multiple_chances(chancelist)
	#acc = train_masked_lower_comb_network(EPOCHS, LRATE, BATCH_SIZE, LOGDIR, COLOUR_PATH, MNIST_PATH, HIDDEN_DIM, VALID, VALID_PATH, chance = CHANCE, logging = True)

	#save_results(res_list, SAVE_PATH)

	#lrates = [0.0001, 0.001,0.01,0.0005,0.00001,0.005]
	#batch_sizes = [10,30,50,100]
	#hidden_dims = [100,300,500,700,900,1500,2000,5000]
	#comb_dims = [200,600,1000,1500,2000,5000]
	#act_funcs = [tf.nn.sigmoid, tf.nn.tanh, tf.nn.relu, tf.identity, tf.nn.elu]
	#comb_funcs = [tf.add, tf.subtract, tf.multiply]
	#hyperparam_grid_search(lrates, batch_sizes, hidden_dims, comb_dims, act_funcs, comb_funcs,savepath = "res/synseparate_grid_search")




	#acc = train_double_network(EPOCHS, LRATE, BATCH_SIZE, LOGDIR, COLOUR_PATH, MNIST_PATH, HIDDEN_DIM, VALID, VALID_PATH, dropout = True, drop_rate = 0.1, variable_weights = True)
	#print acc

	#fig1,fig2, fig3 = double_graph(validation = True)

	#vlist_mnist, vlist_col = train_double_network(EPOCHS, LRATE, BATCH_SIZE, LOGDIR, COLOUR_PATH, MNIST_PATH, HIDDEN_DIM, VALID, VALID_PATH, dropout = True, drop_rate = 0.1, variable_weights = True)
	#print vlist_mnist
	#print "  "
	#print vlist_col

	plot_precision_variables()


if __name__ == '__main__':
	main()

