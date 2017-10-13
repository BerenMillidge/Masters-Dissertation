#Author: Beren Millidge
#Msc Dissertation
#Summer 2017

from __future__ import division
import tensorflow as tf
import numpy as np

def variable_variable(init_val):
	return tf.Variable(tf.to_float(init_val))

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def xavier_init(fan_in, fan_out, constant=1):
	low = -constant*np.sqrt(6.0/(fan_in+fan_out))
	high = constant*np.sqrt(6.0/(fan_in+fan_out))
	return tf.random_uniform((fan_in, fan_out), minval = low, maxval=high, dtype=tf.float32)

def xavier_weight(shape):
	return tf.Variable(xavier_init(shape[0], shape[1]))


# NN Layers:

def fc_layer(input_tensor, input_dim, output_dim, layer_name, act_func=tf.nn.relu):
	with tf.name_scope(layer_name):
		with tf.name_scope('weights'):
			weights = weight_variable([input_dim, output_dim])
			variable_summaries(weights)
		with tf.name_scope('biases'):
			biases = bias_variable([output_dim])
			variable_summaries(biases)
		with tf.name_scope('pre_nonlinearity_activations'):
			pre = tf.matmul(input_tensor, weights) + biases
			tf.summary.histogram('pre_activations', pre)
		acts = act_func(pre, name='activations')
		tf.summary.histogram('activations', acts)
		return acts


def combination_layer(input_tensors, input_dims, output_dim, layer_name, act_func=tf.nn.relu, comb_func=tf.add):
	assert type(input_tensors) == list and len(
		input_tensors) >= 1, 'input tensors should be a list with length greater than or equal to 1'
	assert type(input_dims) == list and len(
		input_dims) >= 1, 'input dims should be a list with length greater than or equal to one' 
	assert type(
		layer_name) == str, 'layer name must be a a string'

	assert len(input_tensors) == len(input_dims), 'your lengths are mismatched' 

	with tf.name_scope(layer_name):
		activations = []  
		for i in xrange(len(input_tensors)):
			with tf.name_scope('comb_' + str(i)):
				with tf.name_scope('weights_' + str(i)):
					weights = weight_variable([input_dims[i], output_dim])
					variable_summaries(weights)

				with tf.name_scope('biases_' + str(i)):
					biases = bias_variable([output_dim])
					variable_summaries(biases)
				with tf.name_scope('pre_nonlinearity_activations_' + str(i)):
					pre = tf.matmul(input_tensors[i], weights) + biases
					tf.summary.histogram('pre_activations_' + str(i), pre)
					activations.append(pre)
		with tf.name_scope('combining'):
			result = activations[0]
			for i in xrange(len(activations) - 1):
				result = comb_func(result, activations[i + 1])
		with tf.name_scope('nonlinearity'):
			ys = act_func(result, name='activations')
			tf.summary.histogram('activations', ys)
		return ys


def masked_comb_layer(input_tensors, input_dims, output_dim, layer_name, mask_flags=None, mask_list=None,
					  mask_func=None, chance=None, act_func=tf.nn.relu, comb_func=tf.add):

	assert type(input_tensors) == list and len(
		input_tensors) >= 1, 'input tensors should be a list with length greater than or equal to 1'
	assert type(input_dims) == list and len(
		input_dims) >= 1, 'input dims should be a list with length greater than or equal to one'
	assert type(layer_name) == str, 'layer name must be a a string'
	assert len(input_tensors) == len(input_dims), 'your lengths are mismatched' 

	# we set up the mutually exclusive asserts for error checking
	if mask_flags:
		assert not (mask_func == None and mask_list == None), 'you need at least the function or the list'
		assert type(mask_flags) == list and len(mask_flags) >= 1, ' mask flags must be list of booleans'

	if not mask_list == None:
		assert mask_func == None, 'you cannot have both a list and the function'
		assert type(mask_list) == list and len(mask_list) >= 1, ' mask list must be list greater length than 1'
		assert len(mask_flags) == len(mask_list) == len(input_tensors), ' all these must be of same length'
		assert mask_flags, ' if you are going to have masks you need the flags'
	if not mask_func == None:
		assert mask_list == None, ' you canot have both the list and the function'
		assert chance <= 1 and chance >= 0, 'chance must exist and be a probability'
		assert mask_flags, ' if you are going to have masks you need the flags'

	with tf.name_scope(layer_name):
		activations = []  
		for i in xrange(len(input_tensors)):
			with tf.name_scope('comb_' + str(i)):
				with tf.name_scope('weights_' + str(i)):
					weights = weight_variable([input_dims[i], output_dim])
					variable_summaries(weights)

				with tf.name_scope('biases_' + str(i)):
					biases = bias_variable([output_dim])
					variable_summaries(biases)
				with tf.name_scope('pre_nonlinearity_activations_' + str(i)):
					if mask_flags:
						with tf.name_scope('apply_mask'):
							if mask_flags[i] == 1:
								if mask_list:
									mask = mask_list[i]
									weights = tf.multiply(mask,weights) 
								if mask_func:
									mask = mask_func(input_dims[i], output_dim, chance)
									weights = tf.multiply(mask, weights)
					pre = tf.matmul(input_tensors[i], weights) + biases
					tf.summary.histogram('pre_activations_' + str(i), pre)
					activations.append(pre)
		with tf.name_scope('combining'):
			result = activations[0]
			for i in xrange(len(activations) - 1):
				result = comb_func(result, activations[i + 1])
		with tf.name_scope('nonlinearity'):
			ys = act_func(result, name='activations')
			tf.summary.histogram('activations', ys)
		return ys


def dropout_layer(inputs, droprate, drop_bool, name, drop_shape = None, seed = None):
	with tf.name_scope(name):
		drop = tf.layers.dropout(inputs, droprate, noise_shape = drop_shape, seed = seed, training = drop_bool, name = name)
	return drop



# Cost functions:

def cross_entropy(y_, y):
	with tf.name_scope('cross_entropy'):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
		tf.summary.scalar('cross_entropy', cross_entropy)
		return cross_entropy


def least_squares(y_, y):
	with tf.name_scope('least_squares'):
		lsq = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_)))
		tf.summary.scalar('least_squares, lsq')
		return lsq


# Regularisers .. 

# This is experimental, might not work
def l1_regulariser(param):
	l1_regulariser = tf.contrib.layers.l1_regularizer(scale=param, scope=None)
	all_vars = tf.trainable_variables()
	reg_penalty = tf.contrib.layers.apply_regularization(l1_regulariser, all_vars)
	return reg_penalty



# Feed dict: - remember the differnt name. this is for namespace reasons
def FeedDict(dataproviders, placeholders):
	assert type(dataproviders) == list and len(
		dataproviders) >= 1, 'you need a list of dataproviders with 1 or more length'
	assert type(placeholders) == list and len(
		placeholders) >= 1, 'you need a list of placeholders with 1 or more length'
	assert len(dataproviders) == len(
		placeholders) / 2, 'your dataprovider and placeholder lists must be the same length otherwise it will not work, obviously' 
	feeddict = {}
	for i in xrange(len(dataproviders)):
		x, y = dataproviders[
			i].next()  
		feeddict[placeholders[2 * i]] = x
		feeddict[placeholders[(2 * i) + 1]] = y
	return feeddict


# MASKS


def generate_random_mask(input_dim, output_dim, chance_on):

	assert chance_on >= 0 and chance_on <= 1, 'probability must be between 0 and 1'

	# initialise array
	mask = np.zeros((input_dim, output_dim))
	# for the loop
	for i in xrange(input_dim):
		for j in xrange(output_dim):
			rand = np.random.uniform(0, 1)
			if rand < chance_on:
				mask[i][j] == 1

	mask = np.array(mask)
	mask = np.ndarray.astype(mask, dtype ='float32') # tensorflow arrays are float32, so we need to  do thisto match, even though it's binary!
	return mask


# Variable Summaries
def variable_summaries(var):
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
	with tf.name_scope('stddev'):
		stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
	tf.summary.scalar('stddev', stddev)
	tf.summary.scalar('max', tf.reduce_max(var))
	tf.summary.scalar('min', tf.reduce_min(var))
	tf.summary.histogram('histogram', var)


def colour_checker(num):
	if num == 0:
		return "Red"
	if num == 1:
		return "Orange"
	if num == 2:
		return "Yellow"
	if num == 3:
		return "Green"
	if num == 4:
		return "Cyan"
	if num == 5:
		return "Blue"
	if num == 6:
		return "Purple"
	if num == 7:
		return "Pink"
	if num == 8:
		return "INDIGO - SOMETHING HAS GONE WEIRD"  # as indigo isn't implemented yet


