#Author: Beren Millidge
#Msc Dissertation
#Summer 2017

#These networks are no longer used.

from __future__ import division
import numpy as np
import tensorflow as tf
import cPickle as pickle
from data_providers import *
from nn_funcs import *


def generate_data_for_integrated(len_data, num_datapoints):
	data1 = []
	data2 = []
	for i in xrange(num_datapoints):
		dat1 = np.random.choice(2, [len_data])
		dat2 = np.random.choice(2, [len_data])
		data1.append(dat1)
		data2.append(dat2)
	data1 = np.array(data1)
	data2 = np.array(data2)
	return data1, data2

def labels_xor_function(datalist):
	d1 = datalist[0]
	d2 = datalist[1]
	labels = []
	assert len(d1) == len(d2)
	for i in xrange(len(d1)):
		labellist = []
		for j in xrange(len(d1[i])):
			if d1[i][j] ==  d2[i][j]:
				labellist.append(1)
			else:
				labellist.append(0)
		labellist = np.array(labellist)
		labels.append(labellist)
	labels = np.array(labels)
	return labels

def labels_or_function(datalist):
	d1 = datalist[0]
	d2 = datalist[1]
	#print d1
	#print d1[0]
	labels = []
	assert len(d1) == len(d2)
	for i in xrange(len(d1)):
		labellist = []
		for j in xrange(len(d1[i])):
			if d1[i][j] == 0 and  d2[i][j] ==0:
				labellist.append(0)
			else:
				labellist.append(1)
		labellist = np.array(labellist)
		labels.append(labellist)
	labels = np.array(labels)
	return labels

def labels_and_function(datalist):
	d1 = datalist[0]
	d2 = datalist[1]
	labels = []
	assert len(d1) == len(d2)
	for i in xrange(len(d1)):
		labellist = []
		for j in xrange(len(d1[i])):
			if d1[i][j] == 1 and  d2[i][j] ==1:
				labellist.append(1)
			else:
				labellist.append(0)
		labellist = np.array(labellist)
		labels.append(labellist)
	labels = np.array(labels)
	return labels

def labels_implication_function(datalist):
	d1 = datalist[0]
	d2 = datalist[1]
	labels = []
	assert len(d1) == len(d2)
	for i in xrange(len(d1)):
		labellist = []
		for j in xrange(len(d1[i])):
			if d1[i][j] == 1 and  d2[i][j] ==0:
				labellist.append(0)
			else:
				labellist.append(1)
		labellist = np.array(labellist)
		labels.append(labellist)
	labels = np.array(labels)
	return labels

def save_generated_data(savepath, datalist, labels):
	assert type(savepath) == str, ' must be str'
	with open(savepath + "_dat1", 'wb') as f:
		pickle.dump(datalist[0],f)
	f.close()
	with open(savepath + "_dat2", 'wb') as f:
		pickle.dump(datalist[1], f)
	f.close()
	with open(savepath + "_labels", 'wb') as f:
		pickle.dump(labels, f)
	f.close()
	print "SAVED!"

def save_invert(savepath, data, labels):
	assert type(savepath) == str, ' must be str'
	with open(savepath + "_invert_data", 'wb') as f:
		pickle.dump(data,f)
	f.close()
	with open(savepath + "_invert_labels", 'wb') as f:
		pickle.dump(labels, f)
	f.close()
	print "SAVED!"


def invert_generated_data(data):
	labels = []
	for i in xrange(len(data)):
		labellist = []
		for j in xrange(len(data[0])):
			if data[i][j] ==1:
				labellist.append(0)
			if data[i][j] == 0:
				labellist.append(1)
		labellist = np.array(labellist)
		labels.append(labellist)
	labels = np.array(labels)
	return labels
	



#dlist = generate_data_for_integrated(20,10000)

#labels = invert_generated_data(dlist[0])

#print dlist[0]
#print "  "
#print dlist[1]
#print "  "
#print labels
#savepath = 'res_pred/gen_invert'
#save_invert(savepath, dlist[0], labels)
#save_generated_data(savepath, dlist, labels) 


def integration_network(savepath):
	#get our data
	d1 = pickle.load(open(savepath + "_dat1",'rb'))
	d2 = pickle.load(open(savepath + "_dat2",'rb'))
	len_data = len(d1[0])
	labels = pickle.load(open(savepath + "_labels",'rb'))

	#set up our dataprovider - luckily our siamese dataprovider will actually work for this, I'm fairly sure!
	dlist = [d1, d2,labels]
	provider = GenericDataProvider(dlist, batch_size = 50, shuffle = False)

	sess = tf.InteractiveSession()

	
	#placeholders
	x1 = tf.placeholder(tf.float32, shape = [None, len_data])
	x2 = tf.placeholder(tf.float32, shape = [None, len_data])
	y_ = tf.placeholder(tf.float32, shape=[None, len_data])

	comb1 = combination_layer([x1,x2], [len_data, len_data], 50, 'combination_layer', act_func = tf.sigmoid)
	h1 = fc_layer(comb1, 50, 50, 'middle_layer', act_func = tf.sigmoid)
	y = fc_layer(h1, 50, len_data, 'output_layer', act_func = tf.identity)

	with tf.name_scope('cross_entropy'):	
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))
		tf.summary.scalar('cross_entropy', cross_entropy)
#and our training step
	with tf.name_scope('train'):
		train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

	log_dir = 'tmp/predictive/integration'
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(log_dir, sess.graph)
	tf.global_variables_initializer().run()

	epochs =100000
	for i in xrange(epochs):
		if i %10 ==0: # we save every 10 epochs
			run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
			run_metadata = tf.RunMetadata()
			x1s, x2s, labels = provider.next()
			feeddict = {x1:x1s, x2: x2s, y_: labels}
			summary, _, cost = sess.run([merged, train_step, cross_entropy], feed_dict = feeddict, options = run_options, run_metadata = run_metadata)
			train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
			train_writer.add_summary(summary,i)
			print('Adding run metadata for', i)
			print cost

	train_writer.close()


#integration_network(savepath)

def inverter_network(savepath):
	d1 = pickle.load(open(savepath + "_invert_data",'rb'))
	len_data = len(d1[0])
	labels = pickle.load(open(savepath + "_invert_labels",'rb'))

	dlist = [d1, labels]
	provider = GenericDataProvider(dlist, batch_size = 50, shuffle = False)

	sess = tf.InteractiveSession()


	
	#placeholders
	x1 = tf.placeholder(tf.float32, shape = [None, len_data])
	y_ = tf.placeholder(tf.float32, shape=[None, len_data])
	
	#layers
	h1 = fc_layer(x1, len_data, 20, 'h1', tf.nn.sigmoid)
	y = fc_layer(h1, 20, len_data, 'output', tf.identity)

	with tf.name_scope('cross_entropy'):	
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))
		tf.summary.scalar('cross_entropy', cross_entropy)
#and our training step
	with tf.name_scope('train'):
		train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

	log_dir = 'tmp/predictive/inversion'
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(log_dir, sess.graph)
	tf.global_variables_initializer().run()

	epochs =100000
	for i in xrange(epochs):
		if i %10 ==0: # we save every 10 epochs
			run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
			run_metadata = tf.RunMetadata()
			x1s, labels = provider.next()
			feeddict = {x1:x1s, y_: labels}
			summary, _, cost = sess.run([merged, train_step, cross_entropy], feed_dict = feeddict, options = run_options, run_metadata = run_metadata)
			train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
			train_writer.add_summary(summary,i)
			print('Adding run metadata for', i)
			print cost

	train_writer.close()


#inverter_network(savepath)


