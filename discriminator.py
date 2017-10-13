#Author: Beren Millidge
#Msc Dissertation
#Summer 2017

from __future__ import division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from nn_funcs import *
from data_providers import *
from tensorflow.examples.tutorials.mnist import input_data
from transformers import *

class Discriminator_network(object):

	def __init__(self, data, learning_rate, batch_size, logdir = None, data_split = "invert", architecture = None, dropout = False, drop_rate = 0.5, drop_bool = True):
		self.learning_rate = learning_rate
		self.batch_size =batch_size
		self.data_split = data_split
		self.architecture = architecture
		self.logging = False

		#check architecture
		if self.architecture is None:
			self.inisialise_architecture()

		if logdir is not None:
			assert type(logdir) == str and len(logdir) >=1,'logdir must be valid string'
			self.logging = True
			self.logdir = logdir

	#check data
		self.data = data
	#if data is string, assume it's a file and load it
		if type(self.data) == str:
			assert len(self.data) >=1
			self.data = pickle.load(open(self.data,'rb'))

	#if it's a list of two - i.e. one labels, one inputs, we split it as such
		if type(self.data) == list and len(self.data) == 2:
			self.inputs1 = self.data[0]
			self.labels1 = self.data[1]
			#now we do the logic to split by whatever it says
			if self.data_split == "invert":
				self.inputs2, self.labels2 = self.invert_data(self.inputs1,self.labels1)

			if self.data_split == "split":
				self.inputs1, self.labels1, self.inputs2, self.labels2 = self.split_data_in_half(self.inputs1, self.labels1)

		if type(self.data) == list and len(self.data) ==4:
			self.inputs1 = self.data[0]
			self.labels1 = self.data[1]
			self.inputs2 = self.data[2]
			self.labels2 = self.data[3]

		self.input_dim = self.inputs1.shape[1]


		self.pairs = self.pair_function(self.labels1, self.labels2)
		#print self.pairs.shape
		#self.pair_dim = self.pairs.shape[1]

		#initialise our generic data provider (we're going to assume no custom one is provided lol atm
		self.data_provider = GenericDataProvider(data =[self.inputs1, self.inputs2,self.pairs], batch_size=self.batch_size)

		#initialise our placeholders
		self.x1 = tf.placeholder(tf.float32, [None, self.input_dim])
		self.x2 = tf.placeholder(tf.float32, [None, self.input_dim])
		self.y = tf.placeholder(tf.float32, [None,1])

		self.hidden_dim = self.architecture['hidden_dim']
		self.act_func =self.architecture['act_func']

		W1 = weight_variable((self.input_dim, self.hidden_dim))
		W2 = weight_variable((self.hidden_dim, self.hidden_dim))
		b1 = bias_variable([self.hidden_dim])
		b2 = bias_variable([self.hidden_dim])

		WD = weight_variable([self.hidden_dim, 1])
		bd = bias_variable([1])

		if dropout == False:
			with tf.name_scope("N1_L1"):
				h1_1 = self.act_func(tf.matmul(self.x1, W1) + b1)
			with tf.name_scope("N1_L2"):
				h1_2 = self.act_func(tf.matmul(h1_1, W2) + b2)

			with tf.name_scope("N2_L1"):
				h2_1 = self.act_func(tf.matmul(self.x2, W1) + b1)
			with tf.name_scope("N2_L2"):
				h2_2 = self.act_func(tf.matmul(h2_1, W2) + b2)

		if dropout == True:
			with tf.name_scope("N1_L1"):
				h1_1 = self.act_func(tf.matmul(self.x1, W1) + b1)
			with tf.name_scope("Drop_1_1"):
				drop1_1 = dropout_layer(h1_1, drop_rate, drop_bool, name = "dropout_N1_L1")
				
			with tf.name_scope("N1_L2"):
				h1_2 = self.act_func(tf.matmul(drop1_1, W2) + b2)
			with tf.name_scope("Drop_1_2"):
				drop1_2 = dropout_layer(h1_2, drop_rate, drop_bool, name = "dropout_N1_L2")

			with tf.name_scope("N2_L1"):
				h2_1 = self.act_func(tf.matmul(self.x2, W1) + b1)
			with tf.name_scope("Drop_1_2"):
				drop2_1 = dropout_layer(h2_1, drop_rate, drop_bool, name = "dropout_N2_L1")

			with tf.name_scope("N2_L2"):
				h2_2 = self.act_func(tf.matmul(drop2_1, W2) + b2)
			with tf.name_scope("Drop_2_2"):
				drop2_2 = dropout_layer(h2_2, drop_rate,drop_bool, name = "dropout_N2_L2")



		#we now compute the distance layer as follows
		if dropout == False:
			with tf.name_scope('distance_layer'):
				d = self.absolute_diff(h1_2, h2_2)
		if dropout == True:
			with tf.name_scope('distance_layer'):
				d = self.absolute_diff(drop1_2, drop2_2)

		with tf.name_scope('output_layer'):
			self.output = tf.identity(tf.matmul(d, WD) + bd)

		#print self.y.eval()
		#print self.output.eval()

		with tf.name_scope('cost'):
			#self.cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels =self.y, logits = self.output))
			self.cost = tf.reduce_sum(tf.square(self.y - self.output)) 

		with tf.name_scope('train'):
			self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

		with tf.name_scope('accuracy'):
			correct_prediction_mnist = tf.equal(tf.argmax(self.output,1), tf.argmax(self.y,1))
			self.accuracy = tf.reduce_sum(tf.cast(correct_prediction_mnist, tf.float32))



		#initialise our session
		self.sess = tf.InteractiveSession()
		init = tf.global_variables_initializer()
		self.sess.run(init)

		if self.logging:
			self.merged = tf.summary.merge_all()
			self.train_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)


	def printresults(self):
		print self.y
		print self.output
		return self.y

	def obtain_batch(self):
		output = self.output.eval()
		labels = self.y.eval()
		return output, labels

	def inisialise_architecture(self):
		self.architecture = {}
		self.architecture['hidden_dim'] = 500
		self.architecture['act_func'] = tf.nn.sigmoid


	def invert_data(self, data, labels):
		#data must be in standard mnist form - i.e. 2d array, length = examples, width = feature dimensions!
		flipped_data = np.flip(data,axis=0)
		flipped_labels = np.flip(labels,axis=0)
		return flipped_data, flipped_labels

	def split_data_in_half(self,data,labels):
		N = len(data)
		half1_data = data[0:N//2]
		half1_labels = labels[0:N//2]
		half2_data = data[N//2:N]
		half2_labels = data[N//2:N]
		return half1_data, half1_labels,half2_data, half2_labels

	def pair_function(self, labels1, labels2):
		N = len(labels1)
		assert N == len(labels2), ' lengths must be the same as logic is based on parrallel arrays'
		pairs =np.zeros(N)
		for i in xrange(N):
			if np.array_equal(labels1[i],labels2[i]):
				pairs[i] =1
		#print "PAIRS"
		#print pairs.shape
		return pairs 

	def Koch_distance(self, l1, l2):
		#I'm calling it the koch distance as it's mr gregory koch whose msc thesis I'm getting this from: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf 
		print "KOCH DISTANCE NOT IMPLEMENTED"
		pass # this gets you an output immediately, which is pretty dangeorous, I do say. so idk

	def absolute_diff(self,l1, l2):
		return tf.abs(tf.subtract(l1,l2))

	def square_distance(self,l1,l2):
		return tf.squre(tf.subtract(l1,l2))
	
	def get_acc(self,output, labels):
		N = len(output)
		assert N == len(labels)
		total = 0
		for i in xrange(N):
			if output[i] > 0.5 and labels[i] == 1:
				total +=1
			if output[i] <=0.5 and labels[i] == 0:
				total +=1
		acc = total/N
		return acc


	def test(self, x1,labels1, x2,labels2):
		N = len(x1)
		assert N == len(x2),'must be same length'
		assert N == len(labels1), 'must be same length'
		pairs = self.pair_function(labels1, labels2)
		total = 0
		for i in xrange(N):
			x1s = x1[i]
			x2s = x2[i]
			x1s = np.reshape(x1s, [1, 784])
			x2s = np.reshape(x2s, [1,784])
			ys = pairs[i]
			out = self.sess.run(self.output, feed_dict={self.x1:x1s, self.x2:x2s})
			if out > 0.5 and ys ==1:
				total +=1
			if out <0.5 and ys == 0:
				total +=1
		acc = total/N
		print acc
		return acc


	def train(self, epochs, batch_size = None, epoch_print =10, logging = False):
		if batch_size is None:
			batch_size =self.data_provider.batch_size

		num_batches = len(self.inputs1)//batch_size
		print num_batches
		#print self.data_provider.get_type()
		if logging == True:
			costlist = []
			acclist = []

		for i in xrange(epochs):
			#print "Epoch " + str(i)
			for j in xrange(num_batches):
				#print self.data_provider
				x1,x2,ys = self.data_provider.next(batch_size)
				#print ys
				ys = np.reshape(ys, [batch_size,1]) 
				feeddict = {self.x1:x1, self.x2:x2, self.y:ys}
				#if i % epoch_print ==0 and j ==1:
				if j ==1 and i % epoch_print ==0:
					cost, _= self.sess.run([self.cost, self.train_step],feed_dict=feeddict)
					
					output = self.sess.run(self.output, feed_dict=feeddict)
					acc = self.get_acc(output, ys)
					
					print "Epoch: " + str(i) + " Cost: " + str(cost) + "Accuracy: " + str(acc)
					print "   "
					print output
					if logging == True:
						costlist.append(cost)
						acclist.append(acc)
					
					
					#print ys


				_ = self.sess.run(self.train_step,feed_dict=feeddict)
		if logging == True:
			return costlist, acclist
		if logging == False:
			return cost, acc

def dropout_grid_search(droplist, save = True, savepath = None):

	print "BEGINNING GRID SEARCH"
	N = len(mnist.test.images)//2
	#train data
	data_mnist_train = mnist.train.images
	data_mnist_labels = mnist.train.labels

	#test data
	data_mnist_test1 = mnist.test.images[0:N]
	labels_mnist_test1 = mnist.test.labels[0:N]
	data_mnist_test2 = mnist.test.images[N:len(mnist.test.images)]
	labesl_mnist_test2 = mnist.test.labels[N:len(mnist.test.images)]
	testlist = []
	costslist = []
	accslist = []

	#set up default savepath
	if save == True and savepath == None:
		savepath = 'res/dropout_grid_search'


	for dropout in droplist:
		disc = Discriminator_network([data_mnist_train,data_mnist_labels],0.001,50, dropout =True, drop_rate = dropout)
		costlist, acclist = disc.train(100, 50)

		acc = disc.test(data_mnist_test1,labels_mnist_test1,data_mnist_test2,labesl_mnist_test2)
		testlist.append(acc)	
		costslist.append(costlist)
		accslist.append(acclist)

	
	#save functionality
	if save == True:
		with open(savepath + "_testlist", 'wb') as f:
			pickle.dump(testlist, f)
		f.close()
		with open(savepath + "_costslist", 'wb') as f:
			pickle.dump(costslist, f)
		f.close()
		with open(savepath + "_accslist", 'wb') as f:
			pickle.dump(accslist, f)
		f.close()
		print "SAVED"
	return testlist, costslist, accslist
	

#droplist = [0,0.001,0.01,0.1,0.2,0.3,0.4,0.5]
#dropout_grid_search(droplist)


def acc_with_img_split(disc,data, shifts, savepath = None, save = True):
	#set up default savepath
	if savepath == None and save == True:
		savepath = 'res/img_shift_siamese'
	
	N = len(data)//5 #we're setting 5 here arbitrarily, might not be worthwhile
	test1 = data[0:N]
	test2 = data[N:2*N]
	test3 = data[2*N:3*N]
	test4 = data[3*N:4*N]
	test5 = data[4*N:5*N]

	shift1 = gen_image_shift(test1, shifts[0])
	shift2 = gen_image_shift(test2, shifts[1])
	shift3 = gen_image_shift(test3, shifts[2])
	shift4 = gen_image_shift(test4, shifts[3])
	shift5 = gen_image_shift(test5, shifts[4])

	labels = np.ones(N)

	

	acc1 = disc.test(test1, labels, shift1, labels)
	acc2 = disc.test(test2, labels, shift2, labels)
	acc3 = disc.test(test3, labels, shift3, labels)
	acc4 = disc.test(test4, labels, shift4, labels)
	acc5 = disc.test(test5, labels, shift5, labels)

	acclist = [acc1, acc2, acc3,acc4,acc5]
	
	if save == True:
		with open(savepath, 'wb') as f:
			pickle.dump(acclist, f)
		f.close()
		print "SAVED"
	return acclist

def acc_with_img_rotate(disc,data, angles, savepath = None, save = True):
	#set up default savepath
	if savepath == None and save == True:
		savepath = 'res/img_rotate_siamese'
	
	N = len(data)//6 #we're setting 5 here arbitrarily
	test1 = data[0:N]
	test2 = data[N:2*N]
	test3 = data[2*N:3*N]
	test4 = data[3*N:4*N]
	test5 = data[4*N:5*N]
	test6 = data[5*N:6*N]

	shift1 = gen_image_rotate(test1, angles[0])
	shift2 = gen_image_rotate(test2, angles[1])
	shift3 = gen_image_rotate(test3, angles[2])
	shift4 = gen_image_rotate(test4, angles[3])
	shift5 = gen_image_rotate(test5, angles[4])
	shift6 = gen_image_rotate(test6, angles[5])

	labels = np.ones(N)

	

	acc1 = disc.test(test1, labels, shift1, labels)
	acc2 = disc.test(test2, labels, shift2, labels)
	acc3 = disc.test(test3, labels, shift3, labels)
	acc4 = disc.test(test4, labels, shift4, labels)
	acc5 = disc.test(test5, labels, shift5, labels)
	acc6 = disc.test(test6, labels, shift6, labels)

	acclist = [acc1, acc2, acc3,acc4,acc5, acc6]
	
	if save == True:
		with open(savepath, 'wb') as f:
			pickle.dump(acclist, f)
		f.close()
		print "SAVED"
	return acclist


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
data_mnist_train = mnist.train.images
data_mnist_labels = mnist.train.labels
logdir = 'tmp/tensorflow/vae/myvae_test'

disc = Discriminator_network([data_mnist_train,data_mnist_labels],0.01,50, dropout =True, drop_rate = 0.01)
disc.train(10, 50)

N = len(mnist.test.images)//2
data_mnist_test1 = mnist.test.images[0:N]
labels_mnist_test1 = mnist.test.labels[0:N]
data_mnist_test2 = mnist.test.images[N:len(mnist.test.images)]
labesl_mnist_test2 = mnist.test.labels[N:len(mnist.test.images)]
acc = siam.test(data_mnist_test1,labels_mnist_test1,data_mnist_test2,labesl_mnist_test2)
print acc

def test_autism_histogram(shifts, save = True, savepath = None):

	if savepath == None and save == True:
		savepath = 'res/img_shift_siamese_test2'
	
	#get data
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	data_mnist_train = mnist.train.images
	data_mnist_labels = mnist.train.labels
	data_mnist_test = mnist.test.images
	logdir = 'tmp/tensorflow/vae/siamese_test'

	#train networks and get results
	#autism
	disc = Discriminator_network([data_mnist_train,data_mnist_labels],0.005,50, dropout =True, drop_rate = 0.05)
	disc.train(200, 50)
	acclist_autism = acc_with_img_split(disc,data_mnist_test,shifts, save = False)

	disc = Discriminator_network([data_mnist_train,data_mnist_labels],0.005,50, dropout =False, drop_rate = 0.5)
	disc.train(200, 50)
	acclist_normal = acc_with_img_split(disc,data_mnist_test,shifts, save = False)

	finalarr = [acclist_autism, acclist_normal]
	finalarr = np.array(finalarr)

	#plot our actual histogram here! this is kind of cool if it ever works. It will be fucking amazing. oh well, could be cool!
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	N = range(len(acclist_autism))
	w = 0.4

	#we don't want this, we want the pixel shifts on the x axis, so let's get them!
	shiftlist = []	
	for shift in shifts:
		shiftlist.append(shift[1])


	
	ax.bar(shiftlist, acclist_autism, w, color='blue', label ='neurotypical', alpha = 0.5)
	ax.bar([shift + w for shift in shiftlist], acclist_normal, w, color='red', label ='autism', alpha = 0.5)
	ax.set_xlabel("pixel shift")
	ax.set_ylabel("proportion correct")
	ax.set_title("test histogram")
	plt.legend()
	plt.show()

	if save == True:
		fig.savefig(savepath)
		plt.close(fig)

	return finalarr, fig

def discriminator_validation_histogram(save = True, savepath = None):
	if savepath == None and save == True:
		savepath = 'res/discriminator_validation_histogram'

	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	data_mnist_train = mnist.train.images
	data_mnist_labels = mnist.train.labels
	data_mnist_test = mnist.test.images
	labels_mnist_test = mnist.test.labels
	logdir = 'tmp/tensorflow/vae/discriminator_test'
	
	N = len(data_mnist_test)//2

	test1_data = data_mnist_test[0:N]
	test2_data = data_mnist_test[N:2*N]

	labels1 = labels_mnist_test[0:N]
	labels2 = labels_mnist_test[N:2*N]

	#train networks and get results
	#autism
	disc = Discriminator_network([data_mnist_train,data_mnist_labels],0.005,50, dropout =True, drop_rate = 0.05)
	disc.train(10, 50)
	acc_autism = disc.test(test1_data, labels1, test2_data, labels2)

	disc = Discriminator_network([data_mnist_train,data_mnist_labels],0.005,50, dropout =False, drop_rate = 0.5)
	disc.train(10, 50)
	acc_normal = disc.test(test1_data, labels1, test2_data, labels2)

	res = [acc_autism, acc_normal]
	vals = ['Autism', 'Neurotypical']

	x = np.arange(len(res))

	fig = plt.figure()
	plt.bar(x,res, align = 'center')
	plt.xticks(x,vals)
	plt.xlabel('Network Type')
	plt.ylabel('Validation Accuracy')
	plt.title('Validation accuracies for autistic and control networks')
	plt.show()

	if save:
		fig.savefig(savepath)
		plt.close(fig)
	return fig

#discriminator_validation_histogram()
	

def test_autism_histogram_rotate(angles, save = True, savepath = None):


	if savepath == None and save == True:
		savepath = 'res/img_shift_discriminator_test_angle'
	
	#get data
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	data_mnist_train = mnist.train.images
	data_mnist_labels = mnist.train.labels
	data_mnist_test = mnist.test.images
	logdir = 'tmp/tensorflow/vae/discriminator_test'

	#train networks and get results
	#autism
	disc = Discriminator_network([data_mnist_train,data_mnist_labels],0.005,50, dropout =True, drop_rate = 0.05)
	disc.train(200, 50)
	acclist_autism = acc_with_img_rotate(disc,data_mnist_test,angles, save = False)

	disc = Discriminator_network([data_mnist_train,data_mnist_labels],0.005,50, dropout =False, drop_rate = 0.5)
	disc.train(200, 50)
	acclist_normal = acc_with_img_rotate(disc,data_mnist_test,angles, save = False)

	finalarr = [acclist_autism, acclist_normal]
	finalarr = np.array(finalarr)

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	N = range(len(acclist_autism))
	w = 0.8

	#shiftlist = []	
	#for shift in shifts:
		#shiftlist.append(shift[1])


	
	ax.bar(angles, acclist_autism, w, color='blue', label ='autism', alpha = 0.5)
	ax.bar([angle + w for angle in angles], acclist_normal, w, color='red', label ='neurotypical', alpha = 0.5)
	ax.set_xlabel("Rotation Angle")
	ax.set_ylabel("Proportion correct")
	ax.set_title("Rotation sensitivity of autistic vs neurotypical discriminator networks")
	plt.legend()
	plt.show()

	if save == True:
		fig.savefig(savepath)
		plt.close(fig)

	return finalarr, fig

#shifts = [[0,1],[0,2],[0,3],[0,4],[0,5]]
#finalarr, fig = test_autism_histogram(shifts)

def get_train_comparison_graphs(savepath = None, save = True, graph = True):
	if savepath == None and save == True:
		savepath = 'res/disc_comparison'
	
	#get data
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	data_mnist_train = mnist.train.images
	data_mnist_labels = mnist.train.labels
	data_mnist_test = mnist.test.images
	logdir = 'tmp/tensorflow/vae/discriminator_test'

	#train networks and get results
	#autism
	disc = Discriminator_network([data_mnist_train,data_mnist_labels],0.005,50, dropout =True, drop_rate = 0.05)
	costlist_autism, acclist_autism = disc.train(200, 50, logging = True, epoch_print =10)

	disc = Discriminator_network([data_mnist_train,data_mnist_labels],0.005,50, dropout =False, drop_rate = 0.5)
	costlist_normal, acclist_normal = disc.train(200, 50, logging = True, epoch_print = 10)

	#we save our data in the usual fashion
	res = [costlist_autism, acclist_autism, costlist_normal, acclist_normal]
	
	if save:
		with open(savepath + '_results', 'wb') as f:
			pickle.dump(res, f)
		f.close()
		print "SAVED"

	if graph:
		#graph 1 - costs
		assert len(costlist_autism) == len(acclist_autism) == len(costlist_normal) == len(acclist_normal), 'all lists should be same length!'

		x = range(len(costlist_autism))

		fig1 = plt.figure()
		ax = fig1.add_subplot(1,1,1)	
		ax.plot(x, costlist_autism, label = 'autism', color = 'red')
		ax.plot(x, costlist_normal, label = 'neurotypical', color = 'blue')
		ax.set_xlabel('Epoch')
		ax.set_ylabel('Error')
		ax.set_title('Training error of autism vs neurotypical discriminator networks')
		plt.legend()
		plt.show()

		if save:
			fig1.savefig(savepath+ '_graph_cost')
			plt.close(fig1)

		#graph 2 - accs
		fig2 = plt.figure()
		ax = fig2.add_subplot(1,1,1)	
		ax.plot(x, acclist_autism, label = 'autism', color = 'red')
		ax.plot(x, acclist_normal, label = 'neurotypical', color = 'blue')
		ax.set_xlabel('Epoch')
		ax.set_ylabel('Accuracy')
		ax.set_title('Training accuracy of autism vs neurotypical discriminator networks')
		plt.legend()
		plt.show()

		if save:
			fig2.savefig(savepath+ '_graph_accuracy')
			plt.close(fig2)

		figs = [fig1, fig2]
		return figs, res


		

#get_train_comparison_graphs()
	


#angles = [-20,-10,-5,5,10,20]
#finalarr, fig = test_autism_histogram_rotate(angles)
#with open('res/autism_discriminator_histogram_rotate2','wb') as f:
#	pickle.dump(finalarr, f)
#
#f.close()





