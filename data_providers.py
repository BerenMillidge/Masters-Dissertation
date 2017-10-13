#Author: Beren Millidge
#Msc Dissertation
#Summer 2017

#Data provider helper class

import numpy as np
import pickle

class DataProvider(object):
	
	def __init__(self,inputs, labels, batch_size,shuffle = False):
		self.inputs = inputs
		self.labels = labels
		self.batch_size = batch_size
		assert type(self.batch_size) == int and self.batch_size >0, 'batch sizes must be an integer greater than 0'
		
		self.current_batch = 0 
		assert len(self.inputs) == len(self.labels), 'inputs and targets not of same length!'
		self.max_batches = len(self.inputs)/self.batch_size 
		self.shuffle = shuffle

	def new_epoch(self):
		self.current_batch = 0	#reset this
		#this shuffles randomly the order of each datapoint in the array
		if self.shuffle:
			new_indices = np.random.choice(len(self.inputs),len(self.inputs),replacement=False)
			new_inputs = []
			new_labels = []
			for index in new_indices:
				new_inputs.append(self.inputs[index])
				new_labels.append(self.labels[index])
			self.inputs = np.array(new_inputs)
			self.labels = np.array(new_labels)
			
			
	
	def next(self):
		#if self.current_batch + 1 > self.max_batches:
            #when we reach the end of the max number of batches
            		#self.new_epoch()
            		#raise StopIteration()
	
		batch_slice = slice(self.current_batch * self.batch_size, (self.current_batch+1)*self.batch_size)
		inputs_batch = self.inputs[batch_slice]
		targets_batch = self.labels[batch_slice]
		self.current_batch +=1
		return inputs_batch, targets_batch

class GenericDataProvider(object):

	def __init__(self, data = None, batch_size = None, shuffle = False):

		#assign our standard variables
		self.current_batch =0
		assert shuffle == False or shuffle == True, 'shuffle must be boolean variable'
		self.shuffle = shuffle

		#initialise flags
		self.supervised = False
		self.unsupervised = False
		self.siamese = False

		if batch_size:
			assert type(batch_size) == int and batch_size >=1, 'invalid batch size; must be an integer greater than or equal to one'
			self.batch_size = batch_size


		assert data != None,' you need to input some data here to the dataprovider else it will fail badly'

		if type(data) == str:
			assert len(data) >=1, 'string must be of length 1 or more'
			#assume if it's a string it's a file, try to load it
			print "Loading data from path"
			try:
				data = pickle.load(open(data, 'rb'))
				print "Loaded data"
			except:
				print "Data load failed"

		if type(data) == list and len(data) ==2:
			#in this case we're going to assume it's a data and labels thing. so we'll do this, and we're going to assume they are numpy arrays.
			self.data = data[0]
			self.labels = data[1]
			#check the types are correct
			assert type(self.data) is np.ndarray and type(self.labels) is np.ndarray, 'your data must be numpy arrays'
			assert len(self.data) == len(self.labels), 'data and labels must be same length!'
			self.supervised = True

		if type(data) is np.ndarray:
			self.data = data
			self.unsupervised = True

		#check to see if the data is in the siamese configuration
		if type(data) == list and len(data) == 3:
			self.data = data[0]
			self.data2 = data[1]
			self.pairs = data[2]
			self.siamese = True



		#calculate max batch here if you have batch size
		if batch_size:
			self.max_batch = len(self.data)//self.batch_size


	def new_epoch(self):
		#reset current batch
		self.current_batch = 0

		#shuffle everything
		if self.shuffle and self.supervised:
			new_indices = np.random.choice(len(self.data),len(self.data),replacement=False)
			new_inputs = []
			new_labels = []
			for index in new_indices:
				new_inputs.append(self.data[index])
				new_labels.append(self.labels[index])
			self.data = np.array(new_inputs)
			self.labels = np.array(new_labels)

		if self.shuffle and self.unsupervised:
			self.data = np.random.shuffle(self.data)

		if self.shuffle and self.siamese:
			new_indices = np.random.choice(len(self.data),len(self.data),replacement=False)
			new_inputs1 = []
			new_inputs2 = []
			new_labels = []
			for index in new_indices:
				new_inputs1.append(self.data[index])
				new_inputs2.append(self.data[index])
				new_labels.append(self.labels[index])
			self.data = np.array(new_inputs1)
			self.data2 = np.array(new_inputs2)
			self.pairs = np.array(new_labels)


	def next(self, batch_size = None):
		if batch_size is not None:
			assert type(batch_size) == int and batch_size >=1, 'invalid batch size; must be an integer greater than or equal to one'
			self.batch_size = batch_size
			self.max_batch = len(self.data)//self.batch_size

		if self.current_batch >= self.max_batch:
			self.new_epoch()

		if self.supervised:
			batch_slice = slice(self.current_batch * self.batch_size, (self.current_batch+1)*self.batch_size)
			inputs_batch = self.data[batch_slice]
			targets_batch = self.labels[batch_slice]
			self.current_batch +=1
			return inputs_batch, targets_batch

		if self.unsupervised:
			batch_slice = slice(self.current_batch * self.batch_size, (self.current_batch+1)*self.batch_size)
			inputs_batch = self.data[batch_slice]
			self.current_batch +=1
			return inputs_batch

		if self.siamese:
			batch_slice = slice(self.current_batch * self.batch_size, (self.current_batch+1)*self.batch_size)
			inputs_batch1 = self.data[batch_slice]
			inputs_batch2 = self.data2[batch_slice]
			targets_batch = self.pairs[batch_slice]
			self.current_batch +=1
			return inputs_batch1, inputs_batch2, targets_batch

	def get_batch_size(self):
		return self.batch_size

	def get_input_data(self):
		return self.data

	def get_current_batch(self):
		return self.current_batch

	def get_shuffle(self):
		return self.shuffle

	def get_max_batch(self):
		return self.max_batch

	def get_type(self):
		if self.supervised:
			return "supervised"
		if self.unsupervised:
			return "unsupervised"
		if self.siamese:
			return "siamese"
		else:
			return "unrecognised"
