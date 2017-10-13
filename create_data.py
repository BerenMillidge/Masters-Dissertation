# Author: Beren Millidge
# MSc Dissertation Project
# Summer 2017

# This script creates training and validation data set

import numpy as np
import pickle
from collections import Counter

BATCH_SIZE_TRAIN = 100
BATCH_SIZE_VALID = 100
BATCH_SIZE_TEST = 2

INPUT_UNITS = 20
INPUT_MIN = 0
INPUT_MAX = 1


def build_input(units, batch_size_train, batch_size_valid=0, batch_size_test=0):
    output = {'train':[], 'valid':[], 'test':[]}
    assert type(batch_size_train) == int and batch_size_train > 0, 'batch_size_train needs to be int and greater than 0, got %s'%str(batch_size_train)

    # This is quite clever: in binary terms (2**X)+1 will always be the length of X so it specifies a range of binary values which are then converted to an array of length X
    rand = np.random.choice((2**INPUT_UNITS)+1, size=batch_size_train+batch_size_valid+batch_size_test, replace=True)

    duplicates = [item for item, count in Counter(rand).iteritems() if count > 1]
    if duplicates:
        print "WARNING: Duplicates in data sets."

    output['train'] = array2input(rand[:batch_size_train], units=units)

    if batch_size_valid:
        assert type(batch_size_valid) == int and batch_size_valid > 0, 'batch_size_valid needs to be int and greater than 0, got %s'%str(batch_size_valid)
        output['valid'] = array2input(rand[batch_size_train:batch_size_train+batch_size_valid], units=units)

    if batch_size_test:
        assert type(batch_size_test) == int and batch_size_test > 0, 'batch_size_test needs to be int and greater than 0, got %s'%str(batch_size_test)
        output['test'] = array2input(rand[batch_size_train+batch_size_valid:], units=units)

    return output


def build_sized_sequence(units, batch_size_train, batch_size_valid = 0, batch_size_test = 0, diff = 0.5):
	output = {'train': [], 'valid': [], 'test': []}

	assert type(batch_size_train) == int and batch_size_train >0, 'batch_size_train needs to be an integer and greater than 0, got %s'%str(batch_size_train)
	assert type(batch_size_valid) == int and batch_size_valid >=0,'batch_size_valid must be a non-negative integer'
	assert type(batch_size_test) == int and batch_size_test >=0,'batch_size_test must be a non-negative integer'
	lengths = [batch_size_train, batch_size_valid, batch_size_test]
	batches = []
	for length in lengths:
		if length > 0:
			sequences = []
			for i in xrange(length):
				seq = []
				for i in xrange(units):
					rand = np.random.uniform()
					if rand <=0.5:
						seq.append(0.5-diff)
					if rand > 0.5:
						seq.append(0.5+ diff)
				seq = np.array(seq)
				sequences.append(seq)
			if length<=0:
				sequences = []
			baches.append(sequences)

	output['train'] = np.array(batches[0],dtype='int16')
	if batch_size_valid:
		assert type(batch_size_valid) == int and batch_size_valid > 0, 'batch_size_valid needs to be int and greater than 0, got %s'%str(batch_size_valid)
		output['valid'] = np.array(batches[1],dtype='int16')
	if batch_size_test:
		assert type(batch_size_test) == int and batch_size_test > 0, 'batch_size_test needs to be int and greater than 0, got %s'%str(batch_size_test)
		output['test'] = np.array(batches[2],dtype='int16')

def build_set_sized_sequence(units, batch_size_train, batch_size_valid = 0, batch_size_test = 0, val = 0.5):
	output = {'train': [], 'valid': [], 'test': []}

	assert type(batch_size_train) == int and batch_size_train >0, 'batch_size_train needs to be an integer and greater than 0, got %s'%str(batch_size_train)
	assert type(batch_size_valid) == int and batch_size_valid >=0,'batch_size_valid must be a non-negative integer'
	assert type(batch_size_test) == int and batch_size_test >=0,'batch_size_test must be a non-negative integer'
	lengths = [batch_size_train, batch_size_valid, batch_size_test]
	batches = []
	for length in lengths:
		if length > 0:
			sequences = []
			for i in xrange(length):
				seq = []
				for i in xrange(units):
					rand = np.random.uniform()
					if rand <=0.5:
						seq.append(val)
					if rand > 0.5:
						seq.append(0)
				seq = np.array(seq)
				sequences.append(seq)
			if length<=0:
				sequences = []
			batches.append(sequences)

	output['train'] = np.array(batches[0],dtype='float32')
	if batch_size_valid:
		assert type(batch_size_valid) == int and batch_size_valid > 0, 'batch_size_valid needs to be int and greater than 0, got %s'%str(batch_size_valid)
		output['valid'] = np.array(batches[1],dtype='int16')
	if batch_size_test:
		assert type(batch_size_test) == int and batch_size_test > 0, 'batch_size_test needs to be int and greater than 0, got %s'%str(batch_size_test)
		output['test'] = np.array(batches[2],dtype='int16')
	return output

def generate_even_sequence(units, batch_size_train, batch_size_valid = 0, batch_size_test = 0):
	#this function generates even sequenes - i.e. 010101010101010101010, which could be its own "skill"
	output = {'train': [], 'valid': [], 'test': []}

	assert type(batch_size_train) == int and batch_size_train >0, 'batch_size_train needs to be an integer and greater than 0, got %s'%str(batch_size_train)
	assert type(batch_size_valid) == int and batch_size_valid >=0,'batch_size_valid must be a non-negative integer'
	assert type(batch_size_test) == int and batch_size_test >=0,'batch_size_test must be a non-negative integer'
	lengths = [batch_size_train, batch_size_valid, batch_size_test]
	batches = []
	for length in lengths:
		if length > 0:
			sequences = []
			for i in xrange(length):
				seq = []
				rand = np.random.uniform()
				if rand <=0.5:
					seq.append(0)
				if rand >0.5:
					seq.append(1)
				for j in xrange(1,units-1):
					prev = seq[j-1]
					if prev ==1:
						seq.append(0)
					if prev ==0:
						seq.append(1)
				seq = np.array(seq)
				sequences.append(seq)
			if length<=0:
				sequences = []
			batches.append(sequences)

	output['train'] = np.array(batches[0],dtype='int16')
	if batch_size_valid:
		assert type(batch_size_valid) == int and batch_size_valid > 0, 'batch_size_valid needs to be int and greater than 0, got %s'%str(batch_size_valid)
		output['valid'] = np.array(batches[1],dtype='int16')
	if batch_size_test:
		assert type(batch_size_test) == int and batch_size_test > 0, 'batch_size_test needs to be int and greater than 0, got %s'%str(batch_size_test)
		output['test'] = np.array(batches[2],dtype='int16')
	return output
				

def build_correlated_sequence(units, batch_size_train, batch_size_valid=0, batch_size_test=0, correlation_coeff = 0):
	output = {'train': [], 'valid': [], 'test': []}
	assert type(batch_size_train) == int and batch_size_train >0, 'batch_size_train needs to be an integer and greater than 0, got %s'%str(batch_size_train)
	assert type(correlation_coeff) == float or type(correlation_coeff) == int and correlation_coeff <=1 and correlation_coeff >=-1, 'correlation coefficient must be a number between -1 and 1.'
	assert type(batch_size_valid) == int and batch_size_valid >=0,'batch_size_valid must be a non-negative integer'
	assert type(batch_size_test) == int and batch_size_test >=0,'batch_size_test must be a non-negative integer'
	lengths = [batch_size_train, batch_size_valid, batch_size_test]
	batches = []
	for length in lengths:
		if length>0:
			sequences = []
			for i in xrange(length):
				#is correlation coefficient negative?
				neg = False
				correlation = correlation_coeff
				if correlation_coeff < 0:
					correlation *=-1
					neg = True
				#get initial value randomly
				seq = []
				rand = np.random.uniform()
				#print neg
				if rand > 0.5:
					seq.append(1)
				if rand <=0.5:
					seq.append(0)
				for i in xrange(units-1):
					#we generate the sequence as a correlated version off the previous input according to rand
					rand = np.random.uniform()
					prev = seq[i-1]
					if rand <= correlation:
						if neg == False:
							seq.append(prev)
						if neg == True:
							if prev==1:
								seq.append(0)
							if prev==0:
								seq.append(1)
					if rand > correlation:
						if neg == True:
							seq.append(prev)
						if neg == False:
							if prev ==1:
								seq.append(0)
							if prev == 0:
								seq.append(1)
				seq = np.array(seq)
				sequences.append(seq)
		if length <=0:
			sequences = []
		batches.append(sequences)
	output['train'] = np.array(batches[0],dtype='int16')
	if batch_size_valid:
		assert type(batch_size_valid) == int and batch_size_valid > 0, 'batch_size_valid needs to be int and greater than 0, got %s'%str(batch_size_valid)
		output['valid'] = np.array(batches[1],dtype='int16')
	if batch_size_test:
		assert type(batch_size_test) == int and batch_size_test > 0, 'batch_size_test needs to be int and greater than 0, got %s'%str(batch_size_test)
		output['test'] = np.array(batches[2],dtype='int16')
	#print batches[0]
	"""bib = np.array(batches[0])
	print bib.shape
	output['train'] = array2input(batches[0], units=units)
	
	if batch_size_valid:
        	assert type(batch_size_valid) == int and batch_size_valid > 0, 'batch_size_valid needs to be int and greater than 0, got %s'%str(batch_size_valid)

		output['valid'] = array2input(batches[1], units=units)
	
	if batch_size_test:
        	assert type(batch_size_test) == int and batch_size_test > 0, 'batch_size_test needs to be int and greater than 0, got %s'%str(batch_size_test)
		output['test'] = array2input(batches[2], units=units)"""

	return output




	""" okay, so the point here is that we've got to take advantage of the actual recurrent structure of the network, as that
	 seems like it could actually be useful in some sense. so we're going to build a correlated
	 across time network, as that's the important thing, and see if it can be learned via the
	 savantism... okay, we built this function... what else do we actually and urgently need to do? I'm pretty uncertain re this. let's get out my daily plans to see what's up w/ that!
"""

def build_correlated_across_time(units, batch_size_train, batch_size_valid=0, batch_size_test=0, correlation_coeff = 0):

	output = {'train': [], 'valid': [], 'test': []}
	assert type(batch_size_train) == int and batch_size_train >0, 'batch_size_train needs to be an integer and greater than 0, got %s'%str(batch_size_train)
	assert type(correlation_coeff) == float or type(correlation_coeff) == int and correlation_coeff <=1 and correlation_coeff >=-1, 'correlation coefficient must be a number between -1 and 1.'
	assert type(batch_size_valid) == int and batch_size_valid >=0,'batch_size_valid must be a non-negative integer'
	assert type(batch_size_test) == int and batch_size_test >=0,'batch_size_test must be a non-negative integer'
	lengths = [batch_size_train, batch_size_valid, batch_size_test]
	batches = []
	for length in lengths:
		if length >0:
			sequences = []
			#rand = np.random.choice(units)
			rand = []
			for k in xrange(units):
				rand.append(np.random.randint(0,2))
			sequences.append(rand)
			neg = False
			correlation = correlation_coeff
			if correlation_coeff <0:
				neg = True
				correlation = correlation_coeff*-1
			for i in xrange(length-1):
				prev = sequences[i]
				cur = []
				for j in xrange(len(prev)):
					r = np.random.uniform()
					assert prev[j] == 0 or prev[j] == 1, 'invalid value in sequence'
					if prev[j] == 1:
						if neg == False:
							if r <= correlation:
								cur.append(1)
							if r > correlation:
								cur.append(0)
						if neg == True:
							if r <=correlation:
								cur.append(0)
							if r > correlation:
								cur.append(1)
					if prev[j] ==0:
						if neg == False:
							if r <=correlation:
								cur.append(0)
							if r > correlation:
								cur.append(1)
						if neg == True:
							if r <=correlation:
								cur.append(1)
							if r > correlation:
								cur.append(0)
				sequences.append(cur)
		if length <=0:
			sequences = []
		batches.append(sequences)
		
	assert len(batches) == 3,'fail w/ batches somehow'
	output['train'] = np.array(batches[0],dtype='int16')
	if batch_size_valid:
		output['valid'] = np.array(batches[1], dtype = 'int16')
	if batch_size_test:
		output['test'] = np.array(batches[2], dtype = 'int16')
	return output



def array2input(array, units, dtype='int16'):

    array = np.array(array) # this is for conversion. I should really do something mor complicated with assertions etc
    print array.shape
    assert array.ndim == 1, 'Expected array to have exactly one dimension, got %.0f'%array.ndim
    input = np.empty(shape=(array.shape[0], units))
    for idx, number in enumerate(array):
        string = "{0:b}".format(number).zfill(units)
        for jdx, char in enumerate(string):
            input[idx,jdx] = char
    return input.astype(dtype)

def test_correlated_sequence():
	coeffs = [1,-1,0,0.5,-0.5]
	for coeff in coeffs:
		data = build_correlated_across_time(20,5,correlation_coeff = coeff)
		print data['train']
		print "  "
		print "  "

def test_seq_func(func):
	coeffs = [1,-1,0,0.5,-0.5]
	for coeff in coeffs:
		data = func(20,5,0,0, coeff)
		print data['train']
		print "  "
		print "  "


def main():
    #np.random.seed([1990, 04, 21])
    #print("Creating data ...")
    #pickle.dump(build_input(INPUT_UNITS, BATCH_SIZE_TRAIN, BATCH_SIZE_VALID, BATCH_SIZE_TEST),
    #    open("data/dat_%s_%s_%s_nodupl3.p"%(str(INPUT_MIN), str(INPUT_MAX), str(INPUT_UNITS)), 'wb'))
   # print("Data saved.")
	
    #test_correlated_sequence()
    test_seq_func(build_set_sized_sequence)
	



if __name__ == '__main__':
    #import pdb
    #pdb.set_trace()
    main()









