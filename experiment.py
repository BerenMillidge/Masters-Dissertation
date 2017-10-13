# Author: Beren Millidge
# MSc Dissertation Project
# Summer 2017

import numpy as np
import pickle
import theano
import theano.tensor as T
import lasagne
import datetime
import pdb
import random

from mask import MaskContainer

def stretched(array, length):
    """ stretches 2-dim array along a new, third axis

        Args:
            array: numpy-array of shape (batch size, input units), or None
            length: length of new dimension

        Returns:
            None, if array is not a numpy array
            output: stretched array of shape (batch size, length, input units), if array is numpy array
    """
    if not type(array) == np.ndarray:
        return None
    #output = np.zeros((array.shape[0], length, array.shape[-1]), dtype='int16')
    output = np.zeros((array.shape[0], length, array.shape[-1]), dtype='float32')
    for k in xrange(length):
        output[:, k, :] = array
    return output

# Create simple network with input layer, one recurrent hidden layer and output layer
# per default: layers have rectifying activation function
# weight initialisation: uniform for W, constant 0 for bias (per default)

def build_network(shape, input_length, init_range_hid, input_var=None):
    """ build a network with three layers: input layer, hidden recurrent layer, output layer

        Args:
            shape: shape of weight matrices, given as tupel ((input,hidden), (hidden,hidden), (hidden, out))
            input_length: length of input (how many time steps is one input presented?)
            init_range: float, hidden weights are initialised between [-init_range, init_range]
            input_var: theano variable representing input values

        Returns:
            l_out: output layer, which contains all previous layers and thus represents the compete network
    """
    input_units = shape[0][0]   # number of input units
    hidden_units = shape[1][0]  # number of hidden units
    output_units = shape[2][1]  # number of output units
	


    # Input layer:
    l_in = lasagne.layers.InputLayer(shape=(None, input_length, input_units), input_var=input_var)

    # Recurrent layer:
    l_hid = lasagne.layers.RecurrentLayer(
            incoming=l_in, num_units=hidden_units,
            W_in_to_hid=lasagne.init.Uniform(range=0.1),    # initialize in_to_hid weights uniformly
            W_hid_to_hid=lasagne.init.Uniform(range=init_range_hid),   # initialize hid_to_hid weights uniformly
            b=lasagne.init.Constant(0.),                    # initialize biases constantly with 0
            nonlinearity=lasagne.nonlinearities.rectify,    # use rectifying activation function
            gradient_steps=-1,                             # number of time stepts to include in backpropagating gradient (if -1, take entire sequence)
            only_return_final=True)                         # only return final sequential output to next layer

	    
    l_out = lasagne.layers.DenseLayer(
            incoming=l_hid, num_units=output_units,
            W=lasagne.init.Uniform(range=0.1),              # initialize weights uniformly
            b=lasagne.init.Constant(0.),                    # initialize biases constantly with 0
            nonlinearity=lasagne.nonlinearities.rectify)    # use rectifying activation function

    return l_out

def apply_weight_mask(network, mask, mymul = False):
    """ use mask to set some weights of the network to 0 while leaving the others unchanged
        mask contains three matrices, identical in shape to the weight matrices of the network
        mask values are either 0 or 1, 0 = reset corresponding value in weight matrix to 0, 1 = leave corresponding value in weight matrix unchanged

        Args:
            network: three-layer network
            mask: mask object

        Returns:
            network: network with updated weights
    """
    mask_in = mask.get_masks()['mask_in']
    mask_hid = mask.get_masks()['mask_hid']
    mask_out = mask.get_masks()['mask_out']

    all_param_values = lasagne.layers.get_all_param_values(network) # gets a list of all network parameters
    W_in = all_param_values[1]  # in-to-hid weight matrix
    W_hid = all_param_values[3] # hid-to-hid weight matrix
    W_out = all_param_values[4] # hid-to-out weight matrix

    # apply masks to weight matrices and update values in list of parameters  
    if mymul == False:
   	 all_param_values[1] = W_in*mask_in
   	 all_param_values[3] = W_hid*mask_hid
    	 all_param_values[4] = W_out*mask_out
    if mymul == True:
	 all_param_values[1] = np.multiply(W_in, mask_in)
   	 all_param_values[3] = np.multiply(W_hid, mask_hid)
    	 all_param_values[4] = np.multiply(W_out, mask_out)
	


    lasagne.layers.set_all_param_values(network, all_param_values)  # reset network parameters

    return network

def save_results(res_train, res_valid, data, maskcont, lrcont, optimizer, init_range_hid, seed, path, dir="res"):
    """ save results to file

        Args:
            res_train: error terms of training data, shape (number of masks, number of networks, number of epochs+1)
            res_valid: error terms of validation data, shape (number of masks, number of networks, number of epochs+1)
            data: DataProvider object (among others, contains information on input length, path of data file)
            maskcont: MaskContainer object (among others, contains information on mask labels, path of mask file)
            learning_rate: learning rate used in training
            seed: seed for random numbers

        Returns:
            None
    """
    now = datetime.datetime.now()
    results = {'res_train': res_train,
               'res_valid': res_valid,
               'data': data,
               'maskcont': maskcont,
               'lrcont': lrcont,
               'optimizer': optimizer,
               'init_range_hid': init_range_hid,
               'seed': seed,
               'time': now.strftime('%Y-%m-%d_%H:%M:%S')}
   # pickle.dump(results, open("%s/res_%s"%(dir, now.strftime('%Y-%m-%d_%H-%M-%S')), 'wb'))
    with open("res/" + path, 'wb') as f:
	pickle.dump(results, f)
    f.close()

class DataProvider:

    def __init__(self, path, input_length, train_on_valid=False):
        """ initialize DataProvider object

        Args:
            path: path to data file (file is a dict with keys "train", "valid", "test", and numpy arrays of shape (batch size, input units) as values
            input_length: number of time steps, each input is presented

        Returns:
            None
        """
        self.path = path
        self.input_length = input_length
	assert type(path) == str or type(path) == dict, 'You have failed badly. your fucking path is of an incompatible type'
	if type(path) == str:
        	data = pickle.load(open(path, "rb"))
	if type(path) == dict:
		data = path

        self.t = {}
        self.x = {}
	
	print type(data)
	print data['train']
	print type(data['train'])
	print data['train'].shape
	print "DATAPROVIDER TESTS ABOVE!"

        if not train_on_valid:
            self.t['train']= data['train']
            self.t['valid'] = data['valid']
        else:
            self.t['train']= data['valid']
            self.t['valid'] = data['train']
        self.x['train'] = stretched(self.t['train'], length=input_length)
        self.x['valid'] = stretched(self.t['valid'], length=input_length)
        self.t['test'] = data['test']
        self.x['test'] = stretched(self.t['test'], length=input_length)

    def get_input_length(self):
        """ return length of input (number of time steps each input is presented)

        Args:
            None

        Returns:
            input_length: int
        """
        return self.input_length

    def get_data(self, set):
        """ return input data and targets

        Args:
            set: string, specifies data set ("train", "valid", "test")

        Returns:
            x: input data, numpy array of shape (batch size, input length, input units)
            t: targets, numpy array of shape (batch size, input units)
        """
        assert set in ['train', 'valid', 'test'], 'set must be one of "train", "valid", "test", got %s'%str(set)
        return self.x[set], self.t[set]

    def get_inputs(self, set):
        """ return input data

        Args:
            set: string, specifies data set ("train", "valid", "test")

        Returns:
            x: input data, numpy array of shape (batch size, input length, input units)
        """
        x = self.get_data(set)[0]
        return x

    def get_targets(self, set):
        """ return target data

        Args:
            set: string, specifies data set ("train", "valid", "test")

        Returns:
            t: targets, numpy array of shape (batch size, input units)
        """
        t = self.get_data(set)[1]
        return t

    def has_data(self, set):
        """ checks, if data set is available

        Args:
            set: string, specifies data set ("train", "valid", "test")

        Returns:
            b: boolean
        """
        b = not isinstance(self.get_inputs(set), type(None))
        return b

class LearningRateScheduler:

    def __init__(self, learning_rate_init, exp_decay=1.0, learning_rate_min=0.0, label=None):
        assert learning_rate_init > 0 and type(learning_rate_init) in [float, int], 'learning_rate must be float greater 0, got %s'%str(learning_rate)
        assert exp_decay > 0 and type(exp_decay) in [float, int], 'exp_decay must be float greater 0, got %s'%str(exp_decay)
        assert learning_rate_min >= 0 and type(learning_rate_min) in [float, int], 'learning_rate_min must be float greater or equal 0, got %s'%str(learning_rate_min)
        self.learning_rate_init = learning_rate_init
        self.exp_decay = exp_decay
        self.learning_rate_min = learning_rate_min
        self.label = label

    def get_learning_rate(self, epoch):
        assert type(epoch) == int and epoch >=0, 'epoch must be int greater or equal 0, got %s'%str(epoch)
        learning_rate = max(self.learning_rate_init*((1-self.exp_decay)**epoch), self.learning_rate_min)
        return learning_rate

    def get_label(self):
        """ return label of scheduler

        Args:
            None

        Returns:
            str
        """
        return self.label

class LearningRateContainer:

    def __init__(self, *args):
        assert all([isinstance(x, LearningRateScheduler) for x in args]), "All arguments must be of type LearningRateScheduler, got %s"%str([type(x) for x in args])
        self.scheduler_list = args
        self.idx = 0

    def get_scheduler(self, idx):
        return self.scheduler_list[idx % len(self.scheduler_list)]

    def get_labels(self):
        """ return labels of all scheulders

        Args:
            None

        Returns:
            labels: list of strings
        """
        labels = [x.get_label() for x in self.scheduler_list]
        return labels

class Experiment:

    def __init__(self, maskcont, data):
        """ initializes Experiment object

        Args:
            maskcont: MaskContainer object
            data: DataProvider object

        Returns:
            None
        """
        assert isinstance(maskcont, MaskContainer), 'maskcont must be instance of class MaskContainer'
        self.maskcont = maskcont
	print type(data)
        assert isinstance(data, list), 'fail!' #'data must be instance of class DataProvider'#array!
	#our previous dataprovider is now an array of dataproviders to enable us to have multiple skills working
	for i in xrange(len(data)):
		assert isinstance(data[i], DataProvider), 'not everything in the data array is a dataprovider!'
        self.data = data

    def run(self, n_networks, n_epochs, lrcont, optimizer, init_range_hid=0.1, verbose=True):
        """ Runs the model

        Args:
            n_networks: int, number of networks trained for each mask
            n_epochs: int, number of epochs for training
            lrcont: LearningRateContainer
            optimizer: int ('sgd', 'adagrad')
            init_range_hid: range of weight initialisation in hidden layer
            verbose: boolean, whether or not to print detailed output to terminal

        Returns:
            res_train: error terms of training data, shape (number of masks, number of networks, number of epochs+1)
            rees_valid: error terms of validation data, shape (number of masks, number of networks, number of epochs+1)
        """
        # initialize results as empty arrays
        res_train = np.empty((self.maskcont.count_masks(), n_networks, n_epochs+1, len(self.data)))
        res_valid = np.empty((self.maskcont.count_masks(), n_networks, n_epochs+1,len(self.data)))
	print verbose

        # iterate over masks
        for idx_mask, mask in enumerate(self.maskcont):

            lrscheduler = lrcont.get_scheduler(idx_mask)

            # repeate for each network:
            for idx_network in xrange(n_networks):
                print 'Mask %.0f/%.0f. Network %.0f/%.0f'%(idx_mask+1, self.maskcont.count_masks(), idx_network+1, n_networks)
                if verbose:
                    print("Building network ...")
                #target_values = T.wmatrix('target_values')
		target_values = T.fmatrix('target_values')
                #input_values = T.wtensor3('input_values')
		input_values = T.ftensor3('input_values')

                mask.reinit()   # reinitialize mask values (nothing happens for ConstantMask)

                network = build_network(self.maskcont.get_shapes(), self.data[0].get_input_length, init_range_hid=init_range_hid, input_var=input_values)
                network = apply_weight_mask(network, mask=mask)

                network_output = lasagne.layers.get_output(network)
                loss = lasagne.objectives.squared_error(network_output, target_values)
                loss = loss.mean()  # objective function is mean squared error
                all_params = lasagne.layers.get_all_params(network)
                lr = theano.shared(lrscheduler.get_learning_rate(epoch=0))
                assert optimizer in ['adagrad', 'sgd'], 'optimizer must be one of: \'adagrad\', \'sgd\', got %s'%str(optimizer)
                if optimizer == 'adagrad':
                    updates = lasagne.updates.adagrad(loss, all_params, lr)
                elif optimizer == 'sgd':
                    updates = lasagne.updates.sgd(loss, all_params, lr)

                if verbose:
                    print("Compiling functions ...")
                train = theano.function([input_values, target_values], loss, updates=updates)
                compute_cost = theano.function([input_values, target_values], loss)
                if verbose and self.data[0].has_data('test'):
                    prediction = theano.function([input_values], network_output)

                # compute objective function once before any training has happened (epoch 0)
                cost_train = compute_cost(*self.data[0].get_data('train'))
		print self.data[0].get_data('train')
		print type(self.data[0].get_data('train'))
		#print len(self.data[0].get_data('train'))
		#print type(self.data[0].get_data('train'))
		bib = self.data[0].get_data('train')
		print type(bib)
		print len(bib)
		print type(bib[0])
		print bib[0]
		print bib[0].shape


                cost_valid = compute_cost(*self.data[0].get_data('valid'))
                #res_train[idx_mask,idx_network,0] = cost_train
                #res_valid[idx_mask,idx_network,0] = cost_valid

                if verbose:
                    print("Training ...")
                    print "COST"
                    print "Epoch\ttrain\tvalid"
                    print "---------------------------"
                    print('%.0f\t%.2f\t%.2f'%(0, cost_train, cost_valid))

                # train network for n_epoch times
                for epoch in xrange(1,n_epochs+1):
		    random.shuffle(self.data)
		    for i in xrange(len(self.data)):
                        train(*self.data[i].get_data('train'))
                        network = apply_weight_mask(network, mask=mask)
                        # compute and save costs:
                        cost_train = compute_cost(*self.data[i].get_data('train'))
                        cost_valid = compute_cost(*self.data[i].get_data('valid'))
                        res_train[idx_mask,idx_network,epoch,i] = cost_train
                        res_valid[idx_mask,idx_network,epoch,i] = cost_valid
                        # reset learning rate
                        lr = lrscheduler.get_learning_rate(epoch=epoch)
                        if verbose:
                            if epoch==n_epochs or epoch/10 == epoch/10.0:
                                print('%.0f\t%.2f\t%.2f'%(epoch, cost_train, cost_valid))

                    # if appropriate, compute and print predictions for test set data
                    if verbose and self.data[0].has_data('test'):
                        test_prediction = prediction(self.data[0].get_inputs('test'))

                        for k in xrange(self.data[0].get_targets('test').shape[0]):
                            print "---------------------------"
                            print "Target:\t\t",
                            for y in np.around(self.data[0].get_targets('test')[k, :], decimals=2):
                                print "%s\t" % str(y),
                            print ""
                            print "Prediction:\t",
                            for y in np.around(test_prediction[k, :], decimals=2):
                                print "%s\t" % str(y),
                            print ""
                        print "Test cost: %.2f" % compute_cost(*self.data[0].get_data('test'))

        return res_train, res_valid



