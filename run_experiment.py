# Author: Beren Millidge, with substantial contributions from past Msc student, Florian Bolenz
# MSc Dissertation
# Summer 2017

import numpy as np
from experiment import DataProvider, Experiment, save_results, LearningRateContainer, LearningRateScheduler
from mask import MaskContainer
from create_data import *
import pickle


SEED = [1, 28, 102] # for experiment A
# SEED = [13, 11, 1415] # for experiment B
# SEED = [622, 622, 522] # for experiments C, E
# SEED = [3, 2121, 1978] # for experiments D
# SEED = [21, 04, 2017] # for experiment F

FILENAME = 'dat_0_1_20_nodupl.p'

#MASKCONTNAME = 'interintra_4m_20iu_16hu_20ou_fullin_fullout_withrec' # for experiment A:
MASKCONTNAME = 'interintra_4m_20iu_16hu_20ou_partialin_partialout_withrec' # for experiments B, C, D, F:
# MASKCONTNAME = 'interintra_2m_20iu_16hu_20ou_partialin_partialout_withrec' # for experiments F:


INPUT_LENGTH = 5        # time steps activity circulates through hidden layer
LEARNING_RATE = 0.005    # initial learning rate 
N_EPOCHS = 2000	#1000
N_NETWORKS = 100	#200
OPTIMIZER = 'adagrad'
INIT_RANGE_HID = 0.05   # upper bound for weigh initialisation in hidden layer - #this is Q
TRAIN_ON_VALID = False  # set true for switching training and validation data sets

SAVE_RESULTS = True
VERBOSE = False

#data characteristics:
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_VALID = 100
BATCH_SIZE_TEST = 2

INPUT_UNITS = 20
INPUT_MIN = 0
INPUT_MAX = 1

COEFFICIENTS = [0, 0.3,0.7,-0.3,-0.7]


def main(data, maskpath, savepath, var = 0):

    np.random.seed(SEED)
    print("Loading data ...")
    if var ==0:
    	data = [DataProvider(path='data/%s'%FILENAME, input_length=INPUT_LENGTH, train_on_valid=TRAIN_ON_VALID)]
    maskcont = MaskContainer(path="masks/%s"%maskpath)

    lrcont = LearningRateContainer(LearningRateScheduler(learning_rate_init=LEARNING_RATE))

    print("Start experiment ...")
    exp = Experiment(maskcont=maskcont, data=data)
    res_train, res_valid = exp.run(N_NETWORKS, N_EPOCHS, lrcont=lrcont, optimizer=OPTIMIZER, init_range_hid=INIT_RANGE_HID, verbose=True)

    if SAVE_RESULTS:
        #try:
           # save_results(res_train, res_valid, data, maskcont, lrcont, OPTIMIZER, INIT_RANGE_HID, SEED, savepath)
            #print "Results saved."
       # except:
            #print 'Results could not be saved'
	save_results(res_train, res_valid, data, maskcont, lrcont, OPTIMIZER, INIT_RANGE_HID, SEED, savepath)

def create_correlated_data(coefficients):
	dataproviders = []
	for coeff in coefficients:
		data = build_correlated_sequence(INPUT_UNITS, BATCH_SIZE_TRAIN, BATCH_SIZE_VALID, BATCH_SIZE_TEST, coeff)


		print("Creating data with coefficient %s"%coeff)
    		pickle.dump(data,open("data/dat_%s_%s_%s_coeff_%s.p"%(str(INPUT_MIN), str(INPUT_MAX), str(INPUT_UNITS), str(coeff)), 'wb'))
    		print("Data saved.")
		provider = DataProvider(data, input_length = INPUT_LENGTH, train_on_valid = TRAIN_ON_VALID)
		dataproviders.append(provider)
	return dataproviders

def create_time_correlated(coefficients):
	dataproviders = []
	for coeff in coefficients:
		data = build_correlated_across_time(INPUT_UNITS, BATCH_SIZE_TRAIN, BATCH_SIZE_VALID, BATCH_SIZE_TEST, coeff)
		print("Creating data with coefficient %s"%coeff)
		#pickle.dump(data,open("data/dat_%s_%s_%s_coeff_%s.p"%(str(INPUT_MIN), str(INPUT_MAX), str(INPUT_UNITS), str(coeff)), 'wb'))
    		#print("Data saved.")
		provider = DataProvider(data, input_length = INPUT_LENGTH, train_on_valid = TRAIN_ON_VALID)
		dataproviders.append(provider)
	return dataproviders

def create_sized_data(coefficients, save = False):
	dataproviders = []
	for coeff in coefficients:
		data = build_set_sized_sequence(INPUT_UNITS, BATCH_SIZE_TRAIN, BATCH_SIZE_VALID, BATCH_SIZE_TEST, coeff)
		print("Creating data with coefficient %s"%coeff)
		if save:
			pickle.dump(data,open("data/dat_%s_%s_%s_coeff_%s.p"%(str(INPUT_MIN), str(INPUT_MAX), str(INPUT_UNITS), str(coeff)), 'wb'))
    			print("Data saved.")
		provider = DataProvider(data, input_length = INPUT_LENGTH, train_on_valid = TRAIN_ON_VALID)
		dataproviders.append(provider)
	return dataproviders
	



if __name__ == '__main__':
    #data = create_correlated_data(COEFFICIENTS)
    #main(data,'interintra7_4m_20iu_16hu_20ou_fullin_fullout_withrec', 'intra7_coeff1',1)
    #main(data,'interintra6_4m_20iu_16hu_20ou_fullin_fullout_withrec', 'intra6_coeff1',1)
    #main(data,'interintra5_4m_20iu_16hu_20ou_fullin_fullout_withrec', 'intra5_coeff1',1)
    #main(data,'interintra4_4m_20iu_16hu_20ou_fullin_fullout_withrec', 'intra4_coeff1',1)
    #main(data,'interintra3_4m_20iu_16hu_20ou_fullin_fullout_withrec', 'intra3_coeff1',1)
    #main(data,'interintra2_4m_20iu_16hu_20ou_fullin_fullout_withrec', 'intra2_coeff1',1)
    #main(data,'interintra1_4m_20iu_16hu_20ou_fullin_fullout_withrec', 'intra1_coeff1',1)

    #data = create_time_correlated(COEFFICIENTS)
   #main(data,'interintra7_4m_20iu_16hu_20ou_fullin_fullout_withrec', 'intra7_seq1',1)
    #main(data,'interintra6_4m_20iu_16hu_20ou_fullin_fullout_withrec', 'intra6_seq1',1)
    #main(data,'interintra5_4m_20iu_16hu_20ou_fullin_fullout_withrec', 'intra5_seq1',1)
    #main(data,'interintra4_4m_20iu_16hu_20ou_fullin_fullout_withrec', 'intra4_seq1',1)
    #main(data,'interintra3_4m_20iu_16hu_20ou_fullin_fullout_withrec', 'intra3_seq1',1)
    #main(data,'interintra2_4m_20iu_16hu_20ou_fullin_fullout_withrec', 'intra2_seq1',1)
    #main(data,'interintra1_4m_20iu_16hu_20ou_fullin_fullout_withrec', 'intra1_seq1',1)

    #shuffled masks
    #data = create_time_correlated(COEFFICIENTS)
    #main(data,'interintra7_4m_20iu_16hu_20ou_fullin_fullout_withrec_SHUFFLE', 'intra7_seq1_shuffle',1)
    #main(data,'interintra6_4m_20iu_16hu_20ou_fullin_fullout_withrec_SHUFFLE', 'intra6_seq1_shuffle',1)
    #main(data,'interintra5_4m_20iu_16hu_20ou_fullin_fullout_withrec_SHUFFLE', 'intra5_seq1_shuffle',1)
    #main(data,'interintra4_4m_20iu_16hu_20ou_fullin_fullout_withrec_SHUFFLE', 'intra4_seq1_shuffle',1)
    #main(data,'interintra3_4m_20iu_16hu_20ou_fullin_fullout_withrec_SHUFFLE', 'intra3_seq1_shuffle',1)
    #main(data,'interintra2_4m_20iu_16hu_20ou_fullin_fullout_withrec_SHUFFLE', 'intra2_seq1_shuffle',1)
    #main(data,'interintra1_4m_20iu_16hu_20ou_fullin_fullout_withrec_SHUFFLE', 'intra1_seq1_shuffle',1)


    #data = create_correlated_data(COEFFICIENTS)
    #main(data,'interintra7_4m_20iu_16hu_20ou_partialin_partialout_withrec', 'intra7_seq1_partial',1)
    #main(data,'interintra6_4m_20iu_16hu_20ou_partialin_partialout_withrec', 'intra6_seq1_partial',1)
    #main(data,'interintra5_4m_20iu_16hu_20ou_partialin_partialout_withrec', 'intra5_seq1_partial',1)
    #main(data,'interintra4_4m_20iu_16hu_20ou_partialin_partialout_withrec', 'intra4_seq1_partial',1)
    #main(data,'interintra3_4m_20iu_16hu_20ou_partialin_partialout_withrec', 'intra3_seq1_partial',1)
    #main(data,'interintra2_4m_20iu_16hu_20ou_partialin_partialout_withrec', 'intra2_seq1_partial',1)
    #main(data,'interintra1_4m_20iu_16hu_20ou_partialin_partialout_withrec', 'intra1_seq1_partial',1)

    #data = create_time_correlated(COEFFICIENTS)
    #main(data,'interintra7_4m_20iu_16hu_20ou_partialin_partialout_withrec', 'intra7_seq_time_partial',1)
    #main(data,'interintra6_4m_20iu_16hu_20ou_partialin_partialout_withrec', 'intra6_seq_time_partial',1)
    #main(data,'interintra5_4m_20iu_16hu_20ou_partialin_partialout_withrec', 'intra5_seq_time_partial',1)
    #main(data,'interintra4_4m_20iu_16hu_20ou_partialin_partialout_withrec', 'intra4_seq_time_partial',1)
    #main(data,'interintra3_4m_20iu_16hu_20ou_partialin_partialout_withrec', 'intra3_seq_time_partial',1)
    #main(data,'interintra2_4m_20iu_16hu_20ou_partialin_partialout_withrec', 'intra2_seq_time_partial',1)
    #main(data,'interintra1_4m_20iu_16hu_20ou_partialin_partialout_withrec', 'intra1_seq_time_partial',1)

    #data = create_correlated_data(COEFFICIENTS)
    #main(data,'interintra7_4m_20iu_16hu_20ou_partialin_partialout_withrec_SHUFFLE', 'intra7_seq1_partial_shuffle',1)
    #main(data,'interintra6_4m_20iu_16hu_20ou_partialin_partialout_withrec_SHUFFLE', 'intra6_seq1_partial_shuffle',1)
    #main(data,'interintra5_4m_20iu_16hu_20ou_partialin_partialout_withrec_SHUFFLE', 'intra5_seq1_partial_shuffle',1)
    #main(data,'interintra4_4m_20iu_16hu_20ou_partialin_partialout_withrec_SHUFFLE', 'intra4_seq1_partial_shuffle',1)
    #main(data,'interintra3_4m_20iu_16hu_20ou_partialin_partialout_withrec_SHUFFLE', 'intra3_seq1_partial_shuffle',1)
    #main(data,'interintra2_4m_20iu_16hu_20ou_partialin_partialout_withrec_SHUFFLE', 'intra2_seq1_partial_shuffle',1)
    #main(data,'interintra1_4m_20iu_16hu_20ou_partialin_partialout_withrec_SHUFFLE', 'intra1_seq1_partial_shuffle',1)

    #data = create_correlated_data(COEFFICIENTS)
    #main(data, 'interintra7_4m_20iu_16hu_20ou_partialin_partialout_EQUAL', 'CONNECTIONS_EQUAL', 1)


    #COEFFS = [0.2,0.4,0.6,0.8,1]
    #data = create_sized_data(COEFFS)
    #main(data,'interintra7_4m_20iu_16hu_20ou_partialin_partialout_withrec', 'intra7_sized1_partial',1)
    #main(data,'interintra6_4m_20iu_16hu_20ou_partialin_partialout_withrec', 'intra6_sized1_partial',1)
    #main(data,'interintra5_4m_20iu_16hu_20ou_partialin_partialout_withrec', 'intra5_sized1_partial',1)
    #main(data,'interintra4_4m_20iu_16hu_20ou_partialin_partialout_withrec', 'intra4_sized1_partial',1)
    #main(data,'interintra3_4m_20iu_16hu_20ou_partialin_partialout_withrec', 'intra3_sized1_partial',1)
    #main(data,'interintra2_4m_20iu_16hu_20ou_partialin_partialout_withrec', 'intra2_sized1_partial',1)
    #main(data,'interintra1_4m_20iu_16hu_20ou_partialin_partialout_withrec', 'intra1_sized1_partial',1)


	#RUN THIS TOMORROW!
    #data = create_sized_data(COEFFS)
    #main(data,'interintra7_4m_20iu_16hu_20ou_partialin_partialout_withrec_SHUFFLE', 'intra7_sized1_partial_shuffle',1)
    #main(data,'interintra6_4m_20iu_16hu_20ou_partialin_partialout_withrec_SHUFFLE', 'intra6_sized1_partial_shuffle',1)
    #main(data,'interintra5_4m_20iu_16hu_20ou_partialin_partialout_withrec_SHUFFLE', 'intra5_sized1_partial_shuffle',1)
    #main(data,'interintra4_4m_20iu_16hu_20ou_partialin_partialout_withrec_SHUFFLE', 'intra4_sized1_partial_shuffle',1)
    #main(data,'interintra3_4m_20iu_16hu_20ou_partialin_partialout_withrec_SHUFFLE', 'intra3_sized1_partial_shuffle',1)
    #main(data,'interintra2_4m_20iu_16hu_20ou_partialin_partialout_withrec_SHUFFLE', 'intra2_sized1_partial_shuffle',1)
    #main(data,'interintra1_4m_20iu_16hu_20ou_partialin_partialout_withrec_SHUFFLE', 'intra1_sized1_partial_shuffle',1)

    #data = create_time_correlated(COEFFICIENTS)
    #main(data, 'NULL_PARTIAL_MASK_TEST', 'NULL_MASK_TEST', 1)


