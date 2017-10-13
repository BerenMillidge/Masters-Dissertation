#Author: Beren Millidge
#Msc Dissertation
#Summer 2017


import numpy as np
import pickle
#import theano
import matplotlib.pyplot as plt

#data = pickle.load(open('res/res_2017-07-01_20-33-48','rb'))

#data = pickle.load(open('res/intra1_seq1','rb'))
#data2 = pickle.load(open('res/intra2_seq1', 'rb'))
#data3 = pickle.load(open('res/intra3_seq1', 'rb'))
#data4 = pickle.load(open('res/intra4_seq1', 'rb'))
#data5 = pickle.load(open('res/intra5_seq1', 'rb'))
#data6 = pickle.load(open('res/intra6_seq1', 'rb'))
#data7 = pickle.load(open('res/intra7_seq1', 'rb'))


#data = pickle.load(open('res/intra1_seq1_partial','rb'))
#data2 = pickle.load(open('res/intra2_seq1_partial', 'rb'))
#data3 = pickle.load(open('res/intra3_seq1_partial', 'rb'))
#data4 = pickle.load(open('res/intra4_seq1_partial', 'rb'))
#data5 = pickle.load(open('res/intra5_seq1_partial', 'rb'))
#data6 = pickle.load(open('res/intra6_seq1_partial', 'rb'))
#data7 = pickle.load(open('res/intra7_seq1_partial', 'rb'))

#data_time = pickle.load(open('res/intra1_seq_time_partial','rb'))
#data2_time = pickle.load(open('res/intra2_seq_time_partial', 'rb'))
#data3_time = pickle.load(open('res/intra3_seq_time_partial', 'rb'))
#data4_time = pickle.load(open('res/intra4_seq_time_partial', 'rb'))
#data5_time = pickle.load(open('res/intra5_seq_time_partial', 'rb'))
#data6_time = pickle.load(open('res/intra6_seq_time_partial', 'rb'))
#data7_time = pickle.load(open('res/intra7_seq_time_partial', 'rb'))

#data_shuffle = pickle.load(open('res/intra1_seq1_partial_shuffle','rb'))
#data2_shuffle = pickle.load(open('res/intra2_seq1_partial_shuffle', 'rb'))
#data3_shuffle = pickle.load(open('res/intra3_seq1_partial_shuffle', 'rb'))
#data4_shuffle = pickle.load(open('res/intra4_seq1_partial_shuffle', 'rb'))
#data5_shuffle = pickle.load(open('res/intra5_seq1_partial_shuffle', 'rb'))
#data6_shuffle = pickle.load(open('res/intra6_seq1_partial_shuffle', 'rb'))
#data7_shuffle = pickle.load(open('res/intra7_seq1_partial_shuffle', 'rb')) 


#data_sized = pickle.load(open('res/intra1_sized1_partial','rb'))
#data2_sized = pickle.load(open('res/intra2_sized1_partial', 'rb'))
#data3_sized = pickle.load(open('res/intra3_sized1_partial', 'rb'))
#data4_sized = pickle.load(open('res/intra4_sized1_partial', 'rb'))
#data5_sized = pickle.load(open('res/intra5_sized1_partial', 'rb'))
#data6_sized = pickle.load(open('res/intra6_sized1_partial', 'rb'))
#data7_sized = pickle.load(open('res/intra7_sized1_partial', 'rb'))


#result = data['res_train']
#result2 = data2['res_train']
#result3 = data3['res_train']
#result4 = data4['res_train']
#result5 = data5['res_train']
#result6 = data6['res_train']
#result7 = data7['res_train']

#result_time = data_time['res_train']
#result2_time = data2_time['res_train']
#result3_time = data3_time['res_train']
#result4_time = data4_time['res_train']
#result5_time = data5_time['res_train']
#result6_time = data6_time['res_train']
#result7_time = data7_time['res_train']

#result_shuffle = data_shuffle['res_train']
#result2_shuffle = data2_shuffle['res_train']
#result3_shuffle = data3_shuffle['res_train']
#result4_shuffle = data4_shuffle['res_train']
#result5_shuffle = data5_shuffle['res_train']
#result6_shuffle = data6_shuffle['res_train']
#result7_shuffle = data7_shuffle['res_train']


#result_sized = data_sized['res_train']
#result2_sized = data2_sized['res_train']
#result3_sized = data3_sized['res_train']
#result4_sized = data4_sized['res_train']
#result5_sized = data5_sized['res_train']
#result6_sized = data6_sized['res_train']
#result7_sized = data7_sized['res_train']

#RESULT_LIST = [result, result2, result3, result4, result5, result6, result7]
#RESULT_LIST_SHUFFLE = [result_shuffle, result2_shuffle, result3_shuffle, result4_shuffle, result5_shuffle, result6_shuffle, result7_shuffle]

#RESULT_LIST_SIZED = [result_sized, result2_sized,result3_sized,result4_sized,result5_sized,result6_sized,result7_sized]

#RESULT_LIST_TIME = [result_time, result2_time,result3_time,result4_time,result5_time,result6_time,result7_time]

data = pickle.load(open('res/CONNECTIONS_EQUAL','rb'))
result = data['res_train']
#print result.shape

def save_fig(fig, path):
	with open(path, 'wb') as f:
		pickle.dump(fig, f)
	f.close()

def save_fig_list(figlist, basepath, fmat = None):
	assert type(basepath) == str and len(basepath) >=1
	assert type(figlist) == list and len(figlist) >=1

	for i in xrange(len(figlist)):
		#path = basepath + str(i) #+ fmat
		#with open(path, 'wb') as f:
			#pickle.dump(figlist[i], f)
		#f.close()

		fig = figlist[i]
		fig.savefig(basepath + str(i))
		plt.close(fig)
		
	print "SAVED ALL"

def single_network_various_graph(result, network_index, mean = False, variance = False):

	#this can calculate either mean or variance really through params. pretty simple really
	assert not (mean == False and variance == False), 'need one or the other'
	if mean == True:
		assert variance == False, 'mutually exclusive, so variance must be false'
	if variance == True:
		assert mean == False, 'mutually exclusive so mean must be false'
	split = result[:,network_index, :,:]
	masks = split.shape[0]
	
	if mean == True:
		name = "Mean "
	if variance == True:
		name = "Variance "
	
	#set up our figure environment
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	epoch_num = split.shape[1]
	x =range(epoch_num)	#get our x-axis aligned
	ax.set_xlabel("Epoch")
	ax.set_ylabel(name)
	# we do one across all masks, so we just have a mega loop here
	for i in xrange(masks):
		base = split[i, :,:]
		
		plot_list = []
		for j in xrange(epoch_num):
			arr = base[j]
			if mean == True:
				m = np.mean(arr)
				plot_list.append(m)
			if variance == True:
				var = np.var(arr)
				plot_list.append(var)
		print len(x)
		print len(plot_list)
		ax.plot(x, plot_list, label = name + "mask " + str(i))
	plt.legend()
	plt.show()
	return fig


def average_networks_various_graph(result, mean = False, variance = False, return_val = False):
	#do our asserts
	assert not (mean == False and variance == False), 'need one or the other'
	if mean == True:
		assert variance == False, 'mutually exclusive, so variance must be false'
	if variance == True:
		assert mean == False, 'mutually exclusive so mean must be false'

	if mean == True:
		name = "Mean "
	if variance == True:
		name = "Variance "

	#get our number variables
	num_masks = result.shape[0]
	network_num = result.shape[1]	
	epoch_num = result.shape[2]
	
	#set up our figure environment
	x = range(epoch_num)
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.set_xlabel('Epoch')
	ax.set_ylabel(name)
	if return_val == True:
		val_list = []
	
	#now for the big mega loop:
	for i in xrange(num_masks):
		split = result[i,:,:,:]
		plot_list = []	
		for j in xrange(epoch_num):
			arr = split[:,j,:]
			op_list = []
			for k in xrange(network_num):
				line = arr[k,:]
				if mean == True:
					line_mean = np.mean(line)
					op_list.append(line_mean)
				if variance== True:
					line_var = np.var(line)
					op_list.append(line_var)

			#we average everything in the oplist, even if variances; I think this is right
			op_list = np.array(op_list)
			m = np.mean(op_list)
			plot_list.append(m)
		ax.plot(x, plot_list, label = name + "mask " + str(i))
		if return_val == True:
			val_list.append(plot_list)
	plt.legend()
	plt.show()
	if return_val == False:
		return fig
	if return_val == True:
		return fig, val_list

average_networks_various_graph(result, mean = True)
average_networks_various_graph(result, variance = True)

def average_over_all_per_epoch(data, mean = False, variance = False):

	assert not (mean == False and variance == False), 'need one or the other'
	if mean == True:
		assert variance == False, 'mutually exclusive, so variance must be false'
	if variance == True:
		assert mean == False, 'mutually exclusive so mean must be false'

	num_masks = data.shape[0]
	num_networks = data.shape[1]
	num_epochs = data.shape[2]
	num_skills = data.shape[3]
	
	total_list = []
	for i in xrange(num_masks):
		split = data[i,:,:,:]
		plot_list = []	
		for j in xrange(num_epochs):
			arr = split[:,j,:]
			op_list = []
			for k in xrange(num_networks):
				line = arr[k,:]
				if mean == True:
					line_mean = np.mean(line)
					op_list.append(line_mean)
				if variance== True:
					line_var = np.var(line)
					op_list.append(line_var)

			#we average everything in the oplist, even if variances; I think this is right
			op_list = np.array(op_list)
			m = np.mean(op_list)
			plot_list.append(m)
		total_list.append(plot_list)
	total_list = np.array(total_list)
	print total_list.shape
	final_list = []
	for l in xrange(num_epochs):
		line = total_list[:,l]
		final_list.append(np.mean(line))	#we do mean here too even if it's variances. not sure if right
	print len(final_list)
	return final_list	

def plot_multiple_averaged_intras(data_list, mean = False, variance = False, label_list = None):

	assert not (mean == False and variance == False), 'need one or the other'
	if mean == True:
		assert variance == False, 'mutually exclusive, so variance must be false'
	if variance == True:
		assert mean == False, 'mutually exclusive so mean must be false'
	assert type(data_list) == list and len(data_list) >=1, 'must be a list of length >=1'
	if label_list:
		assert type(label_list) == list and len(label_list)>=1,'must be a list of length >=1'
		assert len(label_list) == len(data_list),'need as many labels as data and vice versa'

	if mean == True:
		name = "Mean "
	if variance == True:
		name = "Variance "

	N = len(data_list)
	data_one = data_list[0]
	num_epochs = data_one.shape[2] 
	#set up our figure and initial plotting functionality
	x = range(num_epochs)
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.set_xlabel('Epoch')
	ax.set_ylabel(name)

	#okay, here comes the standard loop by now
	for i in xrange(N):
		if mean == True:
			plot_list = average_over_all_per_epoch(data_list[i], mean = True)
		if variance==True:
			plot_list = average_over_all_per_epoch(data_list[i], variance = True)
		if label_list:
			ax.plot(x, plot_list, label = name + "intra " + str(label_list[i]))
		if not label_list:
			ax.plot(x, plot_list, label = name + "intra " + str(i))

	plt.legend()
	plt.show()
	return fig

def avg_network_various_graph_for_all_in_list(result_list, mean = False, variance = False, return_val = False, verbose = True):
	assert type(result_list) == list and len(result_list) >=0, 'must be a list and must work'
	assert mean == True or variance == True, 'must have one of mean or variance be true'
	if mean == True:
		assert variance == False, 'cannot have both mean and variance'
	if variance == True:
		assert mean == False, 'cannot have both variance and mean'

	fig_list = []
	if return_val == True:
		return_list = []
	if return_val == False:
		for i in xrange(len(result_list)):
			print "GRAPH FOR INTRA" + str(i)
			fig = average_networks_various_graph(result_list[i], mean = mean, variance = variance, return_val = False)
			fig_list.append(fig)
		return fig_list
	if return_val == True:
		for i in xrange(len(result_list)):
			print "GRAPH FOR INTRA" + str(i)
			fig, ret_val =  average_networks_various_graph(result_list[i], mean = mean, variance = variance, return_val = True)
			fig_list.append(fig)
			return_list.append(ret_val)

		if verbose == True:
			verbose_averaged_all_list(return_list, verbose = verbose)
		return fig_list, return_list
		

def return_averaged_list(val_list):
	retlist = []
	for i in xrange(len(val_list)):
		retlist.append(np.mean(val_list[i]))
	return retlist

def verbose_averaged_all_list(ret_list, verbose = True):
	totallist = []
	for i in xrange(len(ret_list)):
		avglist = []
		if verbose:
			print "Variances for intra " + str(i)
		for j in xrange(len(ret_list[i])):
			l = ret_list[i][j]
			avg =np.mean(l)
			avglist.append(avg)
			if verbose:
				print "INTER " + str(j) + " :" + str(avg)
		totallist.append(avglist)
	return totallist

def averaged_intras_inters(reslist, mean = False, variance = True, return_val = False):

	#asserts
	assert mean == True or variance == True, 'otherwise, whats the point?'
	if mean == True:
		assert variance == False, 'not both'
	if variance == True:
		assert mean == False, 'not both'

	reslist = np.array(reslist)
	shape = reslist.shape
	
	#define shapes
	num_intras = shape[0]
	num_inters = shape[1]
	num_networks = shape[2]
	num_epochs = shape[3]
	num_skills = shape[4]

	#set up graphing
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	x = range(num_epochs)	

	if return_val:
		plotslist = []
		
	for i in xrange(num_inters):
		intras_net = reslist[:,i,:,:,:]
		print intras_net.shape
		plot_list = []
		#okay, now begins the great averaging:
		for j in xrange(num_epochs): #num epochs!
			op_list = []
			epoch_net = intras_net[:,:,j,:]
			for k in xrange(num_networks): #num_networks
				network_net = epoch_net[:,k,:]
				net_list = []
				for l in xrange(num_intras): #num intras
					line = network_net[l,:]
					if mean == True:
						line_mean = np.mean(line)
						net_list.append(line_mean)
					if variance == True:
						line_var = np.var(line)
						net_list.append(line_var)
				net_list = np.array(net_list)
				net_mean = np.mean(net_list)
				op_list.append(net_mean)
			op_mean = np.mean(op_list)
			plot_list.append(op_mean)
		ax.plot(x, plot_list, label = 'inter' + str(i))
		if return_val:
			plotslist.append(plot_list)
	
	#set labels and endless plotting functionality
	ax.set_xlabel('epochs')
	ax.set_ylabel('error')
	ax.set_title('thing')
	plt.legend()
	plt.show()
	if  return_val:
		return fig, plotslist

	return fig


#averaged_intras_inters(RESULT_LIST_TIME)


#result = result4_time

print "MEAN SINGLE GRAPH NETWORK 76"
#single_network_various_graph(result, 76, mean= True, variance = False)
print "VARIANCE SINGLE GRAPH NETWORK 76"
#single_network_various_graph(result, 76, mean= False, variance = True)

# okay, I can't even remember what's going on here. I should learn
print "AVERAGE OVER NETWORKS VARIOUS GRAPH"
#average_networks_various_graph(result, mean=True, variance= False)
#average_networks_various_graph(result, mean=False, variance= True)

print "AVERAGE OVER ALL PER EPOCH"
#average_over_all_per_epoch(result, mean = True)
#average_over_all_per_epoch(result, variance = True)

print "PLOT MULTIPLE AVERAGED INTRAS"
#plot_multiple_averaged_intras(RESULT_LIST_TIME, mean= True)
#plot_multiple_averaged_intras(RESULT_LIST_TIME, variance= True)

 INTER is actually accounted for. so let's do this now

#figlist, val_list = avg_network_various_graph_for_all_in_list(RESULT_LIST_TIME, variance= True, return_val = True)
#print type(figlist[0])
#print val_list
#print len(figlist)
#print len(val_list)
#avglist = return_averaged_list(val_list)
#print avglist
#save_fig_list(figlist, "res/graphs/all_inter_intra_partial_sequence")


reslist = np.array(RESULT_LIST_TIME)
res1 = reslist[5,:,61,500,:]

input1 = res1[1,:]
input2 = res1[3,:]

print input1
print input2

#input1[2] = 0.164
#input2[1] = 0.181


def double_histogram(input1, input2, save = False, savepath = None):

	if save and savepath is None:
		savepath = 'res/graphs/savantism_double_histogram3'
	#print input1
	N = len(input1)
	lengths = range(N)
	w = 0.4	
	#everything is awesome!
	
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.bar(lengths, input1, w, color='blue', label='savantism', alpha =0.5)
	ax.bar([l + w for l in lengths] ,input2, w, color='red', label= 'neurotypical', alpha = 0.5)
	ax.set_xlabel('Skill')
	ax.set_ylabel('MSE error')
	ax.set_title('Savant network vs non savant')
	plt.legend()
	plt.show()

	if save:
		fig.savefig(savepath)
		plt.close(fig)
	return fig


#double_histogram(input1, input2, save = True)

def get_averaged_skill_list(inter, mean = False, variance = False, epoch_limit = 100):
	assert mean == True or variance == True
	if mean == True:
		assert variance == False
	if variance == True:
		assert mean == False


	#wait, this isn't exactly working perfectly imho... I don't even know!
	shape = inter.shape
	num_intras = shape[0]
	num_networks = shape[1]
	num_epochs  = shape[2]
	num_skills = shape[3]

	inter1 = inter[:,:,num_epochs-epoch_limit:num_epochs, :]
	print inter1.shape
	skill_list = []
	for i in xrange(num_skills): # for each skill, we want an average
		op_list = []
		inter = inter1[:,:,:,i]	
		for j in xrange(num_intras): # for each intra
			intra = inter[j,:,:]
			intralist = []
			for k in xrange(epoch_limit): # for each epoch
				line = intra[:,k]
				if mean == True:
					line_mean = np.mean(line)
					intralist.append(line_mean)
				if variance == True:
					line_var = np.var(line)
					intralist.append(line_var)
			intra_mean = np.mean(intralist)
			op_list.append(intra_mean)
		op_mean = np.mean(op_list)
		skill_list.append(op_mean)
	return skill_list
		

def averaged_double_histogram(res1, res2, mean = True, variance = False, save = True, savepath = None):

	if save == True and savepath is None:
		savepath = 'res/autism_double_averaged_histogram2'

	skill1 = get_averaged_skill_list(res1, mean, variance)
	skill2 = get_averaged_skill_list(res2, mean, variance)
	
	N = len(skill1)
	lengths = range(N)
	w = 0.4	
	
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.bar(lengths, skill1, w, color='blue', label='neurotypical', alpha =0.5)
	ax.bar([l + w for l in lengths] ,skill2, w, color='red', label= 'savantism', alpha = 0.5)
	ax.set_xlabel('Skill')
	ax.set_ylabel('MSE error')
	ax.set_title('Savant network vs non savant')
	plt.legend()
	plt.show()

	if save:
		fig.savefig(savepath)
		plt.close(fig)
	return fig


def averaged_histogram(reslist, epoch_limit = 100, mean = True, variance = False):
	#we're presumably just going to pass this the reslist, so we need to deal with this
	
	reslist = np.array(reslist)	#check list
	#we now need to generate the reslist
	interlist = []
	for i in xrange(len(reslist)):	#too hacky
		interlist.append(reslist[:,i,:,:,:])
	

	shape = interlist[0].shape
	num_intras = shape[0]
	num_networks = shape[1]
	num_epochs  = shape[2]
	num_skills = shape[3]
	
	#graph stuff!
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.set_xlabel('Skill number')
	ax.set_ylabel('Average MSE error')
	ax.set_title('Skill of different interconnections averaged across networks, intras, and epochs')
	#x = range(len(interlist))
	x = np.ones(5)
	w =0.4

	res_list = []
	i = 0
	for inter in interlist:
		inter1 = inter[:,:,num_epochs-epoch_limit:num_epochs, :]
		print inter1.shape
		skill_list = []
		for i in xrange(num_skills): # for each skill, we want an average
			op_list = []
			inter = inter1[:,:,:,i]	
			for j in xrange(num_intras): # for each intra
				intra = inter[j,:,:]
				intralist = []
				for k in xrange(epoch_limit): # for each epoch
					line = intra[:,k]
					if mean == True:
						line_mean = np.mean(line)
						intralist.append(line_mean)
					if variance == True:
						line_var = np.var(line)
						intralist.append(line_var)
				intra_mean = np.mean(intralist)
				op_list.append(intra_mean)
			op_mean = np.mean(op_list)
			skill_list.append(op_mean)
		res_list.append(skill_list)
		
		#we plot each bar on the bar chart here!
		#print skill_list
		#ax.bar([xs + i*w for xs in x], skill_list, label = 'inter' + str(i))
		#i +=1
	#print res_list
	colours = ['blue', 'green', 'red', 'cyan','magenta','yellow', 'black']
	for i in xrange(len(res_list)):
		x = np.ones(5)
		ax.bar([xs + i*w for xs in x], res_list[i], label = 'inter' + str(i), color= colours[i])
		

	plt.legend()
	plt.show()
	return fig, res_list


#RESULT_LIST = np.array(RESULT_LIST)
#RESULT_LIST_SHUFFLE = np.array(RESULT_LIST_SHUFFLE)
#RESULT_LIST_SIZED = np.array(RESULT_LIST_SIZED)
RESULT_LIST_TIME = np.array(RESULT_LIST_TIME)

#averaged_double_histogram(RESULT_LIST_SHUFFLE[:,1,:,:,:], RESULT_LIST_SHUFFLE[:,6,:,:,:], savepath = 'res/autism_double_averaged_histogram_shuffle')

#averaged_double_histogram(RESULT_LIST_SIZED[:,1,:,:,:], RESULT_LIST_SIZED[:,6,:,:,:], savepath = 'res/autism_double_averaged_histogram_sized')

averaged_double_histogram(RESULT_LIST_TIME[:,2,:,:,:], RESULT_LIST_TIME[:,6,:,:,:], savepath = 'res/autism_double_averaged_histogram_time2')

