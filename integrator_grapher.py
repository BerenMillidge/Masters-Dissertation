#Author: Beren Millidge
#Msc Dissertation
#Summer 2017

import numpy as np
import pickle
import matplotlib.pyplot as plt

def histogram_by_digit(digit_data):
	
	#width param	
	width = 0.4
	
	#set up our lists	
	mnist_accs = []
	colour_accs = []
	N = len(digit_data)
	xs = range(N)
	
	
	#fig stuff
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.set_xlabel("colours")
	ax.set_ylabel('Validation Accuracy')
	ax.set_title('Colour performance for MNIST digit')
	for i in xrange(N):
		l  = digit_data[i]
		mnist_accs.append(l[0])
		colour_accs.append(l[1])
	ax.bar(xs, mnist_accs, width, color='blue', label = 'mnist accuracy')
	ax.bar([x + width for x in xs], colour_accs, width, color='red', label = 'colour accuracy')
	plt.legend()
	plt.show()
	return fig
	

def all_digit_histograms_by_number_both(data):
	fig_list = []
	#for the loop
	data = data[1]
	for i in xrange(len(data)):
		fig = histogram_by_digit(data[i])
		fig_list.append(fig)
	
	return fig_list


def variances_across_conditions(results_list):

	#width param	
	width = 0.4	

	#set up figure plotting stuff
	N = len(results_list)
	xs = range(N)
	
	
	#fig stuff
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.set_xlabel("condition")
	ax.set_ylabel('average variance in condition')
	ax.set_title('Variance across conditions')

	#initialise our lists
	var_avg_mnist =[]
	var_avg_col = []
	#run the loop
	for i in xrange(N):
		data = results_list[i][1]
		var_mnist_list = []
		var_col_list = []
		for j in xrange(len(data)):	
			mnist_list = []
			col_list = []
			for k in xrange(len(data[j])): # loop across all colours
				mnist_list.append(data[j][k][0])
				col_list.append(data[j][k][1])
			#we then calculate variances
			var_mnist = np.var(np.array(mnist_list))
			var_col = np.var(np.array(col_list))
			var_mnist_list.append(var_mnist)
			var_col_list.append(var_col)
		#average var lists
		avg_mnist = np.mean(np.array(var_mnist_list))
		avg_col = np.mean(np.array(var_col_list))
		#and append
		var_avg_mnist.append(avg_mnist)
		var_avg_col.append(avg_col)

	ax.bar(xs, var_avg_mnist, width, color='blue', label = 'mnist variance')
	ax.bar([x + width for x in xs], var_avg_col, width, color='red', label = 'colour variance')
	plt.legend()
	plt.show()
	return fig
		
			

def multi_list(results_list):
	assert type(results_list) == list and len(results_list) >=1, 'must be a correct list'

	figs_list = []
	for i in xrange(len(results_list)):
		print "RESULT SET NUMBER " + str(i+1)
		fig_list = all_digit_histograms_by_number_both(results_list[i])
		figs_list.append(fig_list)
	return figs_list



def save_fig(fig, path):
	with open(path, 'wb') as f:
		pickle.dump(fig, f)
	f.close()
	print "SAVED!"

def main():
	#path = 'synaesthesia/synseparate_results'

	#acc_list = pickle.load(open(path, 'rb'))
	#total_acc = acc_list[0]
	#validation_data = acc_list[1]

	#print type(validation_data)
	#print len(validation_data)
	#print len(validation_data[0])
	#print len(validation_data[0][0])

	#all_digit_histograms_by_number_both(validation_data)

	#path = 'synaesthesia/synseparate_results_multi_2'
	#res_list = pickle.load(open(path, 'rb'))	
	#print type(res_list)
	#print len(res_list)
	#figs_list = multi_list(res_list)

	#fig = variances_across_conditions(res_list)
	#save_path = 'synaesthesia/variance_fig1.png'
	#save_fig(fig, save_path)


	#synsearch1 = pickle.load(open('res/synseparate_grid_search','rb'))
	#synsearch2 = pickle.load(open('res/synseparate_grid_search2','rb'))
	#print synsearch1
	#print "  "
	#print synsearch2
	
if __name__ == '__main__':
	main()

