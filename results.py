# Author: Beren Millidge
# MSc Dissertation
# Summer 2017

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pickle
import statsmodels.sandbox.stats.multicomp as multicomp
import statsmodels.graphics.gofplots as gofplots


def mannwhitneyu(x, y, use_continuity=True, alternative='two-sided'):
    # define this function only to declare new default for alternative parameter
    return stats.mannwhitneyu(x, y, use_continuity=True, alternative='two-sided')

def xstring(string):
    """ returns empty string if input is None and returns input string otherwise

        Args:
            string: str or None

        Returns:
            string
    """
    if not string:
        return ""
    return string

def load_results(path, verbose=False):
    """ load results from previous experiment

        Args:
            path: str, path to saved results file (should be a dict)

        Returns:
            res_train: error terms of training data, shape (number of masks, number of networks, number of epochs+1)
            res_valid: error terms of validation data, shape (number of masks, number of networks, number of epochs+1)
            mask_labels: list of string, labels of masks used in experiment
    """
    res = pickle.load(open(path, "rb"))
    res_train = res['res_train']
    res_valid = res['res_valid']
    mask_labels = res['maskcont'].get_labels()
    scheduler_labels = res['lrcont'].get_labels()

    labels = []
    for idx in xrange(len(mask_labels)):
        label = xstring(mask_labels[idx]) + xstring(scheduler_labels[idx % len(scheduler_labels)])
        labels.append((label))

    if verbose:
        print "seed: ", res["seed"]

    return res_train, res_valid, labels


def p_symbol(p):
    """ translate p-value into symbol

        Args:
            p: float

        Returns:
            s: string
    """
    if p < 0.001:
        s = "***"
    elif p < 0.01:
        s = "**"
    elif p < 0.05:
        s = "*"
    elif p < 0.1:
        s = u'\u2020' # dagger
    else:
        s = ""
    return s

def get_binsfd(data, n, minbins=4):
    """ use Freedman-Diaconis rule to get bins for histogram

        Args:
            data: numpy array
            n: number of observations
            minbins: int, minimum number of bins (useful for very small samples)

        Returns:
            bins: one-dim numpy array
    """
    min_value = data.min()
    max_value = data.max()
    q75, q25 = np.percentile(data, [75 ,25])
    iqr = q75 - q25
    bin_width = 2 * iqr * n**(-1/3.0)    #Freedman-Diaconis rule
    bins = np.linspace(min_value, max_value, max(4,(max_value-min_value)/bin_width)) # use max function to avoid too few bins with small samples # max-value is inclusive

    return bins


class Result:

    def __init__(self, data, labels=None, name=None):
        """ initializes Result object

        Args:
            data: error terms over multiple network trainings for constant data set, shape (mask, network, epoch)
            mask_labels: list of strings, specifies labels of all masks used in experiment
            name: name of data

        Returns:
            None
        """
        self.data = data
        self.mean = np.mean(data, axis=1) #  mean for each mask and epoch
        self.var = np.var(data, axis=1, ddof=1) #  variance for each mask and epoch
        self.std = np.std(data, axis=1, ddof=1) #  standard deviation for each mask and epoch
        self.name = name
        self.masks = data.shape[0] # number of masks
        self.labels = labels
        self.colors = ['blue', 'green', 'red', 'black', 'yellow', 'orange', 'purple', 'brown']
        self.epochs = data.shape[2] # number of epochs
        self.n = data.shape[1] # number of network per mask

        plt.ion() # switch interative mode on in order to not block script when showing plots

    def get_mean(self):
        """ return mean errors

        Args:
            None

        Returns:
            numpy array of shape (masks, epochs)
        """
        return self.mean

    def get_var(self):
        """ return variance of errors

        Args:
            None

        Returns:
            numpy array of shape (masks, epochs)
        """
        return self.var

    def get_vr(self, sub=None, bottom=0, top=None, step=1):
        """ return variance ratio of two network groups

        Args:
            sub: list of 2 ints, defines two subgroups
            bottom: which epoch to start (inclusive)
            top: which epoch to end (exclusive)
            step: step size

        Returns:
            array of ratios

        """
        if not sub:
            sub = range(self.masks)
        assert len(sub) == 2, "Number of groups must be 2, got %.0f. Specify sub parameter." % len(sub)

        if not top:
            top = self.epochs
        vr = self.var[sub[0],bottom:top:step]/self.var[sub[1],bottom:top:step]
        return vr

    def get_d(self, sub=None, bottom=0, top=None, step=1):
        """ returns cohens d for two network groups

        Args:
            sub: list of 2 ints, defines two subgroups
            bottom: which epoch to start (inclusive)
            top: which epoch to end (exclusive)
            step: step size

        Returns:
            array of effects
        """
        if not sub:
            sub = range(self.masks)
        assert len(sub) == 2, "Number of groups must be 2, got %.0f. Specify sub parameter." % len(sub)

        if not top:
            top = self.epochs

        d = (self.mean[sub[0],bottom:top:step]-self.mean[sub[1],bottom:top:step])/np.sqrt((self.var[sub[0],bottom:top:step]+self.var[sub[1],bottom:top:step])/2)
        return d

    def count_groups(self):
        return self.masks

    def set_labels(self, labels):
        """ set new labels

        Args:
            labels: list of strings

        Returns:
            None
        """
        assert len(labels) == self.masks, 'Number of labels must be identical to number of groups. Expected %0.f, got %0.f'%(self.masks, len(labels))
        self.labels = labels

    def set_colors(self, colors):
        """ set new colors

        Args:
            colors: list of strings

        Returns:
            None
        """
        assert len(colors) >= self.masks, 'Number of colors must be equal or greater to number of groups. Expected %0.f, got %0.f'%(self.masks, len(colors))
        self.colors = colors

    def set_name(self, name):
        self.name = name

    def rescale(self, new_center, new_std, invert=False, sub=None):
        """ rescales data for each epoch indiviually across masks and networks

        Args:
            center: new mean
            std: new standard deviation
            invert: whether to invert values or not
            sub: specifies subgroups (list of ints)

        Returns:
            None
        """
        if not sub:
            sub = range(self.masks)

        if invert:
            self.data[sub,:,:] = self.data[sub,:,:]*(-1)
        for epoch in xrange(self.epochs):
            mean = np.mean(self.data[sub,:,epoch])
            std = np.std(self.data[sub,:,epoch])
            self.data[sub,:,epoch] = (((self.data[sub,:,epoch]-mean)/std)*new_std)+new_center
        self.mean = np.mean(self.data, axis=1) #  mean for each mask and epoch
        self.var = np.var(self.data, axis=1, ddof=1) #  variance for each mask and epoch
        self.std = np.std(self.data, axis=1, ddof=1) #  standard deviation for each mask and epoch

    def IQtransform(self, sub=None):
        """ rescales data to an IQ scale (mean=100, std=15, higher values mean better performance --> therefore invert)

        Args:
            sub: specifies subgroups (list of ints)

        Returns:
            None
        """
        self.rescale(new_center=100, new_std=15, invert=True, sub=sub)

    def print_mean(self, sub=None, bottom=0, top=None, step=1):
        if not sub:
            sub = range(self.masks)

        if not top:
            top = self.epochs

        print "MEAN: %s"%self.name
        print "Epoch\t",
        for i in sub:
            label = self.labels[i]
            print "%s\t\t"%label[0:9],
        print ""
        for i in xrange(bottom, top, step):
            print "%.0f\t"%i,
            for j in sub:
                print "%.5f\t\t"%self.mean[j,i],
            print ""

    def print_std(self, sub=None, bottom=0, top=None, step=1):
        if not sub:
            sub = range(self.masks)

        if not top:
            top = self.epochs

        print "STANDARD DEVIATION: %s"%self.name
        print "Epoch\t",
        for i in sub:
            label = self.labels[i]
            print "%s\t\t"%label[0:9],
        print ""
        for i in xrange(bottom, top, step):
            print "%.0f\t"%i,
            for j in sub:
                print "%.5f\t\t"%self.std[j,i],
            print ""

    def print_max(self, sub=None, bottom=0, top=None, step=1):
        if not sub:
            sub = range(self.masks)

        if not top:
            top = self.epochs

        print "MAX VALUE: %s"%self.name
        print "Epoch\t",
        for i in sub:
            label = self.labels[i]
            print "%s\t\t"%label[0:9],
        print ""
        for i in xrange(bottom, top, step):
            print "%.0f\t"%i,
            for j in sub:
                print "%.5f\t\t"%max(self.data[j,:,i]),
            print ""

    def print_min(self, sub=None, bottom=0, top=None, step=1):
        if not sub:
            sub = range(self.masks)

        if not top:
            top = self.epochs

        print "MIN VALUE: %s"%self.name
        print "Epoch\t",
        for i in sub:
            label = self.labels[i]
            print "%s\t\t"%label[0:9],
        print ""
        for i in xrange(bottom, top, step):
            print "%.0f\t"%i,
            for j in sub:
                print "%.5f\t\t"%min(self.data[j,:,i]),
            print ""

    def print_max_vr(self, sub=None, bottom=0, top=None):
        """ looks for max variance ratio and print information about epoch

        Args:
            sub: list of ints, defines two groups of networks
            bottom: limits range (lower bound, inclusive)
            top: limits range (upper bound, exclusive)

        Returns:
            None
        """
        if not sub:
            sub = range(self.masks)
        assert len(sub) == 2, "Number of groups must be 2, got %.0f. Specify sub parameter." % len(sub)

        if not top:
            top = self.epochs

        vr = self.get_vr(sub=sub, bottom=bottom, top=top)
        max_idx = np.argmax(vr)
        max_epoch = int(max_idx+bottom)

        print "Maximum variance ratio found at epoch %0.2f (search range: %0.2f - %0.2f)"%(max_epoch, bottom, top-1)
        for i in sub:
            print "Mean for %s: %.4f"%(self.labels[i], self.mean[i,max_epoch])
            print "Std for %s: %.4f"%(self.labels[i], self.std[i,max_epoch])
        print ""
        self.kruskalwallis(sub=sub, bottom=max_epoch, top=max_epoch+1)
        print "d: %.4f"%self.get_d(sub=sub, bottom=max_epoch, top=max_epoch+1)

        print ""
        self.levene(sub=sub, bottom=max_epoch, top=max_epoch+1)
        print "VR: %.4f"%self.get_vr(sub=sub, bottom=max_epoch, top=max_epoch+1)
        print ""

    def print_min_d(self, sub=None, bottom=0, top=None):
        """ looks for min Cohens d and print information about epoch

        Args:
            sub: list of ints, defines two groups of networks
            bottom: limits range (lower bound, inclusive)
            top: limits range (upper bound, exclusive)

        Returns:
            None
        """
        if not sub:
            sub = range(self.masks)
        assert len(sub) == 2, "Number of groups must be 2, got %.0f. Specify sub parameter." % len(sub)

        if not top:
            top = self.epochs

        d = self.get_d(sub=sub, bottom=bottom, top=top)
        max_idx = np.argmax(d)
        max_epoch = int(max_idx+bottom)

        print "Minimum Cohen's d found at epoch %0.2f (search range: %0.2f - %0.2f)"%(max_epoch, bottom, top-1)
        for i in sub:
            print "Mean for %s: %.4f"%(self.labels[i], self.mean[i,max_epoch])
            print "Std for %s: %.4f"%(self.labels[i], self.std[i,max_epoch])
        print ""
        self.kruskalwallis(sub=sub, bottom=max_epoch, top=max_epoch+1)
        print "d: %.4f"%self.get_d(sub=sub, bottom=max_epoch, top=max_epoch+1)

        print ""
        self.levene(sub=sub, bottom=max_epoch, top=max_epoch+1)
        print "VR: %.4f"%self.get_vr(sub=sub, bottom=max_epoch, top=max_epoch+1)
        print ""

    def plot_line_chart(self, sub=None, bottom=0, top=None, yaxis=[None, None], is_subplot=False):
        """ plot mean error over epochs for each mask, show variance as error bars

        Args:
            sub: list of integers, specifies a subsample of the masks

        Returns:
            None
        """
        if not sub:
            sub = range(self.masks)

        if not top:
            top = self.epochs
        assert [type(x) for x in [bottom, top]] == [int, int], 'bottom and top must be of type int, got %s'%str([type(x) for x in [bottom, top]])

        if not is_subplot:
            plt.figure()
            plt.xlabel('Epoch', fontsize=28, labelpad=15)
            plt.ylabel('Mean squared error', fontsize=28, labelpad=15)
            plt.suptitle(self.name, fontsize=30)
        for i in sub:
            mean = self.mean[i,:]
            var = self.std[i,:]
            plt.plot(range(bottom, top), mean[bottom:top], color=self.colors[i], label=self.labels[i], linewidth=5.0)
            plt.fill_between(range(bottom, top), mean[bottom:top]-var[bottom:top], mean[bottom:top]+var[bottom:top], alpha=0.2, facecolor=self.colors[i])
        plt.axis([bottom, top, yaxis[0], yaxis[1]])
        fig = plt.gca()
        fig.tick_params(axis='both', which='major', width=1, length=7, labelsize=24)

        plt.legend(prop={'size':28})
        leg = fig.get_legend()
        llines = leg.get_lines()
        plt.setp(llines, linewidth=5.0)

        for bin in self.get_conditional_bins(sub=sub, bottom=bottom, top=top):
            plt.axvspan(bin[0], bin[1], alpha=0.2, facecolor='grey', edgecolor='none')

        # if len(sub) == 2:
        #     for bin in self.get_conditional_bins(sub=sub[::-1], bottom=bottom, top=top):
        #         plt.axvspan(bin[0], bin[1], alpha=0.3, facecolor='red', edgecolor='none')

    def multiplot_line_chart(self, ref=None, sub_list=None, bottom=0, top=None, yaxis=[None, None]):
        """ plots multiple line charts in one figure

        :param ref: reference group plotted in all line charts
        :param sub_list:
        :param bottom:
        :param top:
        :param yaxis:
        :return:

        """
        assert not(ref and sub_list), "Error: do not specify ref and sub_list simultaneously"


        if sub_list is not None:
            sub_list = sub_list
        elif ref is not None:
            sub_list = [sorted([ref, x]) for x in range(self.masks) if x != ref]
        else:
            sub_list = []
            for i in range(self.masks):
                for j in range(i+1,self.masks):
                    sub_list.append([i,j])

        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111)
        plt.xlabel('Epoch', fontsize=28, labelpad=15)
        plt.ylabel('Mean squared error', fontsize=28, labelpad=35)
        plt.suptitle(self.name, fontsize=30)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
        for idx, sub in enumerate(sub_list):
            fig.add_subplot(len(sub_list),1,idx+1)
            self.plot_line_chart(sub=sub, bottom=bottom, top=top, is_subplot=True, yaxis=yaxis)
        plt.subplots_adjust(top=0.96)


    def plot_distribution(self, epoch, sub=None):
        """ plot histogram of errors in specific epoch for each mask

        Args:
            epoch: int, specifies for which epoch to plot errors
            sub: list of integers, specifies a subsample of the masks

        Returns:
            None
        """
        # compute reasonable bin width from data
        if not sub:
            sub = range(self.masks)

        bins = get_binsfd(self.data[sub,:,epoch], self.n)

        plt.figure()
        plt.suptitle("%s, Epoch %.0f"%(self.name, epoch), fontsize=30)
        for i in sub:
            plt.hist(self.data[i,:,epoch], bins, alpha=0.4, label=self.labels[i],  color=self.colors[i])
        plt.legend(loc='upper right')
        plt.xlabel('Mean squared error', fontsize=28, labelpad=15)
        plt.ylabel('Absolute frequency', fontsize=28, labelpad=15)

        fig = plt.gca()
        fig.tick_params(axis='both', which='major', width=1, length=7, labelsize=24)

        plt.legend(prop={'size':28})
        leg = fig.get_legend()
        llines = leg.get_lines()
        plt.setp(llines, linewidth=2.0)

        max_mse = np.max(self.data[sub,:,epoch])
        min_mse = np.min(self.data[sub,:,epoch])
        plt.xlim(min_mse-0.01, max_mse+0.01)

    def plot_proportion(self, epoch, sub=None):
        """ plots proportion of groups for all histogram bins in specific epoch

        Args:
            epoch: int, specifies for which epoch to plot errors
            sub: list of integers, specifies a subsample of the masks

        Returns:
            None
        """
        if not sub:
            sub = range(self.masks)
        assert len(sub) == 2, "Number of groups must be 2, got %.0f. Specify sub parameter." % len(sub)

        bins = get_binsfd(self.data[sub,:,epoch], self.n)
        binmeans = np.asarray([np.mean([bins[x],bins[x+1]]) for x in range(len(bins)-1)])

        proportions = np.empty((len(sub),len(bins)-1))

        for groupidx in xrange(len(sub)):
            group = sub[groupidx]
            othergroup = [x for x in sub if x != group][0]
            for binidx in xrange(len(bins)-1):
                if binidx < len(bins)-2:
                    groupcount = ((bins[binidx] <= self.data[group,:,epoch]) & (self.data[group,:,epoch] < bins[binidx+1])).sum()
                    othergroupcount = ((bins[binidx] <= self.data[othergroup,:,epoch]) & (self.data[othergroup,:,epoch] < bins[binidx+1])).sum()
                else:
                    # for last binidx, make upper bound inclusive
                    groupcount = ((bins[binidx] <= self.data[group,:,epoch]) & (self.data[group,:,epoch] <= bins[binidx+1])).sum()
                    othergroupcount = ((bins[binidx] <= self.data[othergroup,:,epoch]) & (self.data[othergroup,:,epoch] <= bins[binidx+1])).sum()
                try:
                    proportions[groupidx,binidx] = float(groupcount)/float((groupcount+othergroupcount))
                except: # if there is no instance in sample for this bin, save proportion as None
                    proportions[groupidx,binidx] = None

        plt.figure()
        plt.suptitle("%s, Epoch %.0f"%(self.name, epoch), fontsize=30)
        for iprop, isub in zip(xrange(proportions.shape[0]),sub):
            proportions_mask = np.isfinite(proportions[iprop,:].astype(np.double)) # detects missing values in series
            plt.plot(binmeans[proportions_mask], proportions[iprop,proportions_mask], label=self.labels[isub], marker='o', linewidth=5.0, markersize=15.0,  color=self.colors[isub])
        for bin in bins:
            plt.axvline(x=bin, color='black', linestyle='dotted')
        plt.xlabel('Mean squared error', fontsize=28, labelpad=15)
        plt.ylabel('Proportion', fontsize=28, labelpad=15)
        fig = plt.gca()
        fig.tick_params(axis='both', which='major', width=1, length=7, labelsize=24)

        plt.legend(prop={'size':28})
        leg = fig.get_legend()
        llines = leg.get_lines()
        plt.setp(llines, linewidth=5.0)

        plt.ylim(-0.02, 1.02)
        max_mse = np.max(self.data[sub,:,epoch])
        min_mse = np.min(self.data[sub,:,epoch])
        plt.xlim(min_mse-0.01, max_mse+0.01)

    def plot_pvalues(self, sub=None, corrected=True):
        if not sub:
            sub = range(self.masks)

        _, levene_p, levene_p_corr = self.levene(sub=sub, verbose=False)
        _, kw_p, kw_p_corr = self.kruskalwallis(sub=sub, verbose=False)

        plt.figure()
        plt.suptitle(self.name, fontsize=18)
        if not corrected:
            plt.plot(range(self.epochs), levene_p, linestyle='-', c='black', label='levene')
            plt.plot(range(self.epochs), kw_p, linestyle='-', c='red', label='kw')
        else:
            plt.plot(range(self.epochs), levene_p_corr, c='black', label='levene')
            plt.plot(range(self.epochs), kw_p_corr, c='red', label='kw')

        plt.axhline(y=0.05, color='green', linestyle='-')
        plt.axhline(y=0.1, color='green', linestyle='-')
        plt.legend()

        for bin in self.get_conditional_bins(sub=sub):
            plt.axvspan(bin[0], bin[1], alpha=0.3, facecolor='gray', edgecolor='none')

        if len(sub) == 2:
            for bin in self.get_conditional_bins(sub=sub[::-1]):
                plt.axvspan(bin[0], bin[1], alpha=0.3, facecolor='red', edgecolor='none')

    def plot_vr(self, sub=None, bottom=0, top=None, add_lines=None, add_lables=[None], linestyles=['solid'], long_title=False):

        if not sub:
            sub = range(self.masks)
        assert len(sub) == 2, "Number of groups must be 2, got %.0f. Specify sub parameter." % len(sub)

        if not top:
            top = self.epochs
        assert [type(x) for x in [bottom, top]] == [int, int], 'bottom and top must be of type int, got %s'%str([type(x) for x in [bottom, top]])

        vr = self.get_vr(sub=sub, bottom=bottom, top=top)
        plt.figure()
        plt.xlabel('Epoch', fontsize=28, labelpad=15)
        plt.ylabel('Variance ratio', fontsize=28, labelpad=15)
        if long_title:
            plt.suptitle("%s\n%s vs. %s"%(self.name, self.labels[sub[0]], self.labels[sub[1]]), fontsize=30)
        else:
            plt.suptitle(self.name, fontsize=30)
        plt.subplots_adjust(top=0.9)
        plt.plot(range(bottom, top), vr, c='black', linewidth=2.0, label=add_lables[0], linestyle=linestyles[0])

        fig = plt.gca()

        if add_lines:
            for idx, line in enumerate(add_lines):
                plt.plot(range(bottom, top), line, c='black', linewidth=2.0, label=add_lables[idx+1], linestyle=linestyles[idx+1])
            plt.legend(prop={'size':28}, loc=4)
            leg = fig.get_legend()
            llines = leg.get_lines()
            plt.setp(llines, linewidth=2.0)

        fig.tick_params(axis='both', which='major', width=1, length=7, labelsize=24)

        for bin in self.get_conditional_bins(sub=sub):
            plt.axvspan(bin[0], bin[1], alpha=0.3, facecolor='gray', edgecolor='none')

        if len(sub) == 2:
            for bin in self.get_conditional_bins(sub=sub[::-1], bottom=bottom, top=top):
                plt.axvspan(bin[0], bin[1], alpha=0.3, facecolor='red', edgecolor='none')

    def plot_d(self, sub=None, bottom=0, top=None, add_lines=None, add_lables=[None], linestyles=['solid'], long_title=False):

        if not sub:
            sub = range(self.masks)
        assert len(sub) == 2, "Number of groups must be 2, got %.0f. Specify sub parameter." % len(sub)

        if not top:
            top = self.epochs
        assert [type(x) for x in [bottom, top]] == [int, int], 'bottom and top must be of type int, got %s'%str([type(x) for x in [bottom, top]])

        d = self.get_d(sub=sub, bottom=bottom, top=top)
        plt.figure()
        plt.xlabel('Epoch', fontsize=28, labelpad=15)
        plt.ylabel('Cohen\'s d', fontsize=28, labelpad=15)
        if long_title:
            plt.suptitle("%s\n%s vs. %s"%(self.name, self.labels[sub[0]], self.labels[sub[1]]), fontsize=30)
        else:
            plt.suptitle(self.name, fontsize=30)
        plt.subplots_adjust(top=0.9)
        plt.plot(range(bottom, top), d, c='black', linewidth=2.0, label=add_lables[0], linestyle=linestyles[0])

        fig = plt.gca()

        if add_lines:
            for idx, line in enumerate(add_lines):
                plt.plot(range(bottom, top), line, c='black', linewidth=2.0, label=add_lables[idx+1], linestyle=linestyles[idx+1])
            plt.legend(prop={'size':28})
            leg = fig.get_legend()
            llines = leg.get_lines()
            plt.setp(llines, linewidth=2.0)

        fig.tick_params(axis='both', which='major', width=1, length=7, labelsize=24)

        for bin in self.get_conditional_bins(sub=sub):
            plt.axvspan(bin[0], bin[1], alpha=0.3, facecolor='gray', edgecolor='none')

        if len(sub) == 2:
            for bin in self.get_conditional_bins(sub=sub[::-1], bottom=bottom, top=top):
                plt.axvspan(bin[0], bin[1], alpha=0.3, facecolor='red', edgecolor='none')

    def plot_effects(self, sub=None, bottom=0, top=None):
        # plots variance ratio (variance first group/variance second group) and Cohen's d ((mean first group - mean second group)/pooled standard deviation) across epochs
        if not sub:
            sub = range(self.masks)
        assert len(sub) == 2, "Number of groups must be 2, got %.0f. Specify sub parameter." % len(sub)

        if not top:
            top = self.epochs
        assert [type(x) for x in [bottom, top]] == [int, int], 'bottom and top must be of type int, got %s'%str([type(x) for x in [bottom, top]])

        vr = self.get_vr(sub=sub, bottom=bottom, top=top)
        d = self.get_d(sub=sub, bottom=bottom, top=top)

        fig = plt.figure()
        plt.suptitle(self.name, fontsize=30)
        ax1 = fig.add_subplot(111)
        line1 = ax1.plot(range(bottom, top), vr, color='black', linestyle='solid', label='Variance ratio')
        ax1.set_xlabel('Epoch', fontsize=28)
        ax1.set_ylabel('Variance ratio', fontsize=28)

        ax2 = ax1.twinx()
        line2 = ax2.plot(range(bottom, top), d, color='black', linestyle='dotted', label='Cohen\'s d')
        ax2.set_ylabel('Cohen\'s d', fontsize=28)

        fig = plt.gca()
        fig.tick_params(axis='x', which='major', width=1, length=7, labelsize=24)
        lns = line1+line2
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc=4, prop={'size':28})
        leg = fig.get_legend()
        llines = leg.get_lines()
        plt.setp(llines, linewidth=2.0)


        for bin in self.get_conditional_bins(sub=sub, bottom=bottom, top=top):
            plt.axvspan(bin[0], bin[1], alpha=0.3, facecolor='gray', edgecolor='none')

        if len(sub) == 2:
            for bin in self.get_conditional_bins(sub=sub[::-1], bottom=bottom, top=top):
                plt.axvspan(bin[0], bin[1], alpha=0.3, facecolor='red', edgecolor='none')

    def qqplots(self, epoch, sub=None):
        assert 0 <= epoch <= self.epochs, 'epoch must be between 0 and %.0f, got %.0f'%(self.epochs, epoch)
        if not sub:
            sub = range(self.masks)

        for group in sub:
            gofplots.qqplot(self.data[group,:,epoch], fit=True, line='45')
            plt.suptitle("%s, %s, Epoch %.0f"%(self.name, self.labels[group], epoch), fontsize=18)

    def get_conditional_bins(self, sub, threshold_levene=0.05, threshhold_kw=0.1, bottom=0, top=None):
        # conditions: p of levene test is smaller than threshold, p of kw-test is larger than threshold, variance for sub[0] is larger than for sub[1]
        if not sub:
            sub = range(self.masks)

        if not top:
            top = self.epochs

        _, _, levene_p_corr = self.levene(sub=sub, bottom=bottom, top=top, verbose=False)
        _, _, kw_p_corr = self.kruskalwallis(sub=sub, bottom=bottom, top=top, verbose=False)

        if len(sub) == 1:
            return []
        if len(sub) == 2:
            var1 = self.var[sub[0],bottom:top]
            var2 = self.var[sub[1],bottom:top]
            t_bool_list = [((levene_p_corr[x] < threshold_levene) & (kw_p_corr[x] > threshhold_kw) & (var1[x] > var2[x])) for x in range(len(levene_p_corr))] # bool for each time step, whether criteria are met#
        else:
            t_bool_list = [((levene_p_corr[x] < threshold_levene) & (kw_p_corr[x] > threshhold_kw)) for x in range(len(levene_p_corr))] # bool for each time step, whether criteria are met#

        t_bool_list.append(False) # insert False at end to avoid trouble with index out of range
        bins = []
        t_idx = 0
        while t_idx < len(t_bool_list)-1:
            if t_bool_list[t_idx]:
                next_false = t_bool_list[t_idx:].index(False) + t_idx # position of next false element
                bins.append((max(0,t_idx-0.5)+bottom, min(next_false-0.5,len(t_bool_list)-2)+bottom))
                t_idx = next_false+1
            else:
                t_idx = t_idx+1

        return bins

    def levene(self, sub=None, bottom=0, top=None, step=1, center='median', verbose=True):
        """ perform Levene tests (omnibus test for difference in variance) for each epoch with mask as factor

        Args:
            sub: list of integers, specifies a subsample of the masks
            bottom: lower-range for printing epochs (inclusive)
            top: upper range for printing epochs (exclusive)
            step: step size for priniting epochs

        Returns:
            W
            p-values (uncorrected)
            p-values (corrected)
        """
        if not top:
            top = self.epochs
        assert [type(x) for x in [bottom, top, step]] == [int, int, int], 'bottom, top, step must be of type int, got %s'%str([type(x) for x in [bottom, top, step]])

        if not sub:
            sub = range(self.masks)

        res = [] # will become a list of length = number of epochs. each element in list is a tuple (W, p)
        for i in xrange(self.epochs):
            args = np.split(self.data[sub,:,i], len(sub), axis=0) # split results into multiple arrays, one for each mask
            args = [x.reshape((x.shape[1])) for x in args]
            W, p = stats.levene(*args, center=center) # W: test statistic, p: p-value
            res.append([W,p])

        _, p_corr_list = multicomp.fdrcorrection0([x[1] for x in res], alpha=0.05, method='indep') # correct p-value for multiple comparison (Benjamini-Hochberg)

        if verbose:
            print "LEVENE TEST: %s"%self.name
            print "Epoch\t",
            for i in sub:
                label = self.labels[i]
                print "VAR(%s)\t"%label[0:9],
            print "W\tp\tSignif.\tp corr\tSignif."
            for i in xrange(bottom, top, step):
                print "%.0f\t"%i,
                for j in sub:
                    print "%.4f\t\t"%self.var[j,i],
                W, p = res[i]
                p_corr = p_corr_list[i]
                print "%.2f\t%.4f\t%s\t%.4f\t%s"%(W, p, p_symbol(p), p_corr, p_symbol(p_corr))

        return [x[0] for x in res][bottom:top:step], [x[1] for x in res][bottom:top:step], p_corr_list[bottom:top:step]

    def anova(self, sub=None, bottom=0, top=None, step=1, verbose=True):
        """ perform Anova tests (omnibus test for difference in mean) for each epoch with mask as factor

        Args:
            sub: list of integers, specifies a subsample of the masks
            bottom: lower-range for printing epochs (inclusive)
            top: upper range for printing epochs (exclusive)
            step: step size for priniting epochs

        Returns:
            F
            p-values (uncorrected)
            p-values (corrected)
        """
        if not top:
            top = self.epochs
        assert [type(x) for x in [bottom, top, step]] == [int, int, int], 'bottom, top, step must be of type int, got %s'%str([type(x) for x in [bottom, top, step]])

        if not sub:
            sub = range(self.masks)

        res = [] # will become a list of length = number of epochs. each element in list is a tuple (F, p)
        for i in xrange(self.epochs):
            args = np.split(self.data[sub,:,i], len(sub), axis=0) # split results into multiple arrays, one for each mask
            args = [x.reshape((x.shape[1])) for x in args]
            F, p = stats.f_oneway(*args) # F: test statistic, p: p-value
            res.append([F,p])

        _, p_corr_list = multicomp.fdrcorrection0([x[1] for x in res], alpha=0.05, method='indep') # correct p-value for multiple comparison (Benjamini-Hochberg)

        if verbose:
            print "ANOVA: %s"%self.name
            print "Epoch\t",
            for i in sub:
                label = self.labels[i]
                print "MEAN(%s)\t"%label[0:9],
            print "F\tp\tSignif.\tp corr\tSignif"
            for i in xrange(bottom, top, step):
                print "%.0f\t"%i,
                for j in sub:
                    print "%.4f\t\t"%self.mean[j,i],
                F, p = res[i]
                p_corr = p_corr_list[i]
                print "%.2f\t%.4f\t%s\t%.2f\t%s"%(F, p, p_symbol(p), p_corr, p_symbol(p_corr))

        return [x[0] for x in res][bottom:top:step], [x[1] for x in res][bottom:top:step], p_corr_list[bottom:top:step]

    def kruskalwallis(self, sub=None, bottom=0, top=None, step=1, verbose=True):
        """ perform Kruskal Wallis tests (nonparametric equivalent for ANOVA) for each epoch with mask as factor. If only 2 groups are compared, use Mann-Whitney-U Test instead

        Args:
            sub: list of integers, specifies a subsample of the masks
            bottom: lower-range for printing epochs (inclusive)
            top: upper range for printing epochs (exclusive)
            step: step size for priniting epochs

        Returns:
            H
            p-values (uncorrected)
            p-values (corrected)
        """
        if not top:
            top = self.epochs
        assert [type(x) for x in [bottom, top, step]] == [int, int, int], 'bottom, top, step must be of type int, got %s'%str([type(x) for x in [bottom, top, step]])

        if not sub:
            sub = range(self.masks)

        if len(sub) == 2:
            testfun = mannwhitneyu
            testname = "Mann-Whitney"
            teststatname = 'U'
        elif len(sub) > 2:
            testfun = stats.mstats.kruskalwallis
            testname = "Kruskal-Wallis"
            teststatname = 'H'
        else:
            raise ValueError, 'Length of sub must be greater or equal to 2, got %.0f'%len(sub)
        res = [] # will become a list of length = number of epochs. each element in list is a tuple (F, p)
        for i in xrange(self.epochs):
            args = np.split(self.data[sub,:,i], len(sub), axis=0) # split results into multiple arrays, one for each mask
            args = [x.reshape((x.shape[1])) for x in args]
            H, p = testfun(*args) # H: test statistic, p: p-value
            res.append([H,p])

        _, p_corr_list = multicomp.fdrcorrection0([x[1] for x in res], alpha=0.05, method='indep') # correct p-value for multiple comparison (Benjamini-Hochberg)

        if verbose:
            print "%s: %s"%(testname, self.name)
            print "Epoch\t",
            for i in sub:
                label = self.labels[i]
                print "MEAN(%s)\t"%label[0:9],
            print "%s\tp\tSignif.\tp corr\tSignif"%teststatname
            for i in xrange(bottom, top, step):
                print "%.0f\t"%i,
                for j in sub:
                    print "%.4f\t\t"%self.mean[j,i],
                H, p = res[i]
                p_corr = p_corr_list[i]
                print "%.1f\t%.4f\t%s\t%.4f\t%s"%(H, p, p_symbol(p), p_corr, p_symbol(p_corr))

        return [x[0] for x in res][bottom:top:step], [x[1] for x in res][bottom:top:step], p_corr_list[bottom:top:step]

    def shapiro(self, epoch, sub=None):
        assert 0 <= epoch <= self.epochs, 'epoch must be between 0 and %.0f, got %.0f'%(self.epochs, epoch)
        if not sub:
            sub = range(self.masks)

        print "Shapiro-Wilk: %s, Epoch "%self.name
        print "Group\t\tskew\tkurtos\tW\tp\tSignif."
        for group in sub:
            W, p = stats.shapiro(self.data[group,:,epoch])
            skew = stats.skew(self.data[group,:,epoch])
            # For normally distributed data, the skewness should be about 0. A skewness value > 0 means that there is more weight in the left tail of the distribution.
            kurtosis = stats.kurtosis(self.data[group,:,epoch])
            # This definition is used so that the standard normal distribution has a kurtosis of zero. In addition, with the second definition positive kurtosis indicates a "heavy-tailed" distribution and negative kurtosis indicates a "light tailed" distribution.
            print "%s\t%.2f\t%.2f\t%.4f\t%.4f\t%s"%(self.labels[group], skew, kurtosis, W, p, p_symbol(p))
