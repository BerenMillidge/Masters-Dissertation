# Author: Beren Millidge, with substantial contributions from past Msc student, Florian Bolenz
# MSc Dissertation 
# Summer 2017

import numpy as np
import pickle

def save_masks(mask_list, filename):
    """ save masks to file in directory "masks"

        Args:
            mask_list: list of Mask objects
            filename: str, name of file

        Returns:
            None
    """
    pickle.dump(mask_list, open('masks/%s'%filename, 'wb'))
    print "Data saved."

def create_mask_submatrix(units_in, out_on, out_off, contype='symmetric', shuffle=False):
    """ create matrix that specifies connections from one hemisphere to either itself or the other hemisphere. Is used to build complete mask matrix
        for each input unit i, define a central output unit j:
            a) if i is similar or smaller than the total number of output units, set j=i
            b) if i is greater than the number of output units, set j=i-n*output_units (rotate over output units)
        then switch on connections from i to output evenly around the central output unit j and, if out_on is odd, switch also on connection from i to j

        Args:
            units_in: int, number of input units (that is, units the connections are coming from)
            out_on: int, number of active output units (that is, units the connections are going to) for each input unit
            out_off: int, number of inactive output units for each input unit
            contype: 'symmetric' (connections go uniformly from a unit in both directions), 'asymmetric-withrecurrent' (all units have recurrent connection), 'asymmetric-norecurrent' (no unit has recurrent connection)
            shuffle: shuffle connections after creating matrix (beta, not used)

        Returns:
            submatrix: numpy-array of shape (units_in, out_on+out_off) where all elements are either 0 or 1
    """
    assert type(units_in) == int and units_in > 0, 'units_in must be of type int and greater than 0, got %s'%str(type(units_in))
    assert type(out_on) == int and type(out_off) == int, 'out_on and out_off must be int, got %s and %s'%(str(type(out_on)), str(type(out_off)))
    assert out_on >= 0 and out_off >= 0, 'Values of out_on and out_off must be nonnegative, got %.0f and %0.f'%(out_on, out_off)
    assert contype in ['symmetric', 'asymmetric-withrecurrent', 'asymmetric-norecurrent']
    submatrix = np.zeros((units_in, out_on+out_off))
    for i in xrange(units_in):
        c = i%(out_on+out_off) # define central unit
        if (out_on % 2) or out_off == 0: # if on is odd or all connections are on
            submatrix[i,c] = 1
        for j in xrange(out_on/2):
            dist = j + 1
            submatrix[i,(c+dist)%(out_on+out_off)] = 1
            submatrix[i,(c-dist)%(out_on+out_off)] = 1
        if contype == 'asymmetric-withrecurrent' and not (out_on % 2): # out_on is even
            submatrix[i,c] = 1
            submatrix[i,(c+(out_on/2))%(out_on+out_off)] = 0
        if contype == 'asymmetric-norecurrent' and (out_on % 2): # out_on is odd
            submatrix[i,c] = 0
            submatrix[i,(c+1+(out_on/2))%(out_on+out_off)] = 1

        if shuffle:
            np.random.shuffle(submatrix[i,:])
    return submatrix

def create_random_submatrix(units_in, out_on, out_off, contype="symmetric", shuffle=False):
    assert type(units_in) == int and units_in > 0, 'units_in must be of type int and greater than 0, got %s'%str(type(units_in))
    assert type(out_on) == int and type(out_off) == int, 'out_on and out_off must be int, got %s and %s'%(str(type(out_on)), str(type(out_off)))
    assert out_on >= 0 and out_off >= 0, 'Values of out_on and out_off must be nonnegative, got %.0f and %0.f'%(out_on, out_off)
    assert contype in ['symmetric', 'asymmetric-withrecurrent', 'asymmetric-norecurrent']
    submatrix = np.zeros((units_in, out_on+out_off))
    #we've copied all the asserts, but really that's all we need
    on_fraction = out_on/(out_off+out_on)
    for i in xrange(units_in):
	rand = np.random.uniform()
	if rand <= on_fraction:
		submatrix[i,c] = 1
    if shuffle:
	np.random.shuffle(submatrix[i,:])
    return submatrix

def create_mask_matrix(shape, left_in, left_out, l2l_on, l2r_on, r2l_on, r2r_on, contype='symmetric', shuffle=False):
    """ create matrix that specifies connections within and between left and right hemisphere

        Args:
            shape: tupel of two ints, define shape of matrix (input units, output units)
            left_in: int, number of input units belonging to left hemisphere
            left_out: int, number of output units belonging to left hemisphere
            l2l_on: int, number of active connections from each unit in left hemisphere to all units in left hemisphere
            l2r_on: int, number of active connections from each unit in left hemisphere to all units in right hemisphere
            r2l_on: int, number of active connections from each unit in right hemisphere to all units in left hemisphere
            r2r_on: int, number of active connections from each unit in right hemisphere to all units in right hemisphere
            contype: 'symmetric' (connections go uniformly from a unit in both directions), 'asymmetric-withrecurrent' (all units have recurrent connection), 'asymmetric-norecurrent' (no unit has recurrent connection)
            shuffle: shuffle connections after creating matrix (beta, not used)


        Returns:
            matrix: numpy-array of shape defined by argument where all elements are either 0 or 1
    """
    assert [type(x) for x in shape] == [int, int] and all([x >= 0 for x in shape]), 'shape must contain exactly two positive integers, got %s'%str(shape)
    arg_list = [left_in, left_out, l2l_on, l2r_on, r2l_on, r2r_on]
    assert all([type(x) == int for x in arg_list]), 'All arguments must be int, got %s'%str([type(x) for x in arg_list])
    assert all([x >= 0 for x in arg_list]), 'All arguments must be nonnegative, got %s'%str(arg_list)
    assert all([x >= 1 for x in arg_list[0:2]]), 'left_in and left_out must be greater or equal to 1, got %s'%str(arg_list[0:2])
    assert shape[0]-left_in>=1, 'shape[0] must be greater than left_in, got %.0f <= %.0f'%(shape[0],left_in)
    assert shape[1]-left_out>=1, 'shape[1] must be greater than left_out, got %.0f <= %.0f'%(shape[1],left_out)
    assert left_out>=l2l_on, 'left_out must be greater or equal to l2l_on, got %.0f, %.0f'%(left_out, l2l_on)
    assert left_out>=r2l_on, 'left_out must be greater or equal to r2l_on, got %.0f, %.0f'%(left_out, r2l_on)
    assert shape[1]-left_out>=r2r_on, 'shape[1]-left_out must be greater or equal to r2r_on, got %.0f, %.0f'%(shape[1]-left_out, r2r_on)
    assert shape[1]-left_out>=l2r_on, 'shape[1]-left_out must be greater or equal to l2r_on, got %.0f, %.0f'%(shape[1]-left_out, l2r_on)
    assert contype in ['symmetric', 'asymmetric-withrecurrent', 'asymmetric-norecurrent']
    right_in = shape[0] - left_in
    right_out = shape[1] - left_out
    matrix = np.empty(shape)
    matrix[0:left_in,0:left_out] = create_mask_submatrix(units_in=left_in, out_on=l2l_on, out_off=left_out-l2l_on, contype=contype, shuffle=shuffle)
    matrix[0:left_in,left_out:] = create_mask_submatrix(units_in=left_in, out_on=l2r_on, out_off=right_out-l2r_on, contype=contype, shuffle=shuffle)
    matrix[left_in:,0:left_out] = create_mask_submatrix(units_in=right_in, out_on=r2l_on, out_off=left_out-r2l_on, contype=contype, shuffle=shuffle)
    matrix[left_in:,left_out:] = create_mask_submatrix(units_in=right_in, out_on=r2r_on, out_off=right_out-r2r_on, contype=contype, shuffle=shuffle)
    #pdb.set_trace()
    return matrix

def create_mixed_matrix(shape, topleft):
    matrix = np.zeros(shape)
    if topleft==1:
        matrix[0::2,0::2] = 1
        matrix[1::2,1::2] = 1
    elif topleft==0:
        matrix[1::2,0::2] = 1
        matrix[0::2,1::2] = 1
    return matrix


class MaskContainer:

    def __init__(self, path):
        """ initialize MaskContainer object

        Args:
            path: path to saved mask file (file should be a list of mask objects)

        Returns:
            None
        """
        self.path = path
        self.mask_list = pickle.load(open(path, "rb"))
        self.check_masks()
        self.idx = 0    # idx points at position in mask_list, when container is iterated over

    def check_masks(self):
        """ check if all elements in mask_list are mask objects and have identical shapes

        Args:
            None

        Returns:
            None
        """
        assert all([issubclass(type(x), Mask) for x in self.mask_list]), 'Args must be of class Mask or a Mask subclass, got %s'%str([type(x) for x in self.mask_list])
        assert all([x.get_shapes() == self.mask_list[0].get_shapes() for x in self.mask_list]), 'Masks must not differ in shapes, got %s'%str(x.get_shapes() for x in self.mask_list)

    def get_labels(self):
        """ return labels of all masks

        Args:
            None

        Returns:
            labels: list of strings
        """
        labels = [x.get_label() for x in self.mask_list]
        return labels

    def __iter__(self):
        """ makes object iterable

        Args:
            None

        Returns:
            self
        """
        return self

    def next(self):
        """ defines output when object is used in iteration: return objects in mask_list successively

        Args:
            None

        Returns:
            mask
        """
        if self.idx >= len(self.mask_list):
            self.idx = 0
            raise StopIteration()
        mask = self.mask_list[self.idx]
        self.idx = self.idx+1
        return mask

    def count_masks(self):
        """ returns number of masks in container

        Args:
            None

        Returns:
            int
        """
        return len(self.mask_list)

    def get_shapes(self):
        """ returns shapes of masks (since all masks have identical shape, just look at first mask)

        Args:
            None

        Returns:
            shapes: tupel of shape ((in,hid), (hid,hid), (hid,out))
        """
        shapes =  self.mask_list[0].get_shapes()
        return shapes

    def print_masks(self):
        """ prints matrices for all masks

        Args:
            None

        Returns:
            None
        """
        for mask in self:
            print mask.get_label()
            print 'mask_in'
            print mask.mask_in
            print 'mask_hid'
            print mask.mask_hid
            print 'mask_out'
            print mask.mask_out
            print "------------------------------"


class Mask(object):
    def __init__(self, mask_in, mask_hid, mask_out, label):
        """ initialize Mask object

        Args:
            mask_in: 2-dim numpy array, defines mask for input-hidden connections
            mask_hidden: 2-dim numpy array, defines mask for hidden-hidden connections
            mask_out: 2-dim numpy array, defines mask for hidden-output connections
            label: string, name of mask

        Returns:
            None
        """
        self.mask_in = mask_in
        self.mask_hid = mask_hid
        self.mask_out = mask_out
        self.label = label

    def get_masks(self):
        """ return masks

        Args:
            None

        Returns:
            dict with keys "mask_in", "mask_hid", "mask_out" and 2-dim numpy arrays as values
        """
        return {'mask_in':self.mask_in, 'mask_hid':self.mask_hid, 'mask_out':self.mask_out}

    def get_label(self):
        """ return label of mask

        Args:
            None

        Returns:
            str
        """
        return self.label

    def get_shapes(self):
        """ return shapes of mask

        Args:
            None

        Returns:
            tupel of shape ((in,hid), (hid,hid), (hid,out))
        """
        return (self.mask_in.shape, self.mask_hid.shape, self.mask_out.shape)

class ConstantMask(Mask):

   def __init__(self, mask_in, mask_hid, mask_out, label=None):
       """ initialize ConstantMask object (mask stays constant over training and multiple networks)

        Args:
            mask_in: 2-dim numpy array, defines mask for input-hidden connections
            mask_hidden: 2-dim numpy array, defines mask for hidden-hidden connections
            mask_out: 2-dim numpy array, defines mask for hidden-output connections
            label: string, name of mask

        Returns:
            None
        """
       assert mask_in.shape[1] == mask_hid.shape[0], 'Dimensions of mask_in and mask_hid do not match, got %s and %s'%(str(mask_in.shape), str(mask_hid.shape))
       assert mask_hid.shape[0] == mask_hid.shape[1], 'mask_hid must be quadratic matrix, got %s'%str(mask_hid.shape)
       assert mask_hid.shape[1] == mask_out.shape[0], 'Dimensions of mask_hid and mask_out do not match, got %s and %s'%(str(mask_hid.shape), str(mask_out.shape))
       super(ConstantMask, self).__init__(mask_in, mask_hid, mask_out, label)

   def reinit(self):
       """ reinitialzes mask values (nothing happens here becaue mask stays constant)

        Args:
            None

        Returns:
            None
        """
       pass

# class ShuffledHiddenMask(ConstantMask):
#
#     def __init__(self, mask_in, mask_hid, mask_out, units_hid_left=None, label=None):
#         super(ShuffledHiddenMask, self).__init__(mask_in, mask_hid, mask_out, label)
#         if units_hid_left is None:
#             units_hid_left = self.get_shapes()[1][0]/2.0
#             assert units_hid_left == int(units_hid_left), 'units_hid_left must be specified when hidden layer has an odd number of units'
#         self.units_hid_left = int(units_hid_left)
#         self.total_units=self.get_shapes()[1][0]
#
#     def shuffle_hid_submatrix(self, topleft, bottomright):
#         for i in range(topleft[0], bottomright[0]+1):
#             np.random.shuffle(self.mask_hid[i, topleft[1]:bottomright[1]+1])
#
#     def reinit(self):
#         self.shuffle_hid_submatrix(topleft=(0,0), bottomright=(self.units_hid_left-1,self.units_hid_left-1)) #l2l
#         self.shuffle_hid_submatrix(topleft=(self.units_hid_left,self.units_hid_left), bottomright=(self.total_units-1,self.total_units-1)) #r2r
#         self.shuffle_hid_submatrix(topleft=(0,self.units_hid_left), bottomright=(self.units_hid_left-1,self.total_units-1)) #l2r
#         self.shuffle_hid_submatrix(topleft=(self.units_hid_left,0), bottomright=(self.total_units-1,self.units_hid_left-1))
#
#
# class RandomHiddenMask(Mask):
#
#     def __init__(self, mask_in, mask_out, units_left, units_right, l2l_on, l2r_on, r2l_on, r2r_on, label):
#         self.units_left = units_left
#         self.units_right = units_right
#         self.units = units_left + units_right
#         assert mask_in.shape[1] == self.units, 'Dimensions of mask_in and self.units do not match, got %s and %s'%(str(mask_in.shape), str(self.units))
#         assert self.units == mask_out.shape[0], 'Dimensions of self.units and mask_out do not match, got %s and %s'%(str(self.units), str(mask_out.shape))
#         assert self.units_left**2 >= l2l_on, 'More left-to-left connections than expected. Maximum %.0f, got %.0f'%(self.units_left**2, l2l_on)
#         assert self.units_left*self.units_right >= l2r_on, 'More left-to-right connections than expected. Maximum %.0f, got %.0f'%(self.units_left*self.units_right, l2r_on)
#         assert self.units_left*self.units_right >= r2l_on, 'More right-to-left connections than expected. Maximum %.0f, got %.0f'%(self.units_left*self.units_right, r2l_on)
#         assert self.units_right**2 >= r2r_on, 'More right-to-right connections than expected. Maximum %.0f, got %.0f'%(self.units_left**2, r2r_on)
#
#         self.l2l_on = l2l_on
#         self.l2r_on = l2r_on
#         self.r2l_on = r2l_on
#         self.r2r_on = r2r_on
#         super(RandomHiddenMask, self).__init__(mask_in, np.empty((self.units, self.units)), mask_out, label)
#
#     def get_submatrix(self, shape, on):
#         total = shape[0]*shape[1]
#         off = total - on
#         connections = np.concatenate((np.ones(on), np.zeros(off)))
#         np.random.shuffle(connections)
#         submatrix = connections.reshape(shape)
#         return submatrix
#
#     def reinit(self):
#         self.mask_hid[0:self.units_left,0:self.units_left] = self.get_submatrix(shape=(self.units_left, self.units_left), on=self.l2l_on)
#         self.mask_hid[0:self.units_left,self.units_left:] = self.get_submatrix(shape=(self.units_left, self.units_right), on=self.l2r_on)
#         self.mask_hid[self.units_left:,0:self.units_left] = self.get_submatrix(shape=(self.units_right, self.units_left), on=self.r2l_on)
#         self.mask_hid[self.units_left:,self.units_left:] = self.get_submatrix(shape=(self.units_right, self.units_right), on=self.r2r_on)


















