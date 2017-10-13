# Author: Beren Millidge
# MSc Dissertation
# Summer 2017

from mask import *
import pdb

def main():

    # 8 hidden units:
    #
    # mask_in_full = create_mask_matrix(shape=(10,8), left_in=5, left_out=4, l2l_on=4, l2r_on=4, r2l_on=4, r2r_on=4)
    # mask_in_partial = create_mask_matrix(shape=(10,8), left_in=5, left_out=4, l2l_on=4, l2r_on=0, r2l_on=0, r2r_on=4)
    # mask_out_full = create_mask_matrix(shape=(8,10), left_in=4, left_out=5, l2l_on=5, l2r_on=5, r2l_on=5, r2r_on=5)
    # mask_out_partial = create_mask_matrix(shape=(8,10), left_in=4, left_out=5, l2l_on=0, l2r_on=5, r2l_on=5, r2r_on=0)
    # mask_in_zero = create_mask_matrix(shape=(10,8), left_in=5, left_out=4, l2l_on=0, l2r_on=0, r2l_on=0, r2r_on=0)
    #
    # mask_hid_intra3 = create_mask_matrix(shape=(8,8), left_in=4, left_out=4, l2l_on=3, l2r_on=1, r2l_on=1, r2r_on=3)
    # mask_hid_intra2 = create_mask_matrix(shape=(8,8), left_in=4, left_out=4, l2l_on=2, l2r_on=2, r2l_on=2, r2r_on=2)
    # mask_hid_full =  create_mask_matrix(shape=(8,8), left_in=4, left_out=4, l2l_on=4, l2r_on=4, r2l_on=4, r2r_on=4)
    #
    # mask_intra3_full = ConstantMask(mask_in_full, mask_hid_intra3, mask_out_full, label="intra3f")
    # mask_intra2_full = ConstantMask(mask_in_full, mask_hid_intra2, mask_out_full, label="intra2f")
    # mask_intra3_partial = ConstantMask(mask_in_partial, mask_hid_intra3, mask_out_partial, label="intra3p")
    # mask_intra2_partial = ConstantMask(mask_in_partial, mask_hid_intra2, mask_out_partial, label="intra2p")
    # mask_zero =  ConstantMask(mask_in_zero, mask_hid_full, mask_out_full, label="zero_in")
    #
    # mask_list_fullzero = [mask_intra3_full, mask_intra2_full, mask_zero]
    # mask_list_full = [mask_intra3_full, mask_intra2_full]
    # mask_list_partial = [mask_intra3_partial, mask_intra2_partial]
    #
    # save_masks(mask_list_fullzero, 'interintrazero_3m_10iu_8hu_10ou_fullin_fullout')
    # save_masks(mask_list_full, 'interintra_2m_10iu_8hu_10ou_fullin_fullout')
    # save_masks(mask_list_partial, 'interintra_2m_10iu_8hu_10ou_partialin_partialout')

    # 12 hidden units (20 input units):

    # mask_in_partial = create_mask_matrix(shape=(20,12), left_in=10, left_out=6, l2l_on=6, l2r_on=0, r2l_on=0, r2r_on=6)
    # mask_out_partial = create_mask_matrix(shape=(12,20), left_in=6, left_out=10, l2l_on=0, l2r_on=10, r2l_on=10, r2r_on=0)
    #
    # mask_hid_intra5 = create_mask_matrix(shape=(12,12), left_in=6, left_out=6, l2l_on=5, l2r_on=1, r2l_on=1, r2r_on=5)
    # mask_hid_intra4 = create_mask_matrix(shape=(12,12), left_in=6, left_out=6, l2l_on=4, l2r_on=2, r2l_on=2, r2r_on=4)
    # mask_hid_intra3 = create_mask_matrix(shape=(12,12), left_in=6, left_out=6, l2l_on=3, l2r_on=3, r2l_on=3, r2r_on=3)
    #
    # mask_intra5_partial = ConstantMask(mask_in_partial, mask_hid_intra5, mask_out_partial, label="intra5p")
    # mask_intra4_partial = ConstantMask(mask_in_partial, mask_hid_intra4, mask_out_partial, label="intra4p")
    # mask_intra3_partial = ConstantMask(mask_in_partial, mask_hid_intra3, mask_out_partial, label="intra3p")
    #
    # mask_list_partial = [mask_intra5_partial, mask_intra4_partial, mask_intra3_partial]
    # save_masks(mask_list_partial, 'interintra_3m_20iu_12hu_20ou_partialin_partialout')


    # # 14 hidden units (20 input units):
    #
    # mask_in_partial = create_mask_matrix(shape=(20,14), left_in=10, left_out=7, l2l_on=7, l2r_on=0, r2l_on=0, r2r_on=7)
    # mask_out_partial = create_mask_matrix(shape=(14,20), left_in=7, left_out=10, l2l_on=0, l2r_on=10, r2l_on=10, r2r_on=0)
    #
    # mask_hid_intra6 = create_mask_matrix(shape=(14,14), left_in=7, left_out=7, l2l_on=6, l2r_on=1, r2l_on=1, r2r_on=6)
    # mask_hid_intra5 = create_mask_matrix(shape=(14,14), left_in=7, left_out=7, l2l_on=5, l2r_on=2, r2l_on=2, r2r_on=5)
    # mask_hid_intra4 = create_mask_matrix(shape=(14,14), left_in=7, left_out=7, l2l_on=4, l2r_on=3, r2l_on=3, r2r_on=4)
    #
    # mask_intra6_partial = ConstantMask(mask_in_partial, mask_hid_intra6, mask_out_partial, label="intra6p")
    # mask_intra5_partial = ConstantMask(mask_in_partial, mask_hid_intra5, mask_out_partial, label="intra5p")
    # mask_intra4_partial = ConstantMask(mask_in_partial, mask_hid_intra4, mask_out_partial, label="intra4p")
    #
    # mask_list_partial = [mask_intra6_partial, mask_intra5_partial, mask_intra4_partial]
    # save_masks(mask_list_partial, 'interintra_3m_20iu_14hu_20ou_partialin_partialout')

    # 16 hidden units:

    mask_in_full = create_mask_matrix(shape=(20,16), left_in=10, left_out=8, l2l_on=8, l2r_on=8, r2l_on=8, r2r_on=8)
    #pdb.set_trace()
    mask_in_partial = create_mask_matrix(shape=(20,16), left_in=10, left_out=8, l2l_on=8, l2r_on=0, r2l_on=0, r2r_on=8)
    mask_in_mixed = create_mixed_matrix(shape=(20,16), topleft=0)
    mask_out_full = create_mask_matrix(shape=(16,20), left_in=8, left_out=10, l2l_on=10, l2r_on=10, r2l_on=10, r2r_on=10)
    mask_out_partial = create_mask_matrix(shape=(16,20), left_in=8, left_out=10, l2l_on=0, l2r_on=10, r2l_on=10, r2r_on=0)
    mask_out_mixed = create_mixed_matrix(shape=(16,20), topleft=1)

    # mask_hid_intra7 = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=1, r2l_on=1, r2r_on=7)
    # mask_hid_intra6 = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=2, r2l_on=2, r2r_on=6)
    # mask_hid_intra5 = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=3, r2l_on=3, r2r_on=5)
    # mask_hid_intra4 = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=4, r2l_on=4, r2r_on=4)

    # mask_hid_intra7_norec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=1, r2l_on=1, r2r_on=7, contype='asymmetric-norecurrent')
    # mask_hid_intra6_norec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=2, r2l_on=2, r2r_on=6, contype='asymmetric-norecurrent')
    # mask_hid_intra5_norec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=3, r2l_on=3, r2r_on=5, contype='asymmetric-norecurrent')
    # mask_hid_intra4_norec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=4, r2l_on=4, r2r_on=4, contype='asymmetric-norecurrent')

    """mask_hid_intra7_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=1, r2l_on=1, r2r_on=7, contype='asymmetric-withrecurrent')
    mask_hid_intra6_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=2, r2l_on=2, r2r_on=6, contype='asymmetric-withrecurrent')
    mask_hid_intra5_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=3, r2l_on=3, r2r_on=5, contype='asymmetric-withrecurrent')
    mask_hid_intra4_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=4, r2l_on=4, r2r_on=4, contype='asymmetric-withrecurrent')"""

    # mask_intra7_full = ConstantMask(mask_in_full, mask_hid_intra7, mask_out_full, label="intra7f")
    # mask_intra6_full = ConstantMask(mask_in_full, mask_hid_intra6, mask_out_full, label="intra6f")
    # mask_intra5_full = ConstantMask(mask_in_full, mask_hid_intra5, mask_out_full, label="intra5f")
    # mask_intra4_full = ConstantMask(mask_in_full, mask_hid_intra4, mask_out_full, label="intra4f")

    # mask_intra7_partial = ConstantMask(mask_in_partial, mask_hid_intra7, mask_out_partial, label="intra7p")
    # mask_intra6_partial = ConstantMask(mask_in_partial, mask_hid_intra6, mask_out_partial, label="intra6p")
    # mask_intra5_partial = ConstantMask(mask_in_partial, mask_hid_intra5, mask_out_partial, label="intra5p")
    # mask_intra4_partial = ConstantMask(mask_in_partial, mask_hid_intra4, mask_out_partial, label="intra4p")

    # mask_intra7_partial_shuffled = ShuffledHiddenMask(mask_in_partial, mask_hid_intra7, mask_out_partial, label="intra7p_s")
    # mask_intra6_partial_shuffled = ShuffledHiddenMask(mask_in_partial, mask_hid_intra6, mask_out_partial, label="intra6p_s")
    # mask_intra5_partial_shuffled = ShuffledHiddenMask(mask_in_partial, mask_hid_intra5, mask_out_partial, label="intra5p_s")
    # mask_intra4_partial_shuffled = ShuffledHiddenMask(mask_in_partial, mask_hid_intra4, mask_out_partial, label="intra4p_s")

    # mask_intra7_mixed = ConstantMask(mask_in_mixed, mask_hid_intra7, mask_out_mixed, label="intra7m")
    # mask_intra6_mixed = ConstantMask(mask_in_mixed, mask_hid_intra6, mask_out_mixed, label="intra6m")
    # mask_intra5_mixed = ConstantMask(mask_in_mixed, mask_hid_intra5, mask_out_mixed, label="intra5m")
    # mask_intra4_mixed = ConstantMask(mask_in_mixed, mask_hid_intra4, mask_out_mixed, label="intra4m")

    # mask_intra7_partial_norec = ConstantMask(mask_in_partial, mask_hid_intra7_norec, mask_out_partial, label="intra7p_norec")
    # mask_intra6_partial_norec = ConstantMask(mask_in_partial, mask_hid_intra6_norec, mask_out_partial, label="intra6p_norec")
    # mask_intra5_partial_norec = ConstantMask(mask_in_partial, mask_hid_intra5_norec, mask_out_partial, label="intra5p_norec")
    # mask_intra4_partial_norec = ConstantMask(mask_in_partial, mask_hid_intra4_norec, mask_out_partial, label="intra4p_norec")
    """

    mask_intra7_partial_withrec = ConstantMask(mask_in_partial, mask_hid_intra7_withrec, mask_out_partial, label="intra7p_withrec")
    mask_intra6_partial_withrec = ConstantMask(mask_in_partial, mask_hid_intra6_withrec, mask_out_partial, label="intra6p_withrec")
    mask_intra5_partial_withrec = ConstantMask(mask_in_partial, mask_hid_intra5_withrec, mask_out_partial, label="intra5p_withrec")
    mask_intra4_partial_withrec = ConstantMask(mask_in_partial, mask_hid_intra4_withrec, mask_out_partial, label="intra4p_withrec")

    mask_intra7_full_withrec = ConstantMask(mask_in_full, mask_hid_intra7_withrec, mask_out_full, label="intra7f_withrec")
    mask_intra6_full_withrec = ConstantMask(mask_in_full, mask_hid_intra6_withrec, mask_out_full, label="intra6f_withrec")
    mask_intra5_full_withrec = ConstantMask(mask_in_full, mask_hid_intra5_withrec, mask_out_full, label="intra5f_withrec")
    mask_intra4_full_withrec = ConstantMask(mask_in_full, mask_hid_intra4_withrec, mask_out_full, label="intra4f_withrec")

    # mask_list_full = [mask_intra7_full, mask_intra6_full, mask_intra5_full, mask_intra4_full]
    # mask_list_partial = [mask_intra7_partial, mask_intra6_partial, mask_intra5_partial, mask_intra4_partial]
    # mask_list_partial_shuffled = [mask_intra7_partial_shuffled, mask_intra6_partial_shuffled, mask_intra5_partial_shuffled, mask_intra4_partial_shuffled]
    # mask_list_mixed = [mask_intra7_mixed, mask_intra6_mixed, mask_intra5_mixed, mask_intra4_mixed]
    # mask_list_partial_norec = [mask_intra7_partial_norec, mask_intra6_partial_norec, mask_intra5_partial_norec, mask_intra4_partial_norec]
    mask_list_partial_withrec = [mask_intra7_partial_withrec, mask_intra6_partial_withrec, mask_intra5_partial_withrec, mask_intra4_partial_withrec]
    mask_list_full_withrec = [mask_intra7_full_withrec, mask_intra6_full_withrec, mask_intra5_full_withrec, mask_intra4_full_withrec]
    mask_list_partial_withrec_short = [mask_intra7_partial_withrec, mask_intra5_partial_withrec]
    """
    

    # save_masks(mask_list_full, 'interintra_4m_20iu_16hu_20ou_fullin_fullout')
    # save_masks(mask_list_partial, 'interintra_4m_20iu_16hu_20ou_partialin_partialout')
    # save_masks(mask_list_partial_shuffled, 'interintra_4m_shuffle_20iu_16hu_20ou_partialin_partialout')
    # save_masks(mask_list_mixed, 'interintra_4m_20iu_16hu_20ou_mixedin_mixedout')
    # save_masks(mask_list_partial_norec, 'interintra_4m_20iu_16hu_20ou_partialin_partialout_norec')
    #save_masks(mask_list_partial_withrec, 'interintra_4m_20iu_16hu_20ou_partialin_partialout_withrec')
    #save_masks(mask_list_full_withrec, 'interintra_4m_20iu_16hu_20ou_fullin_fullout_withrec')
    #save_masks(mask_list_partial_withrec_short, 'interintra_2m_20iu_16hu_20ou_partialin_partialout_withrec')"""


    
    """
	#intra7 - all
    mask_hid_intra7_inter1_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=1, r2l_on=1, r2r_on=7, contype='asymmetric-withrecurrent')
    mask_hid_intra7_inter2_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=2, r2l_on=2, r2r_on=7, contype='asymmetric-withrecurrent')
    mask_hid_intra7_inter3_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=3, r2l_on=3, r2r_on=7, contype='asymmetric-withrecurrent')
    mask_hid_intra7_inter4_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=4, r2l_on=4, r2r_on=7, contype='asymmetric-withrecurrent')
    mask_hid_intra7_inter5_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=5, r2l_on=5, r2r_on=7, contype='asymmetric-withrecurrent')
    mask_hid_intra7_inter6_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=6, r2l_on=6, r2r_on=7, contype='asymmetric-withrecurrent')
    mask_hid_intra7_inter7_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=7, r2l_on=7, r2r_on=7, contype='asymmetric-withrecurrent')

    #Intra6 - all
    mask_hid_intra6_inter1_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=1, r2l_on=1, r2r_on=6, contype='asymmetric-withrecurrent')
    mask_hid_intra6_inter2_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=2, r2l_on=2, r2r_on=6, contype='asymmetric-withrecurrent')
    mask_hid_intra6_inter3_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=3, r2l_on=3, r2r_on=6, contype='asymmetric-withrecurrent')
    mask_hid_intra6_inter4_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=4, r2l_on=4, r2r_on=6, contype='asymmetric-withrecurrent')
    mask_hid_intra6_inter5_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=5, r2l_on=5, r2r_on=6, contype='asymmetric-withrecurrent')
    mask_hid_intra6_inter6_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=6, r2l_on=6, r2r_on=6, contype='asymmetric-withrecurrent')
    mask_hid_intra6_inter7_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=7, r2l_on=7, r2r_on=6, contype='asymmetric-withrecurrent')


    #Intra5-all
    mask_hid_intra5_inter1_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=1, r2l_on=1, r2r_on=5, contype='asymmetric-withrecurrent')
    mask_hid_intra5_inter2_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=2, r2l_on=2, r2r_on=5, contype='asymmetric-withrecurrent')
    mask_hid_intra5_inter3_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=3, r2l_on=3, r2r_on=5, contype='asymmetric-withrecurrent')
    mask_hid_intra5_inter4_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=4, r2l_on=4, r2r_on=5, contype='asymmetric-withrecurrent')
    mask_hid_intra5_inter5_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=5, r2l_on=5, r2r_on=5, contype='asymmetric-withrecurrent')
    mask_hid_intra5_inter6_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=6, r2l_on=6, r2r_on=5, contype='asymmetric-withrecurrent')
    mask_hid_intra5_inter7_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=7, r2l_on=7, r2r_on=5, contype='asymmetric-withrecurrent')


    #Intra4 - all
    mask_hid_intra4_inter1_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=1, r2l_on=1, r2r_on=4, contype='asymmetric-withrecurrent')
    mask_hid_intra4_inter2_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=2, r2l_on=2, r2r_on=4, contype='asymmetric-withrecurrent')
    mask_hid_intra4_inter3_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=3, r2l_on=3, r2r_on=4, contype='asymmetric-withrecurrent')
    mask_hid_intra4_inter4_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=4, r2l_on=4, r2r_on=4, contype='asymmetric-withrecurrent')
    mask_hid_intra4_inter5_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=5, r2l_on=5, r2r_on=4, contype='asymmetric-withrecurrent')
    mask_hid_intra4_inter6_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=6, r2l_on=6, r2r_on=4, contype='asymmetric-withrecurrent')
    mask_hid_intra4_inter7_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=7, r2l_on=7, r2r_on=4, contype='asymmetric-withrecurrent')


   #Intra3 - all
    mask_hid_intra3_inter1_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=3, l2r_on=1, r2l_on=1, r2r_on=3, contype='asymmetric-withrecurrent')
    mask_hid_intra3_inter2_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=3, l2r_on=2, r2l_on=2, r2r_on=3, contype='asymmetric-withrecurrent')
    mask_hid_intra3_inter3_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=3, l2r_on=3, r2l_on=3, r2r_on=3, contype='asymmetric-withrecurrent')
    mask_hid_intra3_inter4_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=3, l2r_on=4, r2l_on=4, r2r_on=3, contype='asymmetric-withrecurrent')
    mask_hid_intra3_inter5_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=3, l2r_on=5, r2l_on=5, r2r_on=3, contype='asymmetric-withrecurrent')
    mask_hid_intra3_inter6_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=3, l2r_on=6, r2l_on=6, r2r_on=3, contype='asymmetric-withrecurrent')
    mask_hid_intra3_inter7_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=3, l2r_on=7, r2l_on=7, r2r_on=3, contype='asymmetric-withrecurrent')

    #Intra2 - all
    mask_hid_intra2_inter1_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=2, l2r_on=1, r2l_on=1, r2r_on=2, contype='asymmetric-withrecurrent')
    mask_hid_intra2_inter2_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=2, l2r_on=2, r2l_on=2, r2r_on=2, contype='asymmetric-withrecurrent')
    mask_hid_intra2_inter3_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=2, l2r_on=3, r2l_on=3, r2r_on=2, contype='asymmetric-withrecurrent')
    mask_hid_intra2_inter4_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=2, l2r_on=4, r2l_on=4, r2r_on=2, contype='asymmetric-withrecurrent')
    mask_hid_intra2_inter5_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=2, l2r_on=5, r2l_on=5, r2r_on=2, contype='asymmetric-withrecurrent')
    mask_hid_intra2_inter6_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=2, l2r_on=6, r2l_on=6, r2r_on=2, contype='asymmetric-withrecurrent')
    mask_hid_intra2_inter7_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=2, l2r_on=7, r2l_on=7, r2r_on=2, contype='asymmetric-withrecurrent')

    #Intra 1 - all
    mask_hid_intra1_inter1_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=1, l2r_on=1, r2l_on=1, r2r_on=1, contype='asymmetric-withrecurrent')
    mask_hid_intra1_inter2_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=1, l2r_on=2, r2l_on=2, r2r_on=1, contype='asymmetric-withrecurrent')
    mask_hid_intra1_inter3_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=1, l2r_on=3, r2l_on=3, r2r_on=1, contype='asymmetric-withrecurrent')
    mask_hid_intra1_inter4_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=1, l2r_on=4, r2l_on=4, r2r_on=1, contype='asymmetric-withrecurrent')
    mask_hid_intra1_inter5_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=1, l2r_on=5, r2l_on=5, r2r_on=1, contype='asymmetric-withrecurrent')
    mask_hid_intra1_inter6_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=1, l2r_on=6, r2l_on=6, r2r_on=1, contype='asymmetric-withrecurrent')
    mask_hid_intra1_inter7_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=1, l2r_on=7, r2l_on=7, r2r_on=1, contype='asymmetric-withrecurrent')






    #Create the constant masks for them - full atm. we'll add partial if we need to - not sure what that does!
    #Intra7
    mask_intra7_inter1_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra7_inter1_withrec, mask_out_partial, label="intra7f_inter1_withrec")
    mask_intra7_inter2_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra7_inter2_withrec, mask_out_partial, label="intra7f_inter2_withrec")
    mask_intra7_inter3_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra7_inter3_withrec, mask_out_partial, label="intra7f_inter3_withrec")
    mask_intra7_inter4_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra7_inter4_withrec, mask_out_partial, label="intra7f_inter4_withrec")
    mask_intra7_inter5_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra7_inter5_withrec, mask_out_partial, label="intra7f_inter5_withrec")
    mask_intra7_inter6_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra7_inter6_withrec, mask_out_partial, label="intra7f_inter6_withrec")
    mask_intra7_inter7_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra7_inter7_withrec, mask_out_partial, label="intra7f_inter7_withrec")

    #Intra6
    mask_intra6_inter1_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra6_inter1_withrec, mask_out_partial, label="intra6f_inter1_withrec")
    mask_intra6_inter2_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra6_inter2_withrec, mask_out_partial, label="intra6f_inter2_withrec")
    mask_intra6_inter3_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra6_inter3_withrec, mask_out_partial, label="intra6f_inter3_withrec")
    mask_intra6_inter4_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra6_inter4_withrec, mask_out_partial, label="intra6f_inter4_withrec")
    mask_intra6_inter5_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra6_inter5_withrec, mask_out_partial, label="intra6f_inter5_withrec")
    mask_intra6_inter6_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra6_inter6_withrec, mask_out_partial, label="intra6f_inter6_withrec")
    mask_intra6_inter7_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra6_inter7_withrec, mask_out_partial, label="intra6f_inter7_withrec")

    #Intra5
    mask_intra5_inter1_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra5_inter1_withrec, mask_out_partial, label="intra5f_inter1_withrec")
    mask_intra5_inter2_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra5_inter2_withrec, mask_out_partial, label="intra5f_inter2_withrec")
    mask_intra5_inter3_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra5_inter3_withrec, mask_out_partial, label="intra5f_inter3_withrec")
    mask_intra5_inter4_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra5_inter4_withrec, mask_out_partial, label="intra5f_inter4_withrec")
    mask_intra5_inter5_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra5_inter5_withrec, mask_out_partial, label="intra5f_inter5_withrec")
    mask_intra5_inter6_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra5_inter6_withrec, mask_out_partial, label="intra5f_inter6_withrec")
    mask_intra5_inter7_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra5_inter7_withrec, mask_out_partial, label="intra5f_inter7_withrec")

    #Intra4
    mask_intra4_inter1_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra4_inter1_withrec, mask_out_partial, label="intra4f_inter1_withrec")
    mask_intra4_inter2_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra4_inter2_withrec, mask_out_partial, label="intra4f_inter2_withrec")
    mask_intra4_inter3_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra4_inter3_withrec, mask_out_partial, label="intra4f_inter3_withrec")
    mask_intra4_inter4_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra4_inter4_withrec, mask_out_partial, label="intra4f_inter4_withrec")
    mask_intra4_inter5_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra4_inter5_withrec, mask_out_partial, label="intra4f_inter5_withrec")
    mask_intra4_inter6_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra4_inter6_withrec, mask_out_partial, label="intra4f_inter6_withrec")
    mask_intra4_inter7_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra4_inter7_withrec, mask_out_partial, label="intra4f_inter7_withrec")

    #Intra3
    mask_intra3_inter1_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra3_inter1_withrec, mask_out_partial, label="intra3f_inter1_withrec")
    mask_intra3_inter2_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra3_inter2_withrec, mask_out_partial, label="intra3f_inter2_withrec")
    mask_intra3_inter3_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra3_inter3_withrec, mask_out_partial, label="intra3f_inter3_withrec")
    mask_intra3_inter4_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra3_inter4_withrec, mask_out_partial, label="intra3f_inter4_withrec")
    mask_intra3_inter5_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra3_inter5_withrec, mask_out_partial, label="intra3f_inter5_withrec")
    mask_intra3_inter6_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra3_inter6_withrec, mask_out_partial, label="intra3f_inter6_withrec")
    mask_intra3_inter7_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra3_inter7_withrec, mask_out_partial, label="intra3f_inter7_withrec")

     #Intra2
    mask_intra2_inter1_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra2_inter1_withrec, mask_out_partial, label="intra2f_inter1_withrec")
    mask_intra2_inter2_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra2_inter2_withrec, mask_out_partial, label="intra2f_inter2_withrec")
    mask_intra2_inter3_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra2_inter3_withrec, mask_out_partial, label="intra2f_inter3_withrec")
    mask_intra2_inter4_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra2_inter4_withrec, mask_out_partial, label="intra2f_inter4_withrec")
    mask_intra2_inter5_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra2_inter5_withrec, mask_out_partial, label="intra2f_inter5_withrec")
    mask_intra2_inter6_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra2_inter6_withrec, mask_out_partial, label="intra2f_inter6_withrec")
    mask_intra2_inter7_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra2_inter7_withrec, mask_out_partial, label="intra2f_inter7_withrec")

    #Intra1
    mask_intra1_inter1_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra1_inter1_withrec, mask_out_partial, label="intra1f_inter1_withrec")
    mask_intra1_inter2_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra1_inter2_withrec, mask_out_partial, label="intra1f_inter2_withrec")
    mask_intra1_inter3_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra1_inter3_withrec, mask_out_partial, label="intra1f_inter3_withrec")
    mask_intra1_inter4_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra1_inter4_withrec, mask_out_partial, label="intra1f_inter4_withrec")
    mask_intra1_inter5_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra1_inter5_withrec, mask_out_partial, label="intra1f_inter5_withrec")
    mask_intra1_inter6_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra1_inter6_withrec, mask_out_partial, label="intra1f_inter6_withrec")
    mask_intra1_inter7_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra1_inter7_withrec, mask_out_partial, label="intra1f_inter7_withrec")


    #now for the actual mask lists  we'll do one for each inter!
    mask_list_full_intra7_withrec =[mask_intra7_inter1_full_withrec,
mask_intra7_inter2_full_withrec,
mask_intra7_inter3_full_withrec,
mask_intra7_inter4_full_withrec,
mask_intra7_inter5_full_withrec,
mask_intra7_inter6_full_withrec,
mask_intra7_inter7_full_withrec]
    mask_list_full_intra6_withrec =[mask_intra6_inter1_full_withrec,
mask_intra6_inter2_full_withrec,
mask_intra6_inter3_full_withrec,
mask_intra6_inter4_full_withrec,
mask_intra6_inter5_full_withrec,
mask_intra6_inter6_full_withrec,
mask_intra6_inter7_full_withrec]
    mask_list_full_intra5_withrec =[mask_intra5_inter1_full_withrec,
mask_intra5_inter2_full_withrec,
mask_intra5_inter3_full_withrec,
mask_intra5_inter4_full_withrec,
mask_intra5_inter5_full_withrec,
mask_intra5_inter6_full_withrec,
mask_intra5_inter7_full_withrec]
    mask_list_full_intra4_withrec =[mask_intra4_inter1_full_withrec,
mask_intra4_inter2_full_withrec,
mask_intra4_inter3_full_withrec,
mask_intra4_inter4_full_withrec,
mask_intra4_inter5_full_withrec,
mask_intra4_inter6_full_withrec,
mask_intra4_inter7_full_withrec]

    mask_list_full_intra3_withrec =[mask_intra3_inter1_full_withrec,
mask_intra3_inter2_full_withrec,
mask_intra3_inter3_full_withrec,
mask_intra3_inter4_full_withrec,
mask_intra3_inter5_full_withrec,
mask_intra3_inter6_full_withrec,
mask_intra3_inter7_full_withrec]

    mask_list_full_intra2_withrec =[mask_intra2_inter1_full_withrec,
mask_intra2_inter2_full_withrec,
mask_intra2_inter3_full_withrec,
mask_intra2_inter4_full_withrec,
mask_intra2_inter5_full_withrec,
mask_intra2_inter6_full_withrec,
mask_intra2_inter7_full_withrec]

    mask_list_full_intra1_withrec =[mask_intra1_inter1_full_withrec,
mask_intra1_inter2_full_withrec,
mask_intra1_inter3_full_withrec,
mask_intra1_inter4_full_withrec,
mask_intra1_inter5_full_withrec,
mask_intra1_inter6_full_withrec,
mask_intra1_inter7_full_withrec]

#now we save our crazy masklists

    save_masks(mask_list_full_intra7_withrec, 'interintra7_4m_20iu_16hu_20ou_partialin_partialout_withrec')
    save_masks(mask_list_full_intra6_withrec, 'interintra6_4m_20iu_16hu_20ou_partialin_partialout_withrec')
    save_masks(mask_list_full_intra5_withrec, 'interintra5_4m_20iu_16hu_20ou_partialin_partialout_withrec')
    save_masks(mask_list_full_intra4_withrec, 'interintra4_4m_20iu_16hu_20ou_partialin_partialout_withrec')
    save_masks(mask_list_full_intra3_withrec, 'interintra3_4m_20iu_16hu_20ou_partialin_partialout_withrec')
    save_masks(mask_list_full_intra2_withrec, 'interintra2_4m_20iu_16hu_20ou_partialin_partialout_withrec')
    save_masks(mask_list_full_intra1_withrec, 'interintra1_4m_20iu_16hu_20ou_partialin_partialout_withrec')
    """


    


	#CREATE PARTIAL SHUFFLED MASKS!
	#intra7 - all
    mask_in_partial = create_mask_matrix(shape=(20,16), left_in=10, left_out=8, l2l_on=8, l2r_on=0, r2l_on=0, r2r_on=8, shuffle = True)
    mask_out_partial = create_mask_matrix(shape=(16,20), left_in=8, left_out=10, l2l_on=0, l2r_on=10, r2l_on=10, r2r_on=0, shuffle = True)


    #create equal masks
    mask_hid_intra7_inter1_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=1, r2l_on=1, r2r_on=7, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra6_inter2_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=2, r2l_on=2, r2r_on=6, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra5_inter3_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=3, r2l_on=3, r2r_on=5, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra4_inter4_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=4, r2l_on=4, r2r_on=4, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra3_inter5_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=3, l2r_on=5, r2l_on=5, r2r_on=3, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra2_inter6_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=2, l2r_on=6, r2l_on=6, r2r_on=2, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra1_inter7_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=1, l2r_on=7, r2l_on=7, r2r_on=1, contype='asymmetric-withrecurrent', shuffle = True)

    mask_intra7_inter1_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra7_inter1_withrec, mask_out_partial, label="intra7f_inter1_withrec")
    mask_intra6_inter2_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra6_inter2_withrec, mask_out_partial, label="intra7f_inter2_withrec")
    mask_intra5_inter3_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra5_inter3_withrec, mask_out_partial, label="intra7f_inter3_withrec")
    mask_intra4_inter4_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra4_inter4_withrec, mask_out_partial, label="intra7f_inter4_withrec")
    mask_intra3_inter5_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra3_inter5_withrec, mask_out_partial, label="intra7f_inter5_withrec")
    mask_intra2_inter6_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra2_inter6_withrec, mask_out_partial, label="intra7f_inter6_withrec")
    mask_intra1_inter7_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra1_inter7_withrec, mask_out_partial, label="intra7f_inter7_withrec")

    mask_list_equal =[mask_intra7_inter1_full_withrec,
mask_intra6_inter2_full_withrec,
mask_intra5_inter3_full_withrec,
mask_intra4_inter4_full_withrec,
mask_intra3_inter5_full_withrec,
mask_intra2_inter6_full_withrec,
mask_intra1_inter7_full_withrec]

    save_masks(mask_list_equal, 'interintra7_4m_20iu_16hu_20ou_partialin_partialout_EQUAL')
    print "SAVED!"


    """
    mask_hid_intra7_inter1_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=1, r2l_on=1, r2r_on=7, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra7_inter2_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=2, r2l_on=2, r2r_on=7, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra7_inter3_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=3, r2l_on=3, r2r_on=7, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra7_inter4_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=4, r2l_on=4, r2r_on=7, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra7_inter5_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=5, r2l_on=5, r2r_on=7, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra7_inter6_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=6, r2l_on=6, r2r_on=7, contype='asymmetric-withrecurrent', shuffle = True)

    mask_hid_intra7_inter7_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=7, r2l_on=7, r2r_on=7, contype='asymmetric-withrecurrent', shuffle = True)

    #Intra6 - all
    mask_hid_intra6_inter1_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=1, r2l_on=1, r2r_on=6, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra6_inter2_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=2, r2l_on=2, r2r_on=6, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra6_inter3_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=3, r2l_on=3, r2r_on=6, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra6_inter4_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=4, r2l_on=4, r2r_on=6, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra6_inter5_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=5, r2l_on=5, r2r_on=6, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra6_inter6_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=6, r2l_on=6, r2r_on=6, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra6_inter7_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=7, r2l_on=7, r2r_on=6, contype='asymmetric-withrecurrent', shuffle = True)


    #Intra5-all
    mask_hid_intra5_inter1_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=1, r2l_on=1, r2r_on=5, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra5_inter2_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=2, r2l_on=2, r2r_on=5, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra5_inter3_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=3, r2l_on=3, r2r_on=5, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra5_inter4_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=4, r2l_on=4, r2r_on=5, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra5_inter5_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=5, r2l_on=5, r2r_on=5, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra5_inter6_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=6, r2l_on=6, r2r_on=5, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra5_inter7_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=7, r2l_on=7, r2r_on=5, contype='asymmetric-withrecurrent', shuffle = True)


    #Intra4 - all
    mask_hid_intra4_inter1_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=1, r2l_on=1, r2r_on=4, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra4_inter2_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=2, r2l_on=2, r2r_on=4, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra4_inter3_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=3, r2l_on=3, r2r_on=4, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra4_inter4_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=4, r2l_on=4, r2r_on=4, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra4_inter5_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=5, r2l_on=5, r2r_on=4, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra4_inter6_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=6, r2l_on=6, r2r_on=4, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra4_inter7_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=7, r2l_on=7, r2r_on=4, contype='asymmetric-withrecurrent', shuffle = True)


   #Intra3 - all
    mask_hid_intra3_inter1_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=3, l2r_on=1, r2l_on=1, r2r_on=3, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra3_inter2_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=3, l2r_on=2, r2l_on=2, r2r_on=3, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra3_inter3_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=3, l2r_on=3, r2l_on=3, r2r_on=3, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra3_inter4_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=3, l2r_on=4, r2l_on=4, r2r_on=3, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra3_inter5_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=3, l2r_on=5, r2l_on=5, r2r_on=3, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra3_inter6_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=3, l2r_on=6, r2l_on=6, r2r_on=3, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra3_inter7_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=3, l2r_on=7, r2l_on=7, r2r_on=3, contype='asymmetric-withrecurrent', shuffle = True)

    #Intra2 - all
    mask_hid_intra2_inter1_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=2, l2r_on=1, r2l_on=1, r2r_on=2, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra2_inter2_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=2, l2r_on=2, r2l_on=2, r2r_on=2, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra2_inter3_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=2, l2r_on=3, r2l_on=3, r2r_on=2, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra2_inter4_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=2, l2r_on=4, r2l_on=4, r2r_on=2, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra2_inter5_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=2, l2r_on=5, r2l_on=5, r2r_on=2, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra2_inter6_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=2, l2r_on=6, r2l_on=6, r2r_on=2, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra2_inter7_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=2, l2r_on=7, r2l_on=7, r2r_on=2, contype='asymmetric-withrecurrent', shuffle = True)

    #Intra 1 - all
    mask_hid_intra1_inter1_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=1, l2r_on=1, r2l_on=1, r2r_on=1, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra1_inter2_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=1, l2r_on=2, r2l_on=2, r2r_on=1, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra1_inter3_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=1, l2r_on=3, r2l_on=3, r2r_on=1, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra1_inter4_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=1, l2r_on=4, r2l_on=4, r2r_on=1, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra1_inter5_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=1, l2r_on=5, r2l_on=5, r2r_on=1, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra1_inter6_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=1, l2r_on=6, r2l_on=6, r2r_on=1, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra1_inter7_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=1, l2r_on=7, r2l_on=7, r2r_on=1, contype='asymmetric-withrecurrent', shuffle = True)






    #Create the constant masks for them - full atm. we'll add partial if we need to - not sure what that does!
    #Intra7
    mask_intra7_inter1_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra7_inter1_withrec, mask_out_partial, label="intra7f_inter1_withrec")
    mask_intra7_inter2_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra7_inter2_withrec, mask_out_partial, label="intra7f_inter2_withrec")
    mask_intra7_inter3_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra7_inter3_withrec, mask_out_partial, label="intra7f_inter3_withrec")
    mask_intra7_inter4_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra7_inter4_withrec, mask_out_partial, label="intra7f_inter4_withrec")
    mask_intra7_inter5_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra7_inter5_withrec, mask_out_partial, label="intra7f_inter5_withrec")
    mask_intra7_inter6_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra7_inter6_withrec, mask_out_partial, label="intra7f_inter6_withrec")
    mask_intra7_inter7_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra7_inter7_withrec, mask_out_partial, label="intra7f_inter7_withrec")

    #Intra6
    mask_intra6_inter1_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra6_inter1_withrec, mask_out_partial, label="intra6f_inter1_withrec")
    mask_intra6_inter2_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra6_inter2_withrec, mask_out_partial, label="intra6f_inter2_withrec")
    mask_intra6_inter3_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra6_inter3_withrec, mask_out_partial, label="intra6f_inter3_withrec")
    mask_intra6_inter4_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra6_inter4_withrec, mask_out_partial, label="intra6f_inter4_withrec")
    mask_intra6_inter5_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra6_inter5_withrec, mask_out_partial, label="intra6f_inter5_withrec")
    mask_intra6_inter6_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra6_inter6_withrec, mask_out_partial, label="intra6f_inter6_withrec")
    mask_intra6_inter7_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra6_inter7_withrec, mask_out_partial, label="intra6f_inter7_withrec")

    #Intra5
    mask_intra5_inter1_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra5_inter1_withrec, mask_out_partial, label="intra5f_inter1_withrec")
    mask_intra5_inter2_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra5_inter2_withrec, mask_out_partial, label="intra5f_inter2_withrec")
    mask_intra5_inter3_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra5_inter3_withrec, mask_out_partial, label="intra5f_inter3_withrec")
    mask_intra5_inter4_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra5_inter4_withrec, mask_out_partial, label="intra5f_inter4_withrec")
    mask_intra5_inter5_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra5_inter5_withrec, mask_out_partial, label="intra5f_inter5_withrec")
    mask_intra5_inter6_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra5_inter6_withrec, mask_out_partial, label="intra5f_inter6_withrec")
    mask_intra5_inter7_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra5_inter7_withrec, mask_out_partial, label="intra5f_inter7_withrec")

    #Intra4
    mask_intra4_inter1_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra4_inter1_withrec, mask_out_partial, label="intra4f_inter1_withrec")
    mask_intra4_inter2_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra4_inter2_withrec, mask_out_partial, label="intra4f_inter2_withrec")
    mask_intra4_inter3_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra4_inter3_withrec, mask_out_partial, label="intra4f_inter3_withrec")
    mask_intra4_inter4_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra4_inter4_withrec, mask_out_partial, label="intra4f_inter4_withrec")
    mask_intra4_inter5_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra4_inter5_withrec, mask_out_partial, label="intra4f_inter5_withrec")
    mask_intra4_inter6_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra4_inter6_withrec, mask_out_partial, label="intra4f_inter6_withrec")
    mask_intra4_inter7_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra4_inter7_withrec, mask_out_partial, label="intra4f_inter7_withrec")

    #Intra3
    mask_intra3_inter1_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra3_inter1_withrec, mask_out_partial, label="intra3f_inter1_withrec")
    mask_intra3_inter2_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra3_inter2_withrec, mask_out_partial, label="intra3f_inter2_withrec")
    mask_intra3_inter3_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra3_inter3_withrec, mask_out_partial, label="intra3f_inter3_withrec")
    mask_intra3_inter4_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra3_inter4_withrec, mask_out_partial, label="intra3f_inter4_withrec")
    mask_intra3_inter5_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra3_inter5_withrec, mask_out_partial, label="intra3f_inter5_withrec")
    mask_intra3_inter6_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra3_inter6_withrec, mask_out_partial, label="intra3f_inter6_withrec")
    mask_intra3_inter7_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra3_inter7_withrec, mask_out_partial, label="intra3f_inter7_withrec")

     #Intra2
    mask_intra2_inter1_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra2_inter1_withrec, mask_out_partial, label="intra2f_inter1_withrec")
    mask_intra2_inter2_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra2_inter2_withrec, mask_out_partial, label="intra2f_inter2_withrec")
    mask_intra2_inter3_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra2_inter3_withrec, mask_out_partial, label="intra2f_inter3_withrec")
    mask_intra2_inter4_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra2_inter4_withrec, mask_out_partial, label="intra2f_inter4_withrec")
    mask_intra2_inter5_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra2_inter5_withrec, mask_out_partial, label="intra2f_inter5_withrec")
    mask_intra2_inter6_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra2_inter6_withrec, mask_out_partial, label="intra2f_inter6_withrec")
    mask_intra2_inter7_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra2_inter7_withrec, mask_out_partial, label="intra2f_inter7_withrec")


    #Intra1
    mask_intra1_inter1_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra1_inter1_withrec, mask_out_partial, label="intra1f_inter1_withrec")
    mask_intra1_inter2_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra1_inter2_withrec, mask_out_partial, label="intra1f_inter2_withrec")
    mask_intra1_inter3_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra1_inter3_withrec, mask_out_partial, label="intra1f_inter3_withrec")
    mask_intra1_inter4_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra1_inter4_withrec, mask_out_partial, label="intra1f_inter4_withrec")
    mask_intra1_inter5_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra1_inter5_withrec, mask_out_partial, label="intra1f_inter5_withrec")
    mask_intra1_inter6_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra1_inter6_withrec, mask_out_partial, label="intra1f_inter6_withrec")
    mask_intra1_inter7_full_withrec = ConstantMask(mask_in_partial, mask_hid_intra1_inter7_withrec, mask_out_partial, label="intra1f_inter7_withrec")


    #now for the actual mask lists  we'll do one for each inter!
    mask_list_full_intra7_withrec =[mask_intra7_inter1_full_withrec,
mask_intra7_inter2_full_withrec,
mask_intra7_inter3_full_withrec,
mask_intra7_inter4_full_withrec,
mask_intra7_inter5_full_withrec,
mask_intra7_inter6_full_withrec,
mask_intra7_inter7_full_withrec]
    mask_list_full_intra6_withrec =[mask_intra6_inter1_full_withrec,
mask_intra6_inter2_full_withrec,
mask_intra6_inter3_full_withrec,
mask_intra6_inter4_full_withrec,
mask_intra6_inter5_full_withrec,
mask_intra6_inter6_full_withrec,
mask_intra6_inter7_full_withrec]
    mask_list_full_intra5_withrec =[mask_intra5_inter1_full_withrec,
mask_intra5_inter2_full_withrec,
mask_intra5_inter3_full_withrec,
mask_intra5_inter4_full_withrec,
mask_intra5_inter5_full_withrec,
mask_intra5_inter6_full_withrec,
mask_intra5_inter7_full_withrec]
    mask_list_full_intra4_withrec =[mask_intra4_inter1_full_withrec,
mask_intra4_inter2_full_withrec,
mask_intra4_inter3_full_withrec,
mask_intra4_inter4_full_withrec,
mask_intra4_inter5_full_withrec,
mask_intra4_inter6_full_withrec,
mask_intra4_inter7_full_withrec]

    mask_list_full_intra3_withrec =[mask_intra3_inter1_full_withrec,
mask_intra3_inter2_full_withrec,
mask_intra3_inter3_full_withrec,
mask_intra3_inter4_full_withrec,
mask_intra3_inter5_full_withrec,
mask_intra3_inter6_full_withrec,
mask_intra3_inter7_full_withrec]

    mask_list_full_intra2_withrec =[mask_intra2_inter1_full_withrec,
mask_intra2_inter2_full_withrec,
mask_intra2_inter3_full_withrec,
mask_intra2_inter4_full_withrec,
mask_intra2_inter5_full_withrec,
mask_intra2_inter6_full_withrec,
mask_intra2_inter7_full_withrec]

    mask_list_full_intra1_withrec =[mask_intra1_inter1_full_withrec,
mask_intra1_inter2_full_withrec,
mask_intra1_inter3_full_withrec,
mask_intra1_inter4_full_withrec,
mask_intra1_inter5_full_withrec,
mask_intra1_inter6_full_withrec,
mask_intra1_inter7_full_withrec]

#now we save our crazy masklists

    save_masks(mask_list_full_intra7_withrec, 'interintra7_4m_20iu_16hu_20ou_partialin_partialout_withrec_SHUFFLE')
    save_masks(mask_list_full_intra6_withrec, 'interintra6_4m_20iu_16hu_20ou_partialin_partialout_withrec_SHUFFLE')
    save_masks(mask_list_full_intra5_withrec, 'interintra5_4m_20iu_16hu_20ou_partialin_partialout_withrec_SHUFFLE')
    save_masks(mask_list_full_intra4_withrec, 'interintra4_4m_20iu_16hu_20ou_partialin_partialout_withrec_SHUFFLE')
    save_masks(mask_list_full_intra3_withrec, 'interintra3_4m_20iu_16hu_20ou_partialin_partialout_withrec_SHUFFLE')
    save_masks(mask_list_full_intra2_withrec, 'interintra2_4m_20iu_16hu_20ou_partialin_partialout_withrec_SHUFFLE')
    save_masks(mask_list_full_intra1_withrec, 'interintra1_4m_20iu_16hu_20ou_partialin_partialout_withrec_SHUFFLE')
    
    """
    """
##CREATE THE SHUFFLED MASKS!

	#intra7 - all
    mask_hid_intra7_inter1_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=1, r2l_on=1, r2r_on=7, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra7_inter2_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=2, r2l_on=2, r2r_on=7, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra7_inter3_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=3, r2l_on=3, r2r_on=7, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra7_inter4_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=4, r2l_on=4, r2r_on=7, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra7_inter5_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=5, r2l_on=5, r2r_on=7, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra7_inter6_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=6, r2l_on=6, r2r_on=7, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra7_inter7_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=7, l2r_on=7, r2l_on=7, r2r_on=7, contype='asymmetric-withrecurrent', shuffle = True)

    #Intra6 - all
    mask_hid_intra6_inter1_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=1, r2l_on=1, r2r_on=6, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra6_inter2_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=2, r2l_on=2, r2r_on=6, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra6_inter3_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=3, r2l_on=3, r2r_on=6, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra6_inter4_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=4, r2l_on=4, r2r_on=6, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra6_inter5_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=5, r2l_on=5, r2r_on=6, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra6_inter6_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=6, r2l_on=6, r2r_on=6, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra6_inter7_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=6, l2r_on=7, r2l_on=7, r2r_on=6, contype='asymmetric-withrecurrent', shuffle = True)


    #Intra5-all
    mask_hid_intra5_inter1_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=1, r2l_on=1, r2r_on=5, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra5_inter2_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=2, r2l_on=2, r2r_on=5, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra5_inter3_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=3, r2l_on=3, r2r_on=5, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra5_inter4_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=4, r2l_on=4, r2r_on=5, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra5_inter5_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=5, r2l_on=5, r2r_on=5, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra5_inter6_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=6, r2l_on=6, r2r_on=5, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra5_inter7_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=5, l2r_on=7, r2l_on=7, r2r_on=5, contype='asymmetric-withrecurrent', shuffle = True)


    #Intra4 - all
    mask_hid_intra4_inter1_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=1, r2l_on=1, r2r_on=4, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra4_inter2_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=2, r2l_on=2, r2r_on=4, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra4_inter3_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=3, r2l_on=3, r2r_on=4, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra4_inter4_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=4, r2l_on=4, r2r_on=4, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra4_inter5_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=5, r2l_on=5, r2r_on=4, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra4_inter6_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=6, r2l_on=6, r2r_on=4, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra4_inter7_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=4, l2r_on=7, r2l_on=7, r2r_on=4, contype='asymmetric-withrecurrent', shuffle = True)


   #Intra3 - all
    mask_hid_intra3_inter1_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=3, l2r_on=1, r2l_on=1, r2r_on=3, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra3_inter2_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=3, l2r_on=2, r2l_on=2, r2r_on=3, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra3_inter3_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=3, l2r_on=3, r2l_on=3, r2r_on=3, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra3_inter4_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=3, l2r_on=4, r2l_on=4, r2r_on=3, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra3_inter5_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=3, l2r_on=5, r2l_on=5, r2r_on=3, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra3_inter6_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=3, l2r_on=6, r2l_on=6, r2r_on=3, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra3_inter7_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=3, l2r_on=7, r2l_on=7, r2r_on=3, contype='asymmetric-withrecurrent', shuffle = True)

    #Intra2 - all
    mask_hid_intra2_inter1_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=2, l2r_on=1, r2l_on=1, r2r_on=2, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra2_inter2_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=2, l2r_on=2, r2l_on=2, r2r_on=2, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra2_inter3_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=2, l2r_on=3, r2l_on=3, r2r_on=2, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra2_inter4_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=2, l2r_on=4, r2l_on=4, r2r_on=2, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra2_inter5_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=2, l2r_on=5, r2l_on=5, r2r_on=2, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra2_inter6_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=2, l2r_on=6, r2l_on=6, r2r_on=2, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra2_inter7_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=2, l2r_on=7, r2l_on=7, r2r_on=2, contype='asymmetric-withrecurrent', shuffle = True)

    #Intra 1 - all
    mask_hid_intra1_inter1_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=1, l2r_on=1, r2l_on=1, r2r_on=1, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra1_inter2_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=1, l2r_on=2, r2l_on=2, r2r_on=1, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra1_inter3_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=1, l2r_on=3, r2l_on=3, r2r_on=1, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra1_inter4_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=1, l2r_on=4, r2l_on=4, r2r_on=1, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra1_inter5_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=1, l2r_on=5, r2l_on=5, r2r_on=1, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra1_inter6_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=1, l2r_on=6, r2l_on=6, r2r_on=1, contype='asymmetric-withrecurrent', shuffle = True)
    mask_hid_intra1_inter7_withrec = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=1, l2r_on=7, r2l_on=7, r2r_on=1, contype='asymmetric-withrecurrent', shuffle = True)

    #Intra7
    mask_intra7_inter1_full_withrec = ConstantMask(mask_in_full, mask_hid_intra7_inter1_withrec, mask_out_full, label="intra7f_inter1_withrec")
    mask_intra7_inter2_full_withrec = ConstantMask(mask_in_full, mask_hid_intra7_inter2_withrec, mask_out_full, label="intra7f_inter2_withrec")
    mask_intra7_inter3_full_withrec = ConstantMask(mask_in_full, mask_hid_intra7_inter3_withrec, mask_out_full, label="intra7f_inter3_withrec")
    mask_intra7_inter4_full_withrec = ConstantMask(mask_in_full, mask_hid_intra7_inter4_withrec, mask_out_full, label="intra7f_inter4_withrec")
    mask_intra7_inter5_full_withrec = ConstantMask(mask_in_full, mask_hid_intra7_inter5_withrec, mask_out_full, label="intra7f_inter5_withrec")
    mask_intra7_inter6_full_withrec = ConstantMask(mask_in_full, mask_hid_intra7_inter6_withrec, mask_out_full, label="intra7f_inter6_withrec")
    mask_intra7_inter7_full_withrec = ConstantMask(mask_in_full, mask_hid_intra7_inter7_withrec, mask_out_full, label="intra7f_inter7_withrec")

    #Intra6
    mask_intra6_inter1_full_withrec = ConstantMask(mask_in_full, mask_hid_intra6_inter1_withrec, mask_out_full, label="intra6f_inter1_withrec")
    mask_intra6_inter2_full_withrec = ConstantMask(mask_in_full, mask_hid_intra6_inter2_withrec, mask_out_full, label="intra6f_inter2_withrec")
    mask_intra6_inter3_full_withrec = ConstantMask(mask_in_full, mask_hid_intra6_inter3_withrec, mask_out_full, label="intra6f_inter3_withrec")
    mask_intra6_inter4_full_withrec = ConstantMask(mask_in_full, mask_hid_intra6_inter4_withrec, mask_out_full, label="intra6f_inter4_withrec")
    mask_intra6_inter5_full_withrec = ConstantMask(mask_in_full, mask_hid_intra6_inter5_withrec, mask_out_full, label="intra6f_inter5_withrec")
    mask_intra6_inter6_full_withrec = ConstantMask(mask_in_full, mask_hid_intra6_inter6_withrec, mask_out_full, label="intra6f_inter6_withrec")
    mask_intra6_inter7_full_withrec = ConstantMask(mask_in_full, mask_hid_intra6_inter7_withrec, mask_out_full, label="intra6f_inter7_withrec")

    #Intra5
    mask_intra5_inter1_full_withrec = ConstantMask(mask_in_full, mask_hid_intra5_inter1_withrec, mask_out_full, label="intra5f_inter1_withrec")
    mask_intra5_inter2_full_withrec = ConstantMask(mask_in_full, mask_hid_intra5_inter2_withrec, mask_out_full, label="intra5f_inter2_withrec")
    mask_intra5_inter3_full_withrec = ConstantMask(mask_in_full, mask_hid_intra5_inter3_withrec, mask_out_full, label="intra5f_inter3_withrec")
    mask_intra5_inter4_full_withrec = ConstantMask(mask_in_full, mask_hid_intra5_inter4_withrec, mask_out_full, label="intra5f_inter4_withrec")
    mask_intra5_inter5_full_withrec = ConstantMask(mask_in_full, mask_hid_intra5_inter5_withrec, mask_out_full, label="intra5f_inter5_withrec")
    mask_intra5_inter6_full_withrec = ConstantMask(mask_in_full, mask_hid_intra5_inter6_withrec, mask_out_full, label="intra5f_inter6_withrec")
    mask_intra5_inter7_full_withrec = ConstantMask(mask_in_full, mask_hid_intra5_inter7_withrec, mask_out_full, label="intra5f_inter7_withrec")

    #Intra4
    mask_intra4_inter1_full_withrec = ConstantMask(mask_in_full, mask_hid_intra4_inter1_withrec, mask_out_full, label="intra4f_inter1_withrec")
    mask_intra4_inter2_full_withrec = ConstantMask(mask_in_full, mask_hid_intra4_inter2_withrec, mask_out_full, label="intra4f_inter2_withrec")
    mask_intra4_inter3_full_withrec = ConstantMask(mask_in_full, mask_hid_intra4_inter3_withrec, mask_out_full, label="intra4f_inter3_withrec")
    mask_intra4_inter4_full_withrec = ConstantMask(mask_in_full, mask_hid_intra4_inter4_withrec, mask_out_full, label="intra4f_inter4_withrec")
    mask_intra4_inter5_full_withrec = ConstantMask(mask_in_full, mask_hid_intra4_inter5_withrec, mask_out_full, label="intra4f_inter5_withrec")
    mask_intra4_inter6_full_withrec = ConstantMask(mask_in_full, mask_hid_intra4_inter6_withrec, mask_out_full, label="intra4f_inter6_withrec")
    mask_intra4_inter7_full_withrec = ConstantMask(mask_in_full, mask_hid_intra4_inter7_withrec, mask_out_full, label="intra4f_inter7_withrec")

    #Intra3
    mask_intra3_inter1_full_withrec = ConstantMask(mask_in_full, mask_hid_intra3_inter1_withrec, mask_out_full, label="intra3f_inter1_withrec")
    mask_intra3_inter2_full_withrec = ConstantMask(mask_in_full, mask_hid_intra3_inter2_withrec, mask_out_full, label="intra3f_inter2_withrec")
    mask_intra3_inter3_full_withrec = ConstantMask(mask_in_full, mask_hid_intra3_inter3_withrec, mask_out_full, label="intra3f_inter3_withrec")
    mask_intra3_inter4_full_withrec = ConstantMask(mask_in_full, mask_hid_intra3_inter4_withrec, mask_out_full, label="intra3f_inter4_withrec")
    mask_intra3_inter5_full_withrec = ConstantMask(mask_in_full, mask_hid_intra3_inter5_withrec, mask_out_full, label="intra3f_inter5_withrec")
    mask_intra3_inter6_full_withrec = ConstantMask(mask_in_full, mask_hid_intra3_inter6_withrec, mask_out_full, label="intra3f_inter6_withrec")
    mask_intra3_inter7_full_withrec = ConstantMask(mask_in_full, mask_hid_intra3_inter7_withrec, mask_out_full, label="intra3f_inter7_withrec")

     #Intra2
    mask_intra2_inter1_full_withrec = ConstantMask(mask_in_full, mask_hid_intra2_inter1_withrec, mask_out_full, label="intra2f_inter1_withrec")
    mask_intra2_inter2_full_withrec = ConstantMask(mask_in_full, mask_hid_intra2_inter2_withrec, mask_out_full, label="intra2f_inter2_withrec")
    mask_intra2_inter3_full_withrec = ConstantMask(mask_in_full, mask_hid_intra2_inter3_withrec, mask_out_full, label="intra2f_inter3_withrec")
    mask_intra2_inter4_full_withrec = ConstantMask(mask_in_full, mask_hid_intra2_inter4_withrec, mask_out_full, label="intra2f_inter4_withrec")
    mask_intra2_inter5_full_withrec = ConstantMask(mask_in_full, mask_hid_intra2_inter5_withrec, mask_out_full, label="intra2f_inter5_withrec")
    mask_intra2_inter6_full_withrec = ConstantMask(mask_in_full, mask_hid_intra2_inter6_withrec, mask_out_full, label="intra2f_inter6_withrec")
    mask_intra2_inter7_full_withrec = ConstantMask(mask_in_full, mask_hid_intra2_inter7_withrec, mask_out_full, label="intra2f_inter7_withrec")

    #Intra1
    mask_intra1_inter1_full_withrec = ConstantMask(mask_in_full, mask_hid_intra1_inter1_withrec, mask_out_full, label="intra1f_inter1_withrec")
    mask_intra1_inter2_full_withrec = ConstantMask(mask_in_full, mask_hid_intra1_inter2_withrec, mask_out_full, label="intra1f_inter2_withrec")
    mask_intra1_inter3_full_withrec = ConstantMask(mask_in_full, mask_hid_intra1_inter3_withrec, mask_out_full, label="intra1f_inter3_withrec")
    mask_intra1_inter4_full_withrec = ConstantMask(mask_in_full, mask_hid_intra1_inter4_withrec, mask_out_full, label="intra1f_inter4_withrec")
    mask_intra1_inter5_full_withrec = ConstantMask(mask_in_full, mask_hid_intra1_inter5_withrec, mask_out_full, label="intra1f_inter5_withrec")
    mask_intra1_inter6_full_withrec = ConstantMask(mask_in_full, mask_hid_intra1_inter6_withrec, mask_out_full, label="intra1f_inter6_withrec")
    mask_intra1_inter7_full_withrec = ConstantMask(mask_in_full, mask_hid_intra1_inter7_withrec, mask_out_full, label="intra1f_inter7_withrec")


    #now for the actual mask lists
    mask_list_full_intra7_withrec =[mask_intra7_inter1_full_withrec,
mask_intra7_inter2_full_withrec,
mask_intra7_inter3_full_withrec,
mask_intra7_inter4_full_withrec,
mask_intra7_inter5_full_withrec,
mask_intra7_inter6_full_withrec,
mask_intra7_inter7_full_withrec]
    mask_list_full_intra6_withrec =[mask_intra6_inter1_full_withrec,
mask_intra6_inter2_full_withrec,
mask_intra6_inter3_full_withrec,
mask_intra6_inter4_full_withrec,
mask_intra6_inter5_full_withrec,
mask_intra6_inter6_full_withrec,
mask_intra6_inter7_full_withrec]
    mask_list_full_intra5_withrec =[mask_intra5_inter1_full_withrec,
mask_intra5_inter2_full_withrec,
mask_intra5_inter3_full_withrec,
mask_intra5_inter4_full_withrec,
mask_intra5_inter5_full_withrec,
mask_intra5_inter6_full_withrec,
mask_intra5_inter7_full_withrec]
    mask_list_full_intra4_withrec =[mask_intra4_inter1_full_withrec,
mask_intra4_inter2_full_withrec,
mask_intra4_inter3_full_withrec,
mask_intra4_inter4_full_withrec,
mask_intra4_inter5_full_withrec,
mask_intra4_inter6_full_withrec,
mask_intra4_inter7_full_withrec]

    mask_list_full_intra3_withrec =[mask_intra3_inter1_full_withrec,
mask_intra3_inter2_full_withrec,
mask_intra3_inter3_full_withrec,
mask_intra3_inter4_full_withrec,
mask_intra3_inter5_full_withrec,
mask_intra3_inter6_full_withrec,
mask_intra3_inter7_full_withrec]

    mask_list_full_intra2_withrec =[mask_intra2_inter1_full_withrec,
mask_intra2_inter2_full_withrec,
mask_intra2_inter3_full_withrec,
mask_intra2_inter4_full_withrec,
mask_intra2_inter5_full_withrec,
mask_intra2_inter6_full_withrec,
mask_intra2_inter7_full_withrec]

    mask_list_full_intra1_withrec =[mask_intra1_inter1_full_withrec,
mask_intra1_inter2_full_withrec,
mask_intra1_inter3_full_withrec,
mask_intra1_inter4_full_withrec,
mask_intra1_inter5_full_withrec,
mask_intra1_inter6_full_withrec,
mask_intra1_inter7_full_withrec]

#now we save our crazy masklists

    print "SAVING MASKS"
    save_masks(mask_list_full_intra7_withrec, 'interintra7_4m_20iu_16hu_20ou_fullin_fullout_withrec_SHUFFLE')
    save_masks(mask_list_full_intra6_withrec, 'interintra6_4m_20iu_16hu_20ou_fullin_fullout_withrec_SHUFFLE')
    save_masks(mask_list_full_intra5_withrec, 'interintra5_4m_20iu_16hu_20ou_fullin_fullout_withrec_SHUFFLE')
    save_masks(mask_list_full_intra4_withrec, 'interintra4_4m_20iu_16hu_20ou_fullin_fullout_withrec_SHUFFLE')
    save_masks(mask_list_full_intra3_withrec, 'interintra3_4m_20iu_16hu_20ou_fullin_fullout_withrec_SHUFFLE')
    save_masks(mask_list_full_intra2_withrec, 'interintra2_4m_20iu_16hu_20ou_fullin_fullout_withrec_SHUFFLE')
    save_masks(mask_list_full_intra1_withrec, 'interintra1_4m_20iu_16hu_20ou_fullin_fullout_withrec_SHUFFLE')
    print "MASKS SAVED SUCCESSFULLY"
"""






    # # 18 hidden units (20 input units):
    #
    # mask_in_partial = create_mask_matrix(shape=(20,18), left_in=10, left_out=9, l2l_on=9, l2r_on=0, r2l_on=0, r2r_on=9)
    # mask_out_partial = create_mask_matrix(shape=(18,20), left_in=9, left_out=10, l2l_on=0, l2r_on=10, r2l_on=10, r2r_on=0)
    #
    # mask_hid_intra8 = create_mask_matrix(shape=(18,18), left_in=9, left_out=9, l2l_on=8, l2r_on=1, r2l_on=1, r2r_on=8)
    # mask_hid_intra7 = create_mask_matrix(shape=(18,18), left_in=9, left_out=9, l2l_on=7, l2r_on=2, r2l_on=2, r2r_on=7)
    # mask_hid_intra6 = create_mask_matrix(shape=(18,18), left_in=9, left_out=9, l2l_on=6, l2r_on=3, r2l_on=3, r2r_on=6)
    # mask_hid_intra5 = create_mask_matrix(shape=(18,18), left_in=9, left_out=9, l2l_on=5, l2r_on=4, r2l_on=4, r2r_on=5)
    #
    # mask_intra8_partial = ConstantMask(mask_in_partial, mask_hid_intra8, mask_out_partial, label="intra8p")
    # mask_intra7_partial = ConstantMask(mask_in_partial, mask_hid_intra7, mask_out_partial, label="intra7p")
    # mask_intra6_partial = ConstantMask(mask_in_partial, mask_hid_intra6, mask_out_partial, label="intra6p")
    # mask_intra5_partial = ConstantMask(mask_in_partial, mask_hid_intra5, mask_out_partial, label="intra5p")
    #
    # mask_list_partial = [mask_intra8_partial, mask_intra7_partial, mask_intra6_partial, mask_intra5_partial]
    # save_masks(mask_list_partial, 'interintra_4m_20iu_18hu_20ou_partialin_partialout')

    # # 20 hidden units (20 input units):
    #
    # mask_in_partial = create_mask_matrix(shape=(20,20), left_in=10, left_out=10, l2l_on=10, l2r_on=0, r2l_on=0, r2r_on=10)
    # mask_out_partial = create_mask_matrix(shape=(20,20), left_in=10, left_out=10, l2l_on=0, l2r_on=10, r2l_on=10, r2r_on=0)
    #
    # mask_hid_intra9 = create_mask_matrix(shape=(20,20), left_in=10, left_out=10, l2l_on=9, l2r_on=1, r2l_on=1, r2r_on=9)
    # mask_hid_intra8 = create_mask_matrix(shape=(20,20), left_in=10, left_out=10, l2l_on=8, l2r_on=2, r2l_on=2, r2r_on=8)
    # mask_hid_intra7 = create_mask_matrix(shape=(20,20), left_in=10, left_out=10, l2l_on=7, l2r_on=3, r2l_on=3, r2r_on=7)
    # mask_hid_intra6 = create_mask_matrix(shape=(20,20), left_in=10, left_out=10, l2l_on=6, l2r_on=4, r2l_on=4, r2r_on=6)
    # mask_hid_intra5 = create_mask_matrix(shape=(20,20), left_in=10, left_out=10, l2l_on=5, l2r_on=5, r2l_on=5, r2r_on=5)
    #
    # mask_intra9_partial = ConstantMask(mask_in_partial, mask_hid_intra9, mask_out_partial, label="intra9p")
    # mask_intra8_partial = ConstantMask(mask_in_partial, mask_hid_intra8, mask_out_partial, label="intra8p")
    # mask_intra7_partial = ConstantMask(mask_in_partial, mask_hid_intra7, mask_out_partial, label="intra7p")
    # mask_intra6_partial = ConstantMask(mask_in_partial, mask_hid_intra6, mask_out_partial, label="intra6p")
    # mask_intra5_partial = ConstantMask(mask_in_partial, mask_hid_intra5, mask_out_partial, label="intra5p")
    #
    # mask_list_partial = [mask_intra9_partial, mask_intra8_partial, mask_intra7_partial, mask_intra6_partial, mask_intra5_partial]
    # save_masks(mask_list_partial, 'interintra_5m_20iu_20hu_20ou_partialin_partialout')


    # 24 hidden units (30 input/output):

    # mask_in_full = create_mask_matrix(shape=(30,24), left_in=15, left_out=12, l2l_on=12, l2r_on=12, r2l_on=12, r2r_on=12)
    # mask_in_partial = create_mask_matrix(shape=(30,24), left_in=15, left_out=12, l2l_on=12, l2r_on=0, r2l_on=0, r2r_on=12)
    # mask_out_full = create_mask_matrix(shape=(24,30), left_in=12, left_out=15, l2l_on=15, l2r_on=15, r2l_on=15, r2r_on=15)
    # mask_out_partial = create_mask_matrix(shape=(24,30), left_in=12, left_out=15, l2l_on=0, l2r_on=15, r2l_on=15, r2r_on=0)
    #
    # mask_hid_intra11 = create_mask_matrix(shape=(24,24), left_in=12, left_out=12, l2l_on=11, l2r_on=1, r2l_on=1, r2r_on=11)
    # mask_hid_intra10 = create_mask_matrix(shape=(24,24), left_in=12, left_out=12, l2l_on=10, l2r_on=2, r2l_on=2, r2r_on=10)
    # mask_hid_intra09 = create_mask_matrix(shape=(24,24), left_in=12, left_out=12, l2l_on=9, l2r_on=3, r2l_on=3, r2r_on=9)
    # mask_hid_intra08 = create_mask_matrix(shape=(24,24), left_in=12, left_out=12, l2l_on=8, l2r_on=4, r2l_on=4, r2r_on=8)
    # mask_hid_intra07 = create_mask_matrix(shape=(24,24), left_in=12, left_out=12, l2l_on=7, l2r_on=5, r2l_on=5, r2r_on=7)
    # mask_hid_intra06 = create_mask_matrix(shape=(24,24), left_in=12, left_out=12, l2l_on=6, l2r_on=6, r2l_on=6, r2r_on=6)
    #
    # mask_intra11_full = ConstantMask(mask_in_full, mask_hid_intra11, mask_out_full, label="intra11f")
    # mask_intra10_full = ConstantMask(mask_in_full, mask_hid_intra10, mask_out_full, label="intra10f")
    # mask_intra09_full = ConstantMask(mask_in_full, mask_hid_intra09, mask_out_full, label="intra09f")
    # mask_intra08_full = ConstantMask(mask_in_full, mask_hid_intra08, mask_out_full, label="intra08f")
    # mask_intra07_full = ConstantMask(mask_in_full, mask_hid_intra07, mask_out_full, label="intra07f")
    # mask_intra06_full = ConstantMask(mask_in_full, mask_hid_intra06, mask_out_full, label="intra06f")
    #
    # mask_intra11_partial = ConstantMask(mask_in_partial, mask_hid_intra11, mask_out_partial, label="intra11p")
    # mask_intra10_partial = ConstantMask(mask_in_partial, mask_hid_intra10, mask_out_partial, label="intra10p")
    # mask_intra09_partial = ConstantMask(mask_in_partial, mask_hid_intra09, mask_out_partial, label="intra09p")
    # mask_intra08_partial = ConstantMask(mask_in_partial, mask_hid_intra08, mask_out_partial, label="intra08p")
    # mask_intra07_partial = ConstantMask(mask_in_partial, mask_hid_intra07, mask_out_partial, label="intra07p")
    # mask_intra06_partial = ConstantMask(mask_in_partial, mask_hid_intra06, mask_out_partial, label="intra06p")
    #
    # mask_list_full = [mask_intra11_full, mask_intra10_full, mask_intra09_full, mask_intra08_full, mask_intra07_full, mask_intra06_full]
    # mask_list_partial = [mask_intra11_partial, mask_intra10_partial, mask_intra09_partial, mask_intra08_partial, mask_intra07_partial, mask_intra06_partial]
    #
    # save_masks(mask_list_full, 'interintra_6m_30iu_24hu_30ou_fullin_fullout')
    # save_masks(mask_list_partial, 'interintra_6m_30iu_24hu_30ou_partialin_partialout')


    # # 32 hidden units:
    #
    # mask_in_full = create_mask_matrix(shape=(40,32), left_in=20, left_out=16, l2l_on=16, l2r_on=16, r2l_on=16, r2r_on=16)
    # mask_in_partial = create_mask_matrix(shape=(40,32), left_in=20, left_out=16, l2l_on=16, l2r_on=0, r2l_on=0, r2r_on=16)
    # mask_out_full = create_mask_matrix(shape=(32,40), left_in=16, left_out=20, l2l_on=20, l2r_on=20, r2l_on=20, r2r_on=20)
    # mask_out_partial = create_mask_matrix(shape=(32,40), left_in=16, left_out=20, l2l_on=0, l2r_on=20, r2l_on=20, r2r_on=0)
    #
    # mask_hid_intra15 = create_mask_matrix(shape=(32,32), left_in=16, left_out=16, l2l_on=15, l2r_on=01, r2l_on=01, r2r_on=15)
    # mask_hid_intra14 = create_mask_matrix(shape=(32,32), left_in=16, left_out=16, l2l_on=14, l2r_on=02, r2l_on=02, r2r_on=14)
    # mask_hid_intra13 = create_mask_matrix(shape=(32,32), left_in=16, left_out=16, l2l_on=13, l2r_on=03, r2l_on=03, r2r_on=13)
    # mask_hid_intra12 = create_mask_matrix(shape=(32,32), left_in=16, left_out=16, l2l_on=12, l2r_on=04, r2l_on=04, r2r_on=12)
    # mask_hid_intra11 = create_mask_matrix(shape=(32,32), left_in=16, left_out=16, l2l_on=11, l2r_on=05, r2l_on=05, r2r_on=11)
    # mask_hid_intra10 = create_mask_matrix(shape=(32,32), left_in=16, left_out=16, l2l_on=10, l2r_on=06, r2l_on=06, r2r_on=10)
    # mask_hid_intra09 = create_mask_matrix(shape=(32,32), left_in=16, left_out=16, l2l_on=9, l2r_on=7, r2l_on=7, r2r_on=9)
    # mask_hid_intra08 = create_mask_matrix(shape=(32,32), left_in=16, left_out=16, l2l_on=8, l2r_on=8, r2l_on=8, r2r_on=8)
    #
    # mask_intra15_partial = ConstantMask(mask_in_partial, mask_hid_intra15, mask_out_partial, label="intra15p")
    # mask_intra14_partial = ConstantMask(mask_in_partial, mask_hid_intra14, mask_out_partial, label="intra14p")
    # mask_intra13_partial = ConstantMask(mask_in_partial, mask_hid_intra13, mask_out_partial, label="intra13p")
    # mask_intra12_partial = ConstantMask(mask_in_partial, mask_hid_intra12, mask_out_partial, label="intra12p")
    # mask_intra11_partial = ConstantMask(mask_in_partial, mask_hid_intra11, mask_out_partial, label="intra11p")
    # mask_intra10_partial = ConstantMask(mask_in_partial, mask_hid_intra10, mask_out_partial, label="intra10p")
    # mask_intra09_partial = ConstantMask(mask_in_partial, mask_hid_intra09, mask_out_partial, label="intra09p")
    # mask_intra08_partial = ConstantMask(mask_in_partial, mask_hid_intra08, mask_out_partial, label="intra08p")
    #
    # mask_list_partial = [mask_intra15_partial, mask_intra14_partial, mask_intra13_partial, mask_intra12_partial, mask_intra11_partial, mask_intra10_partial, mask_intra09_partial, mask_intra08_partial]
    #
    # save_masks(mask_list_partial, 'interintra_8m_40iu_32hu_40ou_partialin_partialout')

def null_mask():
	#this is just a testing function to make sure that the network doesn't work at all with no masks at all. 
    mask_in_full = create_mask_matrix(shape=(20,16), left_in=10, left_out=8, l2l_on=8, l2r_on=8, r2l_on=8, r2r_on=8)
    #pdb.set_trace()
    mask_in_partial = create_mask_matrix(shape=(20,16), left_in=10, left_out=8, l2l_on=8, l2r_on=0, r2l_on=0, r2r_on=8)
    mask_in_mixed = create_mixed_matrix(shape=(20,16), topleft=0)
    mask_out_full = create_mask_matrix(shape=(16,20), left_in=8, left_out=10, l2l_on=10, l2r_on=10, r2l_on=10, r2r_on=10)
    mask_out_partial = create_mask_matrix(shape=(16,20), left_in=8, left_out=10, l2l_on=0, l2r_on=10, r2l_on=10, r2r_on=0)
    mask_out_mixed = create_mixed_matrix(shape=(16,20), topleft=1)

	
    mask_null = create_mask_matrix(shape=(16,16), left_in=8, left_out=8, l2l_on=0, l2r_on=0, r2l_on=0, r2r_on=0, contype='asymmetric-withrecurrent', shuffle = False)
    mask_full_null = ConstantMask(mask_in_full, mask_null, mask_out_full, label="null")
    mask_list_null =[mask_full_null]
    save_masks(mask_list_null, 'NULL_PARTIAL_MASK_TEST')
    print "SAVED NULL MASK"

if __name__ == '__main__':
    main()
    #null_mask()


