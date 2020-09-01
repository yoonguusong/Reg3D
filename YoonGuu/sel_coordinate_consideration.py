
'''
########################################################################################
this is for how did I considered to get error
                                    -YoonGuu Song-
                                    -Gwangju Institute of Science and Technology
########################################################################################
'''

########################check############################
'''
5 xy,xz,yz data was selected but  
(182, 218, 5), (182, 5, 182), (5, 218, 182) --> (5, 218, 182), (5, 182, 182), (5, 218, 182)

'''
MRA_yz = img_MRA_0_array[idx_MRA,...] #(5, 218, 182)
MRA_xy = img_MRA_0_array[...,idx_MRA] #(182, 218, 5)
MRA_xz = img_MRA_0_array[:,idx_MRA,:] #(182, 5, 182)

# MRA_xy_reshape = np.transpose(MRA_xy, (2, 1, 0)) #(5, 218, 182)
# MRA_xz_reshape = np.transpose(MRA_xz, (1, 0, 2)) #(5, 182, 182)

############################################################################



'''yz'''
idx_MRA_0 = np.random.randint(0, img_MRA_0.shape[0], [5,])
print(img_MRA_0_array[idx_MRA_0,...].shape)
MRA_0_yz = [f for f in img_MRA_0_array[idx_MRA_0,...]]
print('MRA_0_yz.shape : ', len(MRA_0_yz), len(MRA_0_yz[0]), len(MRA_0_yz[0][0]))
neuron.plot.slices(MRA_0_yz, cmaps=['gray'], do_colorbars=True);



'''xz'''
idx_MRA_1 = np.random.randint(0, img_MRA_0.shape[1], [5,])
MRA_0_xz = [f for f in img_MRA_0_array[:,idx_MRA_1,:]]
print('MRA_0_xz.shape : ', len(MRA_0_xz), len(MRA_0_xz[0]), len(MRA_0_xz[0][0]))
########################dim check####################
np.transpose(img_MRA_0_array[:,idx_MRA_1,:], (0, 1, 2)).shape #(182, 5, 182)
np.transpose(img_MRA_0_array[:,idx_MRA_1,:], (0, 2, 1)).shape #(182, 182, 5)
np.transpose(img_MRA_0_array[:,idx_MRA_1,:], (1, 0, 2)).shape #(5, 182, 182) --> 90 degreee rotated
np.transpose(img_MRA_0_array[:,idx_MRA_1,:], (1, 2, 0)).shape #(5, 182, 182) --> x,y flip
np.transpose(img_MRA_0_array[:,idx_MRA_1,:], (2, 1, 0)).shape #(182, 5, 182)
np.transpose(img_MRA_0_array[:,idx_MRA_1,:], (2, 0, 1)).shape #(182, 182, 5)

MRA_0_xz_rotated = np.transpose(img_MRA_0_array[:,idx_MRA_1,:], (1, 0, 2)) #(5, 182, 182)
print('MRA_0_xz_rotated.shape : ', MRA_0_xz_rotated.shape)
MRA_0_xz_flipped = np.transpose(img_MRA_0_array[:,idx_MRA_1,:], (1, 2, 0)) #(5, 182, 182)
print('MRA_0_xz_flipped.shape : ', MRA_0_xz_flipped.shape)

#swapaxes result is same as rotated
#neuron.plot.slices(img_MRA_0_array[:,idx_MRA_1,:].swapaxes(0,1), cmaps=['gray'], do_colorbars=True);
neuron.plot.slices(MRA_0_xz_rotated, cmaps=['gray'], do_colorbars=True);
neuron.plot.slices(MRA_0_xz_flipped, cmaps=['gray'], do_colorbars=True);

####################################################




'''xy'''
idx_MRA_2 = np.random.randint(0, img_MRA_0.shape[2], [5,])
MRA_xy = img_MRA_0_array[...,idx_MRA]
print('MRA_xy.shape : ', len(MRA_xy), len(MRA_xy[0]), len(MRA_xy[0][0]))
MRA_0_xy = [f for f in img_MRA_0_array[..., idx_MRA_2]]

####################################################

np.transpose(img_MRA_0_array[..., idx_MRA_2], (0, 1, 2)).shape #(182, 218, 5)
np.transpose(img_MRA_0_array[..., idx_MRA_2], (0, 2, 1)).shape #(182, 5, 218)
np.transpose(img_MRA_0_array[..., idx_MRA_2], (1, 0, 2)).shape #(218, 182, 5)
np.transpose(img_MRA_0_array[..., idx_MRA_2], (1, 2, 0)).shape #(218, 5, 182)
np.transpose(img_MRA_0_array[..., idx_MRA_2], (2, 1, 0)).shape #(5, 218, 182)
np.transpose(img_MRA_0_array[..., idx_MRA_2], (2, 0, 1)).shape #(5, 182, 218)

MRA_0_xy_flipped = np.transpose(img_MRA_0_array[..., idx_MRA_2], (2, 1, 0)) #(5, 218, 182)
print('MRA_0_xy_flipped.shape : ', MRA_0_xy_flipped.shape)
MRA_0_xy_rotated = np.transpose(img_MRA_0_array[..., idx_MRA_2], (2, 0, 1)) # (5, 182, 218)
print('MRA_0_xy_rotated.shape : ', MRA_0_xy_rotated.shape)

neuron.plot.slices(MRA_0_xy_flipped, cmaps=['gray'], do_colorbars=True);
neuron.plot.slices(MRA_0_xy_rotated, cmaps=['gray'], do_colorbars=True);
####################################################

