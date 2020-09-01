import os, sys
import numpy as np
import keras.layers
import random
from nibabel.testing import data_path
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt
from matplotlib import pyplot, cm

# local imports.
print('current loc : ', os.getcwd())
dir_VoxelMorph = os.path.join(os.getcwd(), 'VoxelMorph')
dir_voxelmorph = os.path.join(dir_VoxelMorph, 'voxelmorph') #this dir_voxelmorph is diff from dir_VoxelMorph
print('current paths : ', sys.path)

sys.path.append(os.path.join(dir_VoxelMorph, 'voxelmorph','ext','pynd-lib'))
sys.path.append(os.path.join(dir_VoxelMorph, 'voxelmorph','ext','pytools-lib'))
sys.path.append(os.path.join(dir_VoxelMorph, 'voxelmorph','ext','neuron'))
sys.path.append(os.path.join(dir_VoxelMorph, 'voxelmorph'))
sys.path.append(os.path.join(dir_VoxelMorph, 'YoonGuu'))
sys.path.append(os.path.join(dir_VoxelMorph))

for pth in sys.path:
    print("path : ", pth)

from VoxelMorph.voxelmorph.src import losses
from VoxelMorph.voxelmorph.src import networks

#import VoxelMorph.voxelmorph.src as vxm
import neuron
from medpy.io import load, save

################################################
#Registration of Brain MRI
################################################
# we've prepared the data in the following files
# prepared as N x H x W


core_path = 'F:\MPI dataset\Duc_dataset\check'
dir_BE_CN_MRA = 'F:\\MPI dataset\\Duc_dataset\\brain_extraction_MRA_dim_same'
dir_BE_CN_T2 = 'F:\\MPI dataset\\Duc_dataset\\brain_extraction_T2_CN_dim_same'


list_Normal_MRA = os.listdir(dir_BE_CN_MRA)
list_Normal_T2 = os.listdir(dir_BE_CN_T2)



####################################################################################
'''
MRA, T2 data shape --> numbers 
MRA data shape --> {(182, 218, 182): 109}
dic_T2_shape   --> {(182, 218, 182): 107}

'''
dic_MRA_shape = {}
MRA_shape_set = set()
for MRA in list_Normal_MRA:
    dir_MRA_subj = os.path.join(dir_BE_CN_MRA, MRA)
    img_load = nib.load(dir_MRA_subj)
    MRA_shape_set.add(img_load.shape)

for MRA_shape in MRA_shape_set:
    dic_MRA_shape[MRA_shape] = 0

for MRA in list_Normal_MRA:
    dir_MRA_subj = os.path.join(dir_BE_CN_MRA, MRA)
    img_load = nib.load(dir_MRA_subj)
    dic_MRA_shape[img_load.shape] +=1



dic_T2_shape = {}
T2_shape_set = set()
for T2 in list_Normal_T2:
    dir_T2_subj = os.path.join(dir_BE_CN_T2, T2)
    img_load = nib.load(dir_T2_subj)
    T2_shape_set.add(img_load.shape)

for T2_shape in T2_shape_set:
    dic_T2_shape[T2_shape] = 0

for T2 in list_Normal_T2:
    dir_T2_subj = os.path.join(dir_BE_CN_T2, T2)
    img_load = nib.load(dir_T2_subj)
    dic_T2_shape[img_load.shape] +=1




#select random imgs
random_select_MRA = random.choices(list_Normal_MRA, k=5)
random_select_T2 = random.choices(list_Normal_T2, k=5)



random_MRA_0 = os.path.join(dir_BE_CN_MRA, random_select_MRA[0])
random_MRA_1 = os.path.join(dir_BE_CN_MRA, random_select_MRA[1])
random_T2_0 = os.path.join(dir_BE_CN_T2, random_select_T2[0])
random_T2_1 = os.path.join(dir_BE_CN_T2, random_select_T2[1])

# plot nii.gz file
# plotting.plot_img(random_MRA_0 )
# plotting.plot_img(random_MRA_1)
# plotting.plot_img(random_T2_0)
# plotting.plot_img(random_T2_1, cut_coords=(-94,-131,55) )


#load img
img_MRA_0 = nib.load(random_MRA_0)
img_MRA_1 = nib.load(random_MRA_1)
img_T2_0 = nib.load(random_T2_0)
img_T2_1 = nib.load(random_T2_1)


img_MRA_0.shape #(182, 218, 182)
img_MRA_1.shape #(182, 218, 182)
img_T2_0.shape  #(182, 218, 182)
img_T2_1.shape  #(182, 218, 182)

img_MRA_0_array =  np.array(img_MRA_0.dataobj)
img_MRA_1_array =  np.array(img_MRA_1.dataobj)
img_T2_0_array =  np.array(img_T2_0.dataobj)
img_T2_1_array =  np.array(img_T2_1.dataobj)


# code was used in MNIST dataset
#img_MRA_0.shape #(182, 218, 182)

nb_enc_features = [32, 32, 32, 32]
nb_dec_features = [32, 32, 32, 32, 32, 16]

ndims= 2
nb_vis = 5

#from min to max, select 5 numbers
idx_MRA = np.random.randint(0, img_MRA_0.shape[0], [5,])
print(idx_MRA)
example_digits_MRA = [f for f in img_MRA_0_array[...,idx_MRA]]
# len(example_digits_MRA[0][0])



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

#vol_shape_T2 = img_T2_0.shape[1:] #(256, 128)
####################################################



# visualize
# plt.imshow(img_MRA_0.get_data()[100,:,:])
# plt.imshow(img_MRA_0.get_data()[:,:,120])
# plt.imshow(img_MRA_0.get_data()[:,100,:])
# print(img_MRA_0.header)








#dimensions
# MRA_0_yz               # 5 218 182 --> list
# MRA_0_xz_rotated.shape #(5, 182, 182)
# MRA_0_xz_flipped.shape #(5, 182, 182)
# MRA_0_xy_flipped.shape #(5, 218, 182) --> tuple
# MRA_0_xy_rotated.shape #(5, 182, 218) --> tuple

# unet
unet = networks.unet_core(vol_shape_T2, nb_enc_features, nb_dec_features);
disp_tensor = keras.layers.Conv2D(ndims, kernel_size=3, padding='same', name='disp')(unet.output)

# spatial transfomer
spatial_transformer = neuron.layers.SpatialTransformer(name='image_warping')
moved_image_tensor = spatial_transformer([unet.inputs[0], disp_tensor])

# final model
vxm_model = keras.models.Model(unet.inputs, [moved_image_tensor, disp_tensor])

# losses. Keras recognizes the string 'mse' as mean squared error, so we don't have to code it
losses = ['mse', losses.Grad('l2').loss]

# usually, we have to balance the two losses by a hyper-parameter.
lambda_param = 0.01
loss_weights = [1, lambda_param]

vxm_model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss=losses, loss_weights=loss_weights)

vol_shape = img_T2_1.shape[:2] #(448, 128)
zero_phi = np.zeros([32, *vol_shape, ndims])
zero_phi.shape #(32, 192, 256, 2)



def vxm_data_generator(x_data, batch_size=32):
    """
    generator that takes in data of size [N, H, W], and yields data for our vxm model

    Note that we need to provide numpy data for each input, and each output

    inputs:  moving_image [bs, H, W, 1], fixed_image [bs, H, W, 1]
    outputs: moved_image  [bs, H, W, 1], zeros [bs, H, W, 2]
    """
    # preliminary sizing
    vol_shape = x_data.shape[1:]  # extract data shape
    ndims = len(vol_shape)

    # prepare a zero array the size of the deformation. We'll explain this below.
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])

    while True:
        # prepare inputs
        # inputs need to be of the size [batch_size, H, W, number_features]
        #   number_features at input is 1 for us
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[idx1, ..., np.newaxis]
        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[idx2, ..., np.newaxis]
        inputs = [moving_images, fixed_images]

        # outputs
        # we need to prepare the "true" moved image.
        # Of course, we don't have this, but we know we want to compare
        # the resulting moved image with the fixed image.
        # we also wish to penalize the deformation field.
        outputs = [fixed_images, zero_phi]

        yield inputs, outputs

# let's test it
train_generator = vxm_data_generator(img_T2_1_array, batch_size=8)
input_sample, output_sample = next(train_generator)

# visualize
slices_2d = [f[0,...,0] for f in input_sample + output_sample]
titles = ['input_moving', 'input_fixed', 'output_sample_true', 'zero']
neuron.plot.slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True);

nb_epochs = 10
steps_per_epoch = 10



#here error was made
hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2);

# for the purpose of the tutorial we ran very few epochs.
# Here we load a model that was run for 10 epochs and 100 steps per epochs

# this is loading the weights
dir_YoonGuu_models = os.path.join(os.getcwd(), 'VoxelMorph','voxelmorph', 'models', 'YoonGuu_models')
# vxm_model.load_weights(os.path.join(dir_YoonGuu_models, '20_02_22.h5'))

# as before, let's visualize what happened

def plot_history(hist, loss_name='loss'):
    """
    Quick function to plot the history
    """
    plt.figure()
    plt.plot(hist.epoch, hist.history[loss_name], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

plot_history(hist)

val_generator = vxm_data_generator(img_T2_0_array, batch_size = 1)

val_input, _ = next(val_generator)

val_pred = vxm_model.predict(val_input)

# visualize
slices_2d = [f[0,...,0] for f in val_input + val_pred]
titles = ['input_moving', 'input_fixed', 'predicted_moved', 'deformation_x']
neuron.plot.slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True);

flow = val_pred[1].squeeze()[::3,::3]
neuron.plot.flow([flow], width=5);











vxm_model.save_weights(os.path.join(dir_YoonGuu_models, '20_02_22.h5'))


vxm_model.load_weights('/kaggle/input/learn2reg-unsupervised-models/brain_2d_shortrun.h5')
our_val_pred = vxm_model.predict(val_input)

vxm_model.load_weights('/kaggle/input/learn2reg-unsupervised-models/brain_2d_mseonly.h5')
mse_val_pred = vxm_model.predict(val_input)

# visualize both models
slices_2d = [f[0,...,0] for f in [val_input[1]] + our_val_pred ]
titles = ['input_fixed', 'our_pred_moved', 'our_disp_x']
neuron.plot.slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True);

# visualize both models
slices_2d = [f[0,...,0] for f in [val_input[1]] + mse_val_pred]
titles = ['input_fixed', 'mse_pred_moved', 'mse_pred_x']
neuron.plot.slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True);

neuron.plot.flow([f[1].squeeze()[::3,::3] for f in [our_val_pred, mse_val_pred]], width=10);
