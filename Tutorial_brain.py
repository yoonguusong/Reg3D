#imports
import os, sys, random

#3rd party imports
import numpy as np
import keras.layers

from nibabel.testing import data_path
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt
from matplotlib import pyplot, cm
from medpy.io import load, save

# local imports.
print('current loc : ', os.getcwd())
dir_Reg3D = os.getcwd()
dir_voxelmorph = os.path.join(dir_Reg3D, 'voxelmorph') #this dir_voxelmorph is diff from dir_VoxelMorph

sys.path.append(os.path.join(dir_Reg3D, 'voxelmorph','ext','pynd-lib'))
sys.path.append(os.path.join(dir_Reg3D, 'voxelmorph','ext','pytools-lib'))
sys.path.append(os.path.join(dir_Reg3D, 'voxelmorph','ext','neuron'))
sys.path.append(os.path.join(dir_Reg3D, 'voxelmorph'))
sys.path.append(os.path.join(dir_Reg3D, 'YoonGuu'))
sys.path.append(os.path.join(dir_Reg3D))

for pth in sys.path:
    print("path : ", pth)

from voxelmorph.src import losses
from voxelmorph.src import networks
import neuron #import VoxelMorph.voxelmorph.src as vxm
# import img_dim_ck
import sel_coordinate
import Upsampling_YG




################################################
#Registration of Brain MRI
################################################
# we've prepared the data in the following files
# prepared as N x H x W
core_path = 'F:\MPI dataset\Duc_dataset\check'

#error --even tho I did make BE, I didn't applied on
# dir_BE_CN_MRA = 'F:\\MPI dataset\\Duc_dataset\\Not_BE_MRA_dim_same'
# dir_BE_CN_T2 = 'F:\\MPI dataset\\Duc_dataset\\Not_BE_T2_CN_dim_same'

# Brain Extracted one
dir_BE_CN_MRA = 'F:\\Dataset\\Brain\\Duc_dataset\\BE_MRA_CN'
dir_BE_CN_T2 = 'F:\\Dataset\\Brain\\Duc_dataset\\BE_T2_CN'

list_Normal_MRA = os.listdir(dir_BE_CN_MRA)
list_Normal_T2 = os.listdir(dir_BE_CN_T2)

# dimension check
num_MRA_dim,dic_MRA_list = sel_coordinate.img_dim(dir_BE_CN_MRA)
num_T2_dim,dic_T2_list =sel_coordinate.img_dim(dir_BE_CN_T2)
print('num_MRA_dim : ', num_MRA_dim)
print('dic_MRA_list : ', dic_MRA_list)

print('num_T2_dim : ', num_T2_dim)
print('dic_T2_list : ', dic_T2_list)


'''
sel_coord(dir_folder, num_file=5, coordinate = 'xy', visual_num_files = True, visual_3plot= False, visual_plot=False )
 'xy' ---> axial, horizontal
 'yz' ---> sagittal
 'xz' ---> coronal
'''

'''
sel_coordinate.sel_coord(dir_folder = dir_BE_CN_MRA, num_file=5, coordinate = 'xy', visual_num_files=True)
# sel_coordinate.sel_coord(dir_folder = dir_BE_CN_MRA, num_file=5, coordinate = 'yz', visual_num_files=True)
# sel_coordinate.sel_coord(dir_folder = dir_BE_CN_MRA, num_file=5, coordinate = 'xz', visual_num_files=True)

sel_coordinate.sel_coord(dir_folder = dir_BE_CN_MRA, num_file=1, coordinate = 'yz', visual_3plot= True)
sel_coordinate.sel_coord(dir_folder = dir_BE_CN_MRA, num_file=2, coordinate = 'zx', visual_plot= True)
'''


#adjusting part
MRA_use = True
T2_use=True
weights_apply=True

img_idx_range='max'
num_file='max'

LoadFileName ='brain_2d_200703_MRA_T2_wo_loading_Num_max_idx_max_1000300.h5'
SaveFileName ='brain_2d_200703_MRA_T2_w_load_Num_max_idx_max.h5'


#visualize
# MRA_xy_MM_array = sel_coordinate.sel_coord(dir_folder = dir_BE_CN_MRA, img_index_range=1, num_file=5, coordinate = 'xy', visual_num_files=True)
# T2_yz_array_MM = sel_coordinate.sel_coord(dir_folder = dir_BE_CN_T2, img_index_range=1, num_file=5, coordinate = 'xy', visual_num_files=True)

if MRA_use==True:
    '''#MRA data'''
    # MRA_xy_MM_array = sel_coordinate.sel_coord(dir_folder = dir_BE_CN_MRA, img_index_range='max', num_file='max', coordinate = 'xy', visual_num_files=False)
    MRA_xy_MM_array = sel_coordinate.sel_coord(dir_folder=dir_BE_CN_MRA, img_index_range=img_idx_range, num_file=num_file,
                                               coordinate='xy', visual_num_files=False)
    print('MRA_xy_MM_array shape : ', MRA_xy_MM_array.shape) #(7848, 218, 182) --> 109 * 72 = 7848
    dim_MRA_xy =sel_coordinate.dimension_ck(MRA_xy_MM_array) #(218, 182) this looks good

    #check encoding and decoding sample size whether it will be same or not_YG
    correct_dim  = Upsampling_YG.en_decom_size_ck_2D(dim_MRA_xy.shape)



    #divide numpy array into training, validation, test set
    MRA_train, MRA_val, _=sel_coordinate.train_val_test_divide(MRA_xy_MM_array, train_pct=0.8, test_pct=0.1, val_pct=0.1, test_val_combine=True)

    #zeropadding medical images to match the suggested size
    changed_np_MRA_train = sel_coordinate.zeropadding_2Ds(MRA_train, correct_dim)
    changed_np_MRA_val = sel_coordinate.zeropadding_2Ds(MRA_val, correct_dim)


    # MRA_yz_array = sel_coordinate.sel_coord(dir_folder = dir_BE_CN_MRA, num_file=1, coordinate = 'yz', visual_num_files=False)
    # dim_MRA_yz =sel_coordinate.dimension_ck(MRA_yz_array) #(218, 182)
    # MRA_xz_array = sel_coordinate.sel_coord(dir_folder = dir_BE_CN_MRA, num_file=5, coordinate = 'xz', visual_num_files=False)
    # dim_MRA_xz =sel_coordinate.dimension_ck(MRA_xz_array) #(182, 182)




if T2_use==True:
    '''#T2 data'''
    T2_xy_array_MM = sel_coordinate.sel_coord(dir_folder = dir_BE_CN_T2, img_index_range=img_idx_range, num_file=num_file, coordinate = 'xy', visual_num_files=False)
    print('T2_yz_array_MM shape : ', T2_xy_array_MM.shape)
    dim_T2_xy =sel_coordinate.dimension_ck(T2_xy_array_MM)
    # T2_yz_array_MM = sel_coordinate.sel_coord(dir_folder=dir_BE_CN_T2, img_index_range=1, num_file=5,
    #                                           coordinate='xy', visual_num_files=True)

    # T2_yz_array = sel_coordinate.sel_coord(dir_folder = dir_BE_CN_T2, num_file=5, coordinate = 'yz', visual_num_files=True)
    # dim_T2_yz =sel_coordinate.dimension_ck(T2_yz_array)
    # T2_yz_array = sel_coordinate.sel_coord(dir_folder = dir_BE_CN_T2, num_file=6, coordinate = 'xz', visual_num_files=True)
    # dim_T2_xz =sel_coordinate.dimension_ck(T2_yz_array)

    #check encoding and decoding sample size whether it will be same or not_YG
    correct_dim  = Upsampling_YG.en_decom_size_ck_2D(dim_T2_xy.shape)

    #divide numpy array into training, validation, test set
    T2_train, T2_val, _=sel_coordinate.train_val_test_divide(T2_xy_array_MM, train_pct=0.8, test_pct=0.1, val_pct=0.1, test_val_combine=True)

    #zeropadding medical images to match the suggested size
    changed_np_T2_train = sel_coordinate.zeropadding_2Ds(T2_train, correct_dim)
    changed_np_T2_val = sel_coordinate.zeropadding_2Ds(T2_val, correct_dim)





'''
this is the part to control the all projects
'''
# train_data = changed_np_T2_train
# val_data =changed_np_T2_val
print('changed_np_MRA_train.shape : ', changed_np_MRA_train.shape)
print('changed_np_MRA_val.shape : ', changed_np_MRA_val.shape)
print('changed_np_T2_train.shape : ', changed_np_T2_train.shape) #(4, 256, 192)
print('changed_np_T2_val.shape : ', changed_np_T2_val.shape) #(1, 256, 192)

resized_MRA_train= sel_coordinate.modal_resize(changed_np_MRA_train, changed_np_T2_train)
resized_MRA_val= sel_coordinate.modal_resize(changed_np_MRA_val, changed_np_T2_val)
print('resized_MRA_train.shape : ', resized_MRA_train.shape) #(4, 256, 192)
print('resized_MRA_val.shape : ', resized_MRA_val.shape) #(1, 256, 192)

train_data = np.concatenate((resized_MRA_train, changed_np_T2_train))
val_data = np.concatenate((resized_MRA_val, changed_np_T2_val))
print('train_data.shape : ', train_data.shape)
print('val_data.shape : ', val_data.shape)




nb_enc_features = [32, 32, 32, 32]
nb_dec_features = [32, 32, 32, 32, 32, 16]

ndims= 2
nb_vis = 5


# unet
unet = networks.unet_core(correct_dim, nb_enc_features, nb_dec_features)  # --> (None, 24, 24, 32), (None, 23, 23, 32)
print('unet.inputs[0] : ', unet.inputs[0])
print('unet.inputs[1] : ',  unet.inputs[1])
print('output : ', unet.output)

disp_tensor = keras.layers.Conv2D(ndims, kernel_size=3, padding='same', name='disp')(unet.output)
print('displacement tensor:', disp_tensor)
# spatial transfomer
spatial_transformer = neuron.layers.SpatialTransformer(name='image_warping')
print('spatial_transformer', spatial_transformer)
# warp the image
moved_image_tensor = spatial_transformer([unet.inputs[0], disp_tensor])



# final model
vxm_model = keras.models.Model(unet.inputs, [moved_image_tensor, disp_tensor])



# losses. Keras recognizes the string 'mse' as mean squared error, so we don't have to code it
losses = ['mse', losses.Grad('l2').loss]

# usually, we have to balance the two losses by a hyper-parameter.
lambda_param = 0.01
loss_weights = [1, lambda_param]

vxm_model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss=losses, loss_weights=loss_weights)

# vol_shape = img_T2_1.shape[:2] #(448, 128)
# zero_phi = np.zeros([32, *vol_shape, ndims])
# zero_phi.shape #(32, 192, 256, 2)



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
# img_load = nib.load(rand_img)

train_generator = vxm_data_generator(train_data, batch_size=8)
input_sample, output_sample = next(train_generator)

# visualize
slices_2d = [f[0,...,0] for f in input_sample + output_sample]
titles = ['input_moving', 'input_fixed', 'output_sample_true', 'zero']
neuron.plot.slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True);

# original number was 10
# nb_epochs = 10
nb_epochs = 10
steps_per_epoch = 10



#here error was made
hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2);

# for the purpose of the tutorial we ran very few epochs.
# Here we load a model that was run for 10 epochs and 100 steps per epochs

# this is loading the weights
dir_YoonGuu_models = os.path.join(os.getcwd(), 'VoxelMorph', 'voxelmorph', 'models', 'YoonGuu_models')
if weights_apply == True:
    vxm_model.load_weights(os.path.join(dir_YoonGuu_models, 'brain_2d_shortrun.h5'))


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

val_generator = vxm_data_generator(val_data, batch_size = 1)

val_input, _ = next(val_generator)

val_pred = vxm_model.predict(val_input)

# visualize
slices_2d = [f[0,...,0] for f in val_input + val_pred]
titles = ['input_moving', 'input_fixed', 'predicted_moved', 'deformation_x']
neuron.plot.slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True);

flow = val_pred[1].squeeze()[::3,::3]
neuron.plot.flow([flow], width=5);
save_weight = os.path.join(dir_YoonGuu_models, SaveFileName)
vxm_model.save_weights(save_weight)




load_weight = os.path.join(dir_YoonGuu_models, LoadFileName)
vxm_model.load_weights(load_weight)
our_val_pred = vxm_model.predict(val_input)

# dir_weight_val = os.path.join(dir_YoonGuu_models, 'brain_2d_mseonly_200622.h5')
# vxm_model.load_weights(dir_weight_val)
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




