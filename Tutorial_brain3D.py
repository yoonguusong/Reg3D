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
dir_VoxelMorph = os.path.join(os.getcwd(), 'VoxelMorph')
dir_voxelmorph = os.path.join(dir_VoxelMorph, 'voxelmorph') #this dir_voxelmorph is diff from dir_VoxelMorph

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






LoadFileName ='brain_3d_200812.h5'
SaveFileName ='brain_3d_200812.h5'




## for 3D image

vol_shape = [160, 192, 224]
ndims = 3
nb_enc_features = [16, 32, 32, 32]
nb_dec_features = [32, 32, 32, 32, 32, 16, 16]
unet = networks.unet_core(vol_shape, nb_enc_features, nb_dec_features);
#unet.input --> [(?, 160, 192, 224, 1), (?, 160, 192, 224, 1)]
#unet.output --> (?, 160, 192, 224, 16)
disp_tensor = keras.layers.Conv3D(ndims, kernel_size=3, padding='same', name='disp')(unet.output)
#disp_tensor --> (?, 160, 192, 224, 3)



# spatial transfomer
spatial_transformer = neuron.layers.SpatialTransformer(name='image_warping')
moved_image_tensor = spatial_transformer([unet.inputs[0], disp_tensor])
#moved_image_tensor --> (?, 160, 192, 224, 1)


# final model
vxm_model = keras.models.Model(unet.inputs, [moved_image_tensor, disp_tensor])
# vxm_model.input --> [(?, 160, 192, 224, 1), (?, 160, 192, 224, 1)]
# vxm_model.output--> [(?, 160, 192, 224, 1), (?, 160, 192, 224, 3)]


num_MRA_dim,dic_MRA_list = sel_coordinate.img_dim(dir_BE_CN_MRA)
num_T2_dim,dic_T2_list =sel_coordinate.img_dim(dir_BE_CN_T2)
print('num_MRA_dim : ', num_MRA_dim)
print('num_T2_dim : ', num_T2_dim)
print('dic_MRA_list : ', dic_MRA_list)
print('dic_T2_list : ', dic_T2_list)


MRA_4D = sel_coordinate.read_3D(dir_BE_CN_MRA)
print('MRA_4D.shape : ', MRA_4D.shape) #(109, 448, 448, 128)

T2_4D = sel_coordinate.read_3D(dir_BE_CN_T2)
print('T2_4D.shape : ', T2_4D.shape) #(107, 192, 256, 128)
# val_volume_1 = MRA_4D
# val_volume_2 = T2_4D


'''
This line is for when you want to save the resized medical image files into specific directory
MRA_wo_Affine = 'C:\\Users\\YoonGuu Song\\Desktop\\MRA\\w_Affine'
MRA_w_Affine ='C:\\Users\\YoonGuu Song\\Desktop\\MRA\\wo_Affine'

T2_wo_Affine = 'C:\\Users\\YoonGuu Song\\Desktop\\T2\\w_Affine'
T2_w_Affine ='C:\\Users\\YoonGuu Song\\Desktop\\T2\\wo_Affine'
MRA_4D_desired = sel_coordinate.reshape_3D(dir_BE_CN_MRA, desired_dim, MRA_wo_Affine, MRA_w_Affine, name_num=13)
T2_4D_desired = sel_coordinate.reshape_3D( dir_BE_CN_T2, desired_dim,T2_wo_Affine, T2_w_Affine, name_num=12)
'''

desired_dim = (160,192,224)
MRA_4D_desired = sel_coordinate.reshape_3D(dir_BE_CN_MRA, desired_dim)
print(MRA_4D_desired.shape)
T2_4D_desired = sel_coordinate.reshape_3D( dir_BE_CN_T2, desired_dim)
print(T2_4D_desired.shape)

'''
# just for 3D loading
dir_BE_CN_MRA = 'F:\\Dataset\\Brain\\Duc_dataset\\BE_MRA_CN'
list_files = os.listdir(dir_BE_CN_MRA)

dir_img1 = os.path.join(dir_BE_CN_MRA, 'Normal068-MRA.nii.gz')
dir_img2 = os.path.join(dir_BE_CN_MRA, 'Normal001-MRA.nii.gz')
img_load1 = nib.load(dir_img1)
img_load2 = nib.load(dir_img2)

val_np_array1 = img_load1.get_fdata()
val_np_array1.shape
val_volume_1=val_np_array1
val_np_array2 = img_load2.get_fdata()
val_np_array2.shape
val_volume_2= val_np_array2
'''



# val_volume_1 = np.load('/kaggle/input/learn2reg-mri-3d/subject_1_vol.npz')['vol_data']
# seg_volume_1 = np.load('/kaggle/input/learn2reg-mri-3d/subject_1_seg.npz')['vol_data']
# val_volume_2 = np.load('/kaggle/input/learn2reg-mri-3d/atlas_norm_3d.npz')['vol']
# seg_volume_2 = np.load('/kaggle/input/learn2reg-mri-3d/atlas_norm_3d.npz')['seg']

val_volume_1= MRA_4D_desired
val_volume_2= T2_4D_desired
val_input = [val_volume_1[np.newaxis, ..., np.newaxis], val_volume_2[np.newaxis, ..., np.newaxis]]
print('val_input[0].shape', val_input[0].shape)
print('val_input[1].shape', val_input[1].shape)

model_2018 = 'E:\\Pycharm\\PycharmProjects\\MPI_projects\\VoxelMorph\\voxelmorph\\models\\cvpr2018_vm2_cc.h5'

# from keras.models import Sequential, load_model
# load_model1 =load_model(model_2018)
# load_model1.summary()


vxm_model.load_weights(model_2018)



#ValueError: Error when checking input: expected input_1 to have 5 dimensions, but got array with shape (1, 109, 448, 448, 128, 1)
#--> 4D

#ValueError: You are trying to load a weight file containing 12 layers into a model with 11 layers.
val_pred = vxm_model.predict(val_input);
moved_pred = val_pred[0].squeeze()
pred_warp = val_pred[1]
mid_slices_fixed = [np.take(val_volume_2, vol_shape[d]//2, axis=d) for d in range(ndims)]
mid_slices_fixed[1] = np.rot90(mid_slices_fixed[1], 1)
mid_slices_fixed[2] = np.rot90(mid_slices_fixed[2], -1)

mid_slices_pred = [np.take(moved_pred, vol_shape[d]//2, axis=d) for d in range(ndims)]
mid_slices_pred[1] = np.rot90(mid_slices_pred[1], 1)
mid_slices_pred[2] = np.rot90(mid_slices_pred[2], -1)
neuron.plot.slices(mid_slices_fixed + mid_slices_pred, cmaps=['gray'], do_colorbars=True, grid=[2,3]);
warp_model = networks.nn_trf(vol_shape)
warped_seg = warp_model.predict([seg_volume_1[np.newaxis,...,np.newaxis], pred_warp])
from pytools import plotting as pytools_plot
import matplotlib

[ccmap, scrambled_cmap] = pytools_plot.jitter(255, nargout=2)
scrambled_cmap[0, :] = np.array([0, 0, 0, 1])
ccmap = matplotlib.colors.ListedColormap(scrambled_cmap)
mid_slices_fixed = [np.take(seg_volume_1, vol_shape[d]//1.8, axis=d) for d in range(ndims)]
mid_slices_fixed[1] = np.rot90(mid_slices_fixed[1], 1)
mid_slices_fixed[2] = np.rot90(mid_slices_fixed[2], -1)

mid_slices_pred = [np.take(warped_seg.squeeze(), vol_shape[d]//1.8, axis=d) for d in range(ndims)]
mid_slices_pred[1] = np.rot90(mid_slices_pred[1], 1)
mid_slices_pred[2] = np.rot90(mid_slices_pred[2], -1)

slices = mid_slices_fixed + mid_slices_pred
for si, slc  in enumerate(slices):
    slices[si][0] = 255
neuron.plot.slices(slices, cmaps = [ccmap], grid=[2,3]);
