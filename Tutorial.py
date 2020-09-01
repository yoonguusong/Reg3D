import os, sys

import numpy as np
import keras.layers

# local imports.
'''#maybe differenct cuz it is not running in console'''

dir_VoxelMorph = os.path.join(os.getcwd(), 'VoxelMorph')
dir_voxelmorph = os.path.join(dir_VoxelMorph, 'voxelmorph') #this dir_voxelmorph is diff from dir_VoxelMorph
print('current loc : ', os.getcwd())
print('current paths : ', sys.path)

dir1 =os.path.join(os.getcwd(), 'voxelmorph','ext','pynd-lib')
sys.path.append(dir1)
print('dir1 : ', dir1)

dir2 = os.path.join(os.getcwd(), 'voxelmorph','ext','pytools-lib')
sys.path.append(dir2)
print('dir2 : ', dir2)

dir3 = os.path.join(os.getcwd(), 'voxelmorph','ext','neuron')
sys.path.append(dir3)
print('dir3 : ', dir3)

dir4 = os.path.join(os.getcwd(), 'voxelmorph')
sys.path.append(dir4)
print('dir4 : ', dir4)
sys.path.append(os.path.join(dir_VoxelMorph, 'YoonGuu'))
sys.path.append(os.path.join(dir_VoxelMorph))
print('after add paths : ', sys.path)
import voxelmorph as vxm
import neuron

# import
# You should most often have this import together with all other imports at the top,
# but we include here here explicitly to show where data comes from
from keras.datasets import mnist

# load the data.
# `mnist.load_data()` already splits our data into train and test.
# (x_train_load, y_train_load), (x_test_load, y_test_load) = mnist.load_data()

# unfortunately the above seems to fail on the keras kernel
# so we will load it from a pre-downloaded mnist numpy file
mnist_file = 'mnist.npz'
x_train_load = np.load(mnist_file)['x_train']
y_train_load = np.load(mnist_file)['y_train']
x_test_load = np.load(mnist_file)['x_test']
y_test_load = np.load(mnist_file)['y_test']

# extract only instances of the digit 5
x_train = x_train_load[y_train_load==5, ...]
y_train = y_train_load[y_train_load==5]
x_test = x_test_load[y_test_load==5, ...]
y_test = y_test_load[y_test_load==5]

# let's get some shapes to understand what we loaded.
print('shape of x_train: ', x_train.shape)
print('shape of y_train: ', y_train.shape)



nb_val = 1000 # keep 10,000 subjects for validation
x_val = x_train[-nb_val:, ...]  # this indexing means "the last nb_val entries" of the zeroth axis
y_val = y_train[-nb_val:]
x_train = x_train[:-nb_val, ...]
y_train = y_train[:-nb_val]

#visualize data

nb_vis = 5
# choose nb_vis sample indexes
idx = np.random.choice(x_train.shape[0], nb_vis, replace=False)
example_digits = [f for f in x_train[idx, ...]]

# plot
neuron.plot.slices(example_digits, cmaps=['gray'], do_colorbars=True);

# fix data
x_train = x_train.astype('float')/255
x_val = x_val.astype('float')/255
x_test = x_test.astype('float')/255

# verify
print('training maximum value', x_train.max())

# re-visualize
example_digits = [f for f in x_train[idx, ...]]
neuron.plot.slices(example_digits, cmaps=['gray'], do_colorbars=True);

pad_amount = ((0, 0), (2,2), (2,2))

# fix data
x_train = np.pad(x_test, pad_amount, 'constant')
x_val = np.pad(x_test, pad_amount, 'constant')
x_test = np.pad(x_test, pad_amount, 'constant')

# verify
print('shape of training data', x_train.shape)

ndims = 2
vol_shape = x_train.shape[1:]
nb_enc_features = [32, 32, 32, 32]
nb_dec_features = [32, 32, 32, 32, 32, 16]

# first, let's get a unet (before the final layer)
#unet = vxm.src.networks.unet_core(vol_shape, nb_enc_features, nb_dec_features);
from VoxelMorph.voxelmorph.src import networks
unet = vxm.src.networks.unet_core(vol_shape, nb_enc_features, nb_dec_features);