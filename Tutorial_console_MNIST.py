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

#real imports
from VoxelMorph.voxelmorph.src import losses
from VoxelMorph.voxelmorph.src import networks

#import VoxelMorph.voxelmorph.src as vxm
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
mnist_file = 'VoxelMorph\\mnist.npz'
x_train_load = np.load(mnist_file)['x_train']
x_train_load.shape #(60000, 28, 28)
type(x_train_load) #<class 'numpy.ndarray'>

y_train_load = np.load(mnist_file)['y_train']
x_test_load = np.load(mnist_file)['x_test']
y_test_load = np.load(mnist_file)['y_test']
x_test_load.shape #(10000, 28, 28)

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
len(example_digits)

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
print(x_train[idx, ...].shape)
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
print('type(vol_shape)', type(vol_shape), 'vol_shape', vol_shape)
unet = networks.unet_core(vol_shape, nb_enc_features, nb_dec_features);
# vol_size = vol_shape
# enc_nf=nb_enc_features
# dec_nf=nb_dec_features


#here  the codes
from VoxelMorph.voxelmorph import src as vxm
import VoxelMorph.voxelmorph.src as vxm

############################################################
# dir5 = os.path.join(dir_voxelmorph, 'src')
# sys.path.append(dir5)
# print('dir5 : ', dir5)
# from VoxelMorph.voxelmorph import src as vxm
# unet = vxm.networks.unet_core(vol_shape, nb_enc_features, nb_dec_features);
############################################################
# real imports
#from VoxelMorph.voxelmorph.src import losses
#from VoxelMorph.voxelmorph.src import networks
# unet = networks.unet_core(vol_shape, nb_enc_features, nb_dec_features);




# inputs
print('number of inputs', len(unet.inputs))
moving_input_tensor = unet.inputs[0]
fixed_input_tensor = unet.inputs[1]

# output
print('output:', unet.output)

# transform the results into a flow field.
disp_tensor = keras.layers.Conv2D(ndims, kernel_size=3, padding='same', name='disp')(unet.output)

# check
print('displacement tensor:', disp_tensor)

# a cool aspect of keras is that we can easily form new models via tensor pointers:
def_model = keras.models.Model(unet.inputs, disp_tensor)
# def_model will now *share layers* with the UNet -- if we change layer weights
# in the UNet, they change in the def_model

spatial_transformer = neuron.layers.SpatialTransformer(name='spatial_transformer')

# warp the image
moved_image_tensor = spatial_transformer([moving_input_tensor, disp_tensor])

inputs = [moving_input_tensor, fixed_input_tensor]
outputs = [moved_image_tensor, disp_tensor]
vxm_model = keras.models.Model(inputs, outputs)

# losses. Keras recognizes the string 'mse' as mean squared error, so we don't have to code it
losses = ['mse', losses.Grad('l2').loss]

# usually, we have to balance the two losses by a hyper-parameter.
lambda_param = 0.05
loss_weights = [1, lambda_param]

vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)


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
train_generator = vxm_data_generator(x_train)
#x_train.shape --> (892, 32, 32)
#train_generator --> <generator object vxm_data_generator at 0x00000229FD5C6048>
input_sample, output_sample = next(train_generator)
#input_sample[0].shape --> (32, 32, 32, 1)
#input_sample[1].shape --> (32, 32, 32, 1)
#output_sample[0].shape --> (32, 32, 32, 1)
#output_sample[1].shape --> (32, 32, 32, 2)

# visualize
slices_2d = [f[0,...,0] for f in input_sample + output_sample]
# slices_2d[0].shape --> (32, 32)
# slices_2d[1].shape --> (32, 32)
# slices_2d[2].shape --> (32, 32)
# slices_2d[3].shape --> (32, 32)

titles = ['input_moving', 'input_fixed', 'output_moved_ground_truth', 'zero']
neuron.plot.slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True);

nb_epochs = 10
steps_per_epoch = 100
#start training
hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2);

# as with other imports, this import should be at the top, or use notebook matplotlib magic
# we keep it here to be explicit why we need it
import matplotlib.pyplot as plt

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

# let's get some data
val_generator = vxm_data_generator(x_val, batch_size = 1)
val_input, _ = next(val_generator)
len(val_input)
val_input[1].shape
val_pred = vxm_model.predict(val_input)
len(val_pred)
val_pred[1].shape
# %timeit is a 'jupyter magic' that times the given line over several runs
#%timeit vxm_model.predict(val_input)

# visualize
slices_2d = [f[0,...,0] for f in val_input + val_pred]
titles = ['input_moving', 'input_fixed', 'predicted_moved', 'deformation_x']
neuron.plot.slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True);

neuron.plot.flow([val_pred[1].squeeze()], width=5);


# extract only instances of the digit 5
x_sevens = x_train_load[y_train_load==7, ...].astype('float')/255
x_sevens = np.pad(x_sevens, pad_amount, 'constant')

seven_generator = vxm_data_generator(x_sevens, batch_size=1)
seven_sample, _ = next(seven_generator)
seven_pred = vxm_model.predict(seven_sample)

# visualize
slices_2d = [f[0,...,0] for f in seven_sample + seven_pred]
titles = ['input_moving', 'input_fixed', 'predicted_moved', 'deformation_x']
neuron.plot.slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True);


factor = 5
val_pred = vxm_model.predict([f*factor for f in val_input])

# visualizeb
slices_2d = [f[0,...,0] for f in val_input + val_pred]
titles = ['input_moving', 'input_fixed', 'predicted_moved', 'deformation_x']
neuron.plot.slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True);


