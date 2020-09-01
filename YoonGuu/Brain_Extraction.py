import nibabel as nb
from deepbrain import Extractor
import numpy as np
import matplotlib.pyplot as plt

# Load a nifti as 3d numpy image [H, W, D]

img_path = 'F:\\MPI dataset\\Duc_dataset\\brain_extraction_MRA_dim_same
img = nib.load(img_path).get_fdata()

ext = Extractor()

# `prob` will be a 3d numpy image containing probability
# of being brain tissue for each of the voxels in `img`
prob = ext.run(img)

# mask can be obtained as:
mask = prob > 0.5

npy_file_train = 'F:\\Datasets\\CTdata\\img_train11860.npy'
npy_file_label = 'F:\\Datasets\\CTdata\\img_label11860.npy'


npy_file_train = 'F:\\Datasets\\CTBlood vessels data\\img_train11860.npy'
npy_file_label = 'F:\\Datasets\\CTBlood vessels data\\img_label11860.npy'

np_train=np.load(npy_file_train)
np_label=np.load(npy_file_label)

print('np_label.shape : ', np_label.shape)
print('np_train.shape : ', np_train.shape)
np_train05 = np_train[0:5]
np_label05 = np_label[0:5]
print('np_train05.shape : ', np_train05.shape)
print('np_label05.shape : ', np_label05.shape)
neuron.plot.slices(np_train05, cmaps=['gray'], do_colorbars=True)
neuron.plot.slices(np_label05, cmaps=['gray'], do_colorbars=True)

plt.imshow(np_train[0])
plt.show()




# def slices(slices_in,           # the 2D slices
#            titles=None,         # list of titles
#            cmaps=None,          # list of colormaps
#            norms=None,          # list of normalizations
#            do_colorbars=False,  # option to show colorbars on each slice
#            grid=False,          # option to plot the images in a grid or a single row
#            width=15,            # width in in
#            show=True,           # option to actually show the plot (plt.show())
#            imshow_args=None):
#     '''
#     plot a grid of slices (2d images)
#     '''
#
#     # input processing
#     nb_plots = len(slices_in)
#     for si, slice_in in enumerate(slices_in):
#         assert len(slice_in.shape) == 2, 'each slice has to be 2d: 2d channels'
#         slices_in[si] = slice_in.astype('float')