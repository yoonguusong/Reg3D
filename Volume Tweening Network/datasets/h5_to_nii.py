import h5py
import nibabel as nib
import os
import numpy as np
import h5py

filename_abide = 'F:\\Dataset\\Brain\\VTN_dataset\\brain_train\\abide.h5'
filename_abidef = 'F:\\Dataset\\Brain\\VTN_dataset\\brain_train\\abidef.h5'
filename_adhd = 'F:\\Dataset\\Brain\\VTN_dataset\\brain_train\\adhd.h5'
filename_adni = 'F:\\Dataset\\Brain\\VTN_dataset\\brain_train\\adni.h5'

save_dir_ABIDE = 'F:\\Dataset\\Brain\\ABIDE\\Volume Tweening network'
save_dir_ADHD ='F:\\Dataset\\Brain\\ADHD\\Volume Tweening network'
save_dir_ADNI = 'F:\\Dataset\\Brain\\ADNI\\Volume Tweening Network'

#set file_directory & save file directory
filename = filename_adni
save_directory =save_dir_ADNI

def traverse_datasets(hdf_file):

    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            # print('path', path)
            # print('item', item)
            
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    for path, _ in h5py_dataset_iterator(hdf_file):
        # print('path', path)
        yield path

dic_dim_list = {} #dic_MRA_shape
dic_dim_num = {}
set_dim = set() #MRA_shape_set

with h5py.File(filename, 'r') as f:
    print('f: ', f)
    for dset in traverse_datasets(f):
        print('Path:', dset)
        # print('Data type:', f[dset].dtype)
        print('Shape:' , f[dset].shape)

        # data = h5py.File(filename1, 'r')

        volume_ex1 = f.get(f[dset].name).value  # `data` is now an ndarray.
        data_name = f[dset].name
        print('name : ', f[dset].name)
        data_replace = data_name.replace('/', '_')

        new_image = nib.Nifti1Image(volume_ex1, affine=None)
        new_image.header.get_xyzt_units()
        # new_image_name = img[:name_num] + '_AX' + img[name_num:]
        new_image.to_filename(os.path.join(save_directory, data_replace))


        set_dim.add(f[dset].shape)
    for shape in set_dim:
        dic_dim_list[shape] = []
        dic_dim_num[shape]=0

    for dset in traverse_datasets(f):
        dic_dim_list[f[dset].shape].append(dset)
        dic_dim_num[f[dset].shape]+=1

print('dic_dim_list :', dic_dim_list)
print('dic_dim_num : ', dic_dim_num)
print('set_dim : ', set_dim)