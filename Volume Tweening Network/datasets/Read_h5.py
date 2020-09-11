import h5py
import numpy as np
filename1 = 'F:\\Dataset\\Brain\\VTN_dataset\\brain_train\\abide.h5'
filename2 = 'F:\\Dataset\\Brain\\VTN_dataset\\brain_train\\abidef.h5'
hf1 = h5py.File(filename1, 'r')
hf1_list = list(hf1.keys())
hf1_data = hf1['UM-30256']
#hf1_data = hf1['UCLA_1-29753']
#hf1_data.shape #AttributeError: 'Group' object has no attribute 'shape'

hf2 = h5py.File(filename2, 'r')
hf2_list = list(hf2.keys())
#hf2_data = hf1['Yale-50628']
#hf2_data.shape #AttributeError: 'Group' object has no attribute 'shape'

#read_file_name
hf2 = h5py.File(filename2, 'r')
for key in hf2.keys():
    print(key)



def traverse_datasets(hdf_file):

    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            print('path', path)
            print('item', item)
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    for path, _ in h5py_dataset_iterator(hdf_file):
        print('path', path)
        yield path

dic_dim_list = {} #dic_MRA_shape
dic_dim_num = {}
set_dim = set() #MRA_shape_set

with h5py.File(filename1, 'r') as f:
    print('f: ', f)
    for dset in traverse_datasets(f):
        # print('Path:', dset)
        # print('Data type:', f[dset].dtype)
        print('Shape:', f[dset].shape)
        set_dim.add(f[dset].shape)
    for shape in set_dim:
        dic_dim_list[shape] = []
        dic_dim_num[shape]=0

    for dset in traverse_datasets(f):
        dic_dim_list[f[dset].shape].append(dset)
        dic_dim_num[f[dset].shape]+=1

'''
f1 = hf1.get('30256')
f1 = np.array(f1)
print(f1.shape)

f2 = hf2.get('dataset_1')
f2 = np.array(f2)
print(f2.shape)
'''
