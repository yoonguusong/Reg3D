import os
import pandas as pd
import sys
from keras import models
from keras import layers
import SimpleITK as sitk

#important --- medpy
from medpy.io import load, save


print('current loc : ', os.getcwd())
dir_data = 'F:\MPI dataset\Duc_dataset\Healthy MRA MRI Database_data_kitware_com'

#lists inside of dataset
list_data = os.listdir(dir_data)
num_Normal = 0
for i in list_data:
    if i[0:6] == 'Normal':
        dir_Normal = os.path.join(dir_data, i)
        print(dir_Normal)
        num_Normal += 1
        list_Normal_in = os.listdir(dir_Normal)
        for j in list_Normal_in:
            dir_Normal_in = os.path.join(dir_Normal, j)
            print('   ', dir_Normal_in)


            list_MRA = os.listdir(dir_Normal_in)
            # list_MRA_sort_rev = sorted(list_MRA, reverse=True)

            # check whether there is '.nii' file exist or not
            if [k for k in list_MRA if k[-4:] == '.nii']:  # nii file is not exist --> mha file convert
                for l in list_MRA:
                    if l[-4:] == '.nii':
                        print('---running making img file ---')
                        path_nii = os.path.join(dir_Normal_in, l)
                        filename_img = l[: -4] + '.img'
                        filename_hdr = l[: -4] + '.hdr'
                        path_img = os.path.join(dir_Normal_in, filename_img)
                        path_hdr = os.path.join(dir_Normal_in, filename_hdr)
                        image_data, image_header = load(path_nii)
                        save(image_data, path_img, image_header)



#
#
#
#
# dir_Normal = os.path.join(dir_data, 'Normal001-DTI.nii')
# file_dir = r'F:\MPI dataset\Duc_dataset\check\Normal001-DTI.nii'
# # list_data_sort = sorted(list_data)
# # list_data_sort = sorted(list_data, reverse=True)
#
#
#
# image_data, image_header = load(file_dir)
# save(image_data, r'F:\MPI dataset\Duc_dataset\check\Normal001-DTI.img', image_header)

