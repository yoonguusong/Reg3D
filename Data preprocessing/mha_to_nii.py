import os
import pandas as pd
import sys
from keras import models
from keras import layers
import SimpleITK as sitk


#F:\MPI dataset\Duc_dataset\Healthy MRA MRI Database_data_kitware_com
print('current loc : ', os.getcwd())
dir_data = 'F:\MPI dataset\Duc_dataset\Healthy MRA MRI Database_data_kitware_com'

#lists inside of dataset
list_data = os.listdir(dir_data)
# list_data_sort = sorted(list_data)
# list_data_sort = sorted(list_data, reverse=True)

num_Normal = 0
for i in list_data:
    if i[0:6] == 'Normal':

        dir_Normal = os.path.join(dir_data, i)
        print(dir_Normal)
        num_Normal += 1

        #list of inside of Normal --> DTI, MRA, T1-Flash, T2
        list_Normal_in = os.listdir(dir_Normal)

        for j in list_Normal_in:
            dir_Normal_in = os.path.join(dir_Normal, j)
            print(dir_Normal_in)
            #sort should be reversed because nii file last order compare to mha
            list_MRA = os.listdir(dir_Normal_in)
            # list_MRA_sort_rev = sorted(list_MRA, reverse=True)

            #check whether there is '.nii' file exist or not
            if not [k for k in list_MRA if k[-4:] == '.nii']: #nii file is not exist --> mha file convert
                for l in list_MRA :
                    if l[-4:] == '.mha':
                        nii_name = l[: -4] + '.nii'
                        mha_path = os.path.join(dir_Normal_in, l)
                        print('mha_path : ', mha_path)
                        Image = sitk.ReadImage(mha_path)
                        print(Image.GetPixelIDTypeAsString())
                        sitk.WriteImage(Image, os.path.join(dir_Normal_in, nii_name))


            # for k in list_MRA_sort_rev: #.mha file inside the subdirectory
            #     print('k - file name : ', k)
                # print(k[-6:])

                # if 'nii.gz' in k[-6:]:
                #     print('if loop')
                #     print(k)
                #     break
                # else:
                #     print('else statement')
                #     nii_path = k[: -4] + '.nii'
                #     mha_path = os.path.join(dir_Normal_in, k)
                #     print('mha_path : ', mha_path)
                #     Image = sitk.ReadImage(mha_path)
                #     print(Image.GetPixelIDTypeAsString())
                #
                #     sitk.WriteImage(Image, os.path.join(dir_Normal_in, nii_path))

            print()
        print()
    print()
    num_Normal += 1









print('Number of Normal data ', num_Normal)