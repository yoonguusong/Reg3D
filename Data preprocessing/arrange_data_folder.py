'''

arrange the DTI, MRA, T1, T2 data into one folder each

'''

import os
import pandas as pd
import sys
# from keras import models
# from keras import layers
import SimpleITK as sitk
from shutil import copyfile

# F:\MPI dataset\Duc_dataset\Healthy MRA MRI Database_data_kitware_com
print('current loc : ', os.getcwd())
dir_data = 'F:\MPI dataset\Duc_dataset\Healthy MRA MRI Database_data_kitware_com'

# determine which directory you want to put data
dir_dataset = os.path.abspath(os.path.join(dir_data, os.pardir))  # 'F:\\MPI dataset\\Duc_dataset'
dir_arrange = os.path.join(dir_dataset, 'Dataset_Arrange')  # 'F:\\MPI dataset\\Duc_dataset\\Dataset_Arrange'

if not os.path.isdir(dir_arrange):
    os.mkdir(dir_arrange)
    print('dataset arrange directory is made')
else:
    print('dataset arrange directory already exists')

# check whether directory exist and then make directory.
data_modality = ['DTI', 'MRA', 'T1-Flash', 'T1-MPRage', 'T2']
data_format = ['nii', 'mha', 'hdr_img']

for modality in data_modality:
    dir_modality = os.path.join(dir_arrange, modality)
    if not os.path.isdir(dir_modality):
        os.mkdir(dir_modality)
        print(modality, ' modality directory is made')

        for format in data_format:
            dir_format = os.path.join(dir_modality, format)
            if not os.path.isdir(dir_format):
                os.mkdir(dir_format)
                print('   ', format, ' format directory is made')
            else:
                print('   ', format, ' format already exists')

    else:
        print(modality, ' modality directory already exists')

'''
check the the number of dataset modality
like DTI, MRA, T1-Flash, T1_MPRage, T2, AuxillaryData
'''
# lists inside of dataset
list_data = os.listdir(dir_data)
# list_data_sort = sorted(list_data)
# list_data_sort = sorted(list_data, reverse=True)

len_dataset = 0
for i in list_data:
    if i[0:6] == 'Normal':
        dir_Normal = os.path.join(dir_data, i)
        # print(dir_Normal)

        # list of inside of Normal --> DTI, MRA, T1-Flash, T2
        list_Normal_in = os.listdir(dir_Normal)

        ## to check max number of dataset modality
        # if len(list_Normal_in)>= len_dataset:
        #     len_dataset = len(list_Normal_in)

        ## to check when subject's dataset modality is 6--> print subject's number
        if len(list_Normal_in) == 6:
            print('dir_Normal', dir_Normal)

print('max len_data : ', len_dataset)
# max is 6 for num of directory


'''
check the wrong modality directory names
'''
num_format = 0
wrong_modal_name = set()
for i in list_data:
    if i[0:6] == 'Normal':
        dir_Normal = os.path.join(dir_data,i)
        # 'F:\\MPI dataset\\Duc_dataset\\Healthy MRA MRI Database_data_kitware_com\\Normal-109'

        # print(dir_Normal)
        # list of inside of Normal --> DTI, MRA, T1-Flash, T2
        list_Normal_in = os.listdir(dir_Normal)  # ['DTI', 'MRA', 'T1-MPRage', 'T2']

        # data_modality = ['DTI', 'MRA', 'T1_Flash', 'T1_MPRage', 'T2']

        non_coexist_list = [j for j in list_Normal_in if j not in data_modality]

        print(dir_Normal)
        print(non_coexist_list)
        wrong_modal_name.update(non_coexist_list)
print('wrong modal name : ', wrong_modal_name)

'''
real runing code
'''
num_Normal = 0
num_format = 0
for subject in list_data:
    if subject[0:6] == 'Normal':
        dir_Normal = os.path.join(dir_data, subject)
        # 'F:\\MPI dataset\\Duc_dataset\\Healthy MRA MRI Database_data_kitware_com\\Normal-109'

        # print(dir_Normal)
        num_Normal += 1

        # list of inside of Normal --> DTI, MRA, T1-Flash, T2
        list_Normal_in = os.listdir(dir_Normal)  # ['DTI', 'MRA', 'T1-MPRage', 'T2']

        # data_modality = ['DTI', 'MRA', 'T1_Flash', 'T1_MPRage', 'T2']
        coexist_list = [j for j in list_Normal_in if j in data_modality]
        print(dir_Normal)
        #print(coexist_list)

        for modality in coexist_list:  # ['DTI', 'MRA', 'T1-MPRage', 'T2', 'AuxillaryData']
            print('   ', modality)
            dir_Normal_in = os.path.join(dir_Normal, modality)
            # print(dir_Normal_in) #'F:\\MPI dataset\\Duc_dataset\\Healthy MRA MRI Database_data_kitware_com\\Normal-109\\T2'

            # sort should be reversed because nii file last order compare to mha
            list_MRA = os.listdir(dir_Normal_in)
            # ['Normal109-T2.hdr', 'Normal109-T2.img', 'Normal109-T2.mha', 'Normal109-T2.nii']

            for format in list_MRA:
                print('      ', format)
                # dir_dataset = os.path.abspath(os.path.join(dir_data, os.pardir))
                # 'F:\\MPI dataset\\Duc_dataset'

                # dir_arrange = os.path.join(dir_dataset,'Dataset_Arrange')
                # 'F:\\MPI dataset\\Duc_dataset\\Dataset_Arrange'

                dir_src = os.path.join(dir_Normal_in,format)
                # 'F:\\MPI dataset\\Duc_dataset\\Healthy MRA MRI Database_data_kitware_com\\Normal-109\\T2\\Normal109-T2.nii'

                dir_dst = os.path.join(dir_arrange, modality, format[-3:],format)
                # 'F:\\MPI dataset\\Duc_dataset\\Dataset_Arrange\\T2\\nii\\Normal109-T2.nii'

                if not os.path.isfile(dir_dst):
                    if format[-4:] == '.nii':
                        copyfile(dir_src, dir_dst)
                        print(format, ' file copied to ', modality, format[-3:])
                    elif format[-4:] == '.mha':
                        copyfile(dir_src, dir_dst)
                        print(format, ' file copied to ', modality, format[-3:])
                    elif format[-4:] == '.hdr' or format[-4:] == '.img':
                        dir_dst = os.path.join(dir_arrange, modality, 'hdr_img', format)
                        if not os.path.isfile(dir_dst):
                            copyfile(dir_src, dir_dst)
                            print(format, ' file copied to ', modality, 'hdr_img')

            print()
        print()
    print()
    num_Normal += 1
print('Number of Normal data ', num_Normal)

