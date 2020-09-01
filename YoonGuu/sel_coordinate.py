'''
medical img has 3 dimesions
axial, sagittal, coronal
in my data folder,
only nii.gz file exist
select specific dimension among (xy, yz, xz) & proper index value

use specific index, specific dimension images among diverse patients
'''
import random, os, sys

#import 3rd party
import os
import nibabel as nib
import numpy as np
from nilearn import plotting
from nilearn.image import resample_to_img
from skimage.transform import rescale, resize, downscale_local_mean
import cv2

#add paths
dir_VoxelMorph = os.path.join(os.getcwd(), 'VoxelMorph')
sys.path.append(os.path.join(dir_VoxelMorph, 'voxelmorph','ext','neuron'))

#local imports

import neuron
from keras.preprocessing.image import ImageDataGenerator


def select(img_shape, return_num, coord ='yz', percentage_left=20, percentage_right =20):
    '''
    select appropriate index values from 3D img for registration

    :param img_shape: img file's dimesions --> in my code, tuple, (182, 218, 182)
    :param coord: tuple value
    :param return_num: the number of return index
    :param percentage_left: default value is 20 which means how far the value from the center percentage
    :param percentage_right: default value is 20 which means how far the value from the center percentage
    :return: list of index which is the same amount of 'return_num'
    '''


    dic_coord = {'yz' : 0, 'xz': 1, 'xy': 2,'zy' : 0, 'zx': 1, 'yx': 2 }
    assert type(img_shape) is tuple, 'not tuple'

    dim_size = img_shape[dic_coord[coord]]
    dim_center = int(np.round(dim_size/2))
    dim_left = int(np.round(0.01 * percentage_left * dim_size))
    dim_right = int(np.round(0.01 * percentage_right * dim_size))
    dim_scale = range(dim_center-dim_left, dim_center+dim_right )
    if return_num =='max':
        return_num = len(dim_scale)

    return random.sample(dim_scale, return_num)

def img_dim(folder_dir):
    '''
    in my folder
    MRA, T2 data dimesion(shape) --> numbers
    MRA data dimesion(shape)   --> {(182, 218, 182): 109}
    T2  data dimesion(shape)   --> {(182, 218, 182): 107}

    :param folder_dir: folder directory, which only has nii.gz file
    :param orginal_dim: print original dimension
    :param changed_dim: print original dimension
    :return dic_dim_num --> key: file.shape (dimension), value : the number of files
    :return dic_dim_list --> key: file.shape (dimension), value : list of file names

    '''

    #folder_dir = 'F:\\MPI dataset\\Duc_dataset\\brain_extraction_MRA_CN'
    #folder_dir = 'F:\\MPI dataset\\Duc_dataset\\brain_extraction_T2_CN'
    #orginal_dim = True
    #changed_dim = True

    img_list = os.listdir(folder_dir)
    dic_dim_list = {} #dic_MRA_shape
    dic_dim_num = {}
    set_dim = set() #MRA_shape_set

    for img in img_list:
        dir_img = os.path.join(folder_dir, img)
        img_load = nib.load(dir_img)
        set_dim.add(img_load.shape)

    for shape in set_dim:
        dic_dim_list[shape] = []
        dic_dim_num[shape]=0

    for img in img_list:
        dir_img = os.path.join(folder_dir, img)
        img_load = nib.load(dir_img)
        dic_dim_list[img_load.shape].append(img)
        dic_dim_num[img_load.shape]+=1


    # max_value = max(dic_dim_num.values())
    # [max_key] = [k for k, v in dic_dim_num.items() if v==max_value]
    # max_dim_file = dic_dim[max_key][0]
    #
    # dir_max_img = os.path.join(folder_dir, max_dim_file)
    # desired_dim_file = nib.load(dir_max_img)
    # desired_dim_file.shape
    #
    # not_desired_dim_files = [j for i, j in dic_dim.items() if i != max_key]
    # for not_desired_dim_file in not_desired_dim_files:
    #     for not_file in not_desired_dim_file:
    #         resampled_stat_img = resample_to_img(not_file, img_load)

    return dic_dim_num, dic_dim_list





def img_preprocesing(dir_folder, num_file=5, coordinate = 'xy', visual_num_files = False, visual_3plot= False, visual_plot=False ):
    # dir_BE_CN_MRA = 'F:\\MPI dataset\\Duc_dataset\\brain_extraction_MRA_dim_same'
    # dir_folder = dir_BE_CN_MRA
    train_datagen= ImageDataGenerator(rescale=1./255)
    train_generator=train_datagen.flow_from_directory(dir_folder, target_size=(192,160), batch_size=20, class_mode='binary')
    for data_batch, labels_batch in train_generator:
        print('batch size data : ', data_batch.shape)
        print('batch size label : ', labels_batch.shape)
        break
    return

def dimension_ck(list):
    #list = MRA_xy_MM_array
    #list_MRA_yz
    if len(list)==1:
        [list_array] = list

    else:
        list_array = list[0]
    print('shape size:', list_array.shape)
    return list_array

def sel_coord(dir_folder, num_file=1, img_index_range=1 ,coordinate = 'xy', visual_num_files = False, visual_3plot= False, visual_plot=False ):
    '''
    image resampling to the same dimension


    :param dir_folder: folder directory, which only has nii.gz file
    :param num_file: the number of file that you want to randomly get
    :param img_index_range: if image size is (0,100), how many slice index will be obtained
    :param coordinate: coordinate system --> axial, sagittal, coronal
    you can select coordinate from ['yz', 'xz', 'xy']

    :param visual_num_files : visualization of "num_file".
    ex) num_file=5, coordinate = 'xy', 180th index among [0, 383] <-- using select())
        plot 5 images which is 'xy', 180th

    :param visual_3plot: visualization of xy, yz, zx plot in one pop up

    :param visual_plot : visualization of specific coordinate "coordinate" value hold
                        only "one" img come out.

    :return: list of img matrix list
    '''

    # for check the code, input variables
    # dir_folder = dir_BE_CN_MRA
    # visual_3plot= False
    # visual_plot = False
    # visual_1plot=True
    # coordinate = 'xy'
    # visual_num_files = False
    #
    # num_file = 'max'
    # num_file=5
    #
    # img_index_range = 'max'
    # img_index_range = 1




    list_files = os.listdir(dir_folder)
    if num_file == 'max':
        num_file = len(list_files)
    # print('num_file', num_file)

    random_file = random.sample(list_files, k=num_file)
    # random_file
    #image files dimension check
    #this returns a dictionary ={(x_dim, y_dim, z_dim) : the number of files}
    # dic_img_dim = img_dim_ck.img_dim(dir_folder)



    dic_img_dim, dic_file_name_list = img_dim(dir_folder)
    # assert len(dic_img_dim.keys()) == 1, "image files's dimensions are not identical "

    max_value = max(dic_img_dim.values())
    [max_dim_size] = [k for k, v in dic_img_dim.items() if v==max_value]

    coord_index = select(max_dim_size, coord=coordinate, return_num=img_index_range,  percentage_left=20, percentage_right =20)

    # breakone = 0
    # assert breakone == 1, "just break "


    #max index
    #len(coord_index)
    #coord_index = select(img_shape, coord=coordinate, return_num='max', percentage_left=20, percentage_right=20)

    #coord_index=select(img_shape, coord=coordinate, return_num=1, percentage_left=20, percentage_right=20)

    itr = 0
    for img in random_file:
        # print(img)
        rand_img = os.path.join(dir_folder, img)
        img_load = nib.load(rand_img)
        if img_load.shape != max_dim_size:
            print('before resample size :', img_load.shape)


            dir_desired_img = os.path.join(dir_folder, dic_file_name_list[max_dim_size][0])
            desired_dim_file = nib.load(dir_desired_img)
            img_load = resample_to_img(img_load, desired_dim_file)
            print('after resample size :', img_load.shape)




        #img_load.shape --> (182, 218, 182)
        #type(img_load) --> <class 'nibabel.nifti1.Nifti1Image'>

        img_array = np.array(img_load.dataobj)
        #type(img_array) --> <class 'numpy.ndarray'>
        #img_array.shape -->(182, 218, 182)
        #type(img_array.shape) --> <class 'tuple'>



        # visualization of xy, yz, zx dimension in one pop up
        if visual_3plot == True:
            plotting.plot_img(rand_img)


        if coordinate =='yz' or coordinate =='zy':
            img1 = img_array[coord_index, ...]  #  <class 'numpy.ndarray'> (1, 218, 182)
            # neuron.plot.slices(img1, cmaps=['gray'], do_colorbars=True, titles=['yz'])

        elif coordinate =='xz' or coordinate =='zx':

            MRA_xz = img_array[:, coord_index, :]  # (182, 1, 182)
            img1 = np.transpose(img_array[:, coord_index, :], (1, 2, 0))  # (1, 182, 182)
            img2 = np.transpose(img_array[:, coord_index, :], (1, 0, 2))  # (1, 182, 182)

            # print('MRA_0_xz_rotated.shape : ', img1.shape)
            # print('MRA_0_xz_flipped.shape : ', img2.shape)
            # neuron.plot.slices(img1, cmaps=['gray'], do_colorbars=True, titles='xz_rotated')
            # neuron.plot.slices(img2, cmaps=['gray'], do_colorbars=True, titles='xz_flipped')


        elif coordinate =='xy' or coordinate =='yx':
            MRA_xy = img_array[..., coord_index]  # (182, 218, 5)
            img1 = np.transpose(img_array[..., coord_index], (2, 1, 0))  # (5, 218, 182)
            img2 = np.transpose(img_array[..., coord_index], (2, 0, 1))  # (5, 182, 218)

            # print('MRA_0_xy_rotated.shape : ', img1.shape)
            # print('MRA_0_xy_flipped.shape : ', img2.shape)
            # neuron.plot.slices(img1, cmaps=['gray'], do_colorbars=True, titles='xy_rotated');
            # neuron.plot.slices(img2, cmaps=['gray'], do_colorbars=True, titles='xy_rotated');

        #img1 has 3D so have to strip off one bracket

        if itr==0:
            concat_np_array = img1
            # print('init_np_array shape :', concat_np_array.shape)
        else:
            concat_np_array = np.concatenate((concat_np_array, img1))
            # print('concat_np_array shape :', concat_np_array.shape)
        # img1_1 = img1
        # print('img1_1 : ', img1_1.shape)
        # print('img1 : ', img1.shape)
        # concat= np.concatenate((img1_1, img1))
        # print('concat : ', concat.shape)
        #
        # init = np.array([])
        # concat = np.vstack([img1,init])
        # print('concat : ', concat.shape)
        #
        # this makes 1 D
        # np_array = np.append(img1_1, img1)
        # print('np_array : ', np_array.shape)



        # [one_img] =img1
        # list_imgs.append(one_img)

        itr =+1




    ## visualization of one dimension's specific index
    if visual_plot == True :
        # it seems error--> i don't even know what it was_YG
        neuron.plot.slices(img_convert, cmaps=['gray'], do_colorbars=True)



    ###visualization of the number of files(num_file) in right position
    if visual_num_files ==True :
        neuron.plot.slices(concat_np_array, cmaps=['gray'], do_colorbars=True)

    return concat_np_array


def zeropadding_2Ds(np_array, dim_to_change):
    '''

    :param np_array: this will be 3D with (training data number, dim1, dim2)
    :param dim_to_change: (want to change dim1, want to change dim2)
    :return: changed numpy array
    '''

    # np_array= MRA_xy_MM_array
    # dim_to_change= correct_dim

    print('np_array shape : ', np_array.shape) #(7848, 218, 182)
    print('dim_to_change shape : ', dim_to_change) #(224, 192)
    # np_array.shape[0]
    # list_dim_to_change= list(dim_to_change)
    # want_d_list = []
    # want_d_list.append(np_array.shape[0])
    # want_d_list.append(list_dim_to_change[0])
    # want_d_list.append(list_dim_to_change[1])
    # want_d_tuple = tuple(want_d_list)
    #
    # padded_array = np.zeros(want_d_tuple)
    # print('padded_array shape : ', padded_array.shape)

    idx_1 = dim_to_change[0] - np_array.shape[1]
    idx_2 = dim_to_change[1] - np_array.shape[2]

    if idx_1%2 ==0:
        idx_1_pad = idx_1//2
        if idx_2 % 2 == 0:
            idx_2_pad = idx_2 // 2
            changed_array = np.pad(np_array, ((0, 0), (idx_1_pad, idx_1_pad), (idx_2_pad, idx_2_pad)), 'constant')
        else:
            idx_2_pad = idx_2 // 2
            changed_array = np.pad(np_array, ((0, 0), (idx_1_pad, idx_1_pad), (idx_2_pad, idx_2_pad+1)), 'constant')
    else:
        idx_1_pad = idx_1//2
        if idx_2 % 2 == 0:
            idx_2_pad = idx_2 // 2
            changed_array = np.pad(np_array, ((0, 0), (idx_1_pad, idx_1_pad+1), (idx_2_pad, idx_2_pad)), 'constant')
        else:
            idx_2_pad = idx_2 // 2
            changed_array = np.pad(np_array, ((0, 0), (idx_1_pad, idx_1_pad+1), (idx_2_pad, idx_2_pad+1)), 'constant')

    print('changed_array.shape : ', changed_array.shape)

    assert (changed_array.shape[1] == dim_to_change[0]) & (changed_array.shape[2] == dim_to_change[1]), 'Dimension is not matched'
    return changed_array



def train_val_test_divide(np_array, train_pct=0.8, test_pct=0.1, val_pct=0.1, test_val_combine=True):
    '''

    :param np_array: numpy array which have (1000,182, 254) sth like that
    :param train_pct: train data percentage
    :param test_pct: test data percentage
    :param val_pct: validation data percentage
    :param test_val_combine: decide whether test data and validation data combine
    :return: divided data_train, data_val, data_test will be returned
    '''
    # np_array=MRA_xy_MM_array
    print('np_array shape : ', np_array.shape)
    assert train_pct+test_pct+val_pct==1, 'sum of training, validation, test percentage is not 1'
    num_data = np_array.shape[0]
    num_train = round(num_data *train_pct) #6278
    num_val = round(num_data *val_pct) #785
    num_test = num_data - (num_train +num_val) #785

    assert num_data == (num_train +num_test +num_val), 'training, validation, test number is not matched'
    if test_val_combine ==True:
        num_val = num_val+num_test

    data_train, data_val, data_test = np_array[0:num_train], np_array[num_train: num_train+num_val], np_array[num_train+num_val:]

    # data_train= np_array[0:num_train]
    # data_train.shape
    # data_val=np_array[num_train: num_train+num_val]
    # data_val.shape
    # data_test=np_array[num_train + num_val:]
    # data_test.shape

    print('after divide, train : ', data_train.shape[0],  ' val : ', data_val.shape[0], ' test : ',data_test.shape[0], )
    print('sum = ', data_train.shape[0]+ data_test.shape[0] + data_val.shape[0])

    return data_train, data_val, data_test

def modal_resize(changing_data, based_data, interpol = cv2.INTER_CUBIC):
    '''
    change 2d image resize
    ex) 109 dataset, height =120, width=100
    ex) if you want height 100, width1=100
    ex) (109, 120,100) --> (109, 100, 100)
    :param based_data: datasize 3D size ,(number of index, width, height)
    :param changing_data: desired to be changed data
    :param interpol: cv2.resize's interpolation
    :return: changed numpy array which is desired to have desired dimension
    '''
    # based_data =changed_np_T2_train
    # print('based_data.shape :', based_data.shape) #(4, 448, 448)
    # changing_data =changed_np_MRA_train
    # print('changing_data.shape :', changing_data.shape) #(4, 256, 192)


    # based_data[0].shape #(256, 192)
    # changing_data[0].shape #(448, 448)

    assert len(based_data.shape)==3 and len(based_data.shape)==3, 'data size is not 3'

    img_stack_sm = np.zeros((len(changing_data), based_data[0].shape[0], based_data[0].shape[1]))
    # print('img_stack_sm.shape : ', img_stack_sm.shape) # (4, 256, 192)
    for idx in range(len(changing_data)):
        img = changing_data[idx, :, :]
        # print('img.shape : ', img.shape) #(448, 448)
        img_sm = cv2.resize(img, dsize=(based_data[0].shape[1], based_data[0].shape[0]), interpolation=cv2.INTER_CUBIC)
        print('img_sm.shape : ', img_sm.shape) #(192, 256)
        img_stack_sm[idx, :, :] = img_sm

    # img_sm_o = cv2.resize(img, dsize=(based_data[0].shape[1], based_data[0].shape[0]), interpolation=cv2.INTER_CUBIC)
    # img_sm_x = cv2.resize(img, dsize=(based_data[0].shape[0], based_data[0].shape[1]), interpolation=cv2.INTER_CUBIC)
    import matplotlib.pyplot as plt
    return img_stack_sm

def read_3D(dir_folder, return_dim=4, desired_dim=None):
    '''

    :param dir_folder: folder directory
    :param desired_dim: if don't write any desired dimension --> it will match all the data into the max number of dimension
                        if write any desired dimension --> match all the data into the same desired dimension
                        desired dimension can be a list[] or tuple()
    :param return_dim: the np array return dimension
    :return:
    '''
    # dir_folder = 'F:\\Dataset\\Brain\\Duc_dataset\\BE_MRA_CN'
    # desired_dim=[160, 192, 224]
    list_files = os.listdir(dir_folder)
    dic_img_dim, dic_file_name_list = img_dim(dir_folder)
    max_value = max(dic_img_dim.values())
    [max_dim_size] = [k for k, v in dic_img_dim.items() if v == max_value]



    itr = 0
    for img in list_files:
        # print(img)
        rand_img = os.path.join(dir_folder, img)
        img_load = nib.load(rand_img)
        if img_load.shape != max_dim_size:
            # print('before resample size :', img_load.shape)

            dir_desired_img = os.path.join(dir_folder, dic_file_name_list[max_dim_size][0])
            desired_dim_file = nib.load(dir_desired_img)  # (448,448,128)
            img_load = resample_to_img(img_load, desired_dim_file)
            # print('after resample size :', img_load.shape)
            img_array3D = np.array(img_load.dataobj)
            # print('img_array3D.shape :', img_array3D.shape) #(448, 448, 128)
            '''
            new_image = nib.Nifti1Image(img_array3D, affine=None)
            new_image.header.get_xyzt_units()
            new_image_name = img[:13] + '_AX' + img[13:]
            new_image.to_filename(os.path.join('C:\\Users\\YoonGuu Song\\Desktop', new_image_name))
            '''

        else:
            img_array3D = np.array(img_load.dataobj)
            # print('img_array3D.shape : ', img_array3D.shape)

        img_array4D = np.expand_dims(img_array3D, axis=0)
        # print('img_array4D.shape : ', img_array4D.shape)

        # img_array5D = np.expand_dims(img_array4D, axis=0)
        # print('img_array4D.shape : ', img_array5D.shape)

        if itr == 0:
            concat_np_array = img_array4D
            # print('init_np_array shape :', concat_np_array.shape)
        else:
            concat_np_array = np.concatenate((concat_np_array, img_array4D))
        itr = +1
        # print('concat_np_array.shape : ', concat_np_array.shape)


    return concat_np_array


def reshape_3D(dir_folder,desired_dim,to_dir_wo_Affine=None, to_dir_w_Affine=None, name_num=None ):
    '''
    if to_dir is given --> save the medical image file into desired dimension in the to_dir directory
    if to_dir is not given --> just return medical image files in dir_folder as numpy array with wanted desired_dim

    :param dir_folder: directory for medical image files
    :param to_dir_wo_Affine: directory to save the new changed dimension files without Affine
    :param to_dir_w_Affine: directory to save the new changed dimension files with Affine
    :param desired_dim: wanted medical image file dimension
    :param name_num: to change the file name, like Normal001-T2_AX.nii.gz, you should check where you want to edit name

    :return: (number of data, dimension1, dimension2, dimension3) np array
    '''
    # dir_folder = 'F:\\Dataset\\Brain\\Duc_dataset\\BE_MRA_CN'
    # desired_dim = (160,192,224)
    # to_dir_wo_Affine = 'C:\\Users\\YoonGuu Song\\Desktop\\MRA\\w_Affine'
    # to_dir_w_Affine = 'C:\\Users\\YoonGuu Song\\Desktop\\MRA\\wo_Affine'

    assert type(desired_dim) is tuple, 'desired_dim is not tuple'

    list_files = os.listdir(dir_folder)

    itr = 0
    for img in list_files:
        print(img)
        rand_img = os.path.join(dir_folder, img)
        img_load = nib.load(rand_img)
        img_array = np.array(img_load.dataobj)
        # print('img_array.shape', img_array.shape)


        # hdr = img_load.header
        # hdr.get_xyzt_units()
        # raw = hdr.structarr

        image_resized = resize(img_array, desired_dim, anti_aliasing=True)
        # print('image_resized.shape', image_resized.shape)

        if to_dir_wo_Affine != None:
            print('reshaping on ',img,' wo Affine')
            new_image = nib.Nifti1Image(image_resized, affine=None)
            new_image.header.get_xyzt_units()
            new_image_name = img[:name_num] + '_AX' + img[name_num:]
            new_image.to_filename(os.path.join(to_dir_wo_Affine, new_image_name))

        if to_dir_w_Affine != None:
            print('reshaping on ', img, ' w Affine')
            new_image_affine_none = nib.Nifti1Image(image_resized, affine=np.eye(4))
            new_image_affine_none.header.get_xyzt_units()
            new_image_AX_name = img[:name_num] + '_AO'+img[name_num:]
            new_image_affine_none.to_filename(os.path.join(to_dir_w_Affine, new_image_AX_name))



        img_array4D = np.expand_dims(image_resized, axis=0)
        if itr == 0:
            concat_np_array = img_array4D
            print('init_np_array shape :', concat_np_array.shape)
        else:
            concat_np_array = np.concatenate((concat_np_array, img_array4D))
            print('np_array shape :', concat_np_array.shape)
        itr = +1


    return concat_np_array