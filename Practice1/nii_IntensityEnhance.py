import os
import nibabel as nib
import SimpleITK as sitk
from skimage import measure, filters, morphology
import numpy as np
import concurrent.futures
import tqdm
import skimage.io
import skimage.exposure
import scipy
from skimage import measure, filters, morphology
import matplotlib.pyplot as plt


dir_rat= 'F:\\Dataset\\Rat\\Kim hyoungihl lab\\rat data_kim hyung hil\\Rat_upet2nii'
list_dir_rat =os.listdir(dir_rat)
dir_rat_folder = os.path.join(dir_rat,'140811_NV_Opto_140708-1')
list_dir_rat_fol= os.listdir(dir_rat_folder)
dir_rat_CT = os.path.join(dir_rat_folder,'140811_byte.nii')
dir_rat_PET = os.path.join(dir_rat_folder,'static_byte.nii')

package = ['simpleITK', 'nibabel']
sel_package = 'simpleITK'

comment_ONOFF = ['On', 'Off']
sel_comment = 'Off'

if sel_package== 'simpleITK':
    print('sel_package : ', sel_package)


    # case 1
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(dir_rat_CT)
    image = reader.Execute()

    #case 2 --> X
    image = sitk.ReadImage(dir_rat_CT, imageIO="BMPImageIO") #TypeError: ReadImage() got an unexpected keyword argument 'imageIO'

    selected_image = image
    print('Before modification:')
    print('origin: ' + str(selected_image.GetOrigin()))
    print('size: ' + str(selected_image.GetSize()))
    print('spacing: ' + str(selected_image.GetSpacing()))
    print('direction: ' + str(selected_image.GetDirection()))
    print('pixel type: ' + str(selected_image.GetPixelIDTypeAsString()))
    print('number of pixel components: ' + str(selected_image.GetNumberOfComponentsPerPixel()))

    selected_image.SetOrigin((78.0, 76.0, 77.0))
    selected_image.SetSpacing([0.5, 0.5, 3.0])

    print('\nAfter modification:')
    print('origin: ' + str(selected_image.GetOrigin()))
    print('spacing: ' + str(selected_image.GetSpacing()))


    rat_CT = sitk.ImageFileReader()
    rat_CT.SetImageIO("NiftiImageIO")
    rat_CT.SetFileName(dir_rat_CT)
    if sel_comment=='Off' or 'OFF':
        pass
    elif sel_comment=='On' or 'ON':
        rat_CT.MetaDataDictionaryArrayUpdateOn() #AttributeError: 'ImageFileReader' object has no attribute 'MetaDataDictionaryArrayUpdateOn'
        rat_CT.LoadPrivateTagsOn()
        rat_CT_array = sitk.GetArrayFromImage(rat_CT) #AttributeError: 'ImageFileReader' object has no attribute 'GetNumberOfComponentsPerPixel'
        rat_CT.ReadImageInformation()

    image = sitk.ReadImage(dir_rat_CT, imageIO="NiftiImageIO")




    # img_rat_CT = rat_CT.Execute();
    # print('img_rat_CT size: {0}\nimg_rat_CT spacing: {1}'.format(rat_CT.GetSize(), rat_CT.GetSpacing()))
    # print(rat_CT.GetSize())
    # print(rat_CT.GetOrigin())
    # print(rat_CT.GetSpacing())
    # print(rat_CT.GetDirection())
    # # print(rat_CT.GetNumberOfComponentsPerPixel())
    # # print(rat_CT.GetWidth())
    # # print(rat_CT.GetHeight())
    # # print(rat_CT.GetDepth())
    # print(rat_CT.GetDimension())
    # print(rat_CT.GetPixelIDValue())
    # # print(rat_CT.GetPixelIDTypeAsString()) AttributeError: 'ImageFileReader' object has no attribute 'GetNumberOfComponentsPerPixel'




    rat_PET = sitk.ImageFileReader()
    rat_PET.SetImageIO("NiftiImageIO")
    rat_PET.SetFileName(dir_rat_PET)
    image_array = sitk.GetArrayFromImage(rat_PET)
    if sel_comment=='Off' or 'OFF':
        pass
    elif sel_comment=='On' or 'ON':
        rat_PET.MetaDataDictionaryArrayUpdateOn()
        rat_PET.LoadPrivateTagsOn()
        rat_PET_array = sitk.GetArrayFromImage(rat_PET)


    img_rat_PET = rat_PET.Execute();
    print('img_rat_PET size: {0}\nimg_rat_PET spacing: {1}'.format(rat_PET.GetSize(), rat_PET.GetSpacing()))


elif sel_package== 'nibabel':
    print('sel_package : ', sel_package)
    img_rat_CT = nib.load(dir_rat_CT)  # pixdim[1,2,3] should be non-zero; setting 0 dims to 1
    img_rat_CT.shape  # (384, 584, 384, 1)
    np_CT = img_rat_CT.get_data() #(384, 584, 384, 1)
    np_CT_arr4 = np.asarray(np_CT) #(384, 584, 384, 1)
    np_CT_arr3 = np_CT_arr4.reshape((384, 584, 384)) #(384, 584, 384)
    np.max(np_CT_arr3) #31996.009872436523
    np.min(np_CT_arr3) #-30454.033493041992


    plt.figure(1)
    plt.imshow(np_CT_arr3[0, :, :])
    print(np_CT_arr3.shape)

    img_transpose = np.transpose(np_CT_arr3, (2, 1, 0))
    plt.figure(2)
    plt.imshow(img_transpose[:, :, 0])
    print(img_transpose.shape)

    img_reversed = img_transpose[::-1, :, ::-1]
    plt.figure(3)
    plt.imshow(img_reversed[:, :, -1])
    print(img_reversed.shape)



    img_rat_PET = nib.load(dir_rat_PET)
    img_rat_PET.shape  # (256, 159, 256, 1)
    img_data = img_rat_PET.get_data()
    img_data_arr = np.asarray(img_data)

ths = [(80, 140), (110, 160), (70, 90), (60, 80), (50, 70), (40, 60), (30, 50), (20, 40), (10, 30), (140, 180), (160, 200)]
vol = filters.gaussian(img_reversed, sigma = 2, preserve_range = True)
plt.figure(4)
plt.imshow(vol[:, :,-1 ])
print(vol.shape)

print('vol.shape : ', vol.shape) #(256, 256, 20)
mask = np.zeros_like(vol, dtype = np.bool)
print('mask.shape : ', mask.shape) #(256, 256, 20)
bw_copy = np.ones_like(vol, dtype = np.bool)
print('bw_copy.shape : ', bw_copy.shape) #(256, 256, 20)
print('np.sum(bw_copy) : ', np.sum(bw_copy)) #256*256*20 = 1,310,720

max_area = 0
for th_lo, th_hi in ths:
    print(th_lo, th_hi)
    bw = np.ones_like(vol, dtype = np.bool)
    bw[vol < th_lo] = 0
    bw[vol > th_hi] = 0
    if np.sum(bw) <= max_area:
        continue
    with concurrent.futures.ProcessPoolExecutor(8) as executor:
        jobs = list(range(bw.shape[-1]))
        args1 = [bw[:, :, z] for z in jobs]
        # args1[0].shape --> (256,256)
        args2 = [morphology.disk(35) for z in jobs]

        # args2[0].shape -->(71,71)
        # '''
        # this is for checking
        # '''
        # # a= list(map(filters.median, args1, args2))
        # for idx, ret in zip(jobs, map(filters.median, args1, args2)):
        #     print('idx, ret', idx, ret)
        #     a = filters.median(args1, args2)
        #     print('a.shape', a.shape)
        #
        #
        #     fig = plt.figure(idx)
        #     ax1 = fig.add_subplot(2,1,1)
        #     ax1.imshow(a[1])
        #     ax2 = fig.add_subplot(2,1,2)
        #     ax2.imshow(args1[1])
        #
        #
        #     bw[:, :, jobs[idx]] = ret
        #
        #     bw.shape #(256, 256, 20)
        #     plt.imshow(bw[:, :, -1])

        for idx, ret in tqdm.tqdm(zip(jobs, executor.map(filters.median, args1, args2)), total = len(jobs)):
            bw[:, :, jobs[idx]] = ret
    # for z in range(bw.shape[-1]):
    #     bw[:, :, z] = filters.median(bw[:, :, z], morphology.disk(35))
    if np.sum(bw) <= max_area:
        continue
    labeled_seg = measure.label(bw, connectivity=1)
    regions = measure.regionprops(labeled_seg)
    max_region = max(regions, key = lambda x: x.area)
    if max_region.area <= max_area:
        continue
    max_area = max_region.area
    mask = labeled_seg == max_region.label
assert max_area > 0, 'Failed to find the liver area!'
#return mask
#liver_mask = mask

'''
preprocess_dcm result

'returned mask is img' 
'reader is returned'
'''
# img = crop_mask(img, liver_mask) -->img = crop_mask(img, mask)
# def crop_mask(volume, segmentation, target_shape=(128, 128, 128))
target_shape=(128, 128, 128)
indices = np.array(np.nonzero(mask))
bound_r = np.max(indices, axis=-1)
bound_l = np.min(indices, axis=-1)
box_size = bound_r - bound_l + 1
padding = np.maximum( (box_size * 0.1).astype(np.int32), 5)
bound_l = np.maximum(bound_l - padding, 0)
bound_r = np.minimum(bound_r + padding + 1, mask.shape)











# img = img_reversed
# liver_mask = auto_liver_mask(img)
# def auto_liver_mask(vol, ths = [(80, 140), (110, 160), (70, 90), (60, 80), (50, 70), (40, 60), (30, 50), (20, 40), (10, 30), (140, 180), (160, 200)])
ths = [(80, 140), (110, 160), (70, 90), (60, 80), (50, 70), (40, 60), (30, 50), (20, 40), (10, 30), (140, 180), (160, 200)]
vol = filters.gaussian(img_reversed, sigma = 2, preserve_range = True)
plt.figure(4)
plt.imshow(vol[:, :,-1 ])
print(vol.shape)
# plt.figure(1)
# plt.imshow(img_reversed[:, :,-1 ])
# plt.figure(2)
# plt.imshow(vol[:, :,-1 ])

# from skimage.data import astronaut
# image = astronaut()
# plt.figure(1)
# plt.imshow(image)
#
# filtered_img = filters.gaussian(image, sigma=1, multichannel=True)
# plt.figure(2)
# plt.imshow(filtered_img)
print('vol.shape : ', vol.shape) #(256, 256, 20)
mask = np.zeros_like(vol, dtype = np.bool)
print('mask.shape : ', mask.shape) #(256, 256, 20)
bw_copy = np.ones_like(vol, dtype = np.bool)
print('bw_copy.shape : ', bw_copy.shape) #(256, 256, 20)
print('np.sum(bw_copy) : ', np.sum(bw_copy)) #256*256*20 = 1,310,720

max_area = 0
for th_lo, th_hi in ths:
    print(th_lo, th_hi)
    bw = np.ones_like(vol, dtype = np.bool)
    bw[vol < th_lo] = 0
    bw[vol > th_hi] = 0
    if np.sum(bw) <= max_area:
        continue
    with concurrent.futures.ProcessPoolExecutor(8) as executor:
        jobs = list(range(bw.shape[-1]))
        args1 = [bw[:, :, z] for z in jobs]
        args2 = [morphology.disk(35) for z in jobs]

        # a= list(map(filters.median, args1, args2))
        for idx, ret in tqdm.tqdm(zip(jobs, executor.map(filters.median, args1, args2)), total = len(jobs)):
            bw[:, :, jobs[idx]] = ret
    # for z in range(bw.shape[-1]):
    #     bw[:, :, z] = filters.median(bw[:, :, z], morphology.disk(35))
    if np.sum(bw) <= max_area:
        continue
    labeled_seg = measure.label(bw, connectivity=1)
    regions = measure.regionprops(labeled_seg)
    max_region = max(regions, key = lambda x: x.area)
    if max_region.area <= max_area:
        continue
    max_area = max_region.area
    mask = labeled_seg == max_region.label
assert max_area > 0, 'Failed to find the liver area!'
#return mask
#liver_mask = mask

'''
preprocess_dcm result

'returned mask is img' 
'reader is returned'
'''
# img = crop_mask(img, liver_mask) -->img = crop_mask(img, mask)
# def crop_mask(volume, segmentation, target_shape=(128, 128, 128))
target_shape=(128, 128, 128)
indices = np.array(np.nonzero(mask))
bound_r = np.max(indices, axis=-1)
bound_l = np.min(indices, axis=-1)
box_size = bound_r - bound_l + 1
padding = np.maximum( (box_size * 0.1).astype(np.int32), 5)
bound_l = np.maximum(bound_l - padding, 0)
bound_r = np.minimum(bound_r + padding + 1, mask.shape)

def wl_normalization(img, w=290, l=120):
    img = skimage.exposure.rescale_intensity(img, in_range=(l - w / 2, l + w / 2), out_range=(0, 255))
    return img.astype(np.uint8)

def crop(arr, bound_l, bound_r, target_shape, order=1):
    cropped = arr[bound_l[0]: bound_r[0], bound_l[1]: bound_r[1], bound_l[2]: bound_r[2]]
    return scipy.ndimage.zoom(cropped, np.array(target_shape) / np.array(cropped.shape), order = order)

preprocessed_img = wl_normalization(crop(img_reversed, bound_l, bound_r, target_shape)).astype(np.uint8)

#return wl_normalization(crop(volume, bound_l, bound_r, target_shape)).astype(np.uint8)


# show_image(img_fixed, os.path.join(args.output, 'fixed.png'))
# save_dcm(img_fixed, reader_fixed, os.path.join(args.output, 'fixed'))
