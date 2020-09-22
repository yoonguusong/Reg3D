import os
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

dir_h5 ='E:\\Pycharm\\PycharmProjects\\Reg3D\\Volume Tweening Network\\datasets\\adni.h5'
dir_nii = 'F:\\Dataset\\Brain\\Duc_dataset\\BE_T2_CN\\Normal001-T2.nii.gz'
dir_TCIA_dcm = 'F:\\Dataset\\Brain\\Cancer imaging archive(TCIA)\\REMBRANDT\\900-00-1961\\06-21-2005-11987\\5.000000-12296\\1-01.dcm'
dir_TCIA = 'F:\\Dataset\\Brain\\Cancer imaging archive(TCIA)\\REMBRANDT\\900-00-1961\\06-21-2005-11987\\5.000000-12296'
dir_out = 'C:\\Users\\YoonGuu Song\\Desktop'


#img_fixed, reader_fixed = preprocess_dcm(args.fixed)
## def preprocess_dcm(fpath)

###img, reader = load_dcm(fpath)
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(dir_TCIA)
type(dicom_names)
reader.SetFileNames(dicom_names)
reader.MetaDataDictionaryArrayUpdateOn()
reader.LoadPrivateTagsOn()

#error
image = reader.Execute()
image_array = sitk.GetArrayFromImage(image) # z, y, x
### return image_array, reader

# img = np.transpose(img, (2, 1, 0))
# img = img[::-1, :, ::-1]
plt.figure(1)
plt.imshow(image_array[0, :, :])
print(image_array.shape)

img_transpose = np.transpose(image_array, (2, 1, 0))
plt.figure(2)
plt.imshow(img_transpose[:,:,0])
print(img_transpose.shape)

img_reversed = img_transpose[::-1, :, ::-1]
plt.figure(3)
plt.imshow(img_reversed[:, :,-1 ])
print(img_reversed.shape)

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

        a= list(map(filters.median, args1, args2))
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
