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
dir_TCIA_dcm = 'F:\\Dataset\\Brain\\Cancer imaging archive(TCIA)\\REMBRANDT\\900-00-1961\\06-21-2005-11987\\4.000000-85223\\1-01.dcm'
dir_TCIA = 'F:\\Dataset\\Brain\\Cancer imaging archive(TCIA)\\REMBRANDT\\900-00-1961\\06-21-2005-11987\\4.000000-85223'





reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(dir_TCIA)
type(dicom_names)
reader.SetFileNames(dicom_names)
reader.MetaDataDictionaryArrayUpdateOn()
reader.LoadPrivateTagsOn()

#error
image = reader.Execute()
image_array = sitk.GetArrayFromImage(image) # z, y, x






''
plt.figure(1)
plt.imshow(image_array[0, :, :])
print(image_array.shape)

img_transpose = np.transpose(image_array, (2, 1, 0))
plt.figure(2)
plt.imshow(img_transpose[:,:,0])
print(img_transpose.shape)

img_reversed = img_transpose[::-1, :, ::-1]
plt.figure(3)
plt.imshow(img_reversed[:, :, 0])
print(img_reversed.shape)
''

ths = [(80, 140), (110, 160), (70, 90), (60, 80), (50, 70), (40, 60), (30, 50), (20, 40), (10, 30), (140, 180), (160, 200)]
vol = filters.gaussian(img_reversed, sigma = 2, preserve_range = True)
mask = np.zeros_like(vol, dtype = np.bool)
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

'''
preprocess_dcm result

'returned mask is img' 
'reader is returned'
'''