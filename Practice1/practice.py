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




def preprocess_dcm(fpath):
    # fpath = dir_TCIA
    img, reader = load_dcm(fpath)
    print(img.shape)
    plt.figure(1)
    plt.imshow(img[0, :, :])


    img = np.transpose(img, (2, 1, 0))
    plt.figure(2)
    plt.imshow(img[:,:,0])
    print(img.shape)

    img = img[::-1, :, ::-1]
    plt.figure(3)
    plt.imshow(img[:, :, 0])
    print(img.shape)

    #goes weird thing happen
    liver_mask = auto_liver_mask(img)
    img = crop_mask(img, liver_mask)
    return img, reader

def load_dcm(fpath):
    # fpath = dir_TCIA
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(fpath)
    type(dicom_names)
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    #error
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image) # z, y, x
    return image_array, reader

def auto_liver_mask(vol, ths = [(80, 140), (110, 160), (70, 90), (60, 80), (50, 70), (40, 60), (30, 50), (20, 40), (10, 30), (140, 180), (160, 200)]):
    # vol= img
    # print(vol.shape)
    # ths = [(80, 140), (110, 160), (70, 90), (60, 80), (50, 70), (40, 60), (30, 50), (20, 40), (10, 30), (140, 180),(160, 200)]
    vol = filters.gaussian(vol, sigma = 2, preserve_range = True)
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
    return mask

def crop_mask(volume, segmentation, target_shape=(128, 128, 128)):
    indices = np.array(np.nonzero(segmentation))
    bound_r = np.max(indices, axis=-1)
    bound_l = np.min(indices, axis=-1)
    box_size = bound_r - bound_l + 1
    padding = np.maximum( (box_size * 0.1).astype(np.int32), 5)
    bound_l = np.maximum(bound_l - padding, 0)
    bound_r = np.minimum(bound_r + padding + 1, segmentation.shape)
    return wl_normalization(crop(volume, bound_l, bound_r, target_shape)).astype(np.uint8)

def wl_normalization(img, w=290, l=120):
    img = skimage.exposure.rescale_intensity(img, in_range=(l - w / 2, l + w / 2), out_range=(0, 255))
    return img.astype(np.uint8)

def crop(arr, bound_l, bound_r, target_shape, order=1):
    cropped = arr[bound_l[0]: bound_r[0], bound_l[1]: bound_r[1], bound_l[2]: bound_r[2]]
    return scipy.ndimage.zoom(cropped, np.array(target_shape) / np.array(cropped.shape), order = order)


