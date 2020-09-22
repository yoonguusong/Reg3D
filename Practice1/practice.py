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
import time
dir_h5 ='E:\\Pycharm\\PycharmProjects\\Reg3D\\Volume Tweening Network\\datasets\\adni.h5'
dir_nii = 'F:\\Dataset\\Brain\\Duc_dataset\\BE_T2_CN\\Normal001-T2.nii.gz'
dir_TCIA_dcm = 'F:\\Dataset\\Brain\\Cancer imaging archive(TCIA)\\REMBRANDT\\900-00-1961\\06-21-2005-11987\\5.000000-12296\\1-01.dcm'
dir_TCIA = 'F:\\Dataset\\Brain\\Cancer imaging archive(TCIA)\\REMBRANDT\\900-00-1961\\06-21-2005-11987\\5.000000-12296'
dir_out = 'C:\\Users\\YoonGuu Song\\Desktop'



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


def save_dcm(img, series_reader, fpath):
    try:
        os.makedirs(fpath)
    except:
        pass
    img = img[::-1, :, ::-1]
    img = np.transpose(img, (2, 1, 0))
    filtered_image = sitk.GetImageFromArray(img)

    writer = sitk.ImageFileWriter()
    # Use the study/series/frame of reference information given in the meta-data
    # dictionary and not the automatically generated information from the file IO
    writer.KeepOriginalImageUIDOn()

    tags_to_copy = ["0010|0010",  # Patient Name
                    "0010|0020",  # Patient ID
                    "0010|0030",  # Patient Birth Date
                    "0020|000D",  # Study Instance UID, for machine consumption
                    "0020|0010",  # Study ID, for human consumption
                    "0008|0020",  # Study Date
                    "0008|0030",  # Study Time
                    "0008|0050",  # Accession Number
                    "0008|0060"  # Modality
                    ]

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    # Copy some of the tags and add the relevant tags indicating the change.
    # For the series instance UID (0020|000e), each of the components is a number, cannot start
    # with zero, and separated by a '.' We create a unique series ID using the date and time.
    # tags of interest:
    direction = filtered_image.GetDirection()
    # series_tag_values = [(k, series_reader.GetMetaData(0, k)) for k in tags_to_copy if
    #                      series_reader.HasMetaDataKey(0, k)] + \
    #                     [("0008|0031", modification_time),  # Series Time
    #                      ("0008|0021", modification_date),  # Series Date
    #                      ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
    #                      ("0020|000e", "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time),
    #                      # Series Instance UID
    #                      ("0020|0037",
    #                       '\\'.join(map(str, (direction[0], direction[3], direction[6],  # Image Orientation (Patient)
    #                                           direction[1], direction[4], direction[7])))),
    #                      ("0008|103e",
    #                       series_reader.GetMetaData(0, "0008|103e") + " Processed-SimpleITK")]  # Series Description

    series_tag_values = [(k, series_reader.GetMetaData(0, k)) for k in tags_to_copy if
                         series_reader.HasMetaDataKey(0, k)] + \
                        [("0008|0031", modification_time),  # Series Time
                         ("0008|0021", modification_date),  # Series Date
                         ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
                         ("0020|000e", "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time),
                         # Series Instance UID
                         ("0020|0037",
                          '\\'.join(map(str, (direction[0], direction[3], direction[6],  # Image Orientation (Patient)
                                              direction[1], direction[4], direction[7]))))]

    for i in range(filtered_image.GetDepth()):
        image_slice = filtered_image[:, :, i]
        # Tags shared by the series.
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)
        # Slice specific tags.
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time
        image_slice.SetMetaData("0020|0032", '\\'.join(
            map(str, filtered_image.TransformIndexToPhysicalPoint((0, 0, i)))))  # Image Position (Patient)
        image_slice.SetMetaData("0020|0013", str(i))  # Instance Number

        # Write to the output directory and add the extension dcm, to force writing in DICOM format.
        writer.SetFileName(os.path.join(fpath, str(i) + '.dcm'))
        writer.Execute(image_slice)

img_fixed, reader_fixed = preprocess_dcm(dir_TCIA)

save_dcm(img_fixed, reader_fixed, os.path.join(dir_out, 'fixed'))