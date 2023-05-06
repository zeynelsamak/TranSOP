#!/usr/bin/env python
'''
Dataset for training
Written by Whalechen
'''

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk

from preprocessing import to_shape
from sitk_utils import sitk_resample_to_spacing
from torch.utils.data import Dataset

np.seterr('raise')


def is_nan(x):
    return (x != x)


def read_image(path):

    try:
        if '.nii' not in path:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(path)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
        else:
            image = sitk.ReadImage(path)
    except ImportError:
        raise ImportError("problem in the path:", path)

    return image


def flip_numba(image, axis=1):
    """
    Spatial dimension flip

    Args:
        image (np.ndarray): image to be flipped
        axis (int): flip along the specified image axis

    Returns:
        np.ndarray: same image but flipped along the axis
    """

    assert image.ndim > axis, 'Selected axis must be in image dimension range'
    assert isinstance(image, np.ndarray), 'Image must be numpy array'

    flipped_img = np.flip(image, axis=axis)

    assert image.shape == flipped_img.shape, 'Flipped image must be same shape as original, may need select different axis'

    return flipped_img


class MrcleanDataset(Dataset):

    def __init__(self, root_dir, img_list, sets, phase='train', follow_time=None):

        self.follow_time = follow_time
        self.img_list = pd.read_csv(img_list)

        print("Processing {} datas".format(len(self.img_list)))
        self.root_dir = root_dir
        self.sets = sets
        self.phase = phase

        self.hu_range = self.sets.hu_range
        self.input_shape = self.sets.input_shape
        self.resample = self.sets.resample

        if self.sets.clinic:
            self.clinic_data = pd.read_csv('all_data_clinic.csv')
            self.clinic_data = self.clinic_data.fillna(self.clinic_data.median())  # 67 features

    def __nii2tensorarray__(self, data):
        [z, y, x] = self.input_shape  # data.shape
        X_bl = np.zeros((self.input_shape))
        X_bl = data
        new_data = np.reshape(X_bl, [1, z, y, x])
        new_data = new_data.astype("float32")
        return new_data

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        if self.sets.add_skull:
            ext = "_SS.nii.gz"
        else:
            ext = ".nii.gz"

        bl_path = os.path.join(self.root_dir, self.img_list.iloc[idx]['patient'], self.img_list.iloc[idx]
                               ['scan_time'], self.img_list.iloc[idx]['scan_modal'].replace(".nii.gz", ext))

        lbl = int(self.img_list.iloc[idx]['mrs_90d'])

        # print(bl_path)

        if self.sets.clinic:
            data_clinic = self.clinic_data.loc[self.clinic_data.Patient == self.img_list.iloc[idx]['patient']]
            treatment = np.array(data_clinic.values.tolist()[0][7:]).astype(
                np.float32)
        else:
            treatment = np.zeros((2))
            treatment[int(self.img_list.iloc[idx]['treatment'])] = 1
            treatment = treatment.astype("float32")

        if self.sets.num_classes == 2:
            if lbl > 2:
                y = 1
            else:
                y = 0
        else:
            y = lbl

        if self.phase == "train":
            # read image and labels
            try:
                # data processing
                img_array = self.__training_data_process__([bl_path])
                # 2 tensor array
                img_array1 = self.__nii2tensorarray__(img_array[0])
                #print(img_array.shape, img_array1.shape)
            except:
                print('ERROR: in', bl_path)

            return img_array1, treatment, y
        elif self.phase == "val":
            # read image
            try:
                # data processing
                img_array = self.__valid_data_process__([bl_path])
                # 2 tensor array
                img_array1 = self.__nii2tensorarray__(img_array[0])

            except:
                print('ERROR: in', bl_path)

            return img_array1, treatment, y, self.img_list.iloc[idx]['patient']

    def normalise_0_to_1(self, image):

        image = image.astype(np.float32)

        mean = np.mean(image)
        std = np.std(image)

        if std > 0:
            ret = (image - mean) / std
        else:
            ret = image * 0.
            return ret

        out = self.normalise_zero_one(ret)
        if 'tanh' in self.sets.trial:
            out *= 2.
            out -= 1.
        return out

    def normalise_zero_one(self, image):
        """
        Image normalisation. Normalises image to fit [0, 1] range.

        Args:
            image (np.ndarray): image to be normalised

        Returns:
            np.ndarray: normalised image

        """

        assert isinstance(image, np.ndarray), 'Image must be a numpy array'

        image = image.astype(np.float32)

        minimum = np.min(image)
        maximum = np.max(image)

        if maximum > minimum:
            ret = (image - minimum) / (maximum - minimum)

        else:
            ret = image * 0.
        return ret

    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """
        for i in range(volume.shape[0]):
            volume[i, ...] = self.normalise_0_to_1(volume[i, ...])

        return volume

    def crop_image(self, img, tol=0):
        # img is 2D image data
        # tol  is tolerance
        if img.shape[0] > 10:
            mask = img[img.shape[0]//2] > tol
            tmp = img[img.shape[0]//2]
            tmp = tmp[np.ix_(mask.any(1), mask.any(0))]

            new = np.zeros((img.shape[0], *tuple(tmp.shape)))
            for i in range(img.shape[0]):
                tmp = img[i]
                tmp = tmp[np.ix_(mask.any(1), mask.any(0))]
                new[i] = tmp
            return new
        else:
            return img

    def __training_data_process__(self, path, label=None):

        images = []
        for pth in path:
            if pth is not None:
                img = read_image(pth)
                #print('image read', pth)
                # resample volume spacing if specified
                try:
                    if self.resample:
                        img = sitk_resample_to_spacing(img, new_spacing=self.resample)
                except:
                    print('ERROR:', pth)

                # get image array from sitk object
                img = sitk.GetArrayFromImage(img)
                img = self.crop_image(img)
                # print('3',img.shape)
                img = to_shape(img, shape=self.input_shape)
                # print('4',img.shape)
            else:
                img = np.zeros(self.input_shape)
                #print('image not found, zeros filled')

            images.append(img[np.newaxis, ...])

        image = np.concatenate(images, axis=0)
        # print('5',image.shape)

        if self.sets.augment:
            if np.random.rand() < self.sets.augment:
                image = self._augment(image, ratio=self.sets.augment)
            # print('augmented')

        image = self.__itensity_normalize_one_volume__(image)
        # print('6',image.shape)

        return image

    def NOT(self, a):
        if(a == 0):
            return 1
        elif(a == 1):
            return 0

    def _augment(self, img, ratio=0.5):
        """An image augmentation function"""

        if np.random.rand() < ratio:
            img = flip_numba(img, axis=1)

        elif np.random.rand() < ratio:
            img = flip_numba(img, axis=2)

        elif np.random.rand() < ratio:
            img = flip_numba(img, axis=3)

        if np.random.rand() < ratio:
            random_i = 0.3*np.random.rand()+0.7
            img[0, ...] = img[0, ...] * random_i

        return img

    def __testing_data_process__(self, path):
        # crop data according net input size
        images = []
        for pth in path:
            img = read_image(pth)
            #print('image read', pth)
            # resample volume spacing if specified
            try:
                if self.resample:
                    img = sitk_resample_to_spacing(img, new_spacing=self.resample)
            except:
                print('ERROR:', pth)

            # get image array from sitk object
            img = sitk.GetArrayFromImage(img)
            # print(img.shape)
            #img = normalise_zero_one(img)
            img = self.crop_image(img)
            # print('3',img.shape)
            img = to_shape(img, shape=self.input_shape)
            # print('4',img.shape)

            images.append(img[np.newaxis, ...])

        image = np.concatenate(images, axis=0)

        image = self.__itensity_normalize_one_volume__(image)
        return image

    def __valid_data_process__(self, path, label=None):

        images = []
        for pth in path:
            if pth is not None:
                img = read_image(pth)
                #print('image read', pth)
                # resample volume spacing if specified
                try:
                    if self.resample:
                        img = sitk_resample_to_spacing(img, new_spacing=self.resample)
                except:
                    print('ERROR:', pth)

                # get image array from sitk object
                img = sitk.GetArrayFromImage(img)
                # print(img.shape)
                img = self.crop_image(img)
                # print('3',img.shape)
                img = to_shape(img, shape=self.input_shape)
                # print('4',img.shape)
            else:
                img = np.zeros(self.input_shape)
                #print('image not found, zeros filled')

            images.append(img[np.newaxis, ...])

        image = np.concatenate(images, axis=0)

        image = self.__itensity_normalize_one_volume__(image)

        return image
