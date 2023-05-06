#!/usr/bin/env python
# coding: utf-8

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

#from numba import njit
import numpy as np 
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


#@njit(nogil=True)
def whitening(image):
    """
    Whitening. Normalises image to zero mean and unit variance.

    Args:
        image (np.ndarray): image to be normalised

    Returns:
        np.ndarray: normalised image

    """

    assert isinstance(image, np.ndarray), 'Image must be a numpy array'

    image = image.astype(np.float32)

    mean = np.mean(image)
    std = np.std(image)

    if std > 0:
        ret = (image - mean) / std
    else:
        ret = image * 0.
    return ret


#@njit(nogil=True)
def normalise_zero_one(image):
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


#@njit(nogil=True)
def normalise_one_one(image):
    """
    Image normalisation. Normalises image to fit [-1, 1] range.

    Args:
        image (np.ndarray): image to be normalised

    Returns:
        np.ndarray: normalised image

    """

    assert isinstance(image, np.ndarray), 'Image must be a numpy array'

    ret = normalise_zero_one(image)
    ret *= 2.
    ret -= 1.
    return ret


#@njit(nogil=True)
def normalise_channel_wise(image, normalisation='whitening'):

    """
    Image normalisation. Normalise each channel separately.

    Args:
        image (np.ndarray): image to be normalised

    Returns:
        np.ndarray: normalised image, each channel

    """

    assert isinstance(image, np.ndarray), 'Image must be a numpy array'
    
    normalise = None
    if normalisation == 'whitening':
        normalise = whitening
    elif normalisation == 'normalise_zero_one':
        normalise = normalise_zero_one
    elif normalisation == 'normalise_one_one':
        normalise = normalise_one_one

    for i in range(image.shape[-1]):
        image[...,i] = normalise(image[...,i])

    return image


#@njit(nogil=True)
def make_central_crop(image, crop_size):
    """ 
    Make a crop from center of 3D image of given crop_size.

    Args:
        image (ndarray): 3D image of shape `(dim1, dim2, dim3)`.
        crop_size (ndarray or tuple): Size of crop along three dimensions `(int, int, int)`

    Returns:
        ndarray: 3D crop from image.
    """
    crop_size = np.asarray(crop_size)
    crop_halfsize = np.ceil(crop_size / 2).astype(np.int)
    halfsize = np.rint(np.asarray(image.shape) / 2).astype(np.int)
    cropped_img = image[halfsize[0] - crop_halfsize[0]: halfsize[0] + crop_size[0] - crop_halfsize[0],
                        halfsize[1] - crop_halfsize[1]: halfsize[1] + crop_size[1] - crop_halfsize[1],
                        halfsize[2] - crop_halfsize[2]: halfsize[2] + crop_size[2] - crop_halfsize[2]]
    return cropped_img.copy()


#@njit(nogil=True)
def resize_image_with_crop_or_pad(image, img_size=(64, 64, 64), **kwargs):
    """

    Image resizing. Resizes image by cropping or padding dimension
     to fit specified size.

    Args:
        image (np.ndarray): image to be resized
        img_size (list or tuple): new image size
        kwargs (): additional arguments to be passed to np.pad

    Returns:
        np.ndarray: resized image
    """

    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    # Get the image dimensionality
    rank = len(img_size)

    # Create placeholders for the new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    # For each dimensions find whether it is supposed to be cropped or padded
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # Create slicer object to crop or leave each dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # Pad the cropped image to extend the missing dimension
    return np.pad(image[tuple(slicer)], to_padding, **kwargs)


#@njit(nogil=True)
def to_shape(data, shape, padding='constant'):
    """ 
    Crop or pad 3D array to resize it to `shape`

    Args:
        data (ndarray): 3D array for reshaping
        shape (tuple, list or ndarray): data shape after crop or pad
        padding (str): mode of padding, any of the modes of np.pad()
    
    Returns:
        ndarray: cropped and padded data
    """
    # calculate shapes discrepancy
    data_shape = np.asarray(data.shape)
    shape = np.asarray(shape)
    overshoot = data_shape - shape

    # calclate crop params and perform crop
    crop_dims = np.maximum(overshoot, 0)
    crop_first = crop_dims // 2
    crop_trailing = crop_dims - crop_first
    slices = [slice(first, dim_shape - trailing)
              for first, trailing, dim_shape in zip(crop_first, crop_trailing, data_shape)]
    data = data[tuple(slices)]

    # calculate padding params and perform padding
    pad_dims = -np.minimum(overshoot, 0)
    pad_first = pad_dims // 2
    pad_trailing = pad_dims - pad_first
    pad_params = [(first, trailing)
                  for first, trailing in zip(pad_first, pad_trailing)]
    data = np.pad(data, pad_width=pad_params, mode=padding)

    # return cropped/padded array
    return data



def normalize_hu(images, min_max_hu=(-1000, 400), random_shift=None):
        """ 
        Normalize HU-densities to interval [0, 255].

        Trim HU that are outside range [min_hu, max_hu], then scale to [0, 255].

        Args:

            min_hu (int): minimum value for hu that will be used as trimming threshold.
            max_hu (int): maximum value for hu that will be used as trimming threshold.

        Returns:

            nd.array: image with normalised HU

        Examples
        --------
        >>> image = normalize_hu(image, min_hu=-1300, max_hu=600)
        """

        wl, ww = min_max_hu
        min_hu= wl - (ww/2)
        max_hu = wl + (ww/2) 


        if random_shift and np.random.rand()>0.8:
            min_hu = min_hu + random_shift * np.random.uniform(-1,1)
            max_hu = max_hu + random_shift * np.random.uniform(-1,1)
        # trimming and scaling to [0, 1]
        images = (images - min_hu) / (max_hu - min_hu)
        images[images > 1] = 1.
        images[images < 0] = 0.

        # scaling to [0, 255]
        images *= 255
        return images
        
def normalize_hu2(data, min_max_hu=(-1000, 400)):
    """
    Scale pixel intensity data using specified window level, width, and intensity range.
    """
    
    wl, ww= min_max_hu
    
    data[data <= (wl-ww/2.0)] = 0
    data[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))] = ((data[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))]-(wl-0.5))/(ww-1.0)+0.5)*(1-0)+0
    data[data > (wl+ww/2.0)] = 1
    
    return data