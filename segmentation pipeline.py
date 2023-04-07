import numpy as np
import os
from skimage.io import imsave
from skimage.morphology import disk
import skimage.filters as ft
from skimage.exposure import equalize_adapthist
import h5py
import torch
from cellpose import models
import tqdm


def segPipeline3D(image_volume,
                  do_3D=True,
                  flow=0.4,
                  diameter=17,
                  stitch=0.1,
                  net_avg=True,
                  batch_size=6,
                  use_CP=True,
                  min_size=150):

    """
    3D nuclear segmentation pipeline for large single channel 3D image volume
    using the cellpose framework. This method defaults to using the cellpose
    nuclear segmentation model, but can be switched to segment cell bodies
    using the use_CP parameter. Note this method requires a torch enabled GPU.

    Parameters
    ----------
    image_volume : 3D numpy array
        3D image for segmentation with cellpose.

    do_3D : bool
        Default True, boolean to use 3D or 2D cellpose model.

    flow : float
        Default to 0.4, flow error correction rate for cellpose input. Lower
    values will increase speed, higher values will increase accuracy. Valid
    values are [0.0, 1.0).

    diameter : int
        Default to 17. Average estimated diameter in pixels/voxels for objects
    to label. Value may change depending on image resolution.

    stitch : float
        Default to 0.1, stitch tolerance value for generating 3D images from a
    series of 2D segmentations. Note: this parameter is not used in 3D cellpose
    models.

    net_avg : bool,
        Default to True. Tells cellpose to average the results of 4 separate
    networks. Performance is decreased when False, but pipeline is faster.

    batch_size : int
        Default to 6. Batch size for cellpose model, reduce if memory is
    limited.

    use_CP : bool
        Default to True. Determines which cellpose model is used, default is
    to use the pretrained nuclear segmentation model.

    min_size : int
        Default to 150. Minimum segmented object size allowed by cellpose,
    objects less than this value will be discarded from final mask.


    Returns
    -------

    masks : 3D numpy array
        dtype is uint16 or uint32. Segmentation mask of input image_volume,
    each identified object is assigned a unique integer.

    """

    # empty cache of GPU to ensure no memory issues
    torch.cuda.empty_cache()

    # determine which model to load from cellpose
    # default is to use nuclear segmentation model
    if use_CP:
        model = models.Cellpose(gpu=True, model_type='nuclei', net_avg=net_avg)

    # otherwise use the cell body model
    else:
        model = models.CellposeModel(gpu=True,
                                     pretrained_model='cyto',
                                     net_avg=net_avg)

    # generates 3D segmentation masks
    if do_3D:
        # generate masks from 3D cellpose model using image_volume
        masks, flows, styles, diams = model.eval(image_volume,
                                                 batch_size=4,
                                                 channels=[0, 0],
                                                 diameter=diameter,
                                                 do_3D=do_3D,
                                                 flow_threshold=flow,
                                                 min_size=min_size)

    # generates 3D segmentation masks from stitched 2D cellpose model
    else:
        # split 3D image array into a list of 2D images
        image_list = np.split(image_volume, len(image_volume), axis=0)

        # generate masks from list of 2D images using loaded cellpose model
        masks, flows, styles, diams = model.eval(image_list,
                                                 batch_size=batch_size,
                                                 channels=[0, 0],
                                                 diameter=diameter,
                                                 do_3D=do_3D,
                                                 flow_threshold=flow,
                                                 stitch_threshold=stitch,
                                                 normalize=False)

        # compile segmented 2D images into final 3D array
        masks = np.asarray(masks)

    # if number of detected objects in segmentation mask is less than 65535
    #  then return mask as uint16 datatype
    if masks.max() < 65535:
        return masks.astype(np.uint16)

    # otherwise return mask as uint32 datatype
    else:
        return masks.astype(np.uint32)


def Pipeline2D(image, flow = 0.0, diameter=17, net_avg=True, batch_size = 6,
            use_CP = True, min_size=150):
    
    """
    2D nuclear segmentation pipeline for a single plane of a 1-channel image volume
    using the cellpose framework. This method defaults to using the cellpose
    nuclear segmentation model, but can be switched to segment cell bodies
    using the use_CP parameter. Note this method requires a torch enabled GPU.
    
    Parameters
    ----------
    image : 2D numpy array
        image for segmentation with cellpose.

    flow : float
        Default to 0.4, flow error correction rate for cellpose input. Lower
    values will increase speed, higher values will increase accuracy. Valid
    values are [0.0, 1.0).

    diameter : int
        Default to 17. Average estimated diameter in pixels/voxels for objects
    to label. Value may change depending on image resolution.
    
    net_avg : bool,
        Default to True. Tells cellpose to average the results of 4 separate
    networks. Performance is decreased when False, but pipeline is faster.
    
    batch_size : int
        Default to 6. Batch size for cellpose model, reduce if memory is
    limited.

    use_CP : bool
        Default to True. Determines which cellpose model is used, default is
    to use the pretrained nuclear segmentation model.

    min_size : int
        Default to 150. Minimum segmented object size allowed by cellpose,
    objects less than this value will be discarded from final mask.

    
    Returns
    -------

    masks : 2D numpy array
        dtype is uint16 or uint32. Segmentation mask of input image,
    each identified object is assigned a unique integer.
    """
    
    torch.cuda.empty_cache()
    
    model = models.Cellpose(gpu=True,model_type='nuclei', net_avg=net_avg)

    masks, flows, styles, diams = model.eval(image, batch_size=batch_size, channels=[0,0],
                                            diameter=diameter, flow_threshold=flow,
                                            normalize=False)
    masks = np.asarray(masks)
        
    if masks.max() < 65535:
        return masks.astype(np.uint16)
    
    else:

        return masks.astype(np.uint32)

def preProcessImage(image, radius=2):
    """
    Preprocess image volume via median filter with circular structuring element
    followed by CLAHE.


    Parameters
    ----------

    image : 3D numpy array, dtype = np.uint16
        Input 3D grayscale image

    radius : int
        Default to 2, radius of circular structuring element for median filter.

    Returns
    -------

    med_image : 3D numpy array, dtype = np.uint16
        Median filtered image with 3D CLAHE.

    """

    # create circular kernel
    kernel = disk(radius)

    # generate holder image of equal size to input image
    med_image = np.zeros(image.shape, dtype=image.dtype)

    # iterate through image volume and perform median filter with kernel
    for i, z in enumerate(tqdm(image)):
        med_image[i] = ft.rank.median(z, selem=kernel)

    # perform CLAHE on median filtered image
    med_image = (equalize_adapthist(med_image.astype(np.uint16))*image.max())

    # return image as uint16 dtype
    return med_image.astype(np.uint16)

def preProcessImage2D(image, radius=2):
    """
        Equivalent 2D image preprocessing, median filtering with a circular element,
    followed by CLAHE.

    Parameters
    ----------

    image : 2D numpy array
        image to preprocess
    
    radius : int
        Kernel size for median filter.
    
    Returns
    -------

    med_image : 2D numpy array
        Preprocessed image.
    """
    
    kernel = disk(radius)
    med_image = np.zeros(image.shape, dtype=image.dtype)
    
    med_image = ft.rank.median(image, selem=kernel)
    
    med_image = (equalize_adapthist(med_image.astype(np.uint16))*image.max()).astype(np.uint16)
    
    return med_image


def runWorkflow3D(rootdir,
                filename,
                dirname,
                im_range=500,
                z_start=0,
                z_max=4000):
    """
    Runs full pipeline workflow for 3D nuclear segmentation using cellpose
    pipeline.


    Parameters
    ----------

    rootdir : str or pathlike
        Root directory for data

    filename : str
        Hdf5 file for processing. Assumed to have standard file structure with
    nuclear channel as ch00

    dirname : str
        Directory name for saving results. Will be within rootdir path, does
    not need to exist.

    im_range : int, default 500
        First dimension image size to load into memory.

    """

    # filepath for hdf5 image dataa
    data_file = os.path.join(rootdir, filename)

    # create savedir for saving segmentation  masks
    savedir = os.path.join(rootdir, dirname)
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    # create pipe to image file and print shape and dtype
    f = h5py.File(data_file, 'r')
    print(f['t00000/s00/0/cells'].shape, f['t00000/s00/0/cells'].dtype)

    for i in range(z_start, f['t00000/s00/0/cells'].shape[0], im_range):
        # if outside max z_range break out of loop
        if i + im_range > z_max:
            return

        # create safile path
        filename = os.path.join(savedir,
                                'region1_x_{:0>6d}_{:0>6d}_diam17.tif'.format(i, i+im_range))
        print(filename)

        # read in image volume
        image = f['t00000/s00/0/cells'][i:i+im_range].astype(np.uint16)

        # prep image volume for segmentation
        image = preProcessImage(image)

        # generate segmentation masks
        masks = segPipeline3D(image,
                              diameter=17,
                              do_3D=True,
                              net_avg=True,
                              batch_size=6,
                              flow=.4)

        # save masks to disk then delete them to save memory use
        imsave(filename, masks)
        del masks
    return
