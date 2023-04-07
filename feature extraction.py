from skimage.io import imread
from skimage.measure import regionprops_table, shannon_entropy
from skimage.segmentation import clear_border
from skimage.measure import marching_cubes, mesh_surface_area
from skimage.morphology import remove_small_objects
import os
import numpy as np
import scipy.ndimage as ndi
import scipy.stats as stat
import pandas as pd
import glob
import h5py as h5


def surface_area(regionmask):
    """
    Computes mesh surface area for segmentation mask using marching cubes from 
    skimage measure.

    Parameters
    ----------

    regionmask : 3D numpy array
        Mask of object to compute surface area


    Returns
    -------

    mesh_surface_area : float
        Surface area of region mask computed by marching cubes

    """

    # pad with zeros to prevent holes at edges
    regionmask = np.pad(regionmask, [(1, 1), (1, 1), (1, 1)], 'constant')

    # fill holes to generate solid object
    structure = np.ones((3,) * regionmask.ndim)
    filled_image = ndi.binary_fill_holes(regionmask, structure)

    # get mesh surface of object
    verts, faces, normals, values = marching_cubes(filled_image,
                                                   mask=filled_image)

    # return mesh surface area
    return mesh_surface_area(verts, faces)


def std_intensity(regionmask, intensity):
    """
    Returns  standard deviation of intensity region

    Parameters
    ----------

    regionmask : ndarray
        Labeled mask for object region, required by extra_properties
        in regionprops_table.

    intensity : ndarray
        corresponding intensity image for masked region


    Returns
    -------

    std : float
        standard deviation via numpy

    """

    return np.std(intensity)


def var_intensity(regionmask, intensity):
    """
    Returns variance of intensity region


    Parameters
    ----------

    regionmask : ndarray
        Labeled mask for object region, required by extra_properties
        in regionprops_table.

    intensity : ndarray
        corresponding intensity image for masked region


    Returns
    -------

    var : float
        variance via numpy

    """
    return np.var(intensity)


def entropy_intensity(regionmask, intensity):
    """
    Returns shannon entropy of intensity region

    Parameters
    ----------

    regionmask : ndarray
        Labeled mask for object region, required by extra_properties
        in regionprops_table.

    intensity : ndarray
        corresponding intensity image for masked region


    Returns
    -------

    entropy : float
        shannon entropy via skimage.measure
    """

    return shannon_entropy(intensity)


def get3DObjectProperties(labeled_image, intensity_image):

    """
    Measures 3D properties of a labeled image and returns labeled object
    properties in a pandas DataFrame for convienient sorting.


    Parameters
    ----------

    labled_image : 3D numpy array
        Segmented image of nuclei where each individual object has been
        assigned a unique integer


    intensity_image : 3D numpy array
        3D intensity image of nuclei, assumed np.uint16.


    Returns
    -------

    object_props : pd.DataFrame
        DataFrame object with selected properties extracted using
        skimage measure regionprops_table


    """

    # object properties for extraction
    properties = ['equivalent_diameter', 'inertia_tensor_eigvals',
                  'major_axis_length', 'minor_axis_length',
                  'label', 'area',
                  'solidity', 'feret_diameter_max',
                  'centroid', 'bbox',
                  'bbox_area', 'extent',
                  'convex_area', 'min_intensity',
                  'max_intensity', 'mean_intensity',
                  'extent', 'image', 'intensity_image']

    # extract features and return as dataframe
    object_props = pd.DataFrame(
                                regionprops_table(labeled_image,
                                                  intensity_image=intensity_image,
                                                  properties=properties,
                                                  extra_properties = (std_intensity, 
                                                                      var_intensity,
                                                                      entropy_intensity,
                                                                      surface_area)
                                                  )
                                )

    object_props = object_props[object_props.columns.difference(['image',
                                                                'intensity_image'])]

    return object_props

def get2DObjectProperties(labeled_image):
    
    """
    Returns labeled object properties in a pandas DataFrame for convienient sorting.
    
    
    Parameters 
    ----------
    
    labled_image : 2D numpy array
        Segmented image of nuclei where each individual object has been assigned a 
        unique integer idea.
        
    
    Returns
    -------
    
    object_props : pd.DataFrame
        DataFrame object with selected properties extracted using skimage.measure.regionprops_table
    
    """
    
    #clear boundary
    labeled_image = clear_border(labeled_image)
    
    #object properties for extraction
    properties=[ 'equivalent_diameter', 
                'major_axis_length', 
                'minor_axis_length', 
                'label', 
                'area',
                'perimeter',
                'solidity', 
                'feret_diameter_max', 
                'centroid', 
                'bbox', 
                'bbox_area', 
                'extent',
                'convex_area',
                'extent']
    
    #extract features and return as dataframe
    object_props = pd.DataFrame(regionprops_table(labeled_image, 
                                                  properties=properties))
    
    object_props['pa_ratio'] = object_props['perimeter']/object_props['area']
    object_props['aspect ratio'] = object_props['major_axis_length']/object_props['minor_axis_length']
    
    return object_props


def feature3Dworkflow(mask_filename,
                   h5_filename,
                   z_range,
                   xy_range=None,
                   min_size=150):

    """
    Run full feature extraction workfloww on segmented masks and
    3D intensity data. Saves features to csv file.


    Parameters
    ----------

    mask_filename : str or pathlike
        Path to segmented mask file, assumed tif.

    h5_filename : str or pathlike
        Path to intensity image  file, assumed hdf5.

    z_range : tuple
        Tuple of ints for z range to read intensity data from
    h5 file.

    xy_range : None or tuple
        Tuple for xy coordinates to read intenstity data from
    a particular section of h5 file in the other two dimmensions. 

    min_size : int
        Default is 150. Minimum object size for masks, used for
    remove_small_objects from skimage.

    Returns
    -------


    """

    # read input mask from disk
    print('Reading masks from:', mask_filename)
    mask = imread(mask_filename)

    # remove small objects with min_size
    print('Removing small objects from mask')
    mask = clear_border(mask)
    mask = remove_small_objects(mask, min_size=min_size)

    # read in intensity data from input h5 file
    print('Mask datatype:', mask.dtype)
    print('Reading intensity data from: \n %s, \n z-range %s' % (h5_filename,
                                                                 z_range))
    f = h5.File(h5_filename, 'r')

    if xy_range is None:
        intensity_image = f['t00000/s00/0/cells'][z_range[0]:z_range[1]]

    else:
        x_range = xy_range[0]
        y_range = xy_range[1]
        intensity_image = f['t00000/s00/0/cells'][z_range[0]:z_range[1],
                                                  x_range[0]:x_range[1],
                                                  y_range[0]:y_range[1]]
    f.close()

    # extract 3D properties using get3DObjectProperties
    print('Exctracting features')
    properties = get3DObjectProperties(mask, intensity_image)
    properties = getAspectRatios(properties)

    # create save directory for properties
    savedir = mask_filename.split('.tif')[0] + '_features'
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    # save data as csv in savedir
    print('Saving data')
    prop_filename = os.path.join(savedir, 'region_properties.csv')
    properties.to_csv(prop_filename)

    return


def getPropertyDescriptors(input_properties):
    """
    Gets property descriptors for a set of object properties: (volume,
     convex_area, major axis length, minor axis length, surface area,
     solidity, extent) for each property it will get: min/max, mean,
     variance, skewness, kurtosis

    """
    def describeProperty(properties, key):
        propdict = {}

        propdict['nobs'],
        propdict['minmax'],
        propdict['mean'],
        propdict['var'],
        propdict['skew'], propdict['kurt'] = stat.describe(
            properties[key].dropna())

        propdict['IQR'] = stat.iqr(properties[key].dropna())
        propdict['90th'] = np.percentile(properties[key].dropna(), 90)
        propdict['10th'] = np.percentile(properties[key].dropna(), 10)
        return propdict

    properties = {}
    properties['Biopsy_id'] = input_properties.Biopsy_id.iloc[0]
    # volume descriptors
    vol_props = describeProperty(input_properties, 'area')
    properties['vol_mean'] = vol_props['mean']
    properties['vol_var'] = vol_props['var']
    properties['vol_skew'] = vol_props['skew']
    properties['vol_kurt'] = vol_props['kurt']
    properties['vol_IQR'] = vol_props['IQR']

    # convex area descriptors
    convex_props = describeProperty(input_properties, 'convex_area')
    properties['convex_mean'] = convex_props['mean']
    properties['convex_var'] = convex_props['var']
    properties['convex_skew'] = convex_props['skew']
    properties['convex_kurt'] = convex_props['kurt']
    properties['convex_IQR'] = convex_props['IQR']

    # major axis length descriptors
    major_props = describeProperty(input_properties, 'major_axis_length')
    properties['major_axis_mean'] = major_props['mean']
    properties['major_axis_var'] = major_props['var']
    properties['major_axis_skew'] = major_props['skew']
    properties['major_axis_kurt'] = major_props['kurt']
    properties['major_axis_IQR'] = major_props['IQR']

    # minor axis length descriptors
    minor_props = describeProperty(input_properties, 'minor_axis_length')
    properties['minor_axis_mean'] = minor_props['mean']
    properties['minor_axis_var'] = minor_props['var']
    properties['minor_axis_skew'] = minor_props['skew']
    properties['minor_axis_kurt'] = minor_props['kurt']
    properties['minor_axis_IQR'] = minor_props['IQR']

    # surface area descriptors
    SA_props = describeProperty(input_properties, 'surface_area')
    properties['surface_area_mean'] = SA_props['mean']
    properties['surface_area_var'] = SA_props['var']
    properties['surface_area_skew'] = SA_props['skew']
    properties['surface_area_kurt'] = SA_props['kurt']
    properties['surface_area_IQR'] = SA_props['IQR']

    # solidity descriptors
    solidity_props = describeProperty(input_properties, 'solidity')
    properties['solidity_mean'] = solidity_props['mean']
    properties['solidity_var'] = solidity_props['var']
    properties['solidity_skew'] = solidity_props['skew']
    properties['solidity_kurt'] = solidity_props['kurt']
    properties['solidity_IQR'] = solidity_props['IQR']

    # extent descriptors
    extent_props = describeProperty(input_properties, 'extent')
    properties['extent_mean'] = extent_props['mean']
    properties['extent_var'] = extent_props['var']
    properties['extent_skew'] = extent_props['skew']
    properties['extent_kurt'] = extent_props['kurt']
    properties['extent_IQR'] = extent_props['IQR']

    # equivalent diameter
    equiv_props = describeProperty(input_properties, 'equivalent_diameter')
    properties['EQD_mean'] = equiv_props['mean']
    properties['EQD_var'] = equiv_props['var']
    properties['EQD_skew'] = equiv_props['skew']
    properties['EQD_kurt'] = equiv_props['kurt']
    properties['EQD_IQR'] = equiv_props['IQR']

    # area to volume ratio
    av_props = describeProperty(input_properties, 'av ratio')
    properties['av_mean'] = av_props['mean']
    properties['av_var'] = av_props['var']
    properties['av_skew'] = av_props['skew']
    properties['av_kurt'] = av_props['kurt']
    properties['av_IQR'] = av_props['IQR']

    # aspect ratio
    aspect_props = describeProperty(input_properties, 'aspect ratio')
    properties['aspect_mean'] = aspect_props['mean']
    properties['aspect_var'] = aspect_props['var']
    properties['aspect_skew'] = aspect_props['skew']
    properties['aspect_kurt'] = aspect_props['kurt']
    properties['aspect_IQR'] = aspect_props['IQR']

    # inter nuclear distance properties
    ind_props = describeProperty(input_properties, 'avg_IND')
    properties['IND_mean'] = ind_props['mean']
    properties['IND_var'] = ind_props['var']
    properties['IND_skew'] = ind_props['skew']
    properties['IND_kurt'] = ind_props['kurt']
    properties['IND_IQR'] = ind_props['IQR']

    # local nuclear density properties
    density_props = describeProperty(input_properties, 'local_density')
    properties['density_mean'] = density_props['mean']
    properties['density_var'] = density_props['var']
    properties['density_skew'] = density_props['skew']
    properties['density_kurt'] = density_props['kurt']
    properties['density_IQR'] = density_props['IQR']

    # add bcr to property dictionary
    properties['bcr'] = input_properties['bcr'].iloc[0]
    return properties


def getAspectRatios(proptable):
    """
    Gets major/minor axis length ratio and area to volume ratio for 3D
    properties.

    Parameters
    ----------

    proptable : pd.Dataframe
        Property table for 3D extracted features.

    Returns
    -------

    proptable : pd.Dataframe
        Input dataframe with 'av ratio' and 'aspect ratio' columns added.
    """
    proptable['av ratio'] = proptable['surface_area']/proptable['area']
    proptable['aspect ratio'] = proptable['major_axis_length']/proptable['minor_axis_length']

    return proptable


def loadPropTable(filepath):
    """
    Loads property table as pandas dataframe.

    Parameters
    ----------

    filepath : str or pathlike
        Filepath to csv file of region properties

    Returns
    -------

    pd.read_csv(filepath) : pd.Dataframe
        Data frame loaded from csv file.
    """

    return pd.read_csv(filepath)


def getPropfiles(rootdir):
    """
    Gets set of property files from rootdir.

    Parameters
    ----------

    rootdir : str or pathlike
        Root directory with stored csv files

    Returns
    -------

    file_set : list
        List of files to load.

    """
    file_set = sorted(glob.glob(rootdir + os.sep + '*_features' +
                                os.sep + '*.csv'))
    return file_set


def compileProps(rootdir, savename, z_start=0, z_stop=4000):

    propfiles = getPropfiles(rootdir)
    keys = [str(x) + '-' + str(x+500) for x
            in np.arange(z_start, len(propfiles)*500, 500)]

    total_props = []
    for file, key in zip(propfiles, keys):
        print(file, key)
        props = loadPropTable(file)
        print(type(props))
        drop_cols = [f for f in props.columns if 'unnamed' in f.lower()]
        print(drop_cols)
        props = props.drop(drop_cols, axis=1)
        props = getAspectRatios(props)
        props.to_csv(file)
        total_props.append(props)

    result = pd.concat(total_props, keys=keys, names=['Region'])
    drop_cols = [f for f in result.columns if 'unnamed' in f.lower()]
    result = result.drop(drop_cols, axis=1)

    savename = os.path.join(rootdir, savename)
    print(savename)

    result.to_csv(savename)
    return result


def sorted3DWorkflow(rootdir,
                     segmentation_dir,
                     file_title=None,
                     xy_range=None
                     ):
    """
    Runs feature extraction workflow by sorting mask files in root directory.


    Parameters
    ----------

    rootdir : str or pathlike
        Root directory where 3D intensity data (assumed HDF5) and segmented
        mask directory is stored.

    segmentation_dir : str or pathlike
        Directory where segmented masks are stored. Assumed hierarchy is that
        segmentation_dir is within rootdir.

    file_title : str or None (optional)
        Default is none. File title for hdf5 file to read segmented data.


    Returns
    -------

    """

    if file_title is None:
        h5_file = glob.glob(rootdir + os.sep + '*.h5')

    else:
        h5_file = glob.glob(rootdir + os.sep + '*%s*' % (file_title))
        print(h5_file)

    segmented_files = sorted(glob.glob(
                                       rootdir + os.sep + segmentation_dir +
                                       os.sep + '*.tif'))

    for file in segmented_files:
        split_filename = file.split('_')
        z_range = (int(split_filename[-3]), int(split_filename[-2]))
        print(z_range)

        feature3Dworkflow(file, h5_file[0], z_range, xy_range)

    compileProps(os.path.join(rootdir, segmentation_dir), 'complete_props.csv')

    return

 