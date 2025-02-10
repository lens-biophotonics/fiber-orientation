import gc
import tempfile

from multiprocessing import cpu_count
from os import environ, path, unlink
from shutil import rmtree
from time import perf_counter

import numpy as np
from astropy.visualization import make_lupton_rgb
from joblib import dump, load
from matplotlib.colors import hsv_to_rgb
from skimage.filters import (threshold_li, threshold_niblack,
                             threshold_sauvola, threshold_triangle,
                             threshold_yen)


def ceil_to_multiple(number, multiple):
    """
    Round up number to the nearest multiple.

    Parameters
    ----------
    number:
        number to be rounded

    multiple:
        the input number will be rounded
        to the nearest multiple higher than this value

    Returns
    -------
    rounded:
        rounded up number
    """
    rounded = multiple * np.ceil(number / multiple)

    return rounded


def create_background_mask(img, method='yen', black_bg=False):
    """
    Compute background mask.

    Parameters
    ----------
    img: numpy.ndarray (axis order=(Z,Y,X))
        3D microscopy image

    method: str
        image thresholding method

    black_bg: bool
        generate foreground mask

    Returns
    -------
    bg_msk: numpy.ndarray (axis order=(Z,Y,X), dtype=bool)
        boolean background mask
    """
    # select thresholding method
    if method == 'li':
        thresh = threshold_li(img)
    elif method == 'niblack':
        thresh = threshold_niblack(img, window_size=15, k=0.2)
    elif method == 'sauvola':
        thresh = threshold_sauvola(img, window_size=15, k=0.2, r=None)
    elif method == 'triangle':
        thresh = threshold_triangle(img, nbins=256)
    elif method == 'yen':
        thresh = threshold_yen(img, nbins=256)
    else:
        try:
            thresh = float(method)
        except ValueError as ve:
            raise ValueError(f"{ve}\n\n\tUnsupported Frangi filter thresholding method!") from ve

    # compute mask
    bg_msk = img >= thresh if black_bg else img < thresh

    return bg_msk


def create_memory_map(dtype, shape=None, name='tmp', tmp=None, arr=None, mmap_mode='w+'):
    """
    Create a memory-map to an array stored in a binary file on disk.

    Parameters
    ----------    
    dtype:
        data-type used to interpret the file contents

    shape: tuple
        shape of the stored array

    name: str
        optional temporary filename

    tmp: str
        temporary file directory

    arr: numpy.ndarray
        array to be mapped

    mmap_mode: str
        file opening mode

    Returns
    -------
    mmap: NumPy memory map
        memory-mapped array
    """
    if tmp is None:
        tmp = tempfile.mkdtemp()
    mmap_path = path.join(tmp, name + '.mmap')

    if path.exists(mmap_path):
        unlink(mmap_path)

    if arr is None:
        _ = open(mmap_path, mode='w+')
        mmap = np.memmap(mmap_path, dtype=dtype, mode=mmap_mode, shape=shape)
        mmap[:] = 0
    else:
        _ = dump(arr, mmap_path)
        mmap = load(mmap_path, mmap_mode=mmap_mode)
        del arr
        _ = gc.collect()

    return mmap


def delete_tmp_data(tmp_dir, tmp_data):
    """
    Delete temporary folder.

    Parameters
    ----------
    tmp_dir: str
        path to temporary folder to be removed

    tmp_data: tuple
        temporary data dictionaries

    Returns
    -------
    None
    """
    try:
        for data in tmp_data:
            for k in list(data.keys()):
                del data[k]

        rmtree(tmp_dir)

    except OSError:
        pass


def detect_ch_axis(img):
    """
    Detect image channel axis.

    Parameters
    ----------
    img: numpy.ndarray
        3D microscopy image

    Returns
    -------
    ch_ax: int
        channel axis (either 1 or 3)
    """
    if len(img.shape) < 4:
        return None

    ch_ax = (np.array(img.shape) == 3).nonzero()[0]
    ch_ax = ch_ax[np.logical_or(ch_ax == 1, ch_ax == 3)]
    if len(ch_ax) != 1:
        raise ValueError("Ambiguous image axes order: could not determine channel axis!")

    ch_ax = ch_ax[0]

    return ch_ax


def divide_nonzero(nd_array1, nd_array2, new_val=1e-10):
    """
    Divide two arrays handling zero denominator values.

    Parameters
    ----------
    nd_array1: numpy.ndarray
        dividend array

    nd_array2: numpy.ndarray
        divisor array

    new_val: float
        substituted value

    Returns
    -------
    divided: numpy.ndarray
        divided array
    """
    divisor = np.copy(nd_array2)
    divisor[divisor == 0] = new_val
    divided = np.divide(nd_array1, divisor)

    return divided


def elapsed_time(start_time):
    """
    Compute elapsed time from input start reference.

    Parameters
    ----------
    start_time: float
        start time reference

    Returns
    -------
    tot: float
        total time [s]

    hrs: int
        hours

    mins: int
        minutes

    secs: int
        seconds
    """
    stop_time = perf_counter()
    tot = stop_time - start_time

    secs = tot % 86400
    hrs = int(secs // 3600)
    secs %= 3600
    mins = int(secs // 60)
    secs = int(secs % 60)

    return tot, hrs, mins, secs


def fwhm_to_sigma(fwhm):
    """
    Compute the standard deviation of a Gaussian distribution
    from its FWHM value.

    Parameters
    ----------
    fwhm: float
        full width at half maximum

    Returns
    -------
    sigma: float
        standard deviation
    """
    sigma = np.sqrt(np.square(fwhm) / (8 * np.log(2)))

    return sigma


def get_available_cores():
    """
    Return the number of available logical cores.

    Returns
    -------
    num_cpu: int
        number of available cores
    """
    num_cpu = environ.pop('OMP_NUM_THREADS', default=None)
    num_cpu = cpu_count() if num_cpu is None else int(num_cpu)

    return num_cpu


def get_config_label(cli_args):
    """
    Generate the output filename prefix including
    pipeline configuration information.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        updated namespace of command line arguments

    Returns
    -------
    cfg_lbl: str
        Frangi filter configuration label
    """
    cfg_lbl = f'a{cli_args.alpha}_b{cli_args.beta}_g{cli_args.gamma}_t{cli_args.fb_thr}'
    for s in cli_args.scales:
        cfg_lbl += f'_s{s}'

    return cfg_lbl


def get_item_bytes(data):
    """
    Retrieve data item size in bytes.

    Parameters
    ----------
    data: numpy.ndarray or HDF5 dataset
        input data

    Returns
    -------
    bts: int
        item size in bytes
    """
    try:
        bts = int(np.iinfo(data.dtype).bits / 8)
    except ValueError:
        bts = int(np.finfo(data.dtype).bits / 8)

    return bts


def get_item_size(dtype):
    """
    Get the item size in bytes of a data type.

    Parameters
    ----------
    dtype: str
        data type identifier

    Returns
    -------
    item_sz: int
        item size in bytes
    """
    lst_1 = ['uint8', 'int8']
    lst_2 = ['uint16', 'int16', 'float16', np.float16]
    lst_3 = ['uint32', 'int32', 'float32', np.float32]
    lst_4 = ['uint64', 'int64', 'float64', np.float64]

    if dtype in lst_1:
        item_sz = 1
    elif dtype in lst_2:
        item_sz = 2
    elif dtype in lst_3:
        item_sz = 4
    elif dtype in lst_4:
        item_sz = 8
    else:
        raise ValueError("Unsupported data type!")

    return item_sz


def hsv_orient_cmap(vec_img):
    """
    Compute HSV colormap of vector orientations from 3D vector field.

    Parameters
    ----------
    vec_img: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        orientation vectors

    Returns
    -------
    rgb_map: numpy.ndarray (axis order=(Z,Y,X,C), dtype=uint8)
        orientation color map
    """
    # compute the in-plane versor length
    vy, vx = (vec_img[..., 1], vec_img[..., 2])
    vxy_abs = np.sqrt(np.square(vx) + np.square(vy))
    vxy_abs = divide_nonzero(vxy_abs, np.max(vxy_abs))

    # compute the in-plane angular orientation
    vxy_ang = normalize_angle(np.arctan2(vy, vx), lower=0, upper=np.pi, dtype=np.float32)
    vxy_ang = np.divide(vxy_ang, np.pi)

    # generate HSV orientation colormap
    rgb_map = np.zeros(shape=tuple(list(vec_img.shape[:-1]) + [3]), dtype=np.uint8)
    for z in range(vec_img.shape[0]):
        hsv_map = np.stack((vxy_ang[z], vxy_abs[z], vxy_abs[z]), axis=-1)
        rgb_map[z] = (255.0 * hsv_to_rgb(hsv_map)).astype(np.uint8)

    return rgb_map


def normalize_angle(angle, lower=0.0, upper=360.0, dtype=None):
    """
    Normalize angle to [lower, upper) range.

    Parameters
    ----------
    angle: numpy.ndarray (dtype=float)
        angular values to be normalized (in degrees)

    lower: float
        lower limit (default: 0.0)

    upper: float
        upper limit (default: 360.0)

    dtype:
        output data type

    Returns
    -------
    angle: numpy.ndarray (dtype=float)
        angular values (in degrees) normalized within [lower, upper)

    Raises
    ------
    ValueError
      if lower >= upper
    """
    # check input variables
    if lower >= upper:
        raise ValueError(f"Invalid lower and upper angular limits: ({lower}, {upper})")

    # apply corrections
    dvsr = abs(lower) + abs(upper)
    corr_1 = np.logical_or(angle > upper, angle == lower)
    angle[corr_1] = lower + np.abs(angle[corr_1] + upper) % dvsr

    corr_2 = np.logical_or(angle < lower, angle == upper)
    angle[corr_2] = upper - np.abs(angle[corr_2] - lower) % dvsr

    angle[angle == upper] = lower

    # cast to desired data type
    if dtype is not None:
        angle = angle.astype(dtype)

    return angle


def normalize_image(img, min_val=None, max_val=None, max_out_val=255.0, dtype=np.uint8):
    """
    Normalize image data.

    Parameters
    ----------
    img: numpy.ndarray
        input image

    min_val: float
        minimum input value

    max_val: float
        maximum input value

    max_out_val: float
        maximum output value

    dtype:
        output data type

    Returns
    -------
    norm_img: numpy.ndarray
        normalized image
    """
    # get min and max values
    if min_val is None:
        min_val = np.min(img)
    if max_val is None:
        max_val = np.max(img)

    # normalize
    if max_val != 0:
        if max_val != min_val:
            norm_img = (((img - min_val) / (max_val - min_val)) * max_out_val).astype(dtype)
        else:
            norm_img = ((img / max_val) * max_out_val).astype(dtype)
    else:
        norm_img = img.astype(dtype)

    return norm_img


def rgb_orient_cmap(vec_img, minimum=0, stretch=1, q=8):
    """
    Compute RGB colormap of orientation vector components from 3D vector field.

    Parameters
    ----------
    vec_img: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        n-dimensional array of orientation vectors

    minimum: int
        intensity that should be mapped to black (a scalar or array for R, G, B)

    stretch: int
        linear stretch of the image

    q: int
        asinh softening parameter

    Returns
    -------
    rgb_map: numpy.ndarray (axis order=(Z,Y,X,C), dtype=uint8)
        orientation color map
    """
    # initialize colormap
    vec_img = np.abs(vec_img)
    rgb_map = np.zeros(shape=vec_img.shape, dtype=np.uint8)
    for z in range(vec_img.shape[0]):

        # generate RGB orientation colormap
        img_r, img_g, img_b = (vec_img[z, :, :, 2], vec_img[z, :, :, 1], vec_img[z, :, :, 0])
        rgb_map[z] = make_lupton_rgb(img_r, img_g, img_b, minimum=minimum, stretch=stretch, Q=q)

    return rgb_map


def transform_axes(nd_array, flipped=None, swapped=None, expand=None):
    """
    Manipulate axes and dimensions of the input array.
    The transformation sequence is:
    axes flip >>> axes swap >>> dimensions expansion.

    Parameters
    ----------
    nd_array: numpy.ndarray
        input data array

    swapped: tuple (dtype=int)
        axes to be swapped

    flipped: tuple (dtype=int)
        axes to be flipped

    expand: int
        insert new axis at this position

    Returns
    -------
    nd_array: numpy.ndarray
        transformed data array
    """
    if flipped is not None:
        nd_array = np.flip(nd_array, axis=flipped)

    if swapped is not None:
        swap_src, swap_dest = swapped
        nd_array = np.swapaxes(nd_array, swap_src, swap_dest)

    if expand is not None:
        nd_array = np.expand_dims(nd_array, axis=expand)

    return nd_array
