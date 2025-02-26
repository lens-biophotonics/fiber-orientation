from itertools import product

import numpy as np
import psutil

from numba import njit
from foa3d.utils import get_available_cores


def adjust_axis_range(ax_iter, slc_shp, img_shp, slc_per_ax, start, stop, ovlp=0, shp_thr=7):
    """
    Trim slice axis range at image boundaries
    and adjust padded ranges accordingly.

    Parameters
    ----------
    ax_iter: int
        iteration counter along axis

    slc_shp: int
        slice shape along axis [px]

    img_shp: int
        total image shape along axis [px]

    slc_per_ax: int
        slices along axis

    start: int
        start coordinate [px]

    stop: int
        stop coordinate [px]

    ovlp: int
        overlapping range between slices along each axis [px]

    shp_thr: int
        minimum slice side [px]

    Returns
    -------
    start: int
        adjusted start coordinate [px]

    stop: int
        adjusted stop coordinate [px]

    pad: numpy.ndarray (shape=(2,), dtype=int)
        lower and upper padding ranges [px]
    """
    # adjust start coordinate
    pad = np.zeros((2,), dtype=np.int64)
    start -= ovlp
    if start < 0:
        pad[0] = -start
        start = 0

    # adjust stop coordinate
    stop += ovlp
    if stop > img_shp:
        pad[1] = stop - img_shp
        stop = img_shp

    # handle image residuals at boundaries
    if ax_iter == slc_per_ax - 2:
        if np.remainder(img_shp, slc_shp) < shp_thr:
            pad[1] = ovlp
            stop = img_shp

    elif ax_iter == slc_per_ax - 1:
        if np.remainder(img_shp, slc_shp) < shp_thr and slc_per_ax > 1:
            stop = None

    return start, stop, pad


def check_background(img, ts_msk=None, ts_thr=1e-4):
    """
    Check whether the input image predominantly includes background voxels.

    Parameters
    ----------
    img: numpy.ndarray (axis order=(Z,Y,X))

    ts_msk: numpy.ndarray (axis order=(Z,Y,X), dtype=bool)
        tissue reconstruction binary mask

    ts_thr: float
        relative threshold on non-zero voxels

    Returns
    -------
    not_bg: bool
        foreground boolean flag
    """
    not_bg = np.count_nonzero(ts_msk) / np.prod(ts_msk.shape) > ts_thr if ts_msk is not None else np.max(img) != 0

    return not_bg


def compute_axis_range(ax, ax_iter, slc_shp, img_shp, slc_per_dim, ovlp=0, flip=False, shp_thr=7):
    """
    Adjust image slice coordinates at boundaries.

    Parameters
    ----------
    ax: int
        image axis

    ax_iter: int
        iteration counter along axis (see pipeline.iterate_frangi_on_slices)

    slc_shp: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices
        analyzed using parallel threads [px]

    img_shp: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    slc_per_dim: numpy.ndarray (shape=(3,), dtype=int)
        image slices along each axis

    ovlp: int
        overlapping range between slices along each axis [px]

    flip: bool
        flip axes

    shp_thr: int
        minimum slice side [px]

    Returns
    -------
    start: int
        start index [px]

    stop: int
        stop index [px]

    pad: numpy.ndarray (shape=(2,), dtype=int)
        lower and upper padding ranges [px]
    """
    # compute start and stop coordinates
    start = ax_iter[ax] * slc_shp[ax]
    stop = start + slc_shp[ax]

    # flip coordinates if required
    if flip:
        start_tmp = start
        start = img_shp[ax] - stop
        stop = img_shp[ax] - start_tmp

    # adjust start and stop coordinates
    start, stop, pad = adjust_axis_range(ax_iter[ax], slc_shp[ax], img_shp[ax], slc_per_dim[ax], start, stop,
                                         ovlp=ovlp, shp_thr=shp_thr)

    return start, stop, pad


def compute_slice_range(ax_iter, slc_shp, img_shp, slc_per_dim, ovlp=0, flip=False, shp_thr=7):
    """
    Compute basic slice coordinates from microscopy volumetric image.

    Parameters
    ----------
    ax_iter: tuple
        iteration counters along each axis

    slc_shp: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices
        analyzed using parallel threads [px]

    img_shp: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    slc_per_dim: numpy.ndarray (shape=(3,), dtype=int)
        image slices along each axis

    ovlp: int
        overlapping range between slices along each axis [px]

    flip: bool
        flip axes

    shp_thr: int
        minimum slice side [px]

    Returns
    -------
    rng: np.ndarray
        3D slice index ranges [px]

    pad: np.ndarray (shape=(3,2), dtype=int)
        slice padding range [px]
    """
    # generate axis range and padding array
    dim = len(ax_iter)
    slc = tuple()
    pad = np.zeros((dim, 2), dtype=np.int64)
    for ax in range(dim):
        start, stop, pad[ax] = compute_axis_range(ax, ax_iter, slc_shp, img_shp, slc_per_dim,
                                                  ovlp=ovlp, flip=flip, shp_thr=shp_thr)
        if stop is not None:
            slc += (slice(start, stop, 1),)
        else:
            slc += (None,)

    # generate range array
    rng = np.index_exp[slc]

    return rng, pad


def compute_overlap(smooth_sigma, frangi_sigma_um, rsz_ratio=None, trunc=2):
    """
    Compute lateral slice extension range
    for coping with smoothing-related boundary artifacts.

    Parameters
    ----------
    smooth_sigma: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of the smoothing Gaussian filter [px]

    frangi_sigma_um: numpy.ndarray (dtype=float)
        analyzed spatial scales [μm]

    rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D image resize ratio

    trunc: int
        neglect the Gaussian smoothing kernel
        after this many standard deviations

    Returns
    -------
    ovlp: int
        overlapping range between image slices along each axis [px]
    """
    sigma = np.max(frangi_sigma_um) / rsz_ratio
    if smooth_sigma is not None:
        sigma = np.concatenate((smooth_sigma, sigma))

    ovlp = int(np.ceil(2 * trunc * np.max(sigma)) // 2)

    return ovlp


def compute_slice_shape(img_shp, item_sz, slc_sz=1e6, ovlp=0):
    """
    Compute basic image chunk shape depending on its maximum size (in bytes).

    Parameters
    ----------
    img_shp: numpy.ndarray (shape=(3,), dtype=int)
        total image shape [px]

    item_sz: int
        image item size [B]

    slc_sz: float
        maximum memory size of the basic image slices
        analyzed using parallel threads [B]

    ovlp: int
        overlapping range between image slices along each axis [px]

    Returns
    -------
    slc_shp: numpy.ndarray (shape=(3,), dtype=int)
        shape of the basic image slices
        analyzed using parallel threads [px]
    """
    if len(img_shp) == 4:
        item_sz *= 3

    tot_ovlp = 2 * ovlp
    side_xyz = np.floor((slc_sz / item_sz)**(1/3)).astype(int)
    side_z = min(img_shp[0] + tot_ovlp, side_xyz)
    side_xy = np.floor(np.sqrt(np.abs(slc_sz / side_z / item_sz))).astype(int)
    slc_shp = np.array([side_z, side_xy, side_xy]) - tot_ovlp

    return slc_shp


def crop(img, rng, ovlp=None, flip=()):
    """
    Shrink image slice at total volume boundaries, for overall shape consistency.

    Parameters
    ----------
    img: numpy.ndarray (axis order=(Z,Y,X))
        3D microscopy image

    rng: tuple
        3D slice index range

    ovlp: numpy.ndarray (axis order=(Z,Y,X), dtype=int)
        overlapping range between image slices [px]

    flip: tuple
        flipped axes

    Returns
    -------
    img_out: numpy.ndarray
        cropped microscopy image
    """
    # delete overlapping boundaries
    if ovlp is not None:
        img = img[ovlp[0]:img.shape[0] - ovlp[0], ovlp[1]:img.shape[1] - ovlp[1], ovlp[2]:img.shape[2] - ovlp[2]]

    # check image shape and output index ranges
    crop_rng = np.zeros(shape=(3,), dtype=np.int64)
    for s in range(3):
        crop_rng[s] = img.shape[s] - np.arange(rng[s].start, rng[s].stop, rng[s].step).size

    # crop image if required
    img_out = img[crop_rng[0] or None:, ...] if 0 in flip else img[:-crop_rng[0] or None, ...]
    img_out = img_out[:, crop_rng[1] or None:, ...] if 1 in flip else img_out[:, :-crop_rng[1] or None, ...]
    img_out = img_out[:, :, crop_rng[2] or None:, ...] if 2 in flip else img_out[:, :, :-crop_rng[2] or None, ...]

    return img_out


def crop_img_dict(img_dict, rng, ovlp=None, flip=()):
    """
    Shrink list of image slices at total volume boundaries, for overall shape consistency.

    Parameters
    ----------
    img_dict: list
        dictionary of image data to be cropped

    rng: tuple
        3D slice index range

    ovlp: int
        overlapping range between image slices along each axis [px]

    flip: tuple
        flipped axes

    Returns
    -------
    img_dict: list
        dictionary of cropped image slices
    """
    for key in img_dict.keys():
        if img_dict[key] is not None:
            img_dict[key] = crop(img_dict[key], rng, ovlp=ovlp, flip=flip)

    return img_dict


def generate_slice_ranges(in_img, cfg):
    """
    Generate image slice ranges for the Frangi filtering stage.

    Parameters
    ----------
    in_img: dict
        input image dictionary

            data: numpy.ndarray or NumPy memory-map object (axis order=(Z,Y,X) or (Z,Y,X,C) or (Z,C,Y,X))
                3D microscopy image

            ts_msk: numpy.ndarray (dtype=bool)
                tissue reconstruction binary mask

            ch_ax: int
                RGB image channel axis (either 1, 3, or None for grayscale images)

            fb_ch: int
                neuronal fibers channel

            bc_ch: int
                brain cell soma channel

            msk_bc: bool
                if True, mask neuronal bodies within
                the optionally provided channel

            psf_fwhm: numpy.ndarray (shape=(3,), dtype=float)
                3D FWHM of the PSF [μm]

            px_sz: numpy.ndarray (shape=(3,), dtype=float)
                pixel size [μm]

            name: str
                name of the 3D microscopy image

            is_vec: bool
                vector field flag

            shape: numpy.ndarray (shape=(3,), dtype=int)
                total image shape

            shape_um: numpy.ndarray (shape=(3,), dtype=float)
                total image shape [μm]

            item_sz: int
                image item size [B]

    cfg: dict
        Frangi filter configuration

            alpha: float
                plate-like score sensitivity

            beta: float
                blob-like score sensitivity

            gamma: float
                background score sensitivity

            scales_px: numpy.ndarray (dtype=float)
                Frangi filter scales [px]

            scales_um: numpy.ndarray (dtype=float)
                Frangi filter scales [μm]

            smooth_sd: numpy.ndarray (shape=(3,), dtype=int)
                3D standard deviation of the smoothing Gaussian filter [px]

            px_sz: numpy.ndarray (shape=(3,), dtype=float)
                pixel size [μm]

            bc_ch: int
                neuronal bodies channel

            fb_ch: int
                myelinated fibers channel

            msk_bc: bool
                if True, mask neuronal bodies within
                the optionally provided channel

            hsv_cmap: bool
                generate HSV colormap of 3D fiber orientations

            exp_all: bool
                export all images

            rsz: numpy.ndarray (shape=(3,), dtype=float)
                3D image resize ratio

            ram: float
                    maximum RAM available to the Frangi filter stage [B]

            jobs: int
                number of parallel jobs (threads)
                used by the Frangi filter stage

            batch: int
                slice batch size

            slc_shp: numpy.ndarray (shape=(3,), dtype=int)
                shape of the basic image slices
                analyzed using parallel threads [px]

            ovlp: int
                overlapping range between image slices along each axis [px]

            tot_slc: int
                total number of image slices

            z_out: NumPy slice object
                output z-range

    Returns
    -------
    slc_rng: list of dictionaries
        in: np.ndarray
            3D input slice range [px]

        pad: np.ndarray
            3D slice padding range [px]

        out: np.ndarray
            3D output slice range [px]

        bc: np.ndarray
            (optional) brain cell soma slice range
    """
    # compute i/o image slice index ranges
    out_slc_shp = np.ceil(np.multiply(cfg['rsz'], cfg['slc_shp'])).astype(int)
    out_img_shp = np.ceil(np.multiply(cfg['rsz'], in_img['shape'])).astype(int)
    slc_per_dim = np.ceil(np.divide(in_img['shape'], cfg['slc_shp'])).astype(int)
    slc_rng = []
    for zyx in product(range(slc_per_dim[0]), range(slc_per_dim[1]), range(slc_per_dim[2])):
        in_rng, pad = compute_slice_range(zyx, cfg['slc_shp'], in_img['shape'], slc_per_dim, ovlp=cfg['ovlp'])

        if None not in in_rng:
            out_rng, _ = compute_slice_range(zyx, out_slc_shp, out_img_shp, slc_per_dim)

            # (optional) neuronal body channel
            if in_img['msk_bc']:
                bc_rng, _ = compute_slice_range(zyx, cfg['slc_shp'], in_img['shape'], slc_per_dim)
            else:
                bc_rng = None

            slc_rng.append({'in': in_rng, 'out': out_rng, 'pad': pad, 'bc': bc_rng})

    # get total number of processed slices
    cfg.update({'tot_slc': len(slc_rng)})

    return slc_rng


def get_slicing_config(in_img, frangi_cfg, mem_growth=149.7, shp_thr=7):
    """
    Compute size and number of the batches of basic microscopy image slices
    analyzed in parallel.

    Parameters
    ----------
    in_img: dict
        input image dictionary

            data: numpy.ndarray or NumPy memory-map object (axis order=(Z,Y,X) or (Z,Y,X,C) or (Z,C,Y,X))
                3D microscopy image

            ts_msk: numpy.ndarray (dtype=bool)
                tissue reconstruction binary mask

            ch_ax: int
                RGB image channel axis (either 1, 3, or None for grayscale images)

            fb_ch: int
                neuronal fibers channel

            bc_ch: int
                brain cell soma channel

            msk_bc: bool
                if True, mask neuronal bodies within
                the optionally provided channel

            psf_fwhm: numpy.ndarray (shape=(3,), dtype=float)
                3D FWHM of the PSF [μm]

            px_sz: numpy.ndarray (shape=(3,), dtype=float)
                pixel size [μm]

            name: str
                name of the 3D microscopy image

            is_vec: bool
                vector field flag

            shape: numpy.ndarray (shape=(3,), dtype=int)
                total image shape

            shape_um: numpy.ndarray (shape=(3,), dtype=float)
                total image shape [μm]

            item_sz: int
                image item size [B]

    frangi_cfg: dict
        Frangi filter configuration

            alpha: float
                plate-like score sensitivity

            beta: float
                blob-like score sensitivity

            gamma: float
                background score sensitivity

            scales_px: numpy.ndarray (dtype=float)
                Frangi filter scales [px]

            scales_um: numpy.ndarray (dtype=float)
                Frangi filter scales [μm]

            smooth_sd: numpy.ndarray (shape=(3,), dtype=int)
                3D standard deviation of the smoothing Gaussian filter [px]

            px_sz: numpy.ndarray (shape=(3,), dtype=float)
                pixel size [μm]

            bc_ch: int
                neuronal bodies channel

            fb_ch: int
                myelinated fibers channel

            msk_bc: bool
                if True, mask neuronal bodies within
                the optionally provided channel

            hsv_cmap: bool
                generate HSV colormap of 3D fiber orientations

            exp_all: bool
                export all images

            rsz: numpy.ndarray (shape=(3,), dtype=float)
                3D image resize ratio

            ram: float
                    maximum RAM available to the Frangi filter stage [B]

            jobs: int
                number of parallel jobs (threads)
                used by the Frangi filter stage

    mem_growth: float
        empirical memory growth factor
        of the Frangi filtering stage

    shp_thr: int
        minimum slice side [px]

    Returns
    -------
    None
    """
    # maximum RAM not provided: use all
    ram = frangi_cfg['ram']
    if ram is None:
        ram = psutil.virtual_memory()[1]
    ram /= mem_growth

    # number of logical cores
    jobs = frangi_cfg['jobs']
    num_cpu = get_available_cores()
    if jobs is None:
        jobs = num_cpu

    # compute the shape of basic microscopy image slices
    slc_shp = np.array([-1])
    batch_sz = np.min([jobs, num_cpu]).astype(int) + 1
    ovlp = compute_overlap(frangi_cfg['smooth_sd'], frangi_cfg['scales_um'], rsz_ratio=frangi_cfg['rsz'])
    while np.any(slc_shp < shp_thr):
        batch_sz -= 1
        if batch_sz == 0:
            raise ValueError(
                "Basic image slices do not fit the available RAM: decrease spatial scales and/or maximum scale value!")

        slc_shp = compute_slice_shape(in_img['shape'], in_img['item_sz'], slc_sz=ram/batch_sz, ovlp=ovlp)

    # update Frangi filter configuration dictionary
    tot_slc = np.prod(np.floor(np.divide(in_img['shape'], slc_shp)).astype(int))
    frangi_cfg.update({'batch': min(batch_sz, tot_slc), 'slc_shp': slc_shp, 'ovlp': ovlp})


def slice_image(img, rng, ch_ax, ch, ts_msk=None):
    """
    Slice desired channel from input volumetric image.

    Parameters
    ----------
    img: numpy.ndarray
        3D microscopy image

    rng: tuple (dtype=int)
        3D index ranges

    ch: int
        channel

    ch_ax: int
        RGB image channel axis (either 1 or 3, or None for grayscale images)

    ts_msk: numpy.ndarray (dtype=bool)
        tissue binary mask

    Returns
    -------
    img_slc: numpy.ndarray
        image slice

    ts_msk_slc: numpy.ndarray (dtype=bool)
        tissue mask slice
    """
    z_rng, r_rng, c_rng = rng

    if ch_ax is None:
        img_slc = img[z_rng, r_rng, c_rng]
    elif ch_ax == 1:
        img_slc = img[z_rng, ch, r_rng, c_rng]
    else:
        img_slc = img[z_rng, r_rng, c_rng, ch]

    ts_msk_slc = ts_msk[r_rng, c_rng] if ts_msk is not None else None

    return img_slc, ts_msk_slc
