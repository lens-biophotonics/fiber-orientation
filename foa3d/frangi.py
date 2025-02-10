# portions of code adapted from https://github.com/ellisdg/frangi3d (MIT license)
from itertools import combinations_with_replacement

import numpy as np
from scipy import ndimage as ndi

from foa3d.utils import (create_background_mask, create_memory_map,
                         divide_nonzero, hsv_orient_cmap, rgb_orient_cmap)


def analyze_hessian_eigen(img, sigma, trunc=4):
    """
    Compute the eigenvalues of local Hessian matrices
    of the input image array, sorted by absolute value (in ascending order),
    along with the related (dominant) eigenvectors.

    Parameters
    ----------
    img: numpy.ndarray (axis order=(Z,Y,X))
        3D microscopy image

    sigma: int
        spatial scale [px]

    trunc: int
        truncate the Gaussian smoothing kernel at this many standard deviations

    Returns
    -------
    eigval: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        Hessian eigenvalues sorted by absolute value (ascending order)

    dom_eigvec: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        Hessian eigenvectors related to the dominant (minimum) eigenvalue
    """
    hessian = compute_scaled_hessian(img, sigma=sigma, trunc=trunc)
    eigval, dom_eigvec = compute_dominant_eigen(hessian)

    return eigval, dom_eigvec


def compute_dominant_eigen(hessian):
    """
    Compute the eigenvalues (sorted by absolute value)
    of symmetrical Hessian matrix, selecting the eigenvectors related to the dominant ones.

    Parameters
    ----------
    hessian: numpy.ndarray (axis order=(Z,Y,X,C,C), dtype=float)
        input array of local Hessian matrices

    Returns
    -------
    srt_eigval: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        Hessian eigenvalues sorted by absolute value (ascending order)

    dom_eigvec: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        Hessian eigenvectors related to the dominant (minimum) eigenvalue
    """
    eigenval, eigenvec = np.linalg.eigh(hessian)
    srt_eigval, srt_eigvec = sort_eigen(eigenval, eigenvec)
    dom_eigvec = srt_eigvec[..., 0]

    return srt_eigval, dom_eigvec


def compute_fractional_anisotropy(eigenval):
    """
    Compute structure tensor fractional anisotropy
    as in Schilling et al. (2018).

    Parameters
    ----------
    eigenval: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        structure tensor eigenvalues (at the best local spatial scale)

    Returns
    -------
    fa: numpy.ndarray (shape=(3,), dtype=float)
        fractional anisotropy
    """
    fa = np.sqrt(0.5 * divide_nonzero(
                 np.square((eigenval[..., 0] - eigenval[..., 1])) +
                 np.square((eigenval[..., 0] - eigenval[..., 2])) +
                 np.square((eigenval[..., 1] - eigenval[..., 2])),
                 np.sum(eigenval ** 2, axis=-1)))

    return fa


def compute_frangi_features(eigen1, eigen2, eigen3, gamma):
    """
    Compute the basic image features employed by the Frangi filter.

    Parameters
    ----------
    eigen1: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        lowest Hessian eigenvalue (i.e., the dominant eigenvalue)

    eigen2: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        middle Hessian eigenvalue

    eigen3: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        highest Hessian eigenvalue

    gamma: float
        background score sensitivity

    Returns
    -------
    ra: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        plate-like object score

    rb: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        blob-like object score

    s: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
       second-order structureness

    gamma: float
        background score sensitivity
        (automatically computed if not provided as input)
    """
    ra = divide_nonzero(np.abs(eigen2), np.abs(eigen3))
    rb = divide_nonzero(np.abs(eigen1), np.sqrt(np.abs(np.multiply(eigen2, eigen3))))
    s = compute_structureness(eigen1, eigen2, eigen3)

    # compute default gamma sensitivity
    if gamma is None:
        gamma = 0.5 * np.max(s)

    return ra, rb, s, gamma


def compute_scaled_hessian(img, sigma=1, trunc=4):
    """
    Computes the scaled and normalized Hessian matrices of the input image.
    This is then used to estimate Frangi's vesselness probability score.

    Parameters
    ----------
    img: numpy.ndarray (axis order=(Z,Y,X))
        3D microscopy image

    sigma: int
        spatial scale [px]

    trunc: int
        truncate the Gaussian smoothing kernel at this many standard deviations

    Returns
    -------
    hessian: numpy.ndarray (axis order=(Z,Y,X,C,C), dtype=float)
        Hessian matrix of image second derivatives
    """
    # scale selection
    scaled_img = ndi.gaussian_filter(img, sigma=sigma, output=np.float32, truncate=trunc)

    # compute the first order gradients
    gradient_list = np.gradient(scaled_img)

    # compute the Hessian matrix elements
    hessian_elements = [np.gradient(gradient_list[ax0], axis=ax1)
                        for ax0, ax1 in combinations_with_replacement(range(img.ndim), 2)]

    # scale the elements of the Hessian matrix
    corr_factor = sigma ** 2
    hessian_elements = [corr_factor * element for element in hessian_elements]

    # create the Hessian matrix from its basic elements
    hessian = np.zeros((img.ndim, img.ndim) + scaled_img.shape, dtype=scaled_img.dtype)
    for index, (ax0, ax1) in enumerate(combinations_with_replacement(range(img.ndim), 2)):
        element = hessian_elements[index]
        hessian[ax0, ax1, ...] = element
        if ax0 != ax1:
            hessian[ax1, ax0, ...] = element

    # re-arrange axes
    hessian = np.moveaxis(hessian, (0, 1), (-2, -1))

    return hessian


def compute_plate_like_score(ra, alpha):
    return 1 - np.exp(np.divide(np.negative(np.square(ra)), 2 * np.square(alpha)))


def compute_blob_like_score(rb, beta):
    return np.exp(np.divide(np.negative(np.square(rb)), 2 * np.square(beta)))


def compute_background_score(s, gamma):
    return 1 - np.exp(np.divide(np.negative(np.square(s)), 2 * np.square(gamma)))


def compute_structureness(eigen1, eigen2, eigen3):
    return np.sqrt(np.square(eigen1) + np.square(eigen2) + np.square(eigen3))


def compute_scaled_vesselness(eigen1, eigen2, eigen3, alpha, beta, gamma):
    """
    Estimate Frangi's vesselness probability.

    Parameters
    ----------
    eigen1: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        lowest Hessian eigenvalue (i.e., the dominant eigenvalue)

    eigen2: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        middle Hessian eigenvalue

    eigen3: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        highest Hessian eigenvalue

    alpha: float
        plate-like score sensitivity

    beta: float
        blob-like score sensitivity

    gamma: float
        background score sensitivity

    Returns
    -------
    vesselness: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        Frangi's vesselness likelihood image
    """
    ra, rb, s, gamma = compute_frangi_features(eigen1, eigen2, eigen3, gamma)
    plate = compute_plate_like_score(ra, alpha)
    blob = compute_blob_like_score(rb, beta)
    background = compute_background_score(s, gamma)
    vesselness = plate * blob * background

    return vesselness


def compute_scaled_orientation(scale_px, img, alpha=0.001, beta=1, gamma=None):
    """
    Compute fiber orientation vectors at the input spatial scale of interest

    Parameters
    ----------
    scale_px: int
        spatial scale [px]

    img: numpy.ndarray (axis order=(Z,Y,X))
        3D microscopy image

    alpha: float
        plate-like score sensitivity

    beta: float
        blob-like score sensitivity

    gamma: float
        background score sensitivity

    Returns
    -------
    frangi_img: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        Frangi's vesselness likelihood image

    eigvec: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        3D orientation map at the input spatial scale

    eigval: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        Hessian eigenvalues sorted by absolute value (ascending order)
    """
    # compute local Hessian matrices and perform eigenvalue decomposition
    eigval, eigvec = analyze_hessian_eigen(img, scale_px)

    # compute Frangi's vesselness probability image
    eigen1, eigen2, eigen3 = eigval
    vesselness = compute_scaled_vesselness(eigen1, eigen2, eigen3, alpha=alpha, beta=beta, gamma=gamma)
    frangi_img = reject_vesselness_background(vesselness, eigen2, eigen3)
    eigval = np.stack(eigval, axis=-1)

    return frangi_img, eigvec, eigval


def init_frangi_arrays(in_img, cfg, tmp_dir):
    """
    Initialize the output datasets of the Frangi filter stage.

    Parameters
    ----------
    in_img: dict
        input image dictionary

            data: numpy.ndarray (axis order=(Z,Y,X) or (Z,Y,X,C) or (Z,C,Y,X))
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

    tmp_dir: str
        path to temporary folder

    Returns
    -------
    out_img: dict
        output image dictionary

            vec: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float32)
                initialized fiber orientation 3D image

            clr: numpy.ndarray (axis order=(Z,Y,X,C), dtype=uint8)
                initialized orientation colormap image

            fa: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
                initialized fractional anisotropy image

            frangi: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
                initialized Frangi-enhanced image

            iso: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
                initialized fiber image (isotropic resolution)

            fbr_msk: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
                initialized fiber mask image

            bc_msk: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
                initialized soma mask image
    """
    # output shape
    img_shp = in_img['shape'].copy()
    img_shp[0] = cfg['z_out'].stop - cfg['z_out'].start
    rsz_ratio = np.divide(in_img['px_sz'], cfg['px_sz'])
    tot_shp = tuple(np.ceil(rsz_ratio * img_shp).astype(int))
    vec_shp = tot_shp + (len(img_shp),)
    cfg.update({'rsz': rsz_ratio})

    # fiber channel arrays
    iso_fbr = create_memory_map('uint8', shape=tot_shp, name='iso', tmp=tmp_dir)
    if cfg['exp_all']:
        frangi = create_memory_map('uint8', shape=tot_shp, name='frangi', tmp=tmp_dir)
        fbr_msk = create_memory_map('uint8', shape=tot_shp, name='fbr_msk', tmp=tmp_dir)
        fa = create_memory_map('float32', shape=tot_shp, name='fa', tmp=tmp_dir)

        # soma channel array
        if in_img['msk_bc']:
            bc_msk = create_memory_map('uint8', shape=tot_shp, name='bc_msk', tmp=tmp_dir)
        else:
            bc_msk = None
    else:
        frangi, fbr_msk, fa, bc_msk = 4 * (None,)

    # fiber orientation arrays
    fbr_vec = create_memory_map('float32', shape=vec_shp, name='vec', tmp=tmp_dir)
    fbr_clr = create_memory_map('uint8', shape=vec_shp, name='clr', tmp=tmp_dir)

    # fill output image dictionary
    out_img = {'vec': fbr_vec, 'clr': fbr_clr, 'fa': fa, 'frangi': frangi,
               'iso': iso_fbr, 'fbr_msk': fbr_msk, 'bc_msk': bc_msk, 'px_sz': cfg['px_sz']}

    return out_img


def frangi_filter(img, scales_px=1, alpha=0.001, beta=1.0, gamma=None, hsv=False, _fa=False):
    """
    Apply 3D Frangi filter to 3D microscopy image.

    Parameters
    ----------
    img: numpy.ndarray (axis order=(Z,Y,X))
        3D microscopy image

    scales_px: int or numpy.ndarray (dtype=int)
        analyzed spatial scales [px]

    alpha: float
        plate-like score sensitivity

    beta: float
        blob-like score sensitivity

    gamma: float
        background score sensitivity (if None, gamma is automatically tailored)

    hsv: bool
        generate an HSV colormap of 3D fiber orientations

    _fa: bool
        compute fractional anisotropy

    Returns
    -------
    out_slc: dict
        slice output data

            vec: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
                3D fiber orientation field

            clr: numpy.ndarray (axis order=(Z,Y,X,C), dtype=uint8)
                orientation colormap image

            fa: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
                fractional anisotropy image

            frangi: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
                Frangi's vesselness likelihood image
    """
    # single-scale or parallel multi-scale vesselness analysis
    ns = len(scales_px)
    frangi = np.zeros((ns,) + img.shape, dtype='float32')
    eigvec = np.zeros((ns,) + img.shape + (3,), dtype='float32')
    eigval = np.zeros((ns,) + img.shape + (3,), dtype='float32')
    for s in range(ns):
        frangi[s], eigvec[s], eigval[s] = \
            compute_scaled_orientation(scales_px[s], img, alpha=alpha, beta=beta, gamma=gamma)

    # get maximum response across the requested scales
    max_idx = np.expand_dims(np.argmax(frangi, axis=0), axis=0)
    frangi = np.take_along_axis(frangi, max_idx, axis=0).squeeze(axis=0)

    # select fiber orientation vectors (and the associated eigenvalues) among different scales
    max_idx = np.expand_dims(max_idx, axis=-1)
    eigval = np.take_along_axis(eigval, max_idx, axis=0).squeeze(axis=0)
    fbr_vec = np.take_along_axis(eigvec, max_idx, axis=0).squeeze(axis=0)

    # compute fractional anisotropy image and fiber orientation color map
    fa = compute_fractional_anisotropy(eigval) if _fa else None
    fbr_clr = hsv_orient_cmap(fbr_vec) if hsv else rgb_orient_cmap(fbr_vec)

    # fill slice output dictionary
    out_slc = {'vec': fbr_vec, 'clr': fbr_clr, 'fa': fa, 'frangi': frangi}

    return out_slc


def mask_background(out_slc, ref_img=None, method='yen', invert=False, ornt_keys=('vec', 'clr', 'fa')):
    """
    Mask fiber orientation data arrays.

    Parameters
    ----------
    out_slc: dict
        slice output dictionary

            vec: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
                3D fiber orientation field

            clr: numpy.ndarray (axis order=(Z,Y,X,C), dtype=uint8)
                orientation colormap

            fa: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
                fractional anisotropy

            frangi: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
                Frangi-enhanced image slice (fiber probability image)

            iso: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
                isotropic fiber image slice

            ts_msk: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
                tissue mask slice

            fbr_msk: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
                fiber mask slice

            bc_msk: numpy.ndarray (axis order=(Z,Y,X), dtype=bool)
                brain cell mask slice

            rng: NumPy slice object
                output range

    ref_img: numpy.ndarray (axis order=(Z,Y,X)
        reference image used for thresholding

    method: str
        thresholding method (refer to skimage.filters)

    invert: bool
        mask inversion flag

    ornt_keys: tuple
        fiber orientation data keys

    Returns
    -------
    out_slc: dict

    bg: numpy.ndarray (axis order=(Z,Y,X), dtype=bool)
        background mask
    """
    # generate background mask
    if ref_img is None:
        ref_img = out_slc['frangi']
    bg = create_background_mask(ref_img, method=method)

    # apply tissue reconstruction mask, when provided
    if out_slc['ts_msk'] is not None:
        bg = np.logical_or(bg, np.logical_not(out_slc['ts_msk']))

    # invert mask
    if invert:
        bg = np.logical_not(bg)

    # apply mask to input orientation data dictionary
    for key in out_slc.keys():
        if key in ornt_keys and out_slc[key] is not None:
            if out_slc[key].ndim == 3:
                out_slc[key][bg] = 0
            else:
                out_slc[key][bg, :] = 0

    return out_slc, bg


def reject_vesselness_background(vesselness, eigen2, eigen3):
    """
    Reject the fiber background, exploiting the sign of the "secondary"
    eigenvalues λ2 and λ3.

    Parameters
    ----------
    vesselness: numpy.ndarray (axis order=(Z,Y,X))
        Frangi's vesselness likelihood image

    eigen2: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        middle Hessian eigenvalue

    eigen3: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        highest Hessian eigenvalue

    Returns
    -------
    vesselness: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        masked Frangi's vesselness likelihood image
    """
    bg_msk = np.logical_or(np.logical_or(eigen2 > 0, eigen3 > 0), np.isnan(vesselness))
    vesselness[bg_msk] = 0

    return vesselness


def sort_eigen(eigval, eigvec, axis=-1):
    """
    Sort eigenvalue and related eigenvector arrays
    by absolute value along the given axis.

    Parameters
    ----------
    eigval: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        original eigenvalue array

    eigvec: numpy.ndarray (axis order=(Z,Y,X,C,C), dtype=float)
        original eigenvector array

    axis: int
        sorted axis

    Returns
    -------
    srt_eigval: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        sorted eigenvalue array (ascending order)

    srt_eigvec: numpy.ndarray (axis order=(Z,Y,X,C,C), dtype=float)
        sorted eigenvector array
    """
    # sort the eigenvalue array by absolute value (ascending order)
    srt_val_idx = np.abs(eigval).argsort(axis)
    srt_eigval = np.take_along_axis(eigval, srt_val_idx, axis)
    srt_eigval = [np.squeeze(eigval, axis=axis) for eigval in np.split(srt_eigval, srt_eigval.shape[axis], axis=axis)]

    # sort related eigenvectors consistently
    srt_vec_idx = srt_val_idx[:, :, :, np.newaxis, :]
    srt_eigvec = np.take_along_axis(eigvec, srt_vec_idx, axis)

    return srt_eigval, srt_eigvec


def write_frangi_arrays(out_img, out_slc, rng, z_out=None):
    """
    Fill the output arrays of the Frangi filter stage.

    Parameters
    ----------
    out_img: dict
        output image dictionary

            vec: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float32)
                fiber orientation vector field

            clr: numpy.ndarray (axis order=(Z,Y,X,C), dtype=uint8)
                orientation colormap image

            fa: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
                fractional anisotropy image

            frangi: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
                Frangi-enhanced image (fiber probability image)

            iso: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
                isotropic fiber image

            fbr_msk: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
                fiber mask image

            bc_msk: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
                soma mask image

            px_sz: numpy.ndarray (shape=(3,), dtype=float)
                output pixel size [μm]

    out_slc: dict
        slice output dictionary

            vec: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
                3D fiber orientation field

            clr: numpy.ndarray (axis order=(Z,Y,X,C), dtype=uint8)
                orientation colormap

            fa: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
                fractional anisotropy

            frangi: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
                Frangi-enhanced image slice (fiber probability image)

            iso: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
                isotropic fiber image slice

            ts_msk: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
                tissue mask slice

            fbr_msk: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
                fiber mask slice

            bc_msk: numpy.ndarray (axis order=(Z,Y,X), dtype=bool)
                brain cell mask slice

    rng: NumPy slice object
        output range

    z_out: NumPy slice object
        output z-range

    Returns
    -------
    None
    """
    vec_rng = tuple(np.append(rng, slice(0, 3, 1)))
    out_img['vec'][vec_rng] = out_slc['vec'][z_out, ...]
    out_img['clr'][vec_rng] = out_slc['clr'][z_out, ...]
    out_img['iso'][rng] = out_slc['iso'][z_out, ...].astype(np.uint8)

    # optional output images: fractional anisotropy
    if out_img['fa'] is not None:
        out_img['fa'][rng] = out_slc['fa'][z_out, ...]

    # optional output images: Frangi filter response
    if out_img['frangi'] is not None:
        out_img['frangi'][rng] = (255 * out_slc['frangi'][z_out, ...]).astype(np.uint8)

    # optional output images: fiber mask
    if out_img['fbr_msk'] is not None:
        out_img['fbr_msk'][rng] = (255 * (1 - out_slc['fbr_msk'][z_out, ...])).astype(np.uint8)

    # optional output images: neuronal soma mask
    if out_img['bc_msk'] is not None:
        out_img['bc_msk'][rng] = (255 * out_slc['bc_msk'][z_out, ...]).astype(np.uint8)
