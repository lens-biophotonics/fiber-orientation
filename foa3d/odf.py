import numpy as np
import scipy as sp
from numba import njit
from skimage.transform import resize

from foa3d.spharm import fiber_vectors_to_sph_harm, get_sph_harm_ncoeff
from foa3d.utils import create_memory_map, normalize_image, transform_axes


def compute_odf_map(fbr_vec, px_sz, odf, odi, fbr_dnst, vec_tnsr_eig, scale, norm, deg=6, vx_thr=0.5, vec_thr=1e-6):
    """
    Compute the spherical harmonics coefficients iterating over super-voxels
    of fiber orientation vectors.

    Parameters
    ----------
    fbr_vec: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float)
        fiber orientation vectors

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        adjusted isotropic pixel size [μm]

    odf: NumPy memory-map object (axis order=(X,Y,Z,C), dtype=float32)
        initialized array of ODF spherical harmonics coefficients

    odi: dict
        orientation dispersion dictionary

            odi_pri: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
                primary orientation dispersion index

            odi_sec: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
                secondary orientation dispersion index

            odi_tot: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
                total orientation dispersion index

            odi_anis: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
                orientation dispersion index anisotropy

    fbr_dnst: NumPy memory-map object (axis order=(Z,Y,X), dtype=float)
        initialized fiber density image

    vec_tnsr_eig: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
        initialized array of orientation tensor eigenvalues

    scale: int
        side of the ODF super-voxel [px]

    norm: numpy.ndarray (dtype: float)
        2D array of spherical harmonics normalization factors

    deg: int
        degrees of the spherical harmonics series expansion

    vx_thr: float
        minimum relative threshold on the sliced voxel volume

    vec_thr: float
        minimum relative threshold on non-zero orientation vectors

    Returns
    -------
    odf: numpy.ndarray (axis order=(X,Y,Z,C), dtype=float32)
        volumetric map of real-valued spherical harmonics coefficients

    fbr_dnst: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        fiber density image
    """
    # iterate over super-voxels
    px_vol = np.prod(px_sz)
    ref_vx_vol = min(scale, fbr_vec.shape[0]) * scale**2
    for z in range(0, fbr_vec.shape[0], scale):
        z_max = z + scale
        for y in range(0, fbr_vec.shape[1], scale):
            y_max = y + scale
            for x in range(0, fbr_vec.shape[2], scale):
                x_max = x + scale

                # select super-voxel of fiber orientation data
                vec_vx = fbr_vec[z:z_max, y:y_max, x:x_max, :]
                zerovec = np.count_nonzero(np.all(vec_vx == 0, axis=-1))
                vx_vol = np.prod(vec_vx.shape[:-1])

                # compute local fiber density
                zv, yv, xv = z // scale, y // scale, x // scale
                fbr_dnst[zv, yv, xv] = (vx_vol - zerovec) / (vx_vol * px_vol)

                # compute ODF and orientation tensor eigenvalues
                # (skipping boundary voxels and voxels without enough data)
                if vx_vol / ref_vx_vol > vx_thr and 1 - zerovec / vx_vol > vec_thr:
                    vec_arr = vec_vx.ravel()
                    odf[zv, yv, xv, :] = fiber_vectors_to_sph_harm(vec_arr, deg, norm)
                    vec_tnsr_eig[zv, yv, xv, :] = compute_vec_tensor_eigen(vec_arr)

    # compute dispersion and anisotropy parameters
    compute_orientation_dispersion(vec_tnsr_eig, **odi)

    # set dispersion background to 0
    mask_orientation_dispersion(vec_tnsr_eig, odi)

    # manipulate ODF axes (for visualization in MRtrix3)
    odf = transform_axes(odf, swapped=(0, 2), flipped=(1, 2))

    return odf, fbr_dnst


@njit(cache=True)
def compute_orientation_dispersion(vec_tnsr_eig, odi_pri, odi_sec, odi_tot, odi_anis):
    """
    Compute orientation dispersion parameters.

    Parameters
    ----------
    vec_tnsr_eig: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
        orientation tensor eigenvalues
        computed from an ODF super-voxel

    odi_pri: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
        primary orientation dispersion index

    odi_sec: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
        secondary orientation dispersion index

    odi_tot: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
        total orientation dispersion index

    odi_anis: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
        orientation dispersion index anisotropy

    Returns
    -------
    None
    """
    # primary dispersion (0.3183098861837907 = 1/π)
    avte = np.abs(vec_tnsr_eig)
    if odi_pri is not None:
        odi_pri[:] = (0.5 - 0.3183098861837907 * np.arctan2(avte[..., 2], avte[..., 1])).astype(np.float32)

    # secondary dispersion
    if odi_sec is not None:
        odi_sec[:] = (0.5 - 0.3183098861837907 * np.arctan2(avte[..., 2], avte[..., 0])).astype(np.float32)

    # dispersion anisotropy
    if odi_anis is not None:
        odi_anis[:] = (0.5 - 0.3183098861837907 \
            * np.arctan2(avte[..., 2], np.abs(vec_tnsr_eig[..., 1] - vec_tnsr_eig[..., 0]))).astype(np.float32)

    # total dispersion
    odi_tot[:] = (0.5 - 0.3183098861837907 *
                  np.arctan2(avte[..., 2], np.sqrt(np.abs(np.multiply(avte[..., 1], avte[..., 0]))))).astype(np.float32)


def compute_vec_tensor_eigen(fbr_vec):
    """
    Compute the eigenvalues of the 3x3 orientation tensor
    obtained from a reshaped super-voxel of fiber orientation vectors.

    Parameters
    ----------
    fbr_vec: numpy.ndarray (shape=(N,3), dtype=float)
        fiber orientation vectors
        (reshaped super-voxel of shape=(Nz,Ny,Nx), i.e. N=Nz*Ny*Nx)

    Returns
    -------
    vec_tensor_eigen: numpy.ndarray (shape=(3,), dtype=float32)
        orientation tensor eigenvalues in ascending order
    """
    fbr_vec = np.delete(fbr_vec, np.all(fbr_vec == 0, axis=-1), axis=0)
    fbr_vec.shape = (-1, 3)
    t = np.zeros((3, 3))
    for v in fbr_vec:
        t += np.outer(v, v)

    vec_tensor_eigen = sp.linalg.eigh(t, eigvals_only=True).astype(np.float32)

    return vec_tensor_eigen


def generate_odf_background(bg_mrtrix, fbr_vec, scale, iso_fbr=None):
    """
    Generate the down-sampled background image required
    to visualize the 3D ODF map in MRtrix3.

    Parameters
    ----------
    bg_mrtrix: numpy.ndarray (axis order=(X,Y,Z), dtype=uint8)
        initialized background array for ODF visualization in MRtrix3

    fbr_vec: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float32)
        fiber orientation vectors

    scale: int
        side of the ODF super-voxel [px]

    iso_fbr: numpy.ndarray (axis order=(Z,Y,X), dtype=uint8)
        isotropic fiber image

    Returns
    -------
    None
    """
    # image normalization: get global minimum and maximum values
    bg_img = fbr_vec if iso_fbr is None else iso_fbr
    if bg_img.ndim == 3:
        min_glob = np.min(bg_img)
        max_glob = np.max(bg_img)

    # loop over z-slices, rescale and resize them
    for z in range(scale // 2, bg_img.shape[0], scale):
        if bg_img.ndim == 3:
            tmp_slice = normalize_image(bg_img[z], min_val=min_glob, max_val=max_glob)
        elif bg_img.ndim == 4:
            tmp_slice = 255.0 * np.sum(np.abs(bg_img[z, ...]), axis=-1)
            tmp_slice = np.where(tmp_slice <= 255.0, tmp_slice, 255.0)

        tmp_slice = transform_axes(tmp_slice, swapped=(0, 1), flipped=(0, 1))
        bg_mrtrix[..., z // scale] = \
            resize(tmp_slice, output_shape=bg_mrtrix.shape[:-1], anti_aliasing=True, preserve_range=True)


def init_odf_arrays(vec_img_shp, tmp_dir, scale, deg=6, exp_all=False):
    """
    Initialize the output datasets of the ODF analysis stage.

    Parameters
    ----------
    vec_img_shp: tuple
        vector volume shape [px]

    tmp_dir: str
        path to temporary folder

    scale: int
        fiber ODF resolution (super-voxel side [px])

    deg: int
        degrees of the spherical harmonics series expansion

    exp_all: bool
        export all images

    Returns
    -------
    odf: NumPy memory-map object (axis order=(X,Y,Z,C), dtype=float32)
                initialized array of ODF spherical harmonics coefficients

    odi: dict
        dispersion image dictionary

            odi_pri: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
                initialized array of primary orientation dispersion parameters

            odi_sec: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
                initialized array of secondary orientation dispersion parameters

            odi_tot: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
                initialized array of total orientation dispersion parameters

            odi_anis: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
                initialized array of orientation dispersion anisotropy parameters

    fbr_dnst: NumPy memory-map object (axis order=(Z,Y,X), dtype=float)
        initialized fiber density image

    bg_mrtrix: NumPy memory-map object (axis order=(X,Y,Z), dtype=uint8)
        initialized background for ODF visualization in MRtrix3

    vec_tensor_eigen: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
        initialized fiber orientation tensor eigenvalues
    """
    # initialize ODF and orientation dispersion maps
    nc = get_sph_harm_ncoeff(deg)
    odi_shp = tuple(np.ceil(np.divide(vec_img_shp, scale)).astype(int))
    odf = create_memory_map('float32', shape=odi_shp + (nc,), name=f'odf{scale}', tmp=tmp_dir)
    odi_tot = create_memory_map('float32', shape=odi_shp, name=f'odi_tot{scale}', tmp=tmp_dir)
    bg_mrtrix = create_memory_map('uint8', shape=tuple(np.flip(odi_shp)), name=f'bg{scale}', tmp=tmp_dir)
    if exp_all:
        odi_pri = create_memory_map('float32', shape=odi_shp, name=f'odi_pri{scale}', tmp=tmp_dir)
        odi_sec = create_memory_map('float32', shape=odi_shp, name=f'odi_sec{scale}', tmp=tmp_dir)
        odi_anis = create_memory_map('float32', shape=odi_shp, name=f'odi_anis{scale}', tmp=tmp_dir)
    else:
        odi_pri, odi_sec, odi_anis = (None, None, None)

    # initialize fiber orientation tensor, fiber density and orientation dispersion dictionary
    vec_tnsr_eig = create_memory_map('float32', shape=odi_shp + (3,), name=f'tensor{scale}', tmp=tmp_dir)
    fbr_dnst = create_memory_map('float32', shape=odi_shp, name=f'fbr_dnst{scale}', tmp=tmp_dir)
    odi = {'odi_pri': odi_pri, 'odi_sec': odi_sec, 'odi_tot': odi_tot, 'odi_anis': odi_anis}

    return odf, odi, fbr_dnst, bg_mrtrix, vec_tnsr_eig


def mask_orientation_dispersion(vec_tnsr_eig, odi):
    """
    Suppress orientation dispersion background.

    Parameters
    ----------
    vec_tnsr_eig: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
        orientation tensor eigenvalues
        computed from an ODF super-voxel

    odi: dict
        orientation dispersion dictionary

        odi_pri: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
            array of primary orientation dispersion parameters

        odi_sec: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
            array of secondary orientation dispersion parameters

        odi_tot: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
            array of total orientation dispersion parameters

        odi_anis: NumPy memory-map object (axis order=(Z,Y,X), dtype=float32)
            array of orientation dispersion anisotropy parameters

    Returns
    -------
    None
    """
    bg_msk = np.all(vec_tnsr_eig == 0, axis=-1)
    for key in odi.keys():
        if odi[key] is not None:
            odi[key][bg_msk] = 0
