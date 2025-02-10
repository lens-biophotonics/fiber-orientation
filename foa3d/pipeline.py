from time import perf_counter
import numpy as np
from joblib import Parallel, delayed

from foa3d.frangi import (frangi_filter, init_frangi_arrays, mask_background,
                          write_frangi_arrays)
from foa3d.input import get_frangi_config, get_resource_config
from foa3d.odf import (compute_odf_map, generate_odf_background,
                       init_odf_arrays)
from foa3d.output import save_frangi_arrays, save_odf_arrays
from foa3d.preprocessing import correct_anisotropy
from foa3d.printing import (print_flsh, print_frangi_config,
                            print_frangi_progress, print_odf_info)
from foa3d.slicing import (check_background, get_slicing_config,
                           generate_slice_ranges, crop,
                           crop_img_dict, slice_image)
from foa3d.spharm import get_sph_harm_norm_factors
from foa3d.utils import get_available_cores


def parallel_frangi_over_slices(cli_args, save_dirs, in_img):
    """
    Apply 3D Frangi filter to basic microscopy image slices using parallel threads.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    save_dirs: dict
        saving directories

            frangi: Frangi filter

            odf: ODF analysis

            tmp: temporary data

    in_img: dict
        input image dictionary

            data: NumPy memory-map object (axis order=(Z,Y,X) or (Z,Y,X,C) or (Z,C,Y,X))
                3D microscopy image

            ch_ax: int
                RGB image channel axis (either 1, 3, or None for grayscale images)

            ts_msk: numpy.ndarray (dtype=bool)
                tissue reconstruction binary mask

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

            path: str
                path to the 3D microscopy image

            name: str
                name of the 3D microscopy image

            fmt: str
                format of the 3D microscopy image

            is_tiled: bool
                True for tiled reconstructions aligned using ZetaStitcher

            is_vec: bool
                vector field flag

            shape: numpy.ndarray (shape=(3,), dtype=int)
                total image shape

            shape_um: numpy.ndarray (shape=(3,), dtype=float)
                total image shape [μm]

            item_sz: int
                image item size [B]

    Returns
    -------
    out_img: dict
        output image dictionary

            vec: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
                fiber orientation vector field

            clr: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=uint8)
                orientation colormap image

            fa: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
                fractional anisotropy image

            frangi: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
                Frangi-enhanced image (fiber probability image)

            iso: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
                isotropic fiber image

            fbr_msk: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
                fiber mask image

            bc_msk: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
                soma mask image

            px_sz: numpy.ndarray (shape=(3,), dtype=float)
                output pixel size [μm]
    """
    # skip the Frangi filter stage if an orientation vector field was provided as input
    if in_img['is_vec']:
        out_img = {'vec': in_img['data'], 'iso': None, 'px_sz': None}

    else:
        # initialize Frangi filter stage output
        frangi_cfg = get_frangi_config(cli_args, in_img)
        out_img = init_frangi_arrays(in_img, frangi_cfg, save_dirs['tmp'])

        # get parallel processing configuration
        get_resource_config(cli_args, frangi_cfg)
        get_slicing_config(in_img, frangi_cfg)

        # conduct a Frangi-filter-based analysis of fiber orientations using concurrent workers
        print_frangi_config(in_img, frangi_cfg)
        slc_rng = generate_slice_ranges(in_img, frangi_cfg)
        _fa = out_img['fa'] is not None
        t_start = perf_counter()
        with Parallel(n_jobs=frangi_cfg['batch'], prefer='threads') as parallel:
            parallel(delayed(frangi_analysis)(s, in_img, out_img, frangi_cfg, t_start, _fa=_fa) for s in slc_rng)

        # save output arrays to file
        save_frangi_arrays(save_dirs['frangi'], in_img['name'], out_img, ram=frangi_cfg['ram'])

    return out_img


def frangi_analysis(rng, in_img, out_img, cfg, t_start, _fa=False):
    """
    Conduct a Frangi-based fiber orientation analysis
    on basic slices selected from the whole volumetric microscopy image.

    Parameters
    ----------
    rng: dict
        in: np.ndarray
            3D input slice range [px]

        pad: np.ndarray
            3D slice padding range [px]

        out: np.ndarray
            3D output slice range [px]

        bc: np.ndarray
            (optional) brain cell soma slice range

    in_img: dict
        input image dictionary

            data: NumPy memory-map object (axis order=(Z,Y,X) or (Z,Y,X,C) or (Z,C,Y,X))
                3D microscopy image

            ch_ax: int
                RGB image channel axis (either 1, 3, or None for grayscale images)

            ts_msk: numpy.ndarray (dtype=bool)
                tissue reconstruction binary mask

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

            path: str
                path to the 3D microscopy image

            name: str
                name of the 3D microscopy image

            fmt: str
                format of the 3D microscopy image

            is_tiled: bool
                True for tiled reconstructions aligned using ZetaStitcher

            is_vec: bool
                vector field flag

            shape: numpy.ndarray (shape=(3,), dtype=int)
                total image shape

            shape_um: numpy.ndarray (shape=(3,), dtype=float)
                total image shape [μm]

            item_sz: int
                image item size [B]

    out_img: dict
        output image dictionary

            vec: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
                fiber orientation vector field

            clr: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=uint8)
                orientation colormap image

            fa: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
                fractional anisotropy image

            frangi: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
                Frangi-enhanced image (fiber probability image)

            iso: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
                isotropic fiber image

            fbr_msk: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
                fiber mask image

            bc_msk: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
                soma mask image

            px_sz: numpy.ndarray (shape=(3,), dtype=float)
                output pixel size [μm]

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

            fb_thr: float or str
                image thresholding applied to the Frangi filter response

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

    t_start: float
        start time [s]

    _fa: bool
        compute fractional anisotropy

    Returns
    -------
    None
    """
    # extract image slice
    fbr_slc, ts_msk_slc = \
        slice_image(in_img['data'], rng['in'], in_img['ch_ax'], ch=in_img['fb_ch'], ts_msk=in_img['ts_msk'])

    # process non-background image slice
    not_bg = check_background(fbr_slc, ts_msk=ts_msk_slc)
    if not_bg:
        out_slc = analyze_fibers(fbr_slc, cfg, rng['out'], rng['pad'], ts_msk=ts_msk_slc, _fa=_fa)

        # (optional) brain cell soma masking
        if rng['bc'] is not None:
            bc_slc, _ = slice_image(in_img['data'], rng['bc'], in_img['ch_ax'], ch=in_img['bc_ch'])
            out_slc = reject_brain_cells(bc_slc, out_slc, cfg['rsz'], rng['out'])

        # write memory-mapped output arrays
        write_frangi_arrays(out_img, out_slc, rng['out'], z_out=cfg['z_out'])

    print_frangi_progress(t_start, cfg['batch'], cfg['tot_slc'], not_bg)


def analyze_fibers(fbr_slc, cfg, out_rng, pad_rng, ts_msk=None, _fa=False):
    """
    Analyze 3D fiber orientations exploiting a Frangi-filter-based
    unsupervised enhancement of tubular structures.

    Parameters
    ----------
    fbr_slc: numpy.ndarray (axis order=(Z,Y,X))
        fiber image slice

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

            fb_thr: float or str
                image thresholding applied to the Frangi filter response

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

    out_rng: NumPy slice object
        output range

    pad_rng: numpy.ndarray (axis order=(Z,Y,X))
        image padding range

    ts_msk: numpy.ndarray (dtype=bool)
        tissue reconstruction binary mask

    _fa: bool
        compute fractional anisotropy

    Returns
    -------
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
    """
    # preprocess fiber slice (adaptive blurring and downsampling)
    iso_fbr_slc, pad_rsz, ts_msk_rsz = \
        correct_anisotropy(fbr_slc, cfg['rsz'], sigma=cfg['smooth_sd'], pad=pad_rng, ts_msk=ts_msk)

    # pad fiber slice if required
    if pad_rsz is not None:
        iso_fbr_slc = np.pad(iso_fbr_slc, pad_rsz, mode='reflect')

        # pad tissue mask if available
        if ts_msk_rsz is not None:
            ts_msk_rsz = np.pad(ts_msk_rsz, pad_rsz, mode='constant')

    # 3D Frangi filter
    out_slc = frangi_filter(iso_fbr_slc, scales_px=cfg['scales_px'],
                            alpha=cfg['alpha'], beta=cfg['beta'], gamma=cfg['gamma'], hsv=cfg['hsv_cmap'], _fa=_fa)

    # crop resulting slices
    out_slc.update({'iso': iso_fbr_slc, 'ts_msk': ts_msk_rsz})
    ovlp_rsz = np.multiply(cfg['ovlp'] * np.ones((3,)), cfg['rsz']).astype(int)
    out_slc = crop_img_dict(out_slc, out_rng, ovlp_rsz)

    # remove Frangi filter background
    out_slc, msk = mask_background(out_slc, method=cfg['fb_thr'], invert=False)
    out_slc['fbr_msk'] = msk

    return out_slc


def reject_brain_cells(bc_slc, out_slc, rsz_ratio, out_rng, bc_thr='yen'):
    """
    Suppress soma contribution using the optional image channel
    of neuronal bodies.

    Parameters
    ----------
    bc_slc: numpy.ndarray (axis order=(Z,Y,X))
        brain cell soma image slice

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

    rsz_ratio: numpy.ndarray (shape=(3,), dtype=float)
        3D image resize ratio

    out_rng: NumPy slice object
        output range

    bc_thr: str
        thresholding method applied to the channel of brain cell bodies

    Returns
    -------
    out_slc: dict
        (masked) slice output dictionary

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
                brain cell mask
    """
    # resize and crop brain cell soma slice
    iso_bc, _, _ = correct_anisotropy(bc_slc, rsz_ratio)
    iso_bc = crop(iso_bc, out_rng)

    # mask neuronal bodies
    out_slc, msk = mask_background(out_slc, ref_img=iso_bc, method=bc_thr, invert=True)
    out_slc['bc_msk'] = msk

    return out_slc


def parallel_odf_over_scales(cli_args, save_dirs, out_img, img_name):
    """
    Iterate over the required spatial scales and apply the parallel ODF analysis
    implemented in parallel_odf_on_slices().

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    save_dirs: dict
        saving directories

            frangi: Frangi filter

            odf: ODF analysis

            tmp: temporary data

    out_img: dict
        output image dictionary

            vec: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
                fiber orientation vector field

            clr: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=uint8)
                orientation colormap image

            fa: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
                fractional anisotropy image

            frangi: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
                Frangi-enhanced image (fiber probability image)

            iso: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
                isotropic fiber image

            fbr_msk: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
                fiber mask image

            bc_msk: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
                soma mask image

            px_sz: numpy.ndarray (shape=(3,), dtype=float)
                output pixel size [μm]

    img_name: str
        name of the input volume image

    Returns
    -------
    None
    """
    if cli_args.odf_res is not None:
        print_odf_info(cli_args.odf_res, cli_args.odf_deg)

        # get pixel size from CLI arguments
        # if a vector field was directly provided to Foa3D
        px_sz = out_img['px_sz']
        if px_sz is None:
            px_sz = (cli_args.px_size_z, cli_args.px_size_xy, cli_args.px_size_xy)

        # parallel ODF analysis of fiber orientation vectors over the spatial scales of interest
        batch_sz = min(len(cli_args.odf_res), get_available_cores())
        odf_norm = get_sph_harm_norm_factors(cli_args.odf_deg)
        with Parallel(n_jobs=batch_sz, verbose=10, prefer='threads') as parallel:
            parallel(delayed(odf_analysis)(out_img['vec'], out_img['iso'], px_sz, save_dirs, img_name,
                                           odf_norm=odf_norm, odf_deg=cli_args.odf_deg, odf_scale_um=s,
                                           exp_all=cli_args.exp_all) for s in cli_args.odf_res)

        print_flsh(f"\nODF and dispersion maps saved to: {save_dirs['odf']}\n")


def odf_analysis(fbr_vec, iso_fbr, px_sz, save_dirs, img_name, odf_scale_um, odf_norm, odf_deg=6, exp_all=False):
    """
    Estimate 3D fiber ODFs from basic orientation data chunks using parallel threads.

    Parameters
    ----------
    fbr_vec: NumPy memory-map object (axis order=(Z,Y,X,C), dtype=float32)
        fiber orientation vector field

    iso_fbr: NumPy memory-map object (axis order=(Z,Y,X), dtype=uint8)
        isotropic fiber image

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        adjusted isotropic pixel size [μm]

    save_dirs: dict
        saving directories
        ('frangi': Frangi filter, 'odf': ODF analysis)

    img_name: str
        name of the 3D microscopy image

    odf_scale_um: float
        fiber ODF resolution (super-voxel side [μm])

    odf_norm: numpy.ndarray (dtype: float)
        2D array of spherical harmonic normalization factors

    odf_deg: int
        degrees of the spherical harmonic series expansion

    exp_all: bool
        export all images

    Returns
    -------
    None
    """
    # initialize ODF stage output arrays
    odf_scale = int(np.ceil(odf_scale_um / px_sz[0]))
    odf, odi, dnst, bg_mrtrix, vec_tensor_eigen = \
        init_odf_arrays(fbr_vec.shape[:-1], save_dirs['tmp'], scale=odf_scale, deg=odf_deg, exp_all=exp_all)

    # generate ODF coefficients and down-sampled background for visualization in MRtrix3
    generate_odf_background(bg_mrtrix, fbr_vec, scale=odf_scale, iso_fbr=iso_fbr)
    odf, dnst = compute_odf_map(fbr_vec, px_sz, odf, odi, dnst, vec_tensor_eigen, odf_scale, odf_norm, deg=odf_deg)

    # save output arrays to TIFF or NIfTI files
    save_odf_arrays(save_dirs['odf'], img_name, odf_scale_um, px_sz, odf, bg_mrtrix, dnst, **odi)
