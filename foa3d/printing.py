import numpy as np

from foa3d.utils import elapsed_time


# slice progress counter
slc_cnt = 0


def color_text(r, g, b, text):
    """
    Get colored text string.

    Parameters
    ----------
    r: int
        red channel value

    g: int
        green channel value

    b: int
        blue channel value

    text: str
        text string

    Returns
    -------
    clr_text: str
        colored text
    """
    clr_text = f"\033[38;2;{r};{g};{b}m{text} \033[38;2;255;255;255m"

    return clr_text


def print_flsh(string_to_print="", end='\n'):
    """
    Print string and flush output data buffer.

    Parameters
    ----------
    string_to_print: str
        string to be printed

    end: str
        string appended after the last value, default a newline

    Returns
    -------
    None
    """
    print(string_to_print, flush=True, end=end)


def print_blur(sigma_um, psf_fwhm):
    """
    Print the standard deviation of the smoothing Gaussian filter.

    Parameters
    ----------
    sigma_um: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of the smoothing Gaussian filter [μm]
        (resolution anisotropy correction)

    psf_fwhm: numpy.ndarray (shape=(3,), dtype=float)
        3D FWHM of the PSF [μm]

    Returns
    -------
    None
    """
    psf_sz = np.max(psf_fwhm)
    print_flsh(f"Gaussian blur σ      [μm]: ({sigma_um[0]:.3f}, {sigma_um[1]:.3f}, {sigma_um[2]:.3f})\n" +
               f"Adjusted PSF FWHM    [μm]: ({psf_sz:.3f}, {psf_sz:.3f}, {psf_sz:.3f})")


def print_frangi_config(in_img, cfg):
    """
    Print Frangi filter stage heading.

    Parameters
    ----------
    in_img: dict
        input image dictionary (extended)

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

    Returns
    -------
    None
    """
    # print Frangi filter sensitivity, scales and thresholding method
    alpha = cfg['alpha']
    beta = cfg['beta']
    gamma = cfg['gamma']
    scales_um = np.asarray(cfg['scales_um'])
    if gamma is None:
        gamma = 'auto'

    thr = cfg['fb_thr']
    thr_str = "Filter response threshold:"
    thr_str = f"{thr_str} {thr:.2f} (global)\n" if isinstance(thr, float) else f"{thr_str} {thr.capitalize()} (local)\n"

    print_flsh(color_text(0, 191, 255, "\n3D Frangi Filter\n") + "\nSensitivity\n" +
               f"• plate-like \u03B1: {alpha:.1e}\n• blob-like  \u03B2: {beta:.1e}\n• background \u03B3: {gamma}\n\n" +
               f"Enhanced scales      [μm]: {scales_um}\nEnhanced diameters   [μm]: {4 * scales_um}\n\n" + thr_str)

    # print parallel processing information
    batch_sz = cfg['batch']
    slc_shp_um = np.multiply(cfg['px_sz'], cfg['slc_shp'])
    print_slicing_info(in_img['shape_um'], slc_shp_um, in_img['px_sz'], in_img['item_sz'], in_img['msk_bc'])
    print_flsh(f"[Parallel(n_jobs={batch_sz})]: Using backend ThreadingBackend with {batch_sz} concurrent workers.")


def print_frangi_progress(start_time, batch, tot, not_bg, verbose=5):
    """
    Print Frangi filter progress.

    Parameters
    ----------
    start_time: float
        start time [s]

    batch: int
        slice batch size

    tot: int
        total number of image slices

    not_bg: bool
        foreground slice flag

    verbose: int
        verbosity level (print info only every "verbose" slices)

    Returns
    -------
    None
    """
    global slc_cnt
    slc_cnt += 1

    # print only every N=verbose image slices
    if (slc_cnt % verbose == 0 and not_bg) or slc_cnt == tot:
        prog = 100 * slc_cnt / tot
        _, hrs, mins, secs = elapsed_time(start_time)
        print_flsh(
            f"[Parallel(n_jobs={batch})]:\t{slc_cnt}/{tot} done\t|\telapsed: {hrs} hr {mins} min {secs} s\t{prog:.1f}%")


def print_image_info(in_img):
    """
    Print information on the input microscopy image (shape, voxel size, PSF size).

    Parameters
    ----------
    in_img: dict
        input image dictionary

    Returns
    -------
    None
    """
    # get image info
    ch_ax = in_img['ch_ax']
    px_sz = in_img['px_sz']
    psf_fwhm = in_img['psf_fwhm']

    # get image shape (ignore channel axis)
    img_shp = in_img['data'].shape
    if ch_ax is not None:
        img_shp = np.delete(img_shp, ch_ax)

    print_flsh("\n                              Z      Y      X\nImage shape          [μm]: " +
               f"({img_shp[0] * px_sz[0]:.1f}, {img_shp[1] * px_sz[1]:.1f}, {img_shp[2] * px_sz[2]:.1f})\n" +
               f"PSF FWHM             [μm]: ({psf_fwhm[0]:.3f}, {psf_fwhm[1]:.3f}, {psf_fwhm[2]:.3f})\n" +
               f"Pixel size           [μm]: ({px_sz[0]:.3f}, {px_sz[1]:.3f}, {px_sz[2]:.3f})\n")


def print_import_time(start_time):
    """
    Print image import time.

    Parameters
    ----------
    start_time: float
        import start time

    Returns
    -------
    None
    """
    _, _, mins, secs = elapsed_time(start_time)
    print_flsh(f"Image loaded in: {mins} min {secs:3.1f} s")


def print_odf_info(odf_scales_um, odf_degrees):
    """
    Print ODF analysis heading.

    Parameters
    ----------
    odf_scales_um: list (dtype=float)
        fiber ODF resolutions (super-voxel sides [μm])

    odf_degrees: int
        degrees of the spherical harmonics series expansion

    Returns
    -------
    None
    """
    print_flsh(color_text(0, 191, 255, "\n3D ODF Analysis") +
               f"\n\nResolution   [μm]: {odf_scales_um}\nExpansion degrees: {odf_degrees}\n")


def print_pipeline_heading():
    """
    Print Foa3D tool heading.

    Returns
    -------
    None
    """
    print_flsh(color_text(0, 250, 154, "\n3D Fiber Orientation Analysis"))


def print_prepro_heading():
    """
    Print preprocessing heading.

    Returns
    -------
    None
    """
    print_flsh(color_text(0, 191, 255, "\n\nMicroscopy Image Preprocessing\n") +
               "\n                              Z      Y      X")


def print_slicing_info(img_shp_um, slc_shp_um, px_sz, item_sz, msk_bc):
    """
    Print information on the slicing of the basic image sub-volumes processed by the Foa3D tool.

    Parameters
    ----------
    img_shp_um: numpy.ndarray (shape=(3,), dtype=float)
        3D microscopy image [μm]

    slc_shp_um: numpy.ndarray (shape=(3,), dtype=float)
        shape of the analyzed image slices [μm]

    px_sz: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    item_sz: int
        image item size (in bytes)

    msk_bc: bool
        if True, mask neuronal bodies within
        the optionally provided channel

    Returns
    -------
    None
    """
    # adjust slice shape
    if np.any(img_shp_um < slc_shp_um):
        slc_shp_um = img_shp_um

    # get memory sizes
    img_sz = item_sz * np.prod(np.divide(img_shp_um, px_sz))
    max_slc_sz = item_sz * np.prod(np.divide(slc_shp_um, px_sz))

    # print total image and basic slices information
    print_flsh("\n                              Z      Y      X")
    print_flsh(f"Total image shape    [μm]: ({img_shp_um[0]:.1f}, {img_shp_um[1]:.1f}, {img_shp_um[2]:.1f})\n" +
               f"Total image size     [MB]: {np.ceil(img_sz / 1024**2).astype(int)}\n\n" +
               f"Image slice shape    [μm]: ({slc_shp_um[0]:.1f}, {slc_shp_um[1]:.1f}, {slc_shp_um[2]:.1f})\n" +
               f"Image slice size     [MB]: {np.ceil(max_slc_sz / 1024**2).astype(int)}\n")
    if msk_bc:
        print_flsh('Soma mask: active\n')
    else:
        print_flsh('Soma mask: not active\n')
