import argparse

from time import perf_counter
from os import path

import numpy as np
import tifffile as tiff

try:
    from zetastitcher import VirtualFusedVolume
except ImportError:
    pass

from foa3d.output import create_save_dirs
from foa3d.preprocessing import config_anisotropy_correction
from foa3d.printing import (color_text, print_flsh, print_image_info,
                            print_import_time)
from foa3d.utils import (create_background_mask, create_memory_map,
                         detect_ch_axis, get_item_bytes, get_config_label)


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def get_cli_parser():
    """
    Parse command line arguments.

    Returns
    -------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments
    """
    # configure parser object
    cli_parser = argparse.ArgumentParser(
        description='Foa3D: A 3D Fiber Orientation Analysis Pipeline\n'
                    'author:     Michele Sorelli (2022)\n'
                    'references: Frangi  et al.  (1998) '
                    'Multiscale vessel enhancement filtering.'
                    ' In Medical Image Computing and'
                    ' Computer-Assisted Intervention 1998, pp. 130-137.\n'
                    '            Alimi   et al.  (2020) '
                    'Analytical and fast Fiber Orientation Distribution '
                    'reconstruction in 3D-Polarized Light Imaging. '
                    'Medical Image Analysis, 65, pp. 101760.\n'
                    '            Sorelli et al.  (2023) '
                    'Fiber enhancement and 3D orientation analysis '
                    'in label-free two-photon fluorescence microscopy. '
                    'Scientific Reports, 13, pp. 4160.\n',
        formatter_class=CustomFormatter)
    cli_parser.add_argument(dest='image_path',
                            help='path to input 3D microscopy image or 4D array of fiber orientation vectors\n'
                                 '* supported formats:\n'
                                 '  - .tif .tiff (microscopy image or fiber orientation vectors)\n'
                                 '  - .yml (ZetaStitcher\'s stitch file of tiled microscopy reconstruction)\n'
                                 '* image axes order:\n'
                                 '  - grayscale image:      (Z, Y, X)\n'
                                 '  - RGB image:            (Z, Y, X, C) or (Z, C, Y, X)\n'
                                 '  - NumPy vector image:   (Z, Y, X, C) or (Z, C, Y, X)\n'
                                 '  - TIFF  vector image:   (Z, Y, X, C) or (Z, C, Y, X)')
    cli_parser.add_argument('-a', '--alpha', type=float, default=0.001,
                            help='Frangi\'s plate-like object sensitivity')
    cli_parser.add_argument('-b', '--beta', type=float, default=1.0,
                            help='Frangi\'s blob-like object sensitivity')
    cli_parser.add_argument('-g', '--gamma', type=float, default=None,
                            help='Frangi\'s background score sensitivity')
    cli_parser.add_argument('-s', '--scales', nargs='+', type=float, default=[1.25],
                            help='list of Frangi filter scales [μm]')
    cli_parser.add_argument('-j', '--jobs', type=int, default=None,
                            help='number of parallel threads used by the Frangi filter stage: '
                                 'use one thread per logical core if None')
    cli_parser.add_argument('-r', '--ram', type=float, default=None,
                            help='maximum RAM available to the Frangi filter stage [GB]: use all if None')
    cli_parser.add_argument('--px-size-xy', type=float, default=1.0, help='lateral pixel size [μm]')
    cli_parser.add_argument('--px-size-z', type=float, default=1.0, help='longitudinal pixel size [μm]')
    cli_parser.add_argument('--psf-fwhm-x', type=float, default=1.0, help='PSF FWHM along horizontal x-axis [μm]')
    cli_parser.add_argument('--psf-fwhm-y', type=float, default=1.0, help='PSF FWHM along vertical y-axis [μm]')
    cli_parser.add_argument('--psf-fwhm-z', type=float, default=1.0, help='PSF FWHM along depth z-axis [μm]')
    cli_parser.add_argument('--fb-ch', type=int, default=1, help='neuronal fibers channel')
    cli_parser.add_argument('--bc-ch', type=int, default=0, help='brain cell soma channel')
    cli_parser.add_argument('--fb-thr', default='li', type=str,
                            help='Frangi filter probability response threshold (t ∈ [0, 1] or skimage.filters method)')
    cli_parser.add_argument('--z-min', type=float, default=0, help='forced minimum output z-depth [μm]')
    cli_parser.add_argument('--z-max', type=float, default=None, help='forced maximum output z-depth [μm]')
    cli_parser.add_argument('--hsv', action='store_true', default=False,
                            help='generate HSV colormap of 3D fiber orientations')
    cli_parser.add_argument('--odf-res', nargs='+', type=float, help='side of the fiber ODF super-voxels: '
                                                                     'do not generate ODFs if None [μm]')
    cli_parser.add_argument('--odf-deg', type=int, default=6,
                            help='degrees of the spherical harmonics series expansion (even number between 2 and 10)')
    cli_parser.add_argument('-o', '--out', type=str, default=None,
                            help='output directory')
    cli_parser.add_argument('-c', '--cell-msk', action='store_true', default=False,
                            help='apply neuronal body mask (the optional channel of neuronal bodies must be available)')
    cli_parser.add_argument('-t', '--tissue-msk', action='store_true', default=False,
                            help='apply tissue background mask')
    cli_parser.add_argument('-e', '--exp-all', action='store_true', default=False,
                            help='save the full range of images produced by the Frangi filter and ODF stages, '
                                 'e.g. for testing purposes (see documentation)')

    cli_args = cli_parser.parse_args()

    return cli_args


def get_image_size(in_img):
    """
    Get information on the size of the input 3D microscopy image.

    Parameters
    ----------
    in_img: dict
        input image dictionary

            data: numpy.ndarray or NumPy memory-map object (axis order=(Z,Y,X) or (Z,Y,X,C) or (Z,C,Y,X))
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

    Returns
    -------
    None
    """
    # adapt channel axis
    img_shp = np.asarray(in_img['data'].shape)
    if in_img['ch_ax'] is None:
        in_img.update({'fb_ch': None, 'msk_bc': False})
    else:
        img_shp = np.delete(img_shp, in_img['ch_ax'])

    in_img.update({'shape': img_shp,
                   'shape_um': np.multiply(img_shp, in_img['px_sz']),
                   'item_sz': get_item_bytes(in_img['data'])})


def get_image_info(cli_args):
    """
    Get microscopy image file path and format.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    Returns
    -------
    in_img: dict
        input image dictionary
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

    msk_mip: bool
        apply tissue reconstruction mask (binarized MIP)
    """
    # get microscopy image path and name
    img_path = cli_args.image_path
    img_name = path.basename(img_path)
    split_name = img_name.split('.')

    # check image format
    if len(split_name) == 1:
        raise ValueError('Format must be specified for input microscopy images!')
    img_fmt = split_name[-1]
    img_name = img_name.replace(f'.{img_fmt}', '')
    is_tiled = True if img_fmt == 'yml' else False

    # apply tissue reconstruction mask (binarized MIP) and/or brain cell soma mask
    msk_mip = cli_args.tissue_msk
    msk_bc = cli_args.cell_msk

    # append Frangi filter configuration to input image name
    cfg_lbl = get_config_label(cli_args)
    img_name = f'{img_name}_{cfg_lbl}'

    # get microscopy image channels and resolution
    fb_ch = cli_args.fb_ch
    bc_ch = cli_args.bc_ch
    px_sz, psf_fwhm = get_resolution(cli_args)

    # populate input image dictionary
    in_img = {'fb_ch': fb_ch, 'bc_ch': bc_ch, 'msk_bc': msk_bc, 'psf_fwhm': psf_fwhm, 'px_sz': px_sz,
              'path': img_path, 'name': img_name, 'fmt': img_fmt, 'is_tiled': is_tiled}

    return in_img, msk_mip


def get_frangi_config(cli_args, in_img):
    """
    Get Frangi filter configuration.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

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

    Returns
    -------
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

            fb_thr: float or skimage.filters thresholding method
                Frangi filter probability response threshold

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

            z_out: NumPy slice object
                output z-range
    """
    # preprocessing configuration (adaptive smoothing)
    smooth_sd, out_px_sz = config_anisotropy_correction(in_img['px_sz'], in_img['psf_fwhm'])
    scales_px = np.array(cli_args.scales) / np.max(out_px_sz)

    # adapted output z-axis range when required
    z_min = max(0, int(np.floor(cli_args.z_min / np.max(in_img['px_sz']))))
    z_max = min(int(np.ceil(cli_args.z_max / np.max(in_img['px_sz']))), in_img['shape'][0]) \
        if cli_args.z_max is not None else in_img['shape'][0]

    # get threshold applied to the Frangi filter response
    fb_thr = cli_args.fb_thr
    if fb_thr.replace('.', '', 1).isdigit():
        fb_thr = float(fb_thr)

    # compile Frangi filter configuration dictionary
    frangi_cfg = {'alpha': cli_args.alpha, 'beta': cli_args.beta, 'gamma': cli_args.gamma,
                  'scales_um': np.array(cli_args.scales), 'scales_px': scales_px, 'smooth_sd': smooth_sd,
                  'px_sz': out_px_sz, 'fb_thr': fb_thr, 'hsv_cmap': cli_args.hsv,
                  'exp_all': cli_args.exp_all, 'z_out': slice(z_min, z_max, 1)}

    return frangi_cfg


def get_resolution(cli_args):
    """
    Retrieve microscopy resolution information from command line arguments.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    Returns
    -------
    px_sz: tuple (shape=(3,), dtype=float)
        pixel size [μm]

    psf_fwhm: tuple (shape=(3,), dtype=float)
        3D PSF FWHM [μm]
    """
    px_sz = (cli_args.px_size_z, cli_args.px_size_xy, cli_args.px_size_xy)
    psf_fwhm = (cli_args.psf_fwhm_z, cli_args.psf_fwhm_y, cli_args.psf_fwhm_x)

    return px_sz, psf_fwhm


def get_resource_config(cli_args, frangi_cfg):
    """
    Retrieve resource usage configuration of the Foa3D tool.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

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

    Returns
    -------
    None
    """
    jobs = cli_args.jobs
    ram = cli_args.ram
    if ram is not None:
        ram *= 1024**3

    frangi_cfg.update({'jobs': jobs, 'ram': ram})


def load_microscopy_image(cli_args):
    """
    Load 3D microscopy image from TIFF, or ZetaStitcher .yml file.
    Alternatively, the processing pipeline accepts as input TIFF or NumPy
    files of fiber orientation vector data: in this case, the Frangi filter
    stage will be skipped.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    Returns
    -------
    in_img: dict
        input image dictionary

            data: numpy.ndarray (axis order=(Z,Y,X) or (Z,Y,X,C) or (Z,C,Y,X))
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

    save_dirs: dict
        saving directories
        ('frangi': Frangi filter, 'odf': ODF analysis, 'tmp': temporary files)
    """
    # get input information and create saving directories
    in_img, msk_mip = get_image_info(cli_args)
    save_dirs = create_save_dirs(cli_args, in_img)

    # import fiber orientation vector data or raw 3D microscopy image
    tic = perf_counter()
    load_data(in_img, save_dirs['tmp'], msk_mip=msk_mip)

    # print input data information
    get_image_size(in_img)
    print_import_time(tic)
    if not in_img['is_vec']:
        print_image_info(in_img)
    else:
        print_flsh()

    return in_img, save_dirs


def load_data(in_img, tmp_dir, msk_mip=False):
    """
    Load 3D microscopy data.

    Parameters
    ----------
    in_img: dict
        input image dictionary
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

    tmp_dir: str
        path to temporary folder

    msk_mip: bool
        apply tissue reconstruction mask (binarized MIP)

    Returns
    -------
    None
    """
    print_flsh(color_text(0, 191, 255, "\nMicroscopy Image Import\n"))

    # load tiled reconstruction (aligned using ZetaStitcher)
    if in_img['is_tiled']:
        print_flsh(f"Loading {in_img['path']} tiled reconstruction...\n")
        img = VirtualFusedVolume(in_img['path'])
        ch_ax = detect_ch_axis(img)
        is_vec = False

    # load z-stack
    else:
        print_flsh(f"Loading {in_img['path']} z-stack...\n")

        img_fmt = in_img['fmt'].lower()
        if img_fmt in ('tif', 'tiff'):
            img = tiff.imread(in_img['path'])
            ch_ax = detect_ch_axis(img)

            # detect vector field input
            is_vec = img.ndim == 4 and img.dtype in (np.float32, float, 'float32')
            if is_vec:
                if ch_ax != 3:
                    img = np.moveaxis(img, ch_ax, -1)

            img = create_memory_map(img.dtype, name=in_img['name'], tmp=tmp_dir, arr=img, mmap_mode='r')

        else:
            raise ValueError('Unsupported image format!')

    # generate tissue background mask
    if not is_vec and msk_mip:
        dims = len(img.shape)
        if dims == 3:
            img_fbr = img
        elif dims == 4:
            img_fbr = img[:, in_img['fb_ch'], :, :] if ch_ax == 1 else img[..., in_img['fb_ch']]
        else:
            raise ValueError('Invalid image (ndim != 3 and ndim != 4)!')

        # compute MIP (naive for loop to minimize the required RAM)
        ts_mip = np.zeros(img_fbr.shape[1:], dtype=img_fbr.dtype)
        for z in range(img_fbr.shape[0]):
            stk = np.stack((ts_mip, img_fbr[z]))
            ts_mip = np.max(stk, axis=0)

        ts_msk = create_background_mask(ts_mip, method='li', black_bg=True)
    else:
        ts_msk = None

    # update input image dictionary
    in_img.update({'data': img, 'ts_msk': ts_msk, 'ch_ax': ch_ax, 'is_vec': is_vec})
