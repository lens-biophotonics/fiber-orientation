import argparse
import tempfile
from time import perf_counter

import h5py
import numpy as np
import tifffile as tiff

try:
    from zetastitcher import VirtualFusedVolume
except ImportError:
    pass

from os import path

from foa3d.output import create_save_dirs
from foa3d.preprocessing import config_anisotropy_correction
from foa3d.printing import (color_text, print_image_shape, print_import_time,
                            print_resolution)
from foa3d.utils import create_memory_map, get_item_bytes, get_output_prefix


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
                            help='path to input microscopy volume image or to 4D array of fiber orientation vectors\n'
                                 '* supported formats: .tif (image), '
                                 '.npy (image or fiber vectors), .yml (ZetaStitcher stitch file)\n'
                                 '* image  axes order: (Z, Y, X)\n'
                                 '* vector axes order: (Z, Y, X, C)')
    cli_parser.add_argument('-a', '--alpha', type=float, default=0.001,
                            help='Frangi plate-like object sensitivity')
    cli_parser.add_argument('-b', '--beta', type=float, default=1.0,
                            help='Frangi blob-like object sensitivity')
    cli_parser.add_argument('-g', '--gamma', type=float, default=None,
                            help='Frangi background score sensitivity')
    cli_parser.add_argument('-s', '--scales', nargs='+', type=float, default=[1.25],
                            help='list of Frangi filter scales [μm]')
    cli_parser.add_argument('-n', '--neuron-mask', action='store_true', default=False,
                            help='lipofuscin-based neuronal body masking')
    cli_parser.add_argument('-j', '--jobs', type=int, default=None,
                            help='number of parallel threads used by the Frangi filtering stage: '
                                 'use one thread per logical core if None')
    cli_parser.add_argument('-r', '--ram', type=float, default=None,
                            help='maximum RAM available to the Frangi filtering stage [GB]: use all if None')
    cli_parser.add_argument('-m', '--mmap', action='store_true', default=False,
                            help='create a memory-mapped array of the microscopy volume image')
    cli_parser.add_argument('--px-size-xy', type=float, default=0.878, help='lateral pixel size [μm]')
    cli_parser.add_argument('--px-size-z', type=float, default=1.0, help='longitudinal pixel size [μm]')
    cli_parser.add_argument('--psf-fwhm-x', type=float, default=0.692, help='PSF FWHM along the X axis [μm]')
    cli_parser.add_argument('--psf-fwhm-y', type=float, default=0.692, help='PSF FWHM along the Y axis [μm]')
    cli_parser.add_argument('--psf-fwhm-z', type=float, default=2.612, help='PSF FWHM along the Z axis [μm]')
    cli_parser.add_argument('--ch-fiber', type=int, default=1, help='myelinated fibers channel')
    cli_parser.add_argument('--ch-neuron', type=int, default=0, help='neuronal soma channel')
    cli_parser.add_argument('--z-min', type=float, default=0, help='forced minimum output z-depth [μm]')
    cli_parser.add_argument('--z-max', type=float, default=None, help='forced maximum output z-depth [μm]')
    cli_parser.add_argument('-o', '--odf-res', nargs='+', type=float, help='side of the fiber ODF super-voxels: '
                                                                           'do not generate ODFs if None [μm]')
    cli_parser.add_argument('--odf-deg', type=int, default=6,
                            help='degrees of the spherical harmonics series expansion (even number between 2 and 10)')

    # parse arguments
    cli_args = cli_parser.parse_args()

    return cli_args


def get_image_info(img, px_size, ch_fiber, mosaic=False, ch_axis=None):
    """
    Get information on the input microscopy volume image.

    Parameters
    ----------
    img: numpy.ndarray
        microscopy volume image

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    ch_fiber: int
        myelinated fibers channel

    mosaic: bool
        True for tiled reconstructions aligned using ZetaStitcher

    ch_axis: int
        channel axis

    Returns
    -------
    img_shape: numpy.ndarray (shape=(3,), dtype=int)
        volume image shape [px]

    img_shape_um: numpy.ndarray (shape=(3,), dtype=float)
        volume image shape [μm]

    img_item_size: int
        array item size (in bytes)
    """

    # adapt channel axis
    img_shape = np.asarray(img.shape)
    ndim = len(img_shape)
    if ndim == 4:
        ch_axis = 1 if mosaic else -1
    elif ndim == 3:
        ch_fiber = None

    # get info on microscopy volume image
    if ch_axis is not None:
        img_shape = np.delete(img_shape, ch_axis)
    img_shape_um = np.multiply(img_shape, px_size)
    img_item_size = get_item_bytes(img)

    return img_shape, img_shape_um, img_item_size, ch_fiber


def get_image_file(cli_args):
    """
    Get microscopy image file path and format.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    Returns
    -------
    img_path: str
        path to the microscopy volume image

    img_name: str
        name of the microscopy volume image

    img_fmt: str
        format of the microscopy volume image

    mosaic: bool
        True for tiled reconstructions aligned using ZetaStitcher

    in_mmap: bool
        create a memory-mapped array of the microscopy volume image,
        increasing the parallel processing performance
        (the image will be preliminarily loaded to RAM)
    """

    # get microscopy image path and name
    in_mmap = cli_args.mmap
    img_path = cli_args.image_path
    img_name = path.basename(img_path)
    split_name = img_name.split('.')

    # check image format
    if len(split_name) == 1:
        raise ValueError('Format must be specified for input volume images!')
    else:
        img_fmt = split_name[-1]
        img_name = img_name.replace('.' + split_name[-1], '')
        mosaic = True if img_fmt == 'yml' else False

    return img_path, img_name, img_fmt, mosaic, in_mmap


def get_pipeline_config(cli_args, vector, img_name):
    """
    Retrieve the Foa3D pipeline configuration.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    vector: bool
        True for fiber orientation vector data

    img_name: str
        name of the input volume image

    Returns
    -------
    alpha: float
        plate-like score sensitivity

    beta: float
        blob-like score sensitivity

    gamma: float
        background score sensitivity

    smooth_sigma: numpy.ndarray (shape=(3,), dtype=int)
        3D standard deviation of the low-pass Gaussian filter [px]
        (resolution anisotropy correction)

    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    px_size_iso: numpy.ndarray (shape=(3,), dtype=float)
        adjusted isotropic pixel size [μm]

    odf_scales_um: list (dtype: float)
        list of fiber ODF resolution values (super-voxel sides [μm])

    odf_degrees: int
        degrees of the spherical harmonics series expansion

    z_min: int
        minimum output z-depth [px]

    z_max: int
        maximum output z-depth [px]

    ch_neuron: int
        neuronal bodies channel

    ch_fiber: int
        myelinated fibers channel

    lpf_soma_mask: bool
        neuronal body masking

    max_ram_mb: float
        maximum RAM available to the Frangi filtering stage [MB]

    jobs: int
        number of parallel jobs (threads)
        used by the Frangi filtering stage

    img_name: str
        microscopy image filename
    """

    # Frangi filter parameters
    alpha = cli_args.alpha
    beta = cli_args.beta
    gamma = cli_args.gamma
    scales_um = cli_args.scales
    if type(scales_um) is not list:
        scales_um = [scales_um]

    # pipeline parameters
    lpf_soma_mask = cli_args.neuron_mask
    ch_neuron = cli_args.ch_neuron
    ch_fiber = cli_args.ch_fiber
    max_ram = cli_args.ram
    max_ram_mb = None if max_ram is None else max_ram * 1000
    jobs = cli_args.jobs

    # ODF parameters
    odf_scales_um = cli_args.odf_res
    odf_degrees = cli_args.odf_deg

    # microscopy image pixel size and PSF FWHM
    px_size, psf_fwhm = get_resolution(cli_args, vector)

    # forced output z-range
    z_min = cli_args.z_min
    z_max = cli_args.z_max
    z_min = int(np.floor(z_min / px_size[0]))
    if z_max is not None:
        z_max = int(np.ceil(z_max / px_size[0]))

    # preprocessing configuration
    smooth_sigma, px_size_iso = config_anisotropy_correction(px_size, psf_fwhm, vector)

    # add pipeline configuration prefix to output filenames
    if not vector:
        pfx = get_output_prefix(scales_um, alpha, beta, gamma)
        img_name = '{}img{}'.format(pfx, img_name)

    return alpha, beta, gamma, scales_um, smooth_sigma, px_size, px_size_iso, odf_scales_um, odf_degrees, \
        z_min, z_max, ch_neuron, ch_fiber, lpf_soma_mask, max_ram_mb, jobs, img_name


def get_resolution(cli_args, vector):
    """
    Retrieve microscopy resolution information from command line arguments.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    vector: bool
        True for fiber orientation vector data

    Returns
    -------
    px_size: numpy.ndarray (shape=(3,), dtype=float)
        pixel size [μm]

    psf_fwhm: numpy.ndarray (shape=(3,), dtype=float)
        3D PSF FWHM [μm]
    """

    # pixel size
    px_size_z = cli_args.px_size_z
    px_size_xy = cli_args.px_size_xy

    # psf size
    psf_fwhm_z = cli_args.psf_fwhm_z
    psf_fwhm_y = cli_args.psf_fwhm_y
    psf_fwhm_x = cli_args.psf_fwhm_x

    # create arrays
    px_size = np.array([px_size_z, px_size_xy, px_size_xy])
    psf_fwhm = np.array([psf_fwhm_z, psf_fwhm_y, psf_fwhm_x])

    # print resolution info
    if not vector:
        print_resolution(px_size, psf_fwhm)

    return px_size, psf_fwhm


def load_microscopy_image(cli_args):
    """
    Load microscopy volume image from TIFF, NumPy or ZetaStitcher .yml file.
    Alternatively, the processing pipeline accepts as input .npy or HDF5
    files of fiber orientation vector data: in this case, the Frangi filter
    stage will be skipped.

    Parameters
    ----------
    cli_args: see ArgumentParser.parse_args
        populated namespace of command line arguments

    Returns
    -------
    img: numpy.ndarray or NumPy memory-map object
        microscopy volume image or array of fiber orientation vectors

    mosaic: bool
        True for tiled microscopy reconstructions aligned using ZetaStitcher

    skip_frangi: bool
        True when pre-estimated fiber orientation vectors
        are directly provided to the pipeline

    cli_args: see ArgumentParser.parse_args
        updated namespace of command line arguments

    save_subdirs: list (dtype=str)
        saving subdirectory string paths

    tmp_dir: str
        temporary file directory

    img_name: str
        microscopy image filename
    """

    # initialize variables
    skip_frangi = False
    skip_odf = cli_args.odf_res is None
    tmp_dir = tempfile.mkdtemp()

    # retrieve volume path and name
    img_path, img_name, img_fmt, mosaic, in_mmap = get_image_file(cli_args)

    # import start time
    tic = perf_counter()

    # fiber orientation vector data
    if img_fmt == 'npy' or img_fmt == 'h5':

        # print heading
        print(color_text(0, 191, 255, "\nFiber Orientation Data Import\n"))

        # load fiber orientations
        if img_fmt == 'npy':
            img = np.load(img_path, mmap_mode='r')
        else:
            img_file = h5py.File(img_path, 'r')
            img = img_file.get(img_file.keys()[0])

        # check dimensions
        if img.ndim != 4:
            raise ValueError('Invalid fiber orientation dataset (ndim != 4)')
        else:
            skip_frangi = True
            print("Loading {} orientation dataset...\n".format(img_name))

    # microscopy volume image
    else:
        # print heading
        print(color_text(0, 191, 255, "\nMicroscopy Volume Image Import\n"))

        # load microscopy tiled reconstruction (aligned using ZetaStitcher)
        if mosaic:
            print("Loading {} tiled reconstruction...\n".format(img_name))
            img = VirtualFusedVolume(img_path)

        # load microscopy z-stack
        else:
            print("Loading {} z-stack...\n".format(img_name))
            img_fmt = img_fmt.lower()
            if img_fmt == 'npy':
                img = np.load(img_path)
            elif img_fmt == 'tif' or img_fmt == 'tiff':
                img = tiff.imread(img_path)
            else:
                raise ValueError('Unsupported image format!')

        # grey channel fiber image
        if len(img.shape) == 3:
            cli_args.neuron_mask = False

        # create image memory map
        if in_mmap:
            img = create_memory_map(img.shape, dtype=img.dtype, name=img_name, tmp=tmp_dir, arr=img[:], mmap_mode='r')

    # print import time
    print_import_time(tic)

    # print volume image shape
    print_image_shape(cli_args, img, mosaic) if not skip_frangi else print()

    # create saving directory
    save_subdirs = create_save_dirs(img_path, img_name, skip_frangi=skip_frangi, skip_odf=skip_odf)

    return img, mosaic, skip_frangi, cli_args, save_subdirs, tmp_dir, img_name
