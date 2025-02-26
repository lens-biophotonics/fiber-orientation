.. _installation:

Installation
============
Create a virtual Python environment by executing the venv module:

.. code-block:: console

    $ python -m venv .foa3d_env

Activate the newly created environment:

.. code-block:: console

    $ source .foa3d_env/bin/activate

Install the wheel tool:

.. code-block:: console

    $ pip install wheel

Build the Python wheel file by executing:

.. code-block:: console

    $ python setup.py bdist_wheel

Install the wheel using pip:

.. code-block:: console

    $ pip install dist/foa3d-0.1.0-py3-none-any.whl

.. _usage:

Usage
=====

.. _format:

Microscopy image formats
------------------------
Foa3D supports 3D grayscale or RGB image stacks in TIFF format.
Alternatively, a YAML stitch file created by the ZetaStitcher tool for large volumetric stack alignment
and stitching [`ZetaStitcher GitHub <https://github.com/lens-biophotonics/ZetaStitcher>`_] can be provided as input.
To generate this stitch file from a collection of adjacent 3D stacks composing a tiled reconstruction of brain tissue,
refer to the documentation at [`ZetaStitcher GitHub <https://github.com/lens-biophotonics/ZetaStitcher>`_].
In particular, the Foa3D tool uses the 3D stack alignment information contained in such a file to programmatically
access and process basic image sub-volumes of appropriate size, allowing for the analysis of high-resolution mesoscopic
microscopy images that exceed the typical memory available on low-resource machines.
The YAML and image stack files must be located in the same directory.

.. code-block:: console

   $ foa3d path/to/zetastitch.yml

.. _resolution:

Microscopy image resolution
---------------------------
The lateral and longitudinal voxel size (in μm) must be specified via the command line,
along with the 3D full width at half maximum of the point spread function of the employed microscopy apparatus:

.. code-block:: console

   $ ... --px-size-xy 0.4 --px-size-z 1 --psf-fwhm-x 1.5 --psf-fwhm-y 1.4 --psf-fwhm-z 3.1

This information is required at the preprocessing stage of the pipeline to properly isotropize the spatial resolution
of the raw microscopy images. In fact, since two-photon scanning and light-sheet fluorescence microscopes are in
general characterized by a poorer resolution along the direction of the optical axis, the XY-plane of the sliced
image sub-volumes typically needs to be blurred. A tailored Gaussian smoothing kernel is used in this regard.
If not properly corrected, the residual anisotropy would otherwise introduce a systematic bias in the assessed
3D fiber orientations.

.. _frangi:

Frangi filter configuration
---------------------------
Fiber enhancement and segmentation is achieved via a multiscale 3D Frangi filter [`Frangi, et al., 1998 <https://doi.org/10.1007/BFb0056195>`_].
The spatial scales of the filter (in μm) can be provided via the ``-s/--scales`` option.
As discussed in [`Sorelli, et al., 2023 <https://doi.org/10.1038/s41598-023-30953-w>`_],
the optimal scales that best preserve the original intensity
and cross-sectional size of the 3D tubular structures present in the analized images
correspond to half of their expected radius.
The response of the Frangi filter is also affected by three sensitivity parameters, α, β, and γ.
In detail, lower α values tend to amplify the response of the filter to the presence of elongated structures,
whereas an increase in β determines a relatively higher sensitivity to blob-shaped structures.
Usually, the α and β sensitivity parameters need to be heuristically fixed for the specific application
or image modality of interest:
the default values, namely ``α=0.001`` and ``β=1``, were shown to lead to a marked selective enhancement of
tubular fiber structures, and to a considerable rejection of the neuronal soma.
Whereas α and β are linked to grey-level-invariant geometrical features,
the γ sensitivity is related to the image contrast:
if not specified by the user, this parameter is automatically set to half of the maximum Hessian norm computed
at each spatial scale of interest for each sliced image sub-volume.
In the example below, the 3D Frangi filter is tuned so as to favour the enhancement of fiber structures having a
cross-sectional diameter of 5 and 10 μm, with an automatic (local) contrast sensitivity:

.. code-block:: console

   $ ... -a 0.00001 -b 0.1 -s 1.25 2.5

Please keep in mind that the above automatic local adjustment of the γ sensitivity may produce discontinuities
between the fiber orientation vector fields resulting from adjacent image slices handled by separate CPU cores.

.. _parallelization:

Parallelization
---------------
In order to speed up the fiber orientation analysis on large brain tissue sections, the Foa3D tool divides the input 
image reconstruction into basic slices of appropriate shape and assigns them to separate concurrent workers.
By default, Foa3D will use all available logical cores: for instance, a batch of 32 image slices will be simultaneously
processed on a 32-core CPU. The multiscale Frangi filter, on the other hand, is not currently parallelized in order to
avoid unnecessary overhead and resource oversubscription within the nested parallel loop. The size of the basic image
slices is automatically determined by the available RAM. The ``--job`` and ``--ram`` options can be specified
differently via the command line to limit the employed resources.

.. code-block:: console

   $ ... --jobs 8 --ram 32

.. _somamask:

Soma rejection
--------------
A neuronal soma fluorescence channel may be optionally provided to Foa3D for improving the specificity of the resulting 
fiber orientation maps, which is otherwise dependent on the inherent attenuation of non-tubular objects offered by the 
Frangi filter. This is performed via a postprocessing step that further suppresses neuronal bodies by applying Yen's 
automatic thresholding algorithm to an optionally provided channel. The enhanced neuronal body rejection may be 
activated via the ``-c/--cell-msk`` option, modifying, if required, the default channel related to the soma fluorescence:

.. code-block:: console

   $ ... -c --fb-ch 0 --bc-ch 1

.. _odf:

Orientation distribution functions
----------------------------------
High-resolution fiber orientation data obtained at the native pixel size of the imaging system can be integrated into 
orientation distribution functions (ODFs), providing a comprehensive statistical description
of 3D fiber tract orientations within larger spatial compartments or super-voxels.
ODFs are highly suitable for a multimodal quantitative comparison with spatial fiber architectures
mapped by other high-resolution optical modalities, as 3D-Polarized Light Imaging
[`Axer, et al., 2016 <https://doi.org/10.3389/fnana.2016.00040>`_].
Furthermore, the spatial downscaling produced by the ODF estimation allows to bridge the gulf between the meso-
and macro-scale connectomics that is generally targeted by diffusion magnetic resonance imaging (dMRI).
The Foa3D tool features the generation of fiber ODFs from the 3D orientation vector fields returned by
the Frangi filtering stage via the fast analytical approach described in
[`Alimi, et al., 2020 <https://doi.org/10.1016/j.media.2020.101760>`_].
Alimi's method is computationally efficient and is characterized by improved angular precision and resolution
with respect to deriving the ODFs by modeling local directional histograms of discretized fiber orientations.
The multiscale estimation of fiber ODFs may be enabled by providing a list of super-voxel sides (in μm) via
the ``-o/--odf-res`` option:

.. code-block:: console

   $ ... --odf-res 25 50 100 200

Foa3D also provides the possibility to directly execute the multiscale analysis of fiber ODFs,
skipping the Frangi filter stage, on pre-computed 3D fiber orientation vector fields (TIFF format):

.. code-block:: console

   $ foa3d.py path/to/fiber_vector_field.npy --odf-res 500 1000

The fiber ODFs returned by the Foa3D tool may be accessed using the open source MRtrix3 software package
for medical image processing and visualization
[`Tournier, et al., 2019 <https://doi.org/10.1016/j.neuroimage.2019.116137>`_].

Output description
------------------
The Frangi filter and ODF generation stages of the Foa3D tool export the series of TIFF and NIfTI images listed below.
Images exported by default are reported in bold; the remaining ones can be produced as well, e.g. for testing purposes,
by selecting the "export all" option (-e or --exp-all) via CLI.

#. Frangi filter stage:

    * Normalized response of the Frangi filter (*path/to/save_dir/frangi/frangi_filter_\*cfg_sfx\**, type: uint8, format: TIFF)

    * Binarized response of the Frangi filter (*path/to/save_dir/frangi/fiber_msk_\*cfg_sfx\**, type: uint8, format: TIFF)

    * Optional mask of neuronal cell bodies (*path/to/save_dir/frangi/soma_msk_\*cfg_sfx\**, type: uint8, format: TIFF)

    * **Fiber orientation vector field** (*path/to/save_dir/frangi/fiber_vec_\*cfg_sfx\**, type: float32, format: TIFF)

    * **Fiber orientation colormap** (*path/to/save_dir/frangi/fiber_cmap_\*cfg_sfx\**, type: uint8, format: TIFF)

    * Fractional anisotropy (*path/to/save_dir/frangi/frac_anis_\*cfg_sfx\**, type: float32, format: TIFF)

#. Orientation distribution functions (ODF) stage (one file for each super-voxel size requested via CLI):

    * **ODF** (*path/to/save_dir/odf/odf_mrtrixview_\*cfg_sfx\**, type: float32, format: NIfTI)

    * **ODF background** (*path/to/save_dir/odf/bg_mrtrixview_\*cfg_sfx\**, type: uint8, format: NIfTI)

    * **Total fiber orientation dispersion** (*path/to/save_dir/odf/odi_tot_\*cfg_sfx\**, type: float32, format: TIFF)

    * Primary fiber orientation dispersion (*path/to/save_dir/odf/odi_pri_\*cfg_sfx\**, type: float32, format: TIFF)

    * Secondary fiber orientation dispersion (*path/to/save_dir/odf/odi_sec_\*cfg_sfx\**, type: float32, format: TIFF)

    * Fiber orientation dispersion anisotropy (*path/to/save_dir/odf/odi_anis_\*cfg_sfx\**, type: float32, format: TIFF)

The suffix *\*cfg_sfx\** reports information on the particular configuration of the tool, namely:
name of the input microscopy image, scale(s) of the 3D Frangi filter; sensitivity of the filter (α, β, γ);
super-voxel size (ODF stage only).
