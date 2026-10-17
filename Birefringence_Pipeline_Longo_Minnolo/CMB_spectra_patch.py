"""Patch-based CMB cross-spectra, analytic covariances and beta fits.

Stage 1 of the anisotropic cosmic birefringence (CB) pipeline.

Pipeline
--------
1. namaster_patch_definitive.py : patch spectra and covariances (this script)
2. mapbeta.py                   : birefringence-angle (beta) maps
3. spectra_cross_beta.py        : beta power spectra
4. A_CB_stats.py                : amplitude fit and upper limit

Purpose
-------
The sky is divided into patches, defined as the pixels of a low-resolution
HEALPix grid (``nsidepatch``). The CB angle is assumed to be constant within a
patch. For every patch that passes the sky-coverage criterion, this script:

1. builds apodised temperature (T) and polarisation (P) masks;
2. simulates pairs of CMB-plus-white-noise maps (splits A and B) that share the
   same CMB signal but have independent noise;
3. computes with NaMaster the A x B cross-spectra (TT, EE, BB, TE, EB, TB) of
   every simulation, and their analytic Gaussian covariance matrices;
4. validates the spectra against the fiducial model and, as a cross-check, fits
   the birefringence angle beta in every simulation by minimising the
   chi-square of the EB D-estimator (Gruppuso et al. 2016, arXiv:1604.05202).

The beta maps used by the following stages are produced by mapbeta.py from the
spectra and covariances written here.

Conventions
-----------
* Band-powers are D_ell = ell (ell + 1) C_ell / (2 pi), in uK^2.
* Spectra are stored in the order [TT, EE, BB, TE, EB, TB].
* Two indices label the patches: ``i`` is the HEALPix pixel index of the patch
  and ``k`` (called ``ipatch`` in the code) is the running index of the
  retained patches. All the output files are labelled with ``k``.

Inputs
------
mask_in_dir
    Planck 2018 (PR3) common Galactic masks:
    COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits and
    COM_Mask_CMB-common-Mask-Pol_2048_R3.00.fits.
Cls_dir_in
    ffp10_lensedCls.dat, the Planck FFP10 fiducial lensed spectra. The code
    assumes the columns ell, TT, EE, BB, TE, with D_ell in uK^2, starting at
    ell = 2.

Outputs
-------
Patch masks, simulated maps, NaMaster workspaces, covariance matrices (.npy),
cross-spectra (.txt) and validation plots, written to the sub-directories of
``root_dir`` defined in ``main``. These directories must exist before the
script is run.

Usage
-----
The patches are distributed across MPI ranks::

    mpirun -n <n_ranks> python namaster_patch_definitive.py

The run configuration is set in the call to ``main`` at the end of the file.

Requirements
------------
numpy, scipy, healpy, matplotlib, mpi4py and pymaster (NaMaster; Alonso et al.
2019, arXiv:1809.09603).
"""

import numpy as np
import healpy as hp
import matplotlib
import os

# Use a non-interactive backend so that figures can be written on headless
# nodes (e.g. HPC clusters).
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import perf_counter

import pymaster as nmt
from mpi4py import MPI
from scipy.optimize import minimize

# MPI environment: the patches are distributed across the ranks (see main).
# Without mpirun a single rank is used.
comm  = MPI.COMM_WORLD
nproc = comm.Get_size()
rank  = comm.Get_rank()


def read_Cls (Cls_dir_in,lmax,ll):
    """Read the fiducial CMB spectra and convert them to C_ell.

    Parameters
    ----------
    Cls_dir_in : str
        Directory containing ``ffp10_lensedCls.dat``.
    lmax : int
        Maximum multipole.
    ll : numpy.ndarray
        Array ell (ell + 1) / (2 pi) for ell = 0, ..., lmax, used to convert
        D_ell to C_ell.

    Returns
    -------
    numpy.ndarray
        Array of shape (6, lmax + 1) with the C_ell (uK^2) in the order [TT,
        EE, BB, TE, EB, TB]. EB and TB are zero (parity-conserving model), as
        are the monopole and the dipole.
    """

    # Fiducial lensed spectra (Planck FFP10): columns 1-4 hold TT, EE, BB and
    # TE as D_ell in uK^2, starting at ell = 2.
    TT_th,EE_th,BB_th,TE_th=np.loadtxt(Cls_dir_in+'ffp10_lensedCls.dat',unpack=True,usecols=(1,2,3,4))

    # Output in the order [TT, EE, BB, TE, EB, TB], indexed by ell = 0, ...,
    # lmax.
    Cls_th=np.zeros((6,lmax+1))

    # Convert D_ell to C_ell. The monopole and the dipole stay at zero.
    Cls_th[0,2:]=TT_th[0:lmax-1]/ll[2:]
    Cls_th[1,2:]=EE_th[0:lmax-1]/ll[2:]
    Cls_th[2,2:]=BB_th[0:lmax-1]/ll[2:]
    Cls_th[3,2:]=TE_th[0:lmax-1]/ll[2:]

    # EB and TB vanish in the fiducial (parity-conserving) model.
    Cls_th[4, :] = 0.0
    Cls_th[5, :] = 0.0
    return(Cls_th)


def create_mask_patch(mask_in_dir,mask_patch_dir,nsideout,nsidepatch,npix_patch,aposcale,apotype,plots_dir):
        """Build and save the apodised T and P masks of the retained patches.

        Each patch (a pixel of the HEALPix grid of resolution ``nsidepatch``)
        is upgraded to ``nsideout`` and multiplied by the Planck Galactic
        masks. It is retained if at least 50% of its pixels are unmasked in
        both the temperature and the polarisation mask. The masks of the
        retained patches are apodised and saved with a running index k = 0, 1,
        ... (not the HEALPix index).

        Parameters
        ----------
        mask_in_dir : str
            Directory with the Planck PR3 common masks.
        mask_patch_dir : str
            Output directory for the patch masks and the patch count.
        nsideout : int
            HEALPix resolution of the simulated maps and of the masks.
        nsidepatch : int
            HEALPix resolution that defines the patches.
        npix_patch : int
            Number of patches, 12 * nsidepatch^2.
        aposcale : float
            Apodisation scale in degrees.
        apotype : str
            NaMaster apodisation profile (e.g. 'C2').
        plots_dir : str
            Directory for the plots; its sub-directory ``quality_maps/`` must
            exist.

        Notes
        -----
        Writes patch_maskT_nside_<nsidepatch>_index_<k>.fits,
        patch_maskP_nside_<nsidepatch>_index_<k>.fits and
        active_patches_nside_<nsidepatch>.txt (number of retained patches) in
        ``mask_patch_dir``, and a preview map of the mean coverage of each
        patch in ``plots_dir``.
        """

        # Map of the mean unmasked fraction (%) of each patch; the discarded
        # patches stay hp.UNSEEN.
        patch_quality_map = np.full(npix_patch, hp.UNSEEN)

        # Planck PR3 common Galactic masks (intensity and polarisation),
        # degraded to nsideout.
        mask_galT = hp.ud_grade(hp.read_map(mask_in_dir + 'COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits', field=0, dtype=np.float64),nsideout, dtype= np.float64)
        mask_galP = hp.ud_grade(hp.read_map(mask_in_dir + 'COM_Mask_CMB-common-Mask-Pol_2048_R3.00.fits', field=0, dtype=np.float64),nsideout, dtype= np.float64)

        # Binarise the degraded masks at 0.5.
        mask_galT[mask_galT < 0.5] = 0.0
        mask_galT[mask_galT >= 0.5] = 1.0
        mask_galP[mask_galP < 0.5] = 0.0
        mask_galP[mask_galP >= 0.5] = 1.0

        # Running index of the retained patches (distinct from the HEALPix
        # index i).
        k = 0

        # Loop over all the patches (pixels of the HEALPix grid of resolution
        # nsidepatch).
        for i in range(npix_patch):

            print("creazione patch numero   :",i)

            # Binary mask of patch i, upgraded to nsideout.
            mask_patch_LR = np.zeros(npix_patch)
            mask_patch_LR[i] = 1

            mask_patch_HR = hp.ud_grade(mask_patch_LR, nsideout, dtype= np.float64)
            mask_patch_HR[mask_patch_HR < 0.5] = 0.0
            mask_patch_HR[mask_patch_HR >= 0.5] = 1.0

            # Fraction of the pixels of the patch that survive the temperature
            # and polarisation Galactic masks.
            total_patch_pixels = np.sum(mask_patch_HR)
            total_maskT = mask_patch_HR * mask_galT
            total_maskT_pixels = np.sum(total_maskT)
            fractionT = total_maskT_pixels / total_patch_pixels

            total_maskP = mask_patch_HR * mask_galP
            total_maskP_pixels = np.sum(total_maskP)
            fractionP = total_maskP_pixels / total_patch_pixels

            # Retain the patch only if at least 50% of its pixels are unmasked
            # in both T and P.
            if fractionT >= 0.5 and fractionP >= 0.5:

                # Apodise the temperature and polarisation masks of the patch
                # and save them with the running index k. The preview map
                # stores the mean of the T and P coverage (%).
                maskT = total_maskT
                maskT = nmt.mask_apodization(maskT, aposcale, apotype=apotype)
                hp.write_map(mask_patch_dir+'patch_maskT_nside_'+str(nsidepatch)+
                              '_index_'+str(k).zfill(4)+'.fits', maskT, overwrite=True, dtype= np.float64)

                maskP = total_maskP
                patch_quality_map[i] = (fractionP + fractionT)/2 * 100
                maskP = nmt.mask_apodization(maskP, aposcale, apotype=apotype)
                hp.write_map(mask_patch_dir+'patch_maskP_nside_'+str(nsidepatch)+
                '_index_'+str(k).zfill(4)+'.fits', maskP, overwrite=True, dtype= np.float64)
                k = k + 1

            else:
                 patch_quality_map[i] = hp.UNSEEN

        # Preview map of the retained patches and of their mean coverage.
        hp.mollview(patch_quality_map, title="%Average ctive Pixels", unit="%", cmap='viridis', min=50, max=100)
        plt.savefig(plots_dir +'quality_maps/' + 'patch_preview' + '.png')
        plt.close()

        # Store the number of retained patches (read back in main).
        np.savetxt(mask_patch_dir+'active_patches_nside_'+str(nsidepatch)+
                              '.txt', [k], fmt='%d')


def create_maps(Cls_th,lmax,nsideout,nsims,maps_dir,fwhm,var,npix_map,sigma_noise):
    """Simulate pairs of CMB-plus-noise maps (splits A and B).

    Each realisation draws one lensed-CMB signal map and adds two independent
    sets of white noise, which gives two maps with identical signal and
    independent noise. This allows noise-bias-free cross-spectra to be computed
    later.

    Parameters
    ----------
    Cls_th : numpy.ndarray
        Fiducial C_ell of shape (6, lmax + 1) in the order [TT, EE, BB, TE, EB,
        TB] (uK^2), as returned by ``read_Cls``.
    lmax : int
        Maximum multipole of the signal.
    nsideout : int
        HEALPix resolution of the maps.
    nsims : int
        Number of realisations.
    maps_dir : str
        Output directory.
    fwhm : float
        FWHM of the Gaussian beam applied to the signal, in degrees.
    var : float
        Per-pixel white-noise variance of the temperature maps (uK^2). The
        variance of the Q and U noise is 2 * var.
    npix_map : int
        Number of pixels of the maps.
    sigma_noise : float
        Noise level in uK-arcmin; used only to label the output files.

    Notes
    -----
    Each map is saved as a FITS file (T, Q, U; float64) named
    mapA_nside<nside>_whitenoise<sigma>microK_arcmin_sim<i>.fits, and likewise
    for B. The noise is added after the beam convolution, so it is not
    smoothed. The random number generator is not seeded: runs are not bit-wise
    reproducible.
    """

    # Each realisation yields two maps (splits A and B) with the same signal
    # and independent white noise.
    for i in range (nsims):
        # Independent white-noise draws for the two splits: variance var for T
        # and 2 * var for Q and U (two sets of npix_map values for T, four for
        # Q and U).
        noisesT = np.random.normal(loc=0, scale=np.sqrt(var), size=2 * npix_map)
        noisesP = np.random.normal(loc=0, scale=np.sqrt(2 * var), size=4 * npix_map)

        # Common signal: lensed-CMB realisation smoothed with a Gaussian beam
        # of FWHM fwhm (deg) and the HEALPix pixel window.
        m_sign=hp.synfast(cls=Cls_th ,lmax=lmax,nside=nsideout,pol=True,new=True,fwhm=np.deg2rad(fwhm),
            pixwin=True)

        # Split maps = signal + noise (the noise is added after the beam, so it
        # is not smoothed).
        ma_TQU = np.zeros_like(m_sign)
        mb_TQU = np.zeros_like(m_sign)

        ma_TQU[0] = m_sign[0] + noisesT[0:npix_map]
        ma_TQU[1] = m_sign[1] + noisesP[0:npix_map]
        ma_TQU[2] = m_sign[2] + noisesP[npix_map:2 * npix_map]

        mb_TQU[0] = m_sign[0] + noisesT[npix_map:2 * npix_map]
        mb_TQU[1] = m_sign[1] + noisesP[2 * npix_map:3 * npix_map]
        mb_TQU[2] = m_sign[2] + noisesP[3 * npix_map:4 * npix_map]

        # Output: one FITS file (T, Q, U) per split and simulation.
        filenameA = (maps_dir +'mapA_nside' + str(nsideout) +'_whitenoise' + str(sigma_noise) +'microK_arcmin' +
                     '_sim' + str(i).zfill(4) + '.fits')

        filenameB = (maps_dir +'mapB_nside' + str(nsideout) +'_whitenoise' + str(sigma_noise) +'microK_arcmin' +
                     '_sim' + str(i).zfill(4) + '.fits')
        hp.write_map(filenameA, ma_TQU, overwrite=True, dtype= np.float64)
        hp.write_map(filenameB, mb_TQU, overwrite=True, dtype= np.float64)

        print(f"[Rank 0] Generazione Mappe: {i} / {nsims}", flush=True)


def create_workspace (lmax,deltal,is_Dell,fwhm,nsideout,maskT_out,maskP_out,purify_e,purify_b,ws_dir,maps_dir,sigma_noise,ipatch):
    """Compute and store the NaMaster workspaces of one patch.

    Writes (i) the mode-coupling matrices of the A x B cross-spectra (00: TT;
    02: TE, TB; 20: ET, BT; 22: EE, EB, BE, BB) and (ii) the workspaces and
    covariance coupling coefficients needed for the analytic Gaussian
    covariance (see ``calculate_cov_th``). They depend only on the masks, beams
    and binning, so they are computed once per patch and re-used for all
    simulations.

    Parameters
    ----------
    lmax : int
        Maximum multipole.
    deltal : int
        Multipole bin width.
    is_Dell : bool
        If True, band-powers are D_ell = ell (ell + 1) C_ell / (2 pi).
    fwhm : float
        Beam FWHM in degrees.
    nsideout : int
        Resolution of the maps (used for the pixel window).
    maskT_out, maskP_out : numpy.ndarray
        Apodised temperature and polarisation masks of the patch.
    purify_e, purify_b : bool
        Enable the E and B purification of the polarisation field.
    ws_dir : str
        Output directory of the workspaces.
    maps_dir : str
        Directory of the simulated maps; simulation 0 is read only to
        instantiate the NaMaster fields.
    sigma_noise : float
        Noise level in uK-arcmin (used to build the file names of the maps).
    ipatch : int
        Running index k of the patch.
    """

    # Linear binning of width deltal; the band-powers are D_ell if is_Dell is
    # True.
    b = nmt.NmtBin.from_lmax_linear(lmax, deltal, is_Dell=is_Dell)

    # Transfer functions: Gaussian beam and pixel window of the temperature
    # (index 0) and polarisation (index 1) maps; sm is their squared product.
    bl = hp.gauss_beam(
        np.deg2rad(fwhm),
        lmax=lmax,
        pol=True
    )
    wpix = hp.pixwin(nsideout, pol=True, lmax=lmax)

    # sm[0] and sm[1]: squared transfer functions (beam x pixel window) of the
    # temperature and polarisation maps.
    sm = np.zeros((2,len(bl)))
    sm[0] = (bl[:,0]*wpix[0])**2
    sm[1] = (bl[:,1]*wpix[1])**2

    # Simulation 0 is read only to instantiate the NaMaster fields: the
    # mode-coupling matrices depend on the masks, beams and binning, not on the
    # map values.
    filenameA = (
        maps_dir
        + 'mapA_nside' + str(nsideout)
        + '_whitenoise' + str(sigma_noise) + 'microK_arcmin'
        + '_sim' + str(0).zfill(4) + '.fits'
    )

    filenameB = (
        maps_dir
        + 'mapB_nside' + str(nsideout)
        + '_whitenoise' + str(sigma_noise) + 'microK_arcmin'
        + '_sim' + str(0).zfill(4) + '.fits'
    )

    ma_TQU = hp.read_map(filenameA, [0, 1, 2], dtype= np.float64)
    mb_TQU = hp.read_map(filenameB, [0, 1, 2], dtype= np.float64)

    # NaMaster fields of splits A and B: spin-0 (temperature) and spin-2
    # (polarisation, with optional E/B purification). The beam is passed so
    # that NaMaster corrects the spectra for it.
    f_0_a = nmt.NmtField(
        maskT_out,
        [ma_TQU[0]],
        beam=np.sqrt(sm[0]),
        lmax=lmax
    )
    f_0_b = nmt.NmtField(
        maskT_out,
        [mb_TQU[0]],
        beam=np.sqrt(sm[0]),
        lmax=lmax
    )
    f_2_a = nmt.NmtField(
        maskP_out,
        [ma_TQU[1], ma_TQU[2]],
        purify_e=purify_e,
        purify_b=purify_b,
        beam=np.sqrt(sm[1]),
        lmax=lmax
    )
    f_2_b = nmt.NmtField(
        maskP_out,
        [mb_TQU[1], mb_TQU[2]],
        purify_e=purify_e,
        purify_b=purify_b,
        beam=np.sqrt(sm[1]),
        lmax=lmax
    )

    # Fields with null maps, used only to compute the coupling coefficients of
    # the analytic covariance (only their masks and beams matter).
    f0_cov = nmt.NmtField(
        maskT_out,
        [np.zeros(maskT_out.shape)],

        beam=np.sqrt(sm[0]),
        lmax=lmax
    )

    f2_cov = nmt.NmtField(
        maskP_out,
        [np.zeros(maskP_out.shape),np.zeros(maskP_out.shape)],
        purify_e=purify_e,
        purify_b=purify_b,
        beam=np.sqrt(sm[1]),
        lmax=lmax
    )

    # Mode-coupling workspaces of the four families of A x B cross-spectra: 00
    # (TT), 02 (TE, TB), 20 (ET, BT) and 22 (EE, EB, BE, BB).
    w_00 = nmt.NmtWorkspace()
    w_02 = nmt.NmtWorkspace()
    w_20 = nmt.NmtWorkspace()
    w_22 = nmt.NmtWorkspace()

    # Compute the mode-coupling matrices and save them for re-use (one set per
    # patch).
    w_00.compute_coupling_matrix(f_0_a, f_0_b, b)
    w_02.compute_coupling_matrix(f_0_a, f_2_b, b)
    w_20.compute_coupling_matrix(f_2_a, f_0_b, b)
    w_22.compute_coupling_matrix(f_2_a, f_2_b, b)

    w_00.write_to(ws_dir + 'w_00_deltaell_' + str(deltal).zfill(1) + '_patch'+str(ipatch).zfill(4)+'.fits')
    w_02.write_to(ws_dir + 'w_02_deltaell_' + str(deltal).zfill(1) +'_patch'+str(ipatch).zfill(4)+ '.fits')
    w_20.write_to(ws_dir + 'w_20_deltaell_' + str(deltal).zfill(1) + '_patch'+str(ipatch).zfill(4)+'.fits')
    w_22.write_to(ws_dir + 'w_22_deltaell_' + str(deltal).zfill(1) + '_patch'+str(ipatch).zfill(4)+'.fits')

    # Workspaces and covariance coupling coefficients for the analytic Gaussian
    # covariance (used by calculate_cov_th).
    w_00_th = nmt.NmtWorkspace()
    w_02_th = nmt.NmtWorkspace()
    w_22_th = nmt.NmtWorkspace()

    Cw_00 = nmt.NmtCovarianceWorkspace()
    Cw_02 = nmt.NmtCovarianceWorkspace()
    Cw_22 = nmt.NmtCovarianceWorkspace()

    w_00_th.compute_coupling_matrix (f0_cov, f0_cov, b)
    w_02_th.compute_coupling_matrix (f0_cov, f2_cov, b)
    w_22_th.compute_coupling_matrix  (f2_cov, f2_cov, b)

    Cw_00.compute_coupling_coefficients   (f0_cov, f0_cov, f0_cov, f0_cov)
    Cw_02.compute_coupling_coefficients   (f0_cov, f2_cov, f0_cov, f2_cov)
    Cw_22.compute_coupling_coefficients   (f2_cov, f2_cov, f2_cov, f2_cov)

    w_00_th.write_to(ws_dir + 'w_00_th_deltaell_' + str(deltal).zfill(1) + '_patch'+str(ipatch).zfill(4)+'.fits')
    w_02_th.write_to(ws_dir + 'w_02_th_deltaell_' + str(deltal).zfill(1) + '_patch'+str(ipatch).zfill(4)+'.fits')
    w_22_th.write_to(ws_dir + 'w_22_th_deltaell_' + str(deltal).zfill(1) + '_patch'+str(ipatch).zfill(4)+'.fits')

    Cw_00.write_to(ws_dir + 'Cw_00_deltaell_' + str(deltal).zfill(1) +'_patch'+str(ipatch).zfill(4)+ '.fits')
    Cw_02.write_to(ws_dir + 'Cw_02_deltaell_' + str(deltal).zfill(1) + '_patch'+str(ipatch).zfill(4)+'.fits')
    Cw_22.write_to(ws_dir + 'Cw_22_deltaell_' + str(deltal).zfill(1) + '_patch'+str(ipatch).zfill(4)+'.fits')


def calculate_cov_th(lmax, fwhm, nsideout,
        ws_dir,deltal,
        is_Dell,Cls_th, cov_dir,sigma_noise,ipatch
    ):
    """Compute the analytic Gaussian covariance matrices of one patch.

    Uses the Gaussian covariance estimator of NaMaster with the fiducial
    spectra (convolved with the beam and the pixel window) plus the white-noise
    power of the splits. Noise is included following the structure of the
    covariance of an A x B cross-spectrum, whose two splits have independent
    noise. The covariance of the symmetrised TE, TB and EB estimators is the
    average of the four corresponding blocks (e.g. TE-TE, TE-ET, ET-TE, ET-ET).
    The matrices, of shape (n_bins, n_bins), are saved as
    cov_ana_<XY>_patch<k>.npy for XY = TT, TE, TB, EE, BB, EB.

    Parameters
    ----------
    lmax : int
        Maximum multipole.
    fwhm : float
        Beam FWHM in degrees.
    nsideout : int
        Resolution of the maps (used for the pixel window).
    ws_dir : str
        Directory with the workspaces written by ``create_workspace``.
    deltal : int
        Multipole bin width.
    is_Dell : bool
        If True, band-powers are D_ell.
    Cls_th : numpy.ndarray
        Fiducial C_ell of shape (6, lmax + 1).
    cov_dir : str
        Output directory of the covariance matrices.
    sigma_noise : float
        White-noise level in uK-arcmin (temperature).
    ipatch : int
        Running index k of the patch.
    """

    b = nmt.NmtBin.from_lmax_linear(lmax, deltal, is_Dell=is_Dell)

    # Beam and pixel-window transfer functions (see create_workspace).
    bl = hp.gauss_beam(
        np.deg2rad(fwhm),
        lmax=lmax,
        pol=True
    )
    wpix = hp.pixwin(nsideout, pol=True, lmax=lmax)

    # Squared transfer functions (beam x pixel window) of T and P.
    sm = np.zeros((2,len(bl)))
    sm[0] = (bl[:,0]*wpix[0])**2

    sm[1] = (bl[:,1]*wpix[1])**2

    ell_arr = b.get_effective_ells()
    n_ell = len(ell_arr)

    # White-noise power of each split: N_ell = (sigma_noise converted from
    # uK-arcmin to uK-radian)^2 for T, and twice that for Q and U.
    Cl_noiseT = np.full(lmax+1,(sigma_noise*np.deg2rad(1/60))**2)
    Cl_noiseP = np.full(lmax+1,(np.sqrt(2)*sigma_noise*np.deg2rad(1/60))**2)

    w_00_th = nmt.NmtWorkspace()
    w_02_th = nmt.NmtWorkspace()
    w_22_th = nmt.NmtWorkspace()

    Cw_00 = nmt.NmtCovarianceWorkspace()
    Cw_02 = nmt.NmtCovarianceWorkspace()
    Cw_22 = nmt.NmtCovarianceWorkspace()

    # Read back the workspaces written by create_workspace.
    w_00_th.read_from(ws_dir + 'w_00_th_deltaell_' + str(deltal).zfill(1) +'_patch'+str(ipatch).zfill(4)+ '.fits')
    w_02_th.read_from(ws_dir + 'w_02_th_deltaell_' + str(deltal).zfill(1) +'_patch'+str(ipatch).zfill(4)+ '.fits')
    w_22_th.read_from(ws_dir + 'w_22_th_deltaell_' + str(deltal).zfill(1) +'_patch'+str(ipatch).zfill(4)+ '.fits')

    Cw_00.read_from(ws_dir + 'Cw_00_deltaell_' + str(deltal).zfill(1) +'_patch'+str(ipatch).zfill(4)+ '.fits')
    Cw_02.read_from(ws_dir + 'Cw_02_deltaell_' + str(deltal).zfill(1) + '_patch'+str(ipatch).zfill(4)+'.fits')
    Cw_22.read_from(ws_dir + 'Cw_22_deltaell_' + str(deltal).zfill(1) + '_patch'+str(ipatch).zfill(4)+'.fits')

    # TT x TT covariance. The function takes the spins of the four fields
    # followed by the four spectra C(a1,b1), C(a1,b2), C(a2,b1) and C(a2,b2).
    # The spectra are the fiducial ones convolved with the beam and the pixel
    # window. The white-noise power is added to the terms that involve the
    # auto-spectrum of a split (a1,b1 and a2,b2) and not to the cross-split
    # terms, since the two splits have independent noise. The other blocks add
    # noise following the same principle.
    covar_00_00 = nmt.gaussian_covariance(Cw_00,
                                      0, 0, 0, 0,
                                      [Cls_th[0,:]*sm[0]+Cl_noiseT],
                                      [Cls_th[0,:]*sm[0]],
                                      [Cls_th[0,:]*sm[0]],
                                      [Cls_th[0,:]*sm[0]+Cl_noiseT],
                                      w_00_th, wb=w_00_th).reshape([n_ell, 1,
                                                            n_ell, 1])

    # Extract the (n_ell x n_ell) block and save it.
    covar_TT_TT = covar_00_00[:, 0, :, 0]
    np.save(cov_dir+'cov_ana_TT'+'_patch'+str(ipatch).zfill(4)+'.npy',covar_TT_TT)

    # Spin-0 x spin-2 blocks (T with E and B). The result has shape (n_ell, 2,
    # n_ell, 2); the two components are TE and TB.
    covar_02_02 = nmt.gaussian_covariance(Cw_02, 0, 2, 0, 2,
                                          [Cls_th[0,:]*sm[0]+Cl_noiseT],
                                          [Cls_th[3,:]*np.sqrt(sm[0])*np.sqrt(sm[1]), Cls_th[5,:]*np.sqrt(sm[0])*np.sqrt(sm[1])],
                                          [Cls_th[3,:]*np.sqrt(sm[0])*np.sqrt(sm[1]), Cls_th[5,:]*np.sqrt(sm[0])*np.sqrt(sm[1])],
                                          [Cls_th[1,:]*sm[1]+Cl_noiseP, Cls_th[4,:]*sm[1],
                                           Cls_th[4,:]*sm[1], Cls_th[2,:]*sm[1]+Cl_noiseP],
                                          w_02_th, wb=w_02_th).reshape([n_ell, 2,
                                                                    n_ell, 2])
    covar_TE_TE = covar_02_02[:, 0, :, 0]
    covar_TE_TB = covar_02_02[:, 0, :, 1]
    covar_TB_TE = covar_02_02[:, 1, :, 0]
    covar_TB_TB = covar_02_02[:, 1, :, 1]

    # Covariance between the T x P and P x T orderings (e.g. TE with ET). It is
    # needed because the TE and TB spectra are the average of the two
    # orderings.
    covar_02_20 = nmt.gaussian_covariance(Cw_02, 0, 2, 0, 2,
                                          [Cls_th[0,:]*sm[0]+Cl_noiseT],
                                          [Cls_th[3,:]*np.sqrt(sm[0])*np.sqrt(sm[1]), Cls_th[5,:]*np.sqrt(sm[0])*np.sqrt(sm[1])],
                                          [Cls_th[3,:]*np.sqrt(sm[0])*np.sqrt(sm[1]), Cls_th[5,:]*np.sqrt(sm[0])*np.sqrt(sm[1])],
                                          [Cls_th[1,:]*sm[1], Cls_th[4,:]*sm[1],
                                           Cls_th[4,:]*sm[1], Cls_th[2,:]*sm[1]],
                                          w_02_th, wb=w_02_th).reshape([n_ell, 2,
                                                            n_ell, 2])
    covar_TE_ET = covar_02_20[:, 0, :, 0]
    covar_TE_BT = covar_02_20[:, 0, :, 1]
    covar_TB_ET = covar_02_20[:, 1, :, 0]
    covar_TB_BT = covar_02_20[:, 1, :, 1]

    # Blocks of the P x T ordering (ET and BT).
    covar_20_02 = nmt.gaussian_covariance(Cw_02, 2, 0, 2, 0,
                                          [Cls_th[1,:]*sm[1], Cls_th[4,:]*sm[1],
                                           Cls_th[4,:]*sm[1], Cls_th[2,:]*sm[1]],
                                          [Cls_th[3,:]*np.sqrt(sm[0])*np.sqrt(sm[1]), Cls_th[5,:]*np.sqrt(sm[0])*np.sqrt(sm[1])],
                                          [Cls_th[3,:]*np.sqrt(sm[0])*np.sqrt(sm[1]), Cls_th[5,:] *np.sqrt(sm[0])*np.sqrt(sm[1])],
                                          [Cls_th[0,:]*sm[0]+Cl_noiseT],
                                          w_02_th, wb=w_02_th).reshape([n_ell, 2,
                                                                n_ell, 2])
    covar_ET_TE = covar_20_02[:, 0, :, 0]
    covar_ET_TB = covar_20_02[:, 0, :, 1]
    covar_BT_TE = covar_20_02[:, 1, :, 0]
    covar_BT_TB = covar_20_02[:, 1, :, 1]

    covar_20_20 = nmt.gaussian_covariance(Cw_02, 2, 0, 2, 0,
                                          [Cls_th[1,:]*sm[1]+Cl_noiseP, Cls_th[4,:]*sm[1],
                                           Cls_th[4,:]*sm[1], Cls_th[2,:]*sm[1]+Cl_noiseP],
                                          [Cls_th[3,:]*np.sqrt(sm[0])*np.sqrt(sm[1]), Cls_th[5,:]*np.sqrt(sm[0])*np.sqrt(sm[1])],
                                          [Cls_th[3,:]*np.sqrt(sm[0])*np.sqrt(sm[1]), Cls_th[5,:]*np.sqrt(sm[0])*np.sqrt(sm[1])],
                                          [Cls_th[0,:]*sm[0]+Cl_noiseT],
                                          w_02_th, wb=w_02_th).reshape([n_ell, 2,
                                                                n_ell, 2])
    covar_ET_ET = covar_20_20[:, 0, :, 0]
    covar_ET_BT = covar_20_20[:, 0, :, 1]
    covar_BT_ET = covar_20_20[:, 1, :, 0]
    covar_BT_BT = covar_20_20[:, 1, :, 1]

    # Covariance of the symmetrised estimators (X_a Y_b + Y_a X_b) / 2 for TE
    # and TB: the average of the four blocks (e.g. TE-TE, TE-ET, ET-TE and
    # ET-ET).
    np.save(cov_dir+'cov_ana_TE'+'_patch'+str(ipatch).zfill(4)+'.npy',(covar_TE_TE+covar_TE_ET+covar_ET_TE+covar_ET_ET)/4.)
    np.save(cov_dir+'cov_ana_TB'+'_patch'+str(ipatch).zfill(4)+'.npy',(covar_TB_TB+covar_TB_BT+covar_BT_TB+covar_BT_BT)/4.)

    # Spin-2 x spin-2 blocks. The result has shape (n_ell, 4, n_ell, 4), with
    # the components ordered as EE, EB, BE, BB.
    covar_22_22 = nmt.gaussian_covariance(Cw_22, 2, 2, 2, 2,
                                              [Cls_th[1,:]*sm[1]+Cl_noiseP, Cls_th[4,:]*sm[1],
                                               Cls_th[4,:]*sm[1], Cls_th[2,:]*sm[1]+Cl_noiseP],
                                              [Cls_th[1,:]*sm[1], Cls_th[4,:]*sm[1],
                                               Cls_th[4,:]*sm[1], Cls_th[2,:]*sm[1]],
                                              [Cls_th[1,:]*sm[1], Cls_th[4,:]*sm[1],
                                               Cls_th[4,:]*sm[1], Cls_th[2,:]*sm[1]],
                                              [Cls_th[1,:]*sm[1]+Cl_noiseP, Cls_th[4,:]*sm[1],
                                               Cls_th[4,:]*sm[1], Cls_th[2,:]*sm[1]+Cl_noiseP],
                                              w_22_th, wb=w_22_th).reshape([n_ell, 4,
                                                                    n_ell, 4])
    covar_EE_EE = covar_22_22[:, 0, :, 0]
    # covar_EE_EB = covar_22_22[:, 0, :, 1]
    # covar_EE_BE = covar_22_22[:, 0, :, 2]
    # covar_EE_BB = covar_22_22[:, 0, :, 3]
    # covar_EB_EE = covar_22_22[:, 1, :, 0]
    covar_EB_EB = covar_22_22[:, 1, :, 1]
    covar_EB_BE = covar_22_22[:, 1, :, 2]
    # covar_EB_BB = covar_22_22[:, 1, :, 3]
    # covar_BE_EE = covar_22_22[:, 2, :, 0]
    covar_BE_EB = covar_22_22[:, 2, :, 1]
    covar_BE_BE = covar_22_22[:, 2, :, 2]
    # covar_BE_BB = covar_22_22[:, 2, :, 3]
    # covar_BB_EE = covar_22_22[:, 3, :, 0]
    # covar_BB_EB = covar_22_22[:, 3, :, 1]
    # covar_BB_BE = covar_22_22[:, 3, :, 2]
    covar_BB_BB = covar_22_22[:, 3, :, 3]

    # Save the EE and BB covariances; the EB covariance is that of the
    # symmetrised estimator (EB + BE) / 2, i.e. the average of the four EB and
    # BE blocks.
    np.save(cov_dir+'cov_ana_EE'+'_patch'+str(ipatch).zfill(4)+'.npy',covar_EE_EE)
    np.save(cov_dir+'cov_ana_BB'+'_patch'+str(ipatch).zfill(4)+'.npy',covar_BB_BB)
    np.save(cov_dir+'cov_ana_EB'+'_patch'+str(ipatch).zfill(4)+'.npy',(covar_EB_EB+covar_EB_BE+covar_BE_EB+covar_BE_BE)/4.)

# Legacy helper (disabled) that built matrices to bin (P) and un-bin (Q)
# band-powers manually. The binning is now handled by NaMaster (NmtBin).
# def PQ_matrix(lmin,lmax,deltal):
#     nbin = int(np.floor((lmax-lmin+1)/deltal))
#     P  = np.zeros((nbin,lmax-lmin+1))
#     Qt = np.zeros((nbin,lmax-lmin+1))
#     j = 0
#     k = 0
#     for i in range(nbin):
#         j = k
#         k += deltal
#         P[i,j:k]  = 1./deltal
#         Qt[i,j:k] = 1.
#     Q  = Qt.T
#     Pt = P.T
#     return (P, Pt, Q, Qt)


def calculate_spectrum_pol(
        lmax, fwhm, nsideout,
        ws_dir, Cls_th, nsims,
        deltal, maps_dir, sigma_noise,
        purify_e, purify_b, maskT_out,
        maskP_out,Cls_out_dir,
        is_Dell,Cls_dir_in,ipatch,nsidepatch,new_spectra
    ):
    """Compute (or load) the A x B cross-spectra of all simulations of a patch.

    For every simulation, the T, Q, U maps of splits A and B are read, masked
    with the patch masks and cross-correlated with NaMaster; the mode coupling
    is removed with the precomputed workspaces. Cross-spectra with swapped
    fields are averaged: TE = (TE + ET) / 2, EB = (EB + BE) / 2 and
    TB = (TB + BT) / 2. The fiducial spectra, coupled and decoupled with the
    same workspaces, are also returned for validation.

    Parameters
    ----------
    lmax : int
        Maximum multipole.
    fwhm : float
        Beam FWHM in degrees.
    nsideout : int
        Resolution of the maps.
    ws_dir : str
        Directory with the mode-coupling workspaces.
    Cls_th : numpy.ndarray
        Fiducial C_ell of shape (6, lmax + 1).
    nsims : int
        Number of simulations.
    deltal : int
        Multipole bin width.
    maps_dir : str
        Directory of the simulated maps.
    sigma_noise : float
        Noise level in uK-arcmin (used to build the file names of the maps).
    purify_e, purify_b : bool
        Enable the E and B purification of the polarisation field.
    maskT_out, maskP_out : numpy.ndarray
        Apodised temperature and polarisation masks of the patch.
    Cls_out_dir : str
        Directory where the spectra of each simulation are written (or read).
    is_Dell : bool
        If True, band-powers are D_ell.
    Cls_dir_in : str
        Not used in this function.
    ipatch : int
        Running index k of the patch.
    nsidepatch : int
        HEALPix resolution of the patches (used in the file names).
    new_spectra : bool
        If True, compute the spectra from the maps and save them; if False,
        read the files saved by a previous run.

    Returns
    -------
    Dls_sims : numpy.ndarray, shape (nsims, 6, n_bins)
        Band-powers of all simulations, in the order [TT, EE, BB, TE, EB, TB].
    mean_Dls : numpy.ndarray, shape (6, n_bins)
        Mean over the simulations.
    std_Dls : numpy.ndarray, shape (6, n_bins)
        Standard deviation over the simulations (ddof = 0).
    err_Dls : numpy.ndarray, shape (6, n_bins)
        Standard error of the mean.
    var_sim : numpy.ndarray, shape (6, n_bins)
        Variance over the simulations.
    dl_TT_th, dl_EE_th, dl_BB_th, dl_TE_th : numpy.ndarray, shape (n_bins,)
        Expected band-powers of the estimator for the fiducial model (coupled
        with the mask and binned).
    ell_arr : numpy.ndarray, shape (n_bins,)
        Effective multipole of each bin.
    """

    b = nmt.NmtBin.from_lmax_linear(lmax, deltal, is_Dell=is_Dell)
    ell_arr = b.get_effective_ells()

    # Beam and pixel-window transfer functions.
    bl = hp.gauss_beam(
        np.deg2rad(fwhm),
        lmax=lmax,
        pol=True
    )
    wpix = hp.pixwin(nsideout, pol=True, lmax=lmax)

    # Transfer functions (beam x pixel window) passed to NaMaster, which
    # corrects the spectra for them.
    beam_T = bl[:, 0] * wpix[0]
    beam_P = bl[:, 1] * wpix[1]

    # Load the mode-coupling workspaces computed by create_workspace for this
    # patch.
    w_00 = nmt.NmtWorkspace()
    w_02 = nmt.NmtWorkspace()
    w_20 = nmt.NmtWorkspace()
    w_22 = nmt.NmtWorkspace()

    w_00.read_from(ws_dir + 'w_00_deltaell_' + str(deltal).zfill(1) +'_patch'+str(ipatch).zfill(4)+ '.fits')
    w_02.read_from(ws_dir + 'w_02_deltaell_' + str(deltal).zfill(1) +'_patch'+str(ipatch).zfill(4)+ '.fits')
    w_20.read_from(ws_dir + 'w_20_deltaell_' + str(deltal).zfill(1) +'_patch'+str(ipatch).zfill(4)+ '.fits')
    w_22.read_from(ws_dir + 'w_22_deltaell_' + str(deltal).zfill(1) +'_patch'+str(ipatch).zfill(4)+ '.fits')

    N_bin = b.get_n_bands()

    # Band-powers D_ell (uK^2) indexed as [simulation, spectrum, bin], with
    # spectrum index 0:TT, 1:EE, 2:BB, 3:TE, 4:EB, 5:TB.
    Dls_sims = np.zeros((nsims, 6, N_bin))

    # Compute the spectra from the maps and save them (True), or read the files
    # written by a previous run (False).
    if new_spectra:

        for j in range(nsims):
                filename_out = (
                    Cls_out_dir
                    + 'NamasterCls_lmax' + str(lmax)
                    + '_deltal' + str(deltal)
                    + '_sim' + str(j).zfill(4) +'_patch'+str(ipatch)+'_nsidepatch'+str(nsidepatch)+'.txt'
                )

                # Disabled option: skip the simulations whose output file
                # already exists, to resume an interrupted run.
                # if os.path.exists(filename_out) and os.path.getsize(filename_out) > 0:
                #     try:
                #         data_loaded = np.loadtxt(filename_out, unpack=True)

                #         Dls_sims[j, 0, :] = data_loaded[1] # TT
                #         Dls_sims[j, 1, :] = data_loaded[2] # EE
                #         Dls_sims[j, 2, :] = data_loaded[3] # BB
                #         Dls_sims[j, 3, :] = data_loaded[4] # TE
                #         Dls_sims[j, 4, :] = data_loaded[5] # EB
                #         Dls_sims[j, 5, :] = data_loaded[6] # TB

                #         print(f"[Rank {rank}] Patch {ipatch} Sim {j} skipped (found).", flush=True)
                #         continue
                #     except:
                #         pass

                # Maps of the two splits for simulation j.
                filenameA = (
                    maps_dir
                    + 'mapA_nside' + str(nsideout)
                    + '_whitenoise' + str(sigma_noise) + 'microK_arcmin'
                    + '_sim' + str(j).zfill(4) + '.fits'
                )

                filenameB = (
                    maps_dir
                    + 'mapB_nside' + str(nsideout)
                    + '_whitenoise' + str(sigma_noise) + 'microK_arcmin'
                    + '_sim' + str(j).zfill(4) + '.fits'
                )

                ma_TQU = hp.read_map(filenameA, [0, 1, 2], dtype= np.float64)
                mb_TQU = hp.read_map(filenameB, [0, 1, 2], dtype= np.float64)

                # Masked temperature (spin-0) and polarisation (spin-2) fields
                # of the two splits.
                f_0_a = nmt.NmtField(
                    maskT_out,
                    [ma_TQU[0]],
                    beam=beam_T,
                    lmax=lmax
                )
                f_0_b = nmt.NmtField(
                    maskT_out,
                    [mb_TQU[0]],
                    beam=beam_T,
                    lmax=lmax
                )
                f_2_a = nmt.NmtField(
                    maskP_out,
                    [ma_TQU[1], ma_TQU[2]],
                    purify_e=purify_e,
                    purify_b=purify_b,
                    beam=beam_P,
                    lmax=lmax
                )
                f_2_b = nmt.NmtField(
                    maskP_out,
                    [mb_TQU[1], mb_TQU[2]],
                    purify_e=purify_e,
                    purify_b=purify_b,
                    beam=beam_P,
                    lmax=lmax
                )

                # Coupled (pseudo-) cross-spectra between split A and split B.
                # The noise of the two splits is independent, so the
                # cross-spectra are free of noise bias.
                dl_coupled_00 = nmt.compute_coupled_cell(f_0_a, f_0_b)
                dl_coupled_02 = nmt.compute_coupled_cell(f_0_a, f_2_b)
                dl_coupled_20 = nmt.compute_coupled_cell(f_2_a, f_0_b)
                dl_coupled_22 = nmt.compute_coupled_cell(f_2_a, f_2_b)

                # Remove the mode coupling and bin. Output components: 00 ->
                # TT; 02 -> TE, TB; 20 -> ET, BT; 22 -> EE, EB, BE, BB.
                dl_00 = w_00.decouple_cell(dl_coupled_00)
                dl_02 = w_02.decouple_cell(dl_coupled_02)
                dl_20 = w_20.decouple_cell(dl_coupled_20)
                dl_22 = w_22.decouple_cell(dl_coupled_22)

                # Write one text file per simulation (columns: ell, TT, EE, BB,
                # TE, EB, TB; D_ell in uK^2). Spectra with swapped fields are
                # averaged: TE = (TE + ET) / 2, EB = (EB + BE) / 2 and
                # TB = (TB + BT) / 2.
                np.savetxt(
                    filename_out,
                    np.column_stack([
                        ell_arr,
                        dl_00[0],
                        dl_22[0],
                        dl_22[3],
                        (dl_02[0] + dl_20[0]) / 2,
                        (dl_22[1] + dl_22[2]) / 2,
                        (dl_02[1] + dl_20[1]) / 2

                    ]),
                    header='#l TT EE BB TE EB TB'
                )

                # Keep the same band-powers in memory.
                Dls_sims[j, 0, :] = dl_00[0]
                Dls_sims[j, 1, :] = dl_22[0]
                Dls_sims[j, 2, :] = dl_22[3]
                Dls_sims[j, 3, :] = 0.5 * (dl_02[0] + dl_20[0])
                Dls_sims[j, 4, :] = 0.5 * (dl_22[1] + dl_22[2])
                Dls_sims[j, 5, :] = 0.5 * (dl_02[1] + dl_20[1])

                print(f"[Rank {rank}] Patch {ipatch} | Sto calcolando sim {j}/{nsims}...", flush=True)

    # Read the spectra saved by a previous run.
    else :
        for j in range(nsims):
            filename_out = (
                    Cls_out_dir
                    + 'NamasterCls_lmax' + str(lmax)
                    + '_deltal' + str(deltal)
                    + '_sim' + str(j).zfill(4) +'_patch'+str(ipatch)+'_nsidepatch'+str(nsidepatch)+'.txt')
            Dls_sims[j, :, :] = np.loadtxt(
                filename_out,
                usecols=(1, 2, 3, 4, 5, 6),
                unpack=True
            )

            print(f"[Rank {rank}] Patch {ipatch} | Sto leggendo sim {j}/{nsims}...", flush=True)

    # Statistics over the simulations: mean, standard deviation (ddof = 0),
    # standard error of the mean and variance of the band-powers.
    mean_Dls = np.mean(Dls_sims, axis=0)
    std_Dls = np.std(Dls_sims, axis=0)
    err_Dls = std_Dls / np.sqrt(nsims)
    var_sim = np.var(Dls_sims,axis=0)

    # Fiducial spectra arranged in the NaMaster component layout of each
    # family: 00 (TT), 02 (TE, TB) and 22 (EE, EB, BE, BB).
    cl_00_th = np.zeros((1, lmax + 1))
    cl_02_th = np.zeros((2, lmax + 1))
    cl_22_th = np.zeros((4, lmax + 1))

    cl_00_th[0, :] = Cls_th[0, :]
    cl_22_th[0, :] = Cls_th[1, :]
    cl_22_th[3, :] = Cls_th[2, :]
    cl_02_th[0, :] = Cls_th[3, :]

    # Expectation value of the estimator: couple the fiducial spectra with the
    # mode-coupling matrix, then decouple and bin them. It is the reference of
    # the validation plots.
    dl_00_th_dec = w_00.decouple_cell(w_00.couple_cell(cl_00_th))
    dl_02_th_dec = w_02.decouple_cell(w_02.couple_cell(cl_02_th))
    dl_22_th_dec = w_22.decouple_cell(w_22.couple_cell(cl_22_th))

    dl_TT_th = dl_00_th_dec[0]
    dl_EE_th = dl_22_th_dec[0]
    dl_BB_th = dl_22_th_dec[3]
    dl_TE_th = dl_02_th_dec[0]

    return (
        Dls_sims,
        mean_Dls,
        std_Dls,
        err_Dls,
        var_sim,
        dl_TT_th,
        dl_EE_th,
        dl_BB_th,
        dl_TE_th,
        ell_arr
    )


def plot_validation (ell_arr,nsideout,deltal,is_Dell, sim, err, th, delta, name_spectra, nsims,
                      lmax,lmin, plots_dir,var_sim,var_master,ipatch,nsidepatch, yscale='log'):
        """Plot the validation of one spectrum for one patch.

        Three panels are saved in a PNG file: the simulated mean with the error
        on the mean and the theory curve (top); the residuals in units of the
        error on the mean with the +/-3 sigma band (middle); the empirical
        variance across the simulations versus the analytic NaMaster variance
        (bottom). Only the bins entirely contained in [lmin, 2 * nsideout] are
        shown.

        Parameters
        ----------
        ell_arr : numpy.ndarray
            Effective multipole of each bin.
        nsideout : int
            HEALPix resolution of the maps; the maximum multipole shown is
            2 * nsideout.
        deltal : int
            Multipole bin width.
        is_Dell : bool
            Not used in this function.
        sim : numpy.ndarray
            Simulated mean band-power in each bin.
        err : numpy.ndarray
            Error on the mean in each bin.
        th : numpy.ndarray
            Expected band-power in each bin (zero for EB and TB).
        delta : numpy.ndarray
            Residuals (sim - th) / err.
        name_spectra : str
            Name of the spectrum (e.g. 'TT'), used in the labels and file name.
        nsims : int
            Number of simulations.
        lmax, lmin : int
            Maximum and minimum multipoles.
        plots_dir : str
            Output directory.
        var_sim : numpy.ndarray
            Empirical variance across the simulations.
        var_master : numpy.ndarray
            Analytic variance (diagonal of the NaMaster covariance matrix).
        ipatch : int
            Running index k of the patch.
        nsidepatch : int
            HEALPix resolution of the patches.
        yscale : {'log', 'linear'}, optional
            Scale of the y axis of the top panel.
        """

        # Keep only the bins entirely contained in [lmin, 2 * nsideout].
        limit_l = 2 * nsideout
        ell_upper_bound = ell_arr + (deltal / 2.0)
        ell_lower_bound = ell_arr - (deltal / 2.0)
        valid_indices = (ell_upper_bound <= limit_l) & (ell_lower_bound >= lmin)
        ell_plot = ell_arr[valid_indices]
        sim_cut = sim[valid_indices]
        err_cut = err[valid_indices]
        th_cut  = th[valid_indices]
        delta_cut = delta[valid_indices]
        var_sim_cut = var_sim[valid_indices]
        var_master_cut = var_master[valid_indices]

        # Top: simulated mean with the error on the mean and the theory curve.
        # Middle: residuals in units of the error on the mean, with the +/-3
        # sigma band. Bottom: empirical variance across the simulations versus
        # the analytic NaMaster variance.
        fig, (ax1, ax2, ax3) = plt.subplots(
             3, 1,
             figsize=(7, 8),
             gridspec_kw={'height_ratios': [3, 1, 3]},
             sharex=False
         )

        ax1.errorbar(
             ell_plot,
             sim_cut,
             yerr=err_cut,
             fmt='o',
             markersize=3,
             label=r'$\langle D_{\ell}\rangle \pm \sigma_{\mu}$'
         )
        ax1.plot(ell_plot, th_cut, label=r'$D_{\ell}^{\mathrm{th}}$'' '+name_spectra)
        ax1.set_yscale(yscale)
        ax1.set_xscale('linear')
        #ax1.tick_params(labelbottom=False)
        ax1.set_ylabel(r'$D_{\ell}$')
        ax1.legend(fontsize=8)
        ax1.grid(True, ls='--', alpha=0.4)

        ax2.axhline(0, color='black', lw=0.8)
        ax2.plot(ell_plot, delta_cut, 'o', markersize=2)

        # Guide lines at +/-3 sigma.
        ax2.axhline(3, color='red', ls='--')
        ax2.axhline(-3, color='red', ls='--')
        ax2.set_xscale('linear')
        #ax2.tick_params(labelbottom=False)
        ax2.set_ylim(-5, 5)
        ax2.set_ylabel(r'$\Delta/\sigma$')
        ax2.grid(True, ls='--', alpha=0.4)
        ax3.plot(ell_plot, var_sim_cut, 'o-', markersize=2,
        label=r'$\sigma^2_{\bar D_\ell}{}^{\mathrm{emp}}$')

        ax3.plot(ell_plot, var_master_cut, 'o-', markersize=2,
        label=r'$\sigma^2_{\bar D_\ell}{}^{\mathrm{th\_master}}$'+name_spectra)

        ax3.legend()
        ax3.set_yscale('log')
        ax3.set_xscale('log')
        ax3.set_xlabel(r'Multipole $\ell$')
        ax3.set_ylabel(r'$\sigma^2_{\bar D_\ell}$')
        ax3.grid(True, ls='--', alpha=0.2)

        # One PNG file per spectrum and patch.
        plt.savefig(plots_dir+'Dl_'+name_spectra+'_nsims' + str(nsims)+
                     '_lmax' + str(lmax)+'_nsidepatch'+str(nsidepatch)+'_index_'+str(ipatch).zfill(4)+'.png')
        plt.tight_layout()
        plt.close()


def validation_spectra(
        mean_Dls, std_Dls, err_Dls,
        ell_arr, plots_dir,
        nsims, lmax,lmin, Cls_dir_in,var_sim,
        cov_dir,sigma_noise,ll,fwhm,nsideout,
        dl_TT_th,dl_EE_th,dl_BB_th,dl_TE_th,ipatch,nsidepatch,deltal,is_Dell
        ):
    """Validate the simulated spectra of one patch against theory and
    covariance.

    Computes the residuals of the simulated mean with respect to the fiducial
    expectation (TT, EE, BB, TE) or to zero (EB, TB), in units of the error on
    the mean, loads the analytic covariance matrices, and produces the
    validation plots of the six spectra with ``plot_validation``.

    Parameters
    ----------
    mean_Dls, err_Dls : numpy.ndarray, shape (6, n_bins)
        Mean band-powers and errors on the mean, in the order [TT, EE, BB, TE,
        EB, TB].
    std_Dls : numpy.ndarray
        Not used in this function.
    ell_arr : numpy.ndarray
        Effective multipole of each bin.
    plots_dir : str
        Output directory of the plots.
    nsims : int
        Number of simulations.
    lmax, lmin : int
        Maximum and minimum multipoles.
    Cls_dir_in : str
        Not used in this function.
    var_sim : numpy.ndarray, shape (6, n_bins)
        Variance of the band-powers across the simulations.
    cov_dir : str
        Directory with the analytic covariance matrices.
    sigma_noise : float
        Not used in this function.
    ll : numpy.ndarray
        Not used in this function.
    fwhm : float
        Beam FWHM in degrees (only used to compute transfer functions that are
        not used afterwards).
    nsideout : int
        HEALPix resolution of the maps.
    dl_TT_th, dl_EE_th, dl_BB_th, dl_TE_th : numpy.ndarray
        Expected band-powers for the fiducial model.
    ipatch : int
        Running index k of the patch.
    nsidepatch : int
        HEALPix resolution of the patches.
    deltal : int
        Multipole bin width.
    is_Dell : bool
        If True, band-powers are D_ell.
    """

    # NOTE: bl, wpix and sm are computed here but not used below.
    bl = hp.gauss_beam(
        np.deg2rad(fwhm),
        lmax=lmax,
        pol=True
    )
    wpix = hp.pixwin(nsideout, pol=True, lmax=lmax)

    sm = np.zeros((2,len(bl)))
    sm[0] = (bl[:,0]*wpix[0])**2
    sm[1] = (bl[:,1]*wpix[1])**2

    Dl_TT_sim = mean_Dls[0]
    Dl_EE_sim = mean_Dls[1]
    Dl_BB_sim = mean_Dls[2]
    Dl_TE_sim = mean_Dls[3]
    Dl_EB_sim = mean_Dls[4]
    Dl_TB_sim = mean_Dls[5]

    err_TT = err_Dls[0]
    err_EE = err_Dls[1]
    err_BB = err_Dls[2]
    err_TE = err_Dls[3]
    err_EB = err_Dls[4]
    err_TB = err_Dls[5]

    # Residuals of the simulated mean with respect to the theory, in units of
    # the error on the mean.
    delta_TT = (Dl_TT_sim - dl_TT_th) / err_TT
    delta_EE = (Dl_EE_sim - dl_EE_th) / err_EE
    delta_BB = (Dl_BB_sim - dl_BB_th) / err_BB
    delta_TE = (Dl_TE_sim - dl_TE_th) / err_TE

    # EB and TB vanish in the fiducial model, so their residual is the mean
    # over its error.
    delta_EB = (Dl_EB_sim ) / err_EB
    delta_TB = (Dl_TB_sim ) / err_TB

    # Analytic covariance matrices computed by calculate_cov_th; only their
    # diagonals (the variances) are compared with the simulations.
    C_master_TT = np.load(cov_dir + 'cov_ana_TT'+'_patch'+str(ipatch).zfill(4) +'.npy')
    C_master_EE = np.load(cov_dir + 'cov_ana_EE'+'_patch'+str(ipatch).zfill(4) +'.npy')
    C_master_BB = np.load(cov_dir + 'cov_ana_BB'+'_patch'+str(ipatch).zfill(4) +'.npy')
    C_master_TE = np.load(cov_dir + 'cov_ana_TE'+'_patch'+str(ipatch).zfill(4) +'.npy')
    C_master_EB = np.load(cov_dir + 'cov_ana_EB'+'_patch'+str(ipatch).zfill(4) +'.npy')
    C_master_TB = np.load(cov_dir + 'cov_ana_TB'+'_patch'+str(ipatch).zfill(4) +'.npy')
    var_master_TT = np.diag(C_master_TT)
    var_master_EE = np.diag(C_master_EE)
    var_master_BB = np.diag(C_master_BB)
    var_master_TE = np.diag(C_master_TE)
    var_master_EB = np.diag(C_master_EB)
    var_master_TB = np.diag(C_master_TB)

    # Validation plots of the six spectra of this patch.
    plot_validation (ell_arr,nsideout,deltal,is_Dell, Dl_TT_sim, err_TT, dl_TT_th, delta_TT, 'TT',nsims, lmax,lmin, plots_dir,var_sim[0,:],var_master_TT,ipatch,nsidepatch)
    plot_validation (ell_arr,nsideout,deltal,is_Dell, Dl_EE_sim, err_EE, dl_EE_th, delta_EE, 'EE',nsims, lmax,lmin, plots_dir,var_sim[1,:],var_master_EE,ipatch,nsidepatch)
    plot_validation (ell_arr,nsideout,deltal,is_Dell, Dl_BB_sim, err_BB, dl_BB_th, delta_BB, 'BB',nsims, lmax,lmin, plots_dir,var_sim[2,:],var_master_BB,ipatch,nsidepatch)
    plot_validation (ell_arr,nsideout,deltal,is_Dell, Dl_TE_sim, err_TE, dl_TE_th, delta_TE, 'TE',nsims, lmax,lmin, plots_dir,var_sim[3,:],var_master_TE,ipatch,nsidepatch ,yscale='linear')
    plot_validation (ell_arr,nsideout,deltal,is_Dell, Dl_EB_sim, err_EB, np.zeros(dl_TT_th.shape[0]), delta_EB, 'EB',nsims, lmax,lmin, plots_dir,var_sim[4,:],var_master_EB,ipatch,nsidepatch,yscale='linear')
    plot_validation (ell_arr,nsideout,deltal,is_Dell, Dl_TB_sim, err_TB, np.zeros(dl_TT_th.shape[0]), delta_TB, 'TB',nsims, lmax,lmin, plots_dir,var_sim[5,:],var_master_TB,ipatch,nsidepatch,yscale='linear')


def D_estimators (Dls_sims,ell_arr,
        nsims,lmax,lmin,nsideout,deltal,Cls_out_dir,ipatch,nsidepatch,plots_dir,vardl,varname

    ):
    """Fit the angle beta in every simulation and plot its distribution.

    For each simulation, the EB D-estimator

        D_ell(beta) = D_ell^EB cos(4 beta) - 0.5 (D_ell^EE - D_ell^BB) sin(4
        beta)

    is evaluated in the bins contained in [lmin, 2 * nsideout], and the angle
    that minimises chi2(beta) = sum_ell D_ell(beta)^2 / var_ell (diagonal
    covariance, null expectation value) is found with a bounded L-BFGS-B
    minimisation (|beta| <= 1.5 degrees). A histogram of the best-fit angles is
    saved as a PNG file. This function is a per-patch validation: the
    production maps of beta are made by mapbeta.py.

    Parameters
    ----------
    Dls_sims : numpy.ndarray, shape (nsims, 6, n_bins)
        Band-powers of all simulations, in the order [TT, EE, BB, TE, EB, TB].
    ell_arr : numpy.ndarray
        Effective multipole of each bin.
    nsims : int
        Number of simulations.
    lmax, lmin : int
        Maximum and minimum multipoles (lmax only labels the output file).
    nsideout : int
        HEALPix resolution of the maps; the maximum multipole of the fit is
        2 * nsideout.
    deltal : int
        Multipole bin width.
    Cls_out_dir : str
        Not used in this function.
    ipatch : int
        Running index k of the patch.
    nsidepatch : int
        Not used in this function.
    plots_dir : str
        Output directory of the histogram.
    vardl : numpy.ndarray, shape (n_bins,)
        Variance of the EB band-powers in each bin.
    varname : str
        Label of the variance used ('var_sim' or 'var_master'), included in the
        plot title and in the file name.
    """

    # Expected value of the D-estimator when beta equals the true angle.
    mean_Dest = 0.0

    # Search range of the angle (+/-1.5 degrees) and grid step (0.001 degrees,
    # used by the disabled grid search), in radians.
    betamin = np.deg2rad(-1.5)
    betamax = np.deg2rad(1.5)
    step = np.deg2rad(0.001)

    # Select the bins entirely contained in [lmin, 2 * nsideout], i.e. the
    # multipole range of the fit.
    limit_l = 2 * nsideout
    ell_upper_bound = ell_arr + (deltal / 2.0)
    ell_lower_bound = ell_arr - (deltal / 2.0)
    valid_indices = (ell_upper_bound <= limit_l) & (ell_lower_bound >= lmin)

    # Variance and band-powers restricted to the selected bins.
    var_cut = vardl[valid_indices]
    dl_EE_sim = Dls_sims[:, 1, valid_indices]
    dl_BB_sim = Dls_sims[:, 2, valid_indices]
    dl_EB_sim = Dls_sims[:, 4, valid_indices]

    # Best-fit angle (radians) of each simulation: grid search (disabled) and
    # minimisation.
    betas_best_grid = np.ones(nsims)
    betas_best_func = np.ones(nsims)

    # Diagonal approximation of the inverse covariance of the D-estimator.
    inv_cov= np.diag(1/var_cut)

    def chi2_beta(beta, dl_EB, dl_EE,dl_BB, inv_cov):
            """Chi-square of the EB D-estimator for the angle beta (radians)."""

            b = beta[0]

            Dest_EB=(dl_EB*np.cos(4*b))-0.5*(dl_EE-dl_BB)*np.sin(4*b)
            x_l = Dest_EB - mean_Dest
            return x_l @ inv_cov @ x_l

    def jacobian (beta, dl_EB, dl_EE,dl_BB,inv_cov):
        """Central finite-difference gradient of chi2_beta (step 1e-10 rad)."""

        grad = np.zeros(beta.size)
        for i in range (grad.size):
            betap=np.copy(beta)
            betam=np.copy(beta)
            betap[i]+=1e-10
            betam[i]-=1e-10
            grad[i]=(chi2_beta(betap, dl_EB, dl_EE,dl_BB, inv_cov)-
                     chi2_beta(betam, dl_EB, dl_EE,dl_BB, inv_cov))/2e-10
        return grad

    # Fit the angle independently in every simulation.
    for i in range(nsims):

        dl_EE = dl_EE_sim[i]
        dl_BB = dl_BB_sim[i]
        dl_EB = dl_EB_sim[i]

        # Angle grid (used only by the disabled grid search below).
        betas = np.arange(betamin, betamax + step,step)
        chi2_grid = np.zeros(len(betas))

        # Disabled alternative: brute-force grid search of the minimum of chi2,
        # kept as a cross-check of the minimisation.
        # for j in range(len(betas)):
        #     beta=betas[j]

        #     Dest_EB=(dl_EB*np.cos(4*beta))-0.5*(dl_EE-dl_BB)*np.sin(4*beta)
        #     x_l=Dest_EB-mean_Dest
        #     chi2_grid[j] = x_l @ inv_cov @ x_l

        # betas_best_grid[i]=betas[np.argmin(chi2_grid)]

        # Bounded quasi-Newton minimisation of chi2, starting from beta = 0,
        # with a finite-difference gradient.
        res = minimize(chi2_beta,x0=np.array([0.0]),args=(dl_EB, dl_EE,dl_BB, inv_cov),
                       method='L-BFGS-B',jac=jacobian, bounds=[(betamin, betamax)])
        betas_best_func[i] = res.x[0]

    # Mean and sample standard deviation (ddof = 1) of the best-fit angles, in
    # degrees. The grid-search counterparts are disabled.
    # sigma_beta_grid = np.rad2deg(np.std(betas_best_grid, ddof=1))
    sigma_beta_func = np.rad2deg(np.std(betas_best_func, ddof=1))
    # mean_beta_grid = np.rad2deg(np.mean(betas_best_grid))
    mean_beta_func = np.rad2deg(np.mean(betas_best_func))

    # Histogram of the best-fit angles, with the mean (dashed line) and +/-1
    # sigma (dotted lines) marked. The ax1 lines are the disabled grid-search
    # counterpart.
    fig, ax2 = plt.subplots(1, 1, figsize=(6, 5))

    nbins = 20
    # ax1.hist(np.rad2deg(betas_best_grid), bins=nbins, edgecolor='black')
    # ax1.set_xlabel(r'$\beta_{\rm best}^{\rm (grid)}$ [deg]')
    # ax1.set_ylabel('Counts')
    # ax1.set_title(r'Distribuzione di $\beta_{\rm best}$ (grid) '+varname)
    # ax1.axvline(mean_beta_grid, color='red', linestyle='--')
    # ax1.axvline(mean_beta_grid + sigma_beta_grid, color='red', linestyle=':')
    # ax1.axvline(mean_beta_grid - sigma_beta_grid, color='red', linestyle=':')
    # ax1.text(0.02, 0.95, rf'$\bar\beta = {mean_beta_grid:.2e}$'
    #      '\n' + rf'$\sigma = {sigma_beta_grid:.2e}$',
    #      transform=ax1.transAxes, va='top')

    ax2.hist(np.rad2deg(betas_best_func), bins=nbins, edgecolor='black')
    ax2.set_xlabel(r'$\beta_{\rm best}^{\rm (minimize)}$ [deg]')
    ax2.set_title(r'Distribuzione di $\beta_{\rm best}$ (minimize) '+varname)
    ax2.axvline(mean_beta_func, color='red', linestyle='--')
    ax2.axvline(mean_beta_func + sigma_beta_func, color='red', linestyle=':')
    ax2.axvline(mean_beta_func - sigma_beta_func, color='red', linestyle=':')
    ax2.text(0.02, 0.95, rf'$\bar\beta = {mean_beta_func:.2e}$'
         '\n' + rf'$\sigma = {sigma_beta_func:.2e}$',
         transform=ax2.transAxes, va='top')

    plt.tight_layout()

    # Number of grid steps; used only to label the output file.
    n_steps_int = int(round((betamax - betamin) / step))
    plt.savefig(plots_dir + 'Dest_EB_hist_nsims_'+varname
            + str(nsims) + '_steps' + str(n_steps_int)
            + '_lmax' + str(lmax) +'_index_' + str(ipatch).zfill(4) +'.png')
    plt.close()


def main(root_dir,new_spectra,new_patches,new_maps,new_workspace,is_Dell,nsideout,nsidepatch,nsims,lmin,deltal,sigma_noise,purify_e,purify_b,apotype):
    """Run the patch-based spectra pipeline (stage 1).

    Rank 0 creates the patch masks (``new_patches``) and simulates the maps
    (``new_maps``). Then every rank processes a contiguous block of patches:
    workspaces and covariances (``new_workspace``), cross-spectra
    (``new_spectra``), validation plots and beta histograms.

    Parameters
    ----------
    root_dir : str
        Root working directory. The sub-directories ``simulazione/{plots, ws,
        cov, maps, spectra_out, mask}`` and ``simulazione/mask/mask_patch``
        must exist, with the Planck masks and the fiducial spectra in place.
    new_spectra : bool
        Compute the spectra from the maps (True) or read them from disk
        (False).
    new_patches : bool
        Create the patch masks (True) or re-use the saved ones (False).
    new_maps : bool
        Simulate new maps (True) or re-use the saved ones (False).
    new_workspace : bool
        Compute the workspaces and covariances (True) or re-use the saved ones
        (False).
    is_Dell : bool
        If True, band-powers are D_ell = ell (ell + 1) C_ell / (2 pi).
    nsideout : int
        HEALPix resolution of the simulated maps.
    nsidepatch : int
        HEALPix resolution that defines the patches.
    nsims : int
        Number of simulations.
    lmin : int
        Minimum multipole used in the validation and in the beta fit.
    deltal : int
        Multipole bin width.
    sigma_noise : float
        White-noise level in uK-arcmin (temperature); the polarisation noise is
        sqrt(2) times larger.
    purify_e, purify_b : bool
        Enable the E and B purification of the polarisation fields.
    apotype : str
        NaMaster apodisation profile of the masks (e.g. 'C2').
    """

    # Directory layout relative to root_dir. These directories must exist
    # before the script is run: it does not create them.
    plots_dir   = root_dir + 'simulazione/plots/namaster_patch/nersc_nside512_nsidepatch8/'
    ws_dir      = root_dir + 'simulazione/ws/'
    cov_dir     = root_dir + 'simulazione/cov/'
    Cls_dir_in  = root_dir+'simulazione/'
    maps_dir    = root_dir+'simulazione/'+'maps/'
    Cls_out_dir = root_dir + 'simulazione/spectra_out/'
    mask_in_dir = root_dir + 'simulazione/mask/'
    mask_patch_dir  = root_dir + 'simulazione/mask/mask_patch/'

    # Derived quantities: number of pixels of the maps and of the patch grid;
    # pixel area (deg^2); per-pixel white-noise variance (uK^2) for the noise
    # level sigma_noise (uK-arcmin); Gaussian beam FWHM (deg), equal to 2.4
    # pixel sizes (about 16.5 arcmin at nside = 512); and the apodisation
    # scale, set equal to the beam FWHM.
    npix_map= hp.nside2npix(nsideout)
    npix_patch= hp.nside2npix(nsidepatch)
    Spix = hp.nside2pixarea(nsideout, degrees=True)
    var = sigma_noise * sigma_noise / (Spix * 3600.)
    fwhm = 2.4 * np.sqrt(Spix)
    aposcale = fwhm

    # Rank 0 prepares the inputs shared by all the ranks: patch masks, fiducial
    # spectra and simulated maps. The other ranks wait at the broadcasts below.
    if (rank==0):

        if new_patches:
            start1=perf_counter()
            create_mask_patch(mask_in_dir,mask_patch_dir,nsideout,nsidepatch,npix_patch,aposcale,apotype,plots_dir)
            end1=perf_counter()
            print("tempo creazione patches:   ",(end1-start1))

        # Number of retained patches (written by create_mask_patch).
        npatches = np.loadtxt(mask_patch_dir+'active_patches_nside_'+str(nsidepatch)+
                              '.txt')

        # Maximum multipole of the simulations and spectra, 3 * nside - 1; ll
        # converts C_ell to D_ell.
        lmax = 3*nsideout-1
        ell=np.arange(0,lmax+1)
        ll=ell*(ell+1)/2/np.pi
        Cls_th=read_Cls(Cls_dir_in,lmax,ll)
        #P, Pt, Q, Qt = PQ_matrix(lmin,lmax,deltal)
        if new_maps:
            start2=perf_counter()
            create_maps(Cls_th,lmax,nsideout,nsims,maps_dir,fwhm,var,npix_map,sigma_noise)
            end2=perf_counter()
            print("tempo creazione mappe:   ",(end2-start2))
    else:
         # The other ranks receive the shared inputs through the broadcasts.
         npatches = None
         lmax     = None
         ell      = None
         ll       = None
         Cls_th   = None

    # Broadcast the shared inputs from rank 0. The broadcasts also synchronise
    # the ranks: the others wait until rank 0 has created the masks and the
    # maps.
    lmax     = comm.bcast(lmax, root=0)
    ell      = comm.bcast(ell, root=0)
    ll       = comm.bcast(ll, root=0)
    Cls_th   = comm.bcast(Cls_th, root=0)
    npatches = comm.bcast(npatches, root=0)

    # Static decomposition: each rank processes a contiguous block of
    # simperproc patches.
    simperproc = int(np.ceil(npatches/nproc))

    start3=perf_counter()

    # Loop over the patches assigned to this rank (ipatch is the running patch
    # index k).
    for ipatch in np.arange(rank*simperproc, (rank+1)*simperproc,1):

        # Skip the padding indices of the last block.
        if (ipatch<npatches):
            print(f"[Rank {rank}] >>> INIZIO Patch {ipatch}", flush=True)

            # Apodised patch masks written by create_mask_patch.
            maskT_out=hp.read_map(mask_patch_dir+'patch_maskT_nside_'+str(nsidepatch)+'_index_'+str(ipatch).zfill(4)+'.fits', dtype=np.float64)
            maskP_out=hp.read_map(mask_patch_dir+'patch_maskP_nside_'+str(nsidepatch)+'_index_'+str(ipatch).zfill(4)+'.fits', dtype=np.float64)

            # Mode-coupling workspaces and analytic covariance matrices of the
            # patch (set new_workspace to False to re-use the existing ones).
            if new_workspace:
                create_workspace (lmax,deltal,is_Dell,fwhm,nsideout,maskT_out,maskP_out,purify_e,purify_b,ws_dir,maps_dir,sigma_noise,ipatch)

                calculate_cov_th(lmax, fwhm, nsideout,
                        ws_dir,deltal,
                        is_Dell, Cls_th, cov_dir,sigma_noise,ipatch)

            # Cross-spectra of all the simulations of this patch (computed or
            # loaded).
            Dls_sims,mean_Dls,std_Dls,err_Dls,var_sim,dl_TT_th,dl_EE_th,dl_BB_th,dl_TE_th,ell_arr=\
                    calculate_spectrum_pol(lmax, fwhm, nsideout,
                    ws_dir, Cls_th, nsims,
                    deltal, maps_dir, sigma_noise,
                    purify_e, purify_b, maskT_out,
                    maskP_out,Cls_out_dir,
                    is_Dell,Cls_dir_in,ipatch,nsidepatch,new_spectra
            )

            # Validation plots: simulated mean versus theory, and empirical
            # versus analytic variances.
            validation_spectra(
                    mean_Dls, std_Dls, err_Dls,
                    ell_arr, plots_dir,
                    nsims, lmax,lmin, Cls_dir_in,var_sim,cov_dir,sigma_noise,ll,fwhm,nsideout,
                    dl_TT_th,dl_EE_th,dl_BB_th,dl_TE_th,ipatch,nsidepatch,deltal,is_Dell
            )

            # Fit beta in every simulation with the empirical variance of the
            # EB band-powers and plot the distribution of the best-fit angles.
            D_estimators (Dls_sims,ell_arr,
                nsims,lmax,lmin,nsideout,deltal,Cls_out_dir,ipatch,nsidepatch,plots_dir,var_sim[4],'var_sim'
                )

            # Repeat the fit with the analytic (NaMaster) variance of the EB
            # band-powers.
            C_master_EB = np.load(cov_dir + 'cov_ana_EB'+'_patch'+str(ipatch).zfill(4) +'.npy')
            var_master_EB = np.diag(C_master_EB)

            D_estimators (Dls_sims,ell_arr,
                nsims,lmax,lmin,nsideout,deltal,Cls_out_dir,ipatch,nsidepatch,plots_dir,var_master_EB,'var_master'
                )

    end3=perf_counter()
    print("tempo elaborazione dati:   ",(end3-start3))


# Run configuration (see the docstring of main for the meaning of each
# parameter). Launch the script with mpirun to distribute the patches.
main(root_dir='/home/matteo/Scrivania/workspace/',
    # Compute the spectra from the maps (True) or read them from disk (False).
    new_spectra = True,
    # Create the patch masks (True) or re-use the saved ones (False).
    new_patches = False,
    # Simulate new maps (True) or re-use the saved ones (False).
    new_maps = False,
    # Compute the workspaces and covariances (True) or re-use the saved ones
    # (False).
    new_workspace = True,
    # Express the band-powers as D_ell = ell (ell + 1) C_ell / (2 pi).
    is_Dell = True,
    # HEALPix resolution of the simulated maps.
    nsideout = 512,
    # HEALPix resolution that defines the patches.
    nsidepatch = 8,
    # Number of simulations processed. The next stage (mapbeta.py) expects
    # 1000.
    nsims = 2,
    # Minimum multipole of the validation and of the beta fit.
    lmin = 50,
    # Multipole bin width.
    deltal = 25,
    # White-noise level in uK-arcmin for temperature; the polarisation noise is
    # sqrt(2) times larger.
    sigma_noise = 2.0,
    # E/B purification of the polarisation fields (here only the B modes are
    # purified).
    purify_e = False,
    purify_b = True,
    # NaMaster apodisation profile of the masks.
    apotype = "C2"
    )
