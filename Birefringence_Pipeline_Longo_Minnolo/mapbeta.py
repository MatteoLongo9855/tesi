"""Birefringence-angle maps from per-patch EB D-estimators.

Stage 2 of the anisotropic cosmic birefringence (CB) pipeline.

Pipeline
--------
1. namaster_patch_definitive.py : patch spectra and covariances
2. mapbeta.py                   : beta maps (this script)
3. spectra_cross_beta.py        : beta power spectra
4. A_CB_stats.py                : amplitude fit and upper limit

Method
------
For every simulation and every retained patch, the birefringence angle beta is
estimated by minimising the chi-square of the EB D-estimator (Gruppuso et al.
2016, arXiv:1604.05202):

    D_ell(beta) = D_ell^EB cos(4 beta) - 0.5 (D_ell^EE - D_ell^BB) sin(4 beta)
    chi2(beta)  = sum_ell D_ell(beta)^2 / var_ell

The band-powers are the cross-spectra of the patch computed in stage 1, and
var_ell is the analytic variance of the EB band-powers (diagonal covariance
approximation). Only the bins entirely contained in [lmin, 2 * nsideout] are
used. The best-fit angles of a simulation fill one HEALPix map of resolution
``nsidepatch`` (one value per patch, in degrees); patches that were not
retained are flagged with hp.UNSEEN.

Inputs
------
Cls_out_dir
    Cross-spectra from stage 1,
    NamasterCls_lmax<lmax>_deltal<deltal>_sim<s>_patch<k>_nsidepatch<n>.txt
    (columns: ell, TT, EE, BB, TE, EB, TB).
cov_dir
    Analytic covariance of the EB band-powers from stage 1,
    cov_ana_EB_patch<k>.npy.
mask_in_dir
    Planck 2018 (PR3) common Galactic masks,
    COM_Mask_CMB-common-Mask-{Int,Pol}_2048_R3.00.fits.

Outputs (written to ``output_maps_dir``)
----------------------------------------
map_beta_sim<i>.fits
    Best-fit beta map of simulation i (deg).
map_beta_mean.fits, map_beta_sigma.fits, map_beta_var.fits
    Mean (deg), standard deviation (deg) and variance (deg^2) over the
    simulations.
map_beta_residui.fits
    Significance of the mean, mean / (sigma / sqrt(nsims)).
maps_plot/
    PDF figures of the maps above and of the mean angle as a function of
    Galactic longitude.

Usage
-----
Set ``do_compute`` to True for the first (slow) run, which writes the FITS
maps, and to False to regenerate the plots from the files already on disk.

Requirements
------------
numpy, scipy, healpy, matplotlib, pymaster (NaMaster).
"""

import numpy as np
import healpy as hp
import matplotlib
import os

# Use a non-interactive backend so that figures can be written on headless
# nodes (e.g. HPC clusters).
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from time import perf_counter

import pymaster as nmt
from mpi4py import MPI
from scipy.optimize import minimize


def D_estimators(all_sims_spectra, ell_arr, nsims, lmin, nsideout, deltal, ipatch, vardl):
    """Fit the birefringence angle of one patch in every simulation.

    For each simulation, chi2(beta) = sum_ell D_ell(beta)^2 / var_ell is
    minimised, where D_ell(beta) is the EB D-estimator. The minimisation is
    bounded (|beta| <= 1.5 degrees) and uses a diagonal covariance. Only the
    bins entirely contained in [lmin, 2 * nsideout] enter the fit.

    Parameters
    ----------
    all_sims_spectra : numpy.ndarray, shape (nsims, npix_patch, 6, n_bins)
        Cross-spectra D_ell of all simulations and patches. The spectrum index
        follows the order TT, EE, BB, TE, EB, TB.
    ell_arr : numpy.ndarray, shape (n_bins,)
        Effective multipole of each bin.
    nsims : int
        Number of simulations.
    lmin : int
        Minimum multipole of the fit.
    nsideout : int
        HEALPix resolution of the simulated CMB maps; the maximum multipole of
        the fit is 2 * nsideout.
    deltal : int
        Multipole bin width.
    ipatch : int
        HEALPix index of the patch.
    vardl : numpy.ndarray, shape (n_bins,)
        Variance of the EB band-powers in each bin of this patch.

    Returns
    -------
    mean_beta_func : float
        Mean of the best-fit angles (deg).
    sigma_beta_func : float
        Standard deviation of the best-fit angles (deg, ddof = 0).
    var_beta_func : float
        Variance of the best-fit angles (deg^2).
    residui : float
        Significance of the mean, mean / (sigma / sqrt(nsims)); a null test of
        the unbiasedness of the estimator.
    betas_best : numpy.ndarray, shape (nsims,)
        Best-fit angle of each simulation (deg).
    """

    # Expected value of the D-estimator when beta equals the true angle.
    mean_Dest = 0.0

    # Allowed range of the angle, in radians: +/-1.5 degrees.
    betamin = np.deg2rad(-1.5)
    betamax = np.deg2rad(1.5)

    # Select the bins entirely contained in [lmin, 2 * nsideout], the multipole
    # range used in the fit.
    limit_l = 2 * nsideout
    ell_upper_bound = ell_arr + (deltal / 2.0)
    ell_lower_bound = ell_arr - (deltal / 2.0)
    valid_indices = (ell_upper_bound <= limit_l) & (ell_lower_bound >= lmin)

    # Variances and spectra restricted to the selected bins.
    var_cut = vardl[valid_indices]
    dl_EE_sim = all_sims_spectra[:, ipatch, 1, valid_indices]
    dl_BB_sim = all_sims_spectra[:, ipatch, 2, valid_indices]
    dl_EB_sim = all_sims_spectra[:, ipatch, 4, valid_indices]

    # Best-fit angle (radians) of each simulation.
    betas_best_rad = np.ones(nsims)

    # Diagonal approximation of the inverse covariance of the D-estimator.
    inv_cov = np.diag(1 / var_cut)

    def chi2_beta(beta, dl_EB, dl_EE, dl_BB, inv_cov):
        """Chi-square of the EB D-estimator for the angle beta (radians)."""

        b = beta[0]
        Dest_EB = (dl_EB * np.cos(4 * b)) - 0.5 * (dl_EE - dl_BB) * np.sin(4 * b)
        x_l = Dest_EB - mean_Dest
        return x_l @ inv_cov @ x_l

    def jacobian(beta, dl_EB, dl_EE, dl_BB, inv_cov):
        """Central finite-difference gradient of chi2_beta (step 1e-10 rad)."""

        grad = np.zeros(beta.size)
        for i in range(grad.size):
            betap = np.copy(beta)
            betam = np.copy(beta)
            betap[i] += 1e-10
            betam[i] -= 1e-10
            grad[i] = (chi2_beta(betap, dl_EB, dl_EE, dl_BB, inv_cov) -
                       chi2_beta(betam, dl_EB, dl_EE, dl_BB, inv_cov)) / 2e-10
        return grad

    # Fit the angle independently in every simulation: bounded L-BFGS-B
    # minimisation starting from beta = 0.
    for i in range(nsims):
        dl_EE = dl_EE_sim[i]
        dl_BB = dl_BB_sim[i]
        dl_EB = dl_EB_sim[i]

        res = minimize(chi2_beta, x0=np.array([0.0]), args=(dl_EB, dl_EE, dl_BB, inv_cov),
                       method='L-BFGS-B', jac=jacobian, bounds=[(betamin, betamax)])
        betas_best_rad[i] = res.x[0]

    # Summary statistics of the best-fit angles in degrees (ddof = 0). The
    # significance is the mean over its standard error.
    betas_best = np.rad2deg(betas_best_rad)
    sigma_beta_func = np.std(betas_best)
    var_beta_func = np.var(betas_best)

    mean_beta_func = np.mean(betas_best)
    residui = mean_beta_func / (sigma_beta_func / np.sqrt(nsims))
    return mean_beta_func, sigma_beta_func, var_beta_func, residui, betas_best

# Execution mode. True: load the spectra and covariances of stage 1, fit beta
# in every patch and simulation, and write the FITS maps (slow). False: skip
# the computation, read the maps written by a previous run and only produce the
# plots. Set it to True for the first run.
do_compute = False

# Directory layout (adjust root_dir to your environment). Inputs: Planck masks
# (mask_in_dir), spectra and analytic covariances from stage 1 (Cls_out_dir,
# cov_dir). Outputs: FITS maps (output_maps_dir) and figures (plot_maps_dir).
root_dir    = '/home/matteo/Scrivania/workspace/'
mask_in_dir = root_dir + 'simulazione/mask/'
Cls_out_dir = root_dir + 'simulazione/spectra_out/'
cov_dir     = root_dir + 'simulazione/cov/'
output_maps_dir = root_dir + 'simulazione/maps_beta/'
plot_maps_dir = output_maps_dir + 'maps_plot/'

os.makedirs(output_maps_dir, exist_ok=True)
os.makedirs(plot_maps_dir, exist_ok=True)

# Analysis settings, which must match those of stage 1: resolution of the
# simulated CMB maps, resolution of the patches (and of the beta maps), maximum
# multipole, multipole bin width, number of simulations and minimum multipole
# of the fit.
nsideout    = 512
nsidepatch  = 8
lmax        = 3 * nsideout - 1
deltal      = 25
nsims       = 1000
lmin        = 50
npix_patch  = hp.nside2npix(nsidepatch)

if do_compute:
    # Recompute which patches were retained in stage 1. The k-th retained patch
    # is the HEALPix pixel active_pixel_indices[k], and the stage-1 files are
    # numbered by k, so the same criterion as in stage 1 must be used here.
    print("Creazione mappa indici attivi...")

    # Planck PR3 common Galactic masks (intensity and polarisation) at
    # nsideout, binarised at 0.5 as in stage 1.
    mask_galT = hp.ud_grade(hp.read_map(mask_in_dir + 'COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits', field=0, dtype=np.float64), nsideout, dtype=np.float64)
    mask_galP = hp.ud_grade(hp.read_map(mask_in_dir + 'COM_Mask_CMB-common-Mask-Pol_2048_R3.00.fits', field=0, dtype=np.float64), nsideout, dtype=np.float64)

    mask_galT[mask_galT < 0.5] = 0.0; mask_galT[mask_galT >= 0.5] = 1.0
    mask_galP[mask_galP < 0.5] = 0.0; mask_galP[mask_galP >= 0.5] = 1.0

    # HEALPix indices of the retained patches, in the order of the stage-1
    # patch index k.
    active_pixel_indices = []

    # A patch is retained if at least 50% of its pixels are unmasked in both
    # the temperature and the polarisation masks.
    for i in range(npix_patch):
        mask_patch_LR = np.zeros(npix_patch); mask_patch_LR[i] = 1
        mask_patch_HR = hp.ud_grade(mask_patch_LR, nsideout, dtype=np.float64)
        mask_patch_HR[mask_patch_HR < 0.5] = 0.0; mask_patch_HR[mask_patch_HR >= 0.5] = 1.0

        total_patch = np.sum(mask_patch_HR)
        if total_patch == 0: continue

        fractionT = np.sum(mask_patch_HR * mask_galT) / total_patch
        fractionP = np.sum(mask_patch_HR * mask_galP) / total_patch

        if fractionT >= 0.5 and fractionP >= 0.5:
            active_pixel_indices.append(i)

    print(f"Mappatura completata: {len(active_pixel_indices)} patch attive su {npix_patch}.")

    # The number of bins is read from the first spectrum file (7 columns: ell
    # and the six spectra).
    first_file = (Cls_out_dir + 'NamasterCls_lmax' + str(lmax) + '_deltal' + str(deltal) +
                  '_sim0000_patch0_nsidepatch' + str(nsidepatch) + '.txt')

    n_bins = np.loadtxt(first_file, unpack=True).shape[1]

    # Cross-spectra of all simulations and patches, indexed as [simulation,
    # patch, spectrum, bin]; the spectrum order is TT, EE, BB, TE, EB, TB.
    all_sims_spectra = np.zeros((nsims, npix_patch, 6, n_bins))

    # Analytic (NaMaster) variance of the EB band-powers for each patch and
    # bin.
    master_var_EB = np.zeros((npix_patch, n_bins))

    print("Caricamento Covarianze Master...")

    # Diagonal of the analytic EB covariance matrix of each retained patch. A
    # missing file is skipped silently: the variance then stays at zero and the
    # patch is flagged as unseen in the output maps.
    for k, pixel_idx in enumerate(active_pixel_indices):
        filename_cov = cov_dir + 'cov_ana_EB_patch' + str(k).zfill(4) + '.npy'
        try:
            cov_matrix = np.load(filename_cov)
            master_var_EB[pixel_idx, :] = np.diag(cov_matrix)
        except:
            pass

    print("Caricamento Spettri Simulazioni...")
    for s in range(nsims):
        if s % 100 == 0: print(f"  - Elaborazione simulazione {s}/{nsims}...")

        # Cross-spectra of simulation s for each retained patch.
        for k, pixel_idx in enumerate(active_pixel_indices):
            filename_spectra = (Cls_out_dir + 'NamasterCls_lmax' + str(lmax) +
                        '_deltal' + str(deltal) +
                        '_sim' + str(s).zfill(4) +
                        '_patch' + str(k) +
                        '_nsidepatch' + str(nsidepatch) + '.txt')
            data = np.loadtxt(filename_spectra, unpack=True, usecols=(1,2,3,4,5,6))
            all_sims_spectra[s, pixel_idx, :, :] = data

    # Rebuild the stage-1 binning to obtain the effective multipole of each
    # bin.
    b = nmt.NmtBin.from_lmax_linear(lmax, deltal, is_Dell=True)
    ell_arr = b.get_effective_ells()

    # Output containers (degrees): the beta map of each simulation and the
    # summary maps over the simulations (mean, standard deviation, variance and
    # significance of the mean).
    maps_beta = np.zeros((nsims, npix_patch))
    map_beta_mean = np.zeros(npix_patch)
    map_sigma = np.zeros(npix_patch)
    map_var = np.zeros(npix_patch)
    map_residui = np.zeros(npix_patch)

    print("\nAvvio estrazione dei parametri Beta (Minimizzazione)...")

    # Fit the angle patch by patch. Patches that were not retained (zero
    # variance) are flagged hp.UNSEEN in all maps.
    for ipatch in range(npix_patch):
        if master_var_EB[ipatch, 0] == 0:
            map_beta_mean[ipatch] = hp.UNSEEN
            map_sigma[ipatch] = hp.UNSEEN
            map_var[ipatch] = hp.UNSEEN
            map_residui[ipatch] = hp.UNSEEN
            maps_beta[:, ipatch] = hp.UNSEEN
            continue

        if ipatch % 10 == 0:
            print(f"  -> Calcolo minimizzazione: Patch {ipatch}/{npix_patch}...")

        # Fit beta in all the simulations of this patch.
        map_beta_mean[ipatch], map_sigma[ipatch], map_var[ipatch], map_residui[ipatch], maps_beta[:,ipatch] = \
            D_estimators(all_sims_spectra, ell_arr,
            nsims, lmin, nsideout, deltal, ipatch, master_var_EB[ipatch,:])

    print("Minimizzazione completata! Salvataggio mappe...")

    # Write the summary maps.
    hp.write_map(output_maps_dir + 'map_beta_mean.fits', map_beta_mean, overwrite=True)
    hp.write_map(output_maps_dir + 'map_beta_sigma.fits', map_sigma, overwrite=True)
    hp.write_map(output_maps_dir + 'map_beta_var.fits', map_var, overwrite=True)
    hp.write_map(output_maps_dir + 'map_beta_residui.fits', map_residui, overwrite=True)

    # Write one FITS file per simulation (beta in degrees).
    for i in range(nsims):
        hp.write_map(output_maps_dir + f'map_beta_sim{i}.fits', maps_beta[i, :], overwrite=True, dtype=np.float64)

else:
    # Plot-only mode: read the summary maps written by a previous run.
    print("Lettura mappe da disco in corso...")
    map_beta_mean = hp.read_map(output_maps_dir + 'map_beta_mean.fits', dtype=np.float64)
    map_sigma     = hp.read_map(output_maps_dir + 'map_beta_sigma.fits', dtype=np.float64)
    map_var       = hp.read_map(output_maps_dir + 'map_beta_var.fits', dtype=np.float64)
    map_residui   = hp.read_map(output_maps_dir + 'map_beta_residui.fits', dtype=np.float64)

# Plots, saved as PDF in plot_maps_dir. The global plot style is set first.
print("Generazione grafici...")

plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'axes.linewidth': 1.5,
})

# Mean beta over the simulations in each patch (deg).
plt.figure(figsize=(12, 8))
hp.mollview(map_beta_mean,
            title='',
            unit='deg',
            cmap='RdBu_r')
plt.gcf().get_axes()[-1].xaxis.label.set_size(22)
plt.savefig(plot_maps_dir + 'plot_beta_mean.pdf', bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.close()

# Standard deviation of beta over the simulations in each patch (deg).
plt.figure(figsize=(12, 8))
hp.mollview(map_sigma,
            title='',
            unit='deg',
            cmap='hot')
cb = plt.gcf().get_axes()[-1]
cb.xaxis.label.set_size(22)
cb.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.savefig(plot_maps_dir + 'plot_beta_sigma.pdf', bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.close()

# Variance of beta over the simulations in each patch (deg^2).
plt.figure(figsize=(12, 8))
hp.mollview(map_var,
            title='',
            unit='deg$^2$',
            cmap='magma')
plt.gcf().get_axes()[-1].xaxis.label.set_size(22)
plt.savefig(plot_maps_dir + 'plot_beta_var.pdf', bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.close()

# Significance of the mean, mean / (sigma / sqrt(nsims)), displayed between -3
# and 3: a null test of the unbiasedness of the estimator.
plt.figure(figsize=(12, 8))
hp.mollview(map_residui,
            title='',
            unit=r'$\sigma$',
            cmap='coolwarm',
            min=-3,
            max=3)
plt.gcf().get_axes()[-1].xaxis.label.set_size(28)
plt.savefig(plot_maps_dir + 'plot_beta_residui.pdf', bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.close()

# Select one example simulation (index 88) to display: from memory if it has
# just been computed, otherwise from disk.
if do_compute:
    map_sim_to_plot = maps_beta[88, :]
else:
    map_sim_to_plot = hp.read_map(output_maps_dir + 'map_beta_sim88.fits', dtype=np.float64)

# Beta map of the example simulation.
plt.figure(figsize=(12, 8))
hp.mollview(map_sim_to_plot,
            title='',
            unit='deg',
            cmap='RdBu_r')
plt.gcf().get_axes()[-1].xaxis.label.set_size(22)
plt.savefig(plot_maps_dir + f'plot_beta_sim100.pdf', bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.close()


# Mean beta as a function of the longitude of the retained patches (longitude
# of the patch centre, in the coordinate system of the maps). Error bars are
# the standard error of the mean, sigma / sqrt(nsims).
valid_mask = (map_beta_mean != hp.UNSEEN)
valid_pixels = np.where(valid_mask)[0]
theta, phi = hp.pix2ang(nsidepatch, valid_pixels)
lon_valid = np.rad2deg(phi)

# Top panel: mean angle of each patch. Bottom panel: significance of the mean,
# with guide lines at +/-3 sigma.
fig, (ax1, ax2) = plt.subplots(
         2, 1,
         figsize=(7.5, 7.5),
         gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05},
         sharex=True
     )

ax1.axhline(0, color='gray', lw=1, zorder=0)
ax1.errorbar(
             lon_valid,
             map_beta_mean[valid_mask],
             yerr=map_sigma[valid_mask]/(np.sqrt(nsims)),
             fmt='o',
             color='navy',
             ecolor='gray',
             elinewidth=1.0,
             capsize=1.5,
             markersize=4,
             zorder=2,
             alpha=0.8
         )
ax1.set_ylabel(r'$\bar{\beta} \ [\mathrm{deg}]$')
ax1.grid(True, ls=':', alpha=0.6)

ax2.axhline(0, color='black', lw=1, zorder=1)
ax2.axhline(3, color='gray', ls='--', lw=1.2, zorder=1)
ax2.axhline(-3, color='gray', ls='--', lw=1.2, zorder=1)
ax2.plot(lon_valid, map_residui[valid_mask], 'o', color='crimson', markersize=4, zorder=2, alpha=0.8)

ax2.set_ylabel(r'$\Delta/\sigma_{\mu}$')
ax2.set_ylim(-5.5, 5.5)
ax2.set_yticks([-5, -3, 0, 3, 5])
ax2.grid(True, ls=':', alpha=0.6)

ax2.set_xlabel(r'Galactic Longitude [deg]')
ax2.set_xlim(-5, 365)

fig.align_ylabels()

plt.savefig(plot_maps_dir + 'plot_beta_longitude.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300)
plt.close()

print(f"Tutto salvato correttamente in: {plot_maps_dir}")
