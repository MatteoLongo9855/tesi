"""Angular power spectra of the birefringence-angle maps (cross-spectra).

Stage 3 of the anisotropic cosmic birefringence (CB) pipeline.

Pipeline
--------
1. namaster_patch_definitive.py : patch spectra and covariances
2. mapbeta.py                   : birefringence-angle (beta) maps
3. spectra_cross_beta.py        : beta power spectra (this script)
4. A_CB_stats.py                : amplitude fit and upper limit

Method
------
The angular power spectrum D_ell = ell (ell + 1) C_ell^{beta beta} / (2 pi), in
deg^2, of the low-resolution beta maps is estimated with NaMaster
(pseudo-C_ell, Alonso et al. 2019, arXiv:1809.09603). To avoid the noise bias
of an auto-spectrum, the ensemble of nsims maps is split into two halves and
map i is cross-correlated with map i + nsims / 2, which gives nsims / 2
independent spectra. The spectra are not binned (bin width of one multipole)
and extend to lmax = 3 * nside - 1 (23 for nside = 8). The mask of the analysed
patches is apodised before the estimation.

The scatter of the spectra is compared with the uncertainties of the Planck PR3
(SMICA) constraints on D_ell^{alpha alpha} of Bortolami et al. (2022),
arXiv:2206.01635 (Table 3), for ell = 1, ..., 23.

Inputs (in ``output_maps_dir``)
-------------------------------
map_beta_sim<i>.fits
    Beta maps of simulation i, i = 0, ..., nsims - 1, written by mapbeta.py. At
    least 1000 maps are required, because map 999 is read to build the
    workspace.
mask_beta.fits
    Mask of the analysed patches. This is an external input; it is not produced
    by the scripts of this pipeline.

Outputs
-------
Cls_out_dir
    cl_beta_lmax<lmax>_sim<i>.txt: D_ell (deg^2) for ell = 0, ..., lmax, for
    the i-th pair of maps (i = 0, ..., nsims / 2 - 1).
ws_dir
    w_00_beta.fits: NaMaster mode-coupling workspace.
output_maps_dir/maps_plot/
    plot_Dl_beta_cross.pdf (mean spectrum, residuals and comparison of the
    uncertainties), plot_sigma_ratio.pdf (improvement factor) and
    sigma_comparison_table.txt.

Requirements
------------
numpy, healpy, matplotlib, pymaster (NaMaster).
"""

import numpy as np
import healpy as hp
import matplotlib
import os

import matplotlib.pyplot as plt
from time import perf_counter

import pymaster as nmt
from scipy.optimize import minimize


def create_workspace(lmax, deltal, mask_beta, output_maps_dir, ws_dir, nsideout):
    """Compute and store the NaMaster workspace for the beta cross-spectra.

    The mode-coupling matrix depends only on the mask and the binning, not on
    the map values, so it is computed once and re-used for all pairs of maps.
    Two maps of the ensemble are read only to instantiate the NaMaster fields.

    Parameters
    ----------
    lmax : int
        Maximum multipole.
    deltal : int
        Multipole bin width (1 gives an unbinned spectrum).
    mask_beta : numpy.ndarray
        Apodised mask of the analysed patches, at the resolution of the maps.
    output_maps_dir : str
        Directory containing the beta maps ``map_beta_sim<i>.fits``.
    ws_dir : str
        Directory where the workspace ``w_00_beta.fits`` is written.
    nsideout : int
        HEALPix resolution of the beta maps (not used in this function).
    """

    # Maps 0 and 999 are used only to instantiate the NaMaster fields. Pixels
    # flagged hp.UNSEEN (discarded patches) are set to zero; they are excluded
    # by the mask.
    map_beta_a = hp.read_map(output_maps_dir + f'map_beta_sim{0}.fits', dtype=np.float64, verbose=False)
    mask_bad_a = (map_beta_a == hp.UNSEEN)
    map_beta_a[mask_bad_a] = 0.0

    map_beta_b = hp.read_map(output_maps_dir + f'map_beta_sim{999}.fits', dtype=np.float64, verbose=False)
    mask_bad_b = (map_beta_b == hp.UNSEEN)
    map_beta_b[mask_bad_b] = 0.0

    # Bin edges. With deltal = 1 every bin holds a single multipole, i.e. the
    # spectrum is not binned (bins cover ell = 0, 1, ..., lmax).
    bpws = np.arange(0, lmax + deltal + 1, deltal)
    b = nmt.NmtBin.from_edges(bpws[:-1], bpws[1:], is_Dell=True)

    # Spin-0 NaMaster fields: the masked beta maps.
    f_0_a = nmt.NmtField(mask_beta, [map_beta_a], lmax=lmax)
    f_0_b = nmt.NmtField(mask_beta, [map_beta_b], lmax=lmax)

    # Compute the mode-coupling matrix of the cross-spectrum and store it for
    # the analysis of all pairs of maps.
    w_00 = nmt.NmtWorkspace()
    w_00.compute_coupling_matrix(f_0_a, f_0_b, b)
    w_00.write_to(ws_dir + 'w_00_beta.fits')


def calculate_spectra_beta(lmax, deltal, ws_dir, nsims, output_maps_dir, mask_beta, Cls_out_dir, root_dir, nsideout):
    """Compute the beta cross-spectra, plots and summary table.

    Map i of the first half of the ensemble is cross-correlated with map
    i + nsims / 2 of the second half. The resulting spectra are saved to disk
    and summarised by their mean, standard deviation and standard error.

    Parameters
    ----------
    lmax : int
        Maximum multipole.
    deltal : int
        Multipole bin width.
    ws_dir : str
        Directory containing the workspace written by ``create_workspace``.
    nsims : int
        Number of beta maps in the ensemble (nsims / 2 spectra are produced).
    output_maps_dir : str
        Directory with the beta maps; the plots and the table are written to
        its sub-directory ``maps_plot/``.
    mask_beta : numpy.ndarray
        Apodised mask of the analysed patches.
    Cls_out_dir : str
        Directory where the spectra ``cl_beta_lmax<lmax>_sim<i>.txt`` are
        written. It must exist.
    root_dir : str
        Not used in this function.
    nsideout : int
        Not used in this function.
    """

    # Same binning as in create_workspace.
    bpws = np.arange(0, lmax + deltal + 1, deltal)
    b = nmt.NmtBin.from_edges(bpws[:-1], bpws[1:], is_Dell=True)

    # Effective multipole of each bin.
    ell_arr = b.get_effective_ells()
    w_00 = nmt.NmtWorkspace()

    # Read the mode-coupling workspace computed by create_workspace.
    w_00.read_from(ws_dir + 'w_00_beta.fits')
    N_bin = b.get_n_bands()

    # One cross-spectrum per pair of maps (i, i + nsims / 2): nsims / 2
    # independent realisations.
    Cls_beta = np.zeros((nsims//2, N_bin))

    # Cross-correlate map i (first half of the ensemble) with map i + nsims / 2
    # (second half). The two maps come from independent simulations, so the
    # cross-spectrum is free of the noise bias of an auto-spectrum. Pixels
    # flagged hp.UNSEEN are set to zero and excluded by the mask.
    for i in range(nsims//2):
        map_beta_a = hp.read_map(output_maps_dir + f'map_beta_sim{i}.fits', dtype=np.float64, verbose=False)
        map_beta_a[map_beta_a == hp.UNSEEN] = 0.0
        f_0_a = nmt.NmtField(mask_beta, [map_beta_a], lmax=lmax)

        map_beta_b = hp.read_map(output_maps_dir + f'map_beta_sim{i+(nsims//2)}.fits', dtype=np.float64, verbose=False)
        map_beta_b[map_beta_b == hp.UNSEEN] = 0.0
        f_0_b = nmt.NmtField(mask_beta, [map_beta_b], lmax=lmax)

        # Pseudo-spectrum of the pair, decoupled and binned with the workspace.
        # The output is D_ell in deg^2 (is_Dell=True).
        cl_coupled = nmt.compute_coupled_cell(f_0_a, f_0_b)
        cl_00 = w_00.decouple_cell(cl_coupled)

        # Save the spectrum (ell = 0, ..., lmax) of this pair; it is read by
        # A_CB_stats.py.
        np.savetxt(Cls_out_dir + 'cl_beta_lmax' + str(lmax) + '_sim' + str(i).zfill(4) + '.txt', cl_00[0])
        Cls_beta[i, :] = cl_00[0]

    # Benchmark: 1-sigma uncertainties of D_ell^{alpha alpha} (deg^2) for
    # ell = 1, ..., 23 from Table 3 of Bortolami et al. (2022),
    # arXiv:2206.01635 (Planck PR3, SMICA).
    std_Bortolami = np.array([0.006, 0.019, 0.028, 0.049, 0.060, 0.083, 0.090, 0.112, 0.137, 0.135, 0.169, 0.188, 0.224, 0.253, 0.266, 0.307, 0.298, 0.357, 0.403, 0.406, 0.450, 0.497, 0.476])

    # Ensemble statistics: mean spectrum, standard deviation across
    # realisations (the sensitivity of a single realisation) and standard error
    # of the mean.
    mean_cl = np.mean(Cls_beta, axis=0)
    std_cl = np.std(Cls_beta, axis=0)
    err_mean = std_cl / np.sqrt(nsims/2)

    # Global plot style.
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16,
        'axes.linewidth': 1.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out'
    })

    def apply_professional_style(ax):
        """Apply a common tick and grid style to a Matplotlib axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to be formatted.
        """

        ax.tick_params(axis='both', which='major', length=6, width=1.5, top=False, right=False)
        ax.tick_params(axis='both', which='minor', length=3, width=1.0, top=False, right=False)
        ax.grid(True, linestyle=':', alpha=0.6)

    # Figure with three stacked panels: mean spectrum with the error on the
    # mean (top); residuals in units of the error on the mean, with the +/-3
    # sigma band (middle); uncertainty of the spectrum compared with Bortolami
    # et al. (2022) (bottom).
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(7.5, 10.5),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1, 1.5], 'hspace': 0.05}
    )

    ax1.errorbar(ell_arr, mean_cl, yerr=err_mean, fmt='o', color='navy', ecolor='gray', markersize=4, capsize=2)
    ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_ylabel(r'$D_{\ell}^{\beta \beta} \, [\mathrm{deg}^2]$')
    apply_professional_style(ax1)

    # Mean spectrum over its standard error: it should be consistent with zero
    # in these CB-free simulations.
    residuals = mean_cl / np.where(err_mean == 0, np.nan, err_mean)
    ax2.plot(ell_arr, residuals, 'o', color='crimson', markersize=4)
    ax2.axhline(0, color='black', lw=1)
    ax2.axhline(3, color='gray', ls='--', lw=1.2)
    ax2.axhline(-3, color='gray', ls='--', lw=1.2)
    ax2.set_ylabel(r'$\Delta / \sigma_{\mu}$')
    ax2.set_ylim(-5, 5)
    ax2.set_yticks([-3, 0, 3])
    apply_professional_style(ax2)

    # The monopole (ell = 0) is skipped so that ell = 1, ..., 23 match the 23
    # benchmark values.
    ax3.plot(ell_arr[1:], std_cl[1:], 's-', color='darkgreen', markersize=4, lw=1.5, label=r'$\sigma_{D_{\ell}}$ This work')
    ax3.plot(ell_arr[1:], std_Bortolami, '^-', color='darkorange', markersize=4, lw=1.5, label=r'$\sigma_{D_{\ell}}$ Bortolami et al. 2022')

    ax3.set_ylabel(r'$\sigma_{D_{\ell}} \, [\mathrm{deg}^2]$')
    ax3.set_xlabel(r'Multipole $\ell$')
    ax3.set_yscale('log')
    ax3.legend(frameon=False, loc='best')
    apply_professional_style(ax3)

    # Output directory for the figures and the table.
    os.makedirs(output_maps_dir + 'maps_plot/', exist_ok=True)
    plt.savefig(output_maps_dir + 'maps_plot/plot_Dl_beta_cross.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300)
    plt.close()

    # Improvement factor sigma_Bortolami / sigma_this_work for each multipole
    # (NaN where the standard deviation is zero).
    safe_std_work = np.where(std_cl[1:] == 0, np.nan, std_cl[1:])
    ratio = std_Bortolami / safe_std_work

    # Figure of the improvement factor as a function of multipole.
    fig_r, axr = plt.subplots(figsize=(7.5, 5))

    axr.plot(ell_arr[1:], ratio, 'o', color='navy', markersize=8)

    axr.set_ylabel(r'$\sigma_{D_{\ell}}^{\mathrm{Bortolami}} / \sigma_{D_{\ell}}^{\mathrm{This\,work}}$')
    axr.set_xlabel(r'Multipole $\ell$')

    axr.minorticks_on()
    apply_professional_style(axr)
    axr.grid(True, linestyle=':', alpha=0.6)
    axr.grid(which='minor', linestyle=':', alpha=0.3)

    plt.savefig(output_maps_dir + 'maps_plot/plot_sigma_ratio.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300)
    plt.close()

    # Table with, for ell = 1, ..., 23: mean D_ell, sigma (this work), sigma
    # (Bortolami et al.) and their ratio. The index j = i + 1 skips the
    # monopole.
    header = "Multipole_l\tD_l_Mean\tSigma_ThisWork\tSigma_Bortolami\tRatio_Bort_Work"
    rows = []
    for i in range(len(std_Bortolami)):
        j = i + 1
        row = [
            f"{ell_arr[j]:.2f}",
            f"{mean_cl[j]:.6e}",
            f"{std_cl[j]:.6e}",
            f"{std_Bortolami[i]:.6e}",
            f"{ratio[i]:.6f}" if not np.isnan(ratio[i]) else "NaN"
        ]
        rows.append("\t".join(row))

    table_path = output_maps_dir + 'maps_plot/sigma_comparison_table.txt'
    np.savetxt(table_path, rows, fmt='%s', header=header, comments='')
    print(f"Tabella dati salvata in: {table_path}")


def main():
    """Run the computation of the beta cross-spectra.

    Sets the directories and the analysis parameters, builds and apodises the
    patch mask, computes the workspace and then the spectra of all pairs of
    maps.
    """

    # Directory layout (adjust to your environment). Inputs are expected in
    # output_maps_dir: map_beta_sim<i>.fits (from mapbeta.py) and
    # mask_beta.fits.
    root_dir = '/home/matteo/Scrivania/workspace/'
    output_maps_dir = root_dir + 'simulazione/maps_beta/'
    ws_dir = root_dir + 'simulazione/ws/'
    Cls_out_dir = root_dir + 'simulazione/spectra_beta/'

    # Number of beta maps (at least 1000: map 999 is read to build the
    # workspace), HEALPix resolution of the beta maps, maximum multipole
    # (3 * nside - 1) and bin width (1 = unbinned spectrum).
    nsims = 1000
    nsideout = 8
    lmax = 3 * nsideout - 1
    deltal = 1

    # Mask of the analysed patches (expected: 1 for the analysed patches, 0
    # elsewhere), read from output_maps_dir and set to the resolution of the
    # beta maps.
    mask_beta = hp.ud_grade(hp.read_map(output_maps_dir + 'mask_beta.fits', field=0, dtype=np.float64, verbose=False), nsideout, dtype=np.float64)
    Spix = hp.nside2pixarea(nsideout, degrees=True)

    # Apodisation scale in degrees: 2.4 times the pixel size at nsideout (the
    # variable is named fwhm).
    fwhm = 2.4 * np.sqrt(Spix)

    # Smooth the mask edges with a C2 apodisation to limit the mode coupling.
    mask_beta = nmt.mask_apodization(mask_beta, fwhm, apotype='C2')

    # Step 1: mode-coupling workspace (computed once).
    create_workspace(lmax, deltal, mask_beta, output_maps_dir, ws_dir, nsideout)

    # Step 2: cross-spectra of the nsims / 2 pairs of maps, plots and
    # comparison table.
    calculate_spectra_beta(lmax, deltal, ws_dir, nsims, output_maps_dir, mask_beta, Cls_out_dir, root_dir, nsideout)
    print("Analisi completata con successo.")

if __name__ == '__main__':
    main()
