"""Amplitude constraint on anisotropic cosmic birefringence.

Stage 4 (final stage) of the anisotropic cosmic birefringence (CB) pipeline.

Pipeline
--------
1. namaster_patch_definitive.py : patch spectra and covariances
2. mapbeta.py                   : birefringence-angle (beta) maps
3. spectra_cross_beta.py        : angular power spectra of the beta maps
4. A_CB_stats.py                : amplitude fit and upper limit (this script)

Method
------
The angular power spectra D_ell = ell (ell + 1) C_ell^{beta beta} / (2 pi), in
deg^2, of the beta maps (one spectrum per pair of simulations) are fitted with
a scale-invariant model, D_ell = A_CB at every multipole. The chi-square is the
one of Eq. (3.10) of Bortolami et al. (2022), arXiv:2206.01635:

    chi2(A) = (D - A 1)^T C^-1 (D - A 1)

where D is the mean spectrum over the realisations, 1 is a vector of ones and C
is the sample covariance matrix of a single realisation. Since the model is
linear in A, the minimum and its uncertainty are analytic:

    A_min   = (1^T C^-1 D) / (1^T C^-1 1)
    sigma_A = (1^T C^-1 1)^(-1/2)

The posterior is proportional to exp(-chi2 / 2) with a flat prior on A_CB >= 0,
i.e. a Gaussian with mean A_min and width sigma_A truncated at zero. The 95%
C.L. upper limit is its 95th percentile.

As a validation, chi2 evaluated at A_CB = 0 for each realisation is compared
with the chi-square distribution with n_ell degrees of freedom, which is
expected if the Gaussian likelihood and the covariance matrix are adequate.

Inputs
------
Cls_out_dir
    cl_beta_lmax<lmax>_sim<i>.txt, i = 0, ..., nsims / 2 - 1, written by
    spectra_cross_beta.py.

Outputs (written to ``acb_out_dir``)
------------------------------------
acb_risultati.txt       : best-fit amplitude, uncertainty and upper limit
acb_chi2_histogram.pdf  : validation histogram of the per-realisation chi2
acb_posterior.pdf       : posterior of A_CB with the 95% C.L. upper limit

Requirements
------------
numpy, scipy, matplotlib.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import truncnorm


def apply_professional_style(ax):
    """Apply a common tick and grid style to a Matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to be formatted.

    Notes
    -----
    Defined for consistency with the other pipeline scripts; it is not called
    in this script.
    """

    ax.tick_params(axis='both', which='major', length=6, width=1.5,
                   top=False, right=False)
    ax.tick_params(axis='both', which='minor', length=3, width=1.0,
                   top=False, right=False)
    ax.grid(True, linestyle=':', alpha=0.6)


def carica_spettri_reali(Cls_out_dir, lmax, n_real):
    """Load the beta cross-spectra of all realisations.

    Parameters
    ----------
    Cls_out_dir : str
        Directory containing the files ``cl_beta_lmax<lmax>_sim<i>.txt``.
    lmax : int
        Maximum multipole. Each file holds lmax + 1 values (ell = 0, ...,
        lmax).
    n_real : int
        Number of realisations (cross-spectra) to load.

    Returns
    -------
    numpy.ndarray
        Array of shape (n_real, lmax) with D_ell^{beta beta} in deg^2 for
        ell = 1, ..., lmax. The monopole (ell = 0) is discarded.
    """

    Cls_beta_raw = np.zeros((n_real,lmax + 1))
    for i in range(n_real):
        paths = Cls_out_dir + f'cl_beta_lmax{lmax}_sim{i:04d}.txt'
        Cls_beta_raw[i, :] = np.loadtxt(paths)
    return Cls_beta_raw[:, 1:]


def chi2(A_cb, D_obs, Cinv, id):
    """Chi-square of the scale-invariant model D_ell = A_cb.

    Parameters
    ----------
    A_cb : float
        Amplitude of the scale-invariant spectrum (deg^2).
    D_obs : numpy.ndarray, shape (n_ell,)
        Observed (ensemble-mean) band-powers (deg^2).
    Cinv : numpy.ndarray, shape (n_ell, n_ell)
        Inverse of the covariance matrix of the band-powers.
    id : numpy.ndarray, shape (n_ell,)
        Vector of ones, i.e. the template of the scale-invariant model.

    Returns
    -------
    float
        (D_obs - A_cb * id)^T Cinv (D_obs - A_cb * id).
    """

    x_l = D_obs - A_cb * id
    return x_l @ Cinv @ x_l


# Root working directory of the analysis (adjust to your environment).
root_dir = '/home/matteo/Scrivania/workspace/'

# Select the configuration to analyse by choosing one pair of input/output
# directories. The active pair (``_sat``) is the ground-based, Simons
# Observatory SAT-like footprint; the commented pair is the satellite-like
# configuration.
#Cls_out_dir = root_dir + 'simulazione/spectra_beta/'
Cls_out_dir = root_dir + 'simulazione/spectra_beta_sat/'
#acb_out_dir = root_dir + 'simulazione/acb/'
acb_out_dir = root_dir + 'simulazione/acb_sat/'

# Create the output directory if it does not exist.
os.makedirs(acb_out_dir, exist_ok=True)

# Settings that must match the previous stages: number of simulated maps,
# HEALPix resolution of the beta maps and maximum multipole
# (3 * nside - 1 = 23). The maps are paired (map i with map i + nsims / 2),
# which yields nsims / 2 independent spectra.
nsims = 1000
nsideout = 8
lmax = 3 * nsideout - 1
n_real = nsims // 2

# Multipoles entering the fit: ell = 1, ..., lmax (the monopole is excluded).
ELL_MIN, ELL_MAX = 1, lmax
ells = np.arange(ELL_MIN, ELL_MAX + 1)
n_ell = len(ells)

# Band-powers of all realisations, shape (n_real, n_ell).
D_sims = carica_spettri_reali(Cls_out_dir, lmax, n_real)

# Ensemble-mean spectrum: the best estimate of the underlying spectrum, which
# is zero in these CB-free (null) simulations.
D_obs = D_sims.mean(axis=0)

# Template of the scale-invariant model: D_ell = A_CB at every multipole.
id = np.ones(n_ell)

# Sample covariance of a single realisation (not divided by the number of
# realisations), i.e. the sensitivity of a single experiment. n_real must
# exceed n_ell for the matrix to be invertible.
Cov = np.cov(D_sims, rowvar=False)
Cinv = np.linalg.inv(Cov)

# Closed-form minimisation of chi2 (the model is linear in A_CB):
# A_min = (1^T C^-1 D) / (1^T C^-1 1) and sigma_A = (1^T C^-1 1)^(-1/2).
w = id @ Cinv
denom = w @ id
A_min = (w @ D_obs) / denom
sigma_A = 1.0 / np.sqrt(denom)
chi2_min = chi2(A_min, D_obs, Cinv, id)

# Chi-square at A_CB = 0 for every realisation, using the same covariance
# matrix. Used below to validate the Gaussian likelihood.
chi2_per_sim = np.einsum('ij,jk,ik->i', D_sims, Cinv, D_sims)

# Summary of the fit, printed and saved to a text file.
testo_output = (
    "stima A_cb tramite formula esatta minimizzazione chi quadro\n"
    f"A_min_CB   = {A_min:+.6e} deg^2\n"
    f"sigma_A_CB = {sigma_A:.6e} deg^2\n"
    f"chi2_min   = {chi2_min:.3f}   (atteso ~ n_ell = {n_ell} se il modello nullo e' corretto)\n"
    f"\nchi2 medio sulle {n_real} realizzazioni = {chi2_per_sim.mean():.2f}\n "
    f"(atteso {n_ell} per una chi2_{{{n_ell}}})\n"
)

print(testo_output)

file_txt_out = acb_out_dir + 'acb_risultati.txt'
with open(file_txt_out, 'w') as f:
    f.write(testo_output)

print(f"Risultati salvati con successo in: {file_txt_out}")

# Plot style; fall back to another style if the seaborn one is not available.
try:
    plt.style.use('seaborn-v0_8-white')
except:
    plt.style.use('white')
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'axes.linewidth': 1.5,
    'text.usetex': False,
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif'
})

# Figure 1: distribution of the per-realisation chi-square compared with the
# theoretical chi-square distribution.
fig1, ax1 = plt.subplots(figsize=(7.5, 5.5))

hist_color = '#483D8B'

# Histogram bars: translucent fill (the outlines are drawn by the second call
# below).
counts, bin_edges, _ = ax1.hist(
    chi2_per_sim,
    bins=40,
    histtype='bar',
    alpha=0.1,
    color=hist_color,
    edgecolor='none'
)

# Outline of each bar (no fill).
ax1.hist(
    chi2_per_sim,
    bins=40,
    histtype='bar',
    facecolor='none',
    edgecolor=hist_color,
    linewidth=1.5
)

# Theoretical chi-square density with n_ell degrees of freedom, rescaled to
# expected counts per bin (n_real x bin width).
bin_width = bin_edges[1] - bin_edges[0]
x_theory = np.linspace(0, chi2_per_sim.max() * 1.05, 500)

theory_counts = stats.chi2.pdf(x_theory, df=n_ell) * n_real * bin_width
ax1.plot(x_theory, theory_counts, color='k', linestyle='-', linewidth=2,
         label=r"$\chi^2$ theory")

ax1.set_xlabel(r"$\chi^2$")
ax1.set_ylabel(r"$\mathrm{Counts}$")

ax1.legend(frameon=False, loc='best')

ax1.minorticks_on()
ax1.tick_params(axis='both', which='major', direction='in', length=6, width=1.2)
ax1.tick_params(axis='both', which='minor', direction='in', length=3, width=1.0)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

fig1.tight_layout()
fig1.savefig(acb_out_dir + 'acb_chi2_histogram.pdf',
             bbox_inches='tight', pad_inches=0.01)
plt.close(fig1)

# Posterior of A_CB: Gaussian N(A_min, sigma_A^2) truncated to A_CB >= 0 (flat
# prior on non-negative amplitudes). SciPy's truncnorm expects the truncation
# bounds in units of sigma relative to the mean.
a_trunc = (0.0 - A_min) / sigma_A
b_trunc = np.inf

# 95% C.L. upper limit: 95th percentile of the truncated posterior.
A_95 = truncnorm.ppf(0.95, a=a_trunc, b=b_trunc, loc=A_min, scale=sigma_A)

testo_posterior = (
    "\n--- Posterior e limite superiore ---\n"
    f"A_CB < {A_95:.6e} deg^2   (95% C.L., prior A_CB >= 0)\n"
)
print(testo_posterior)
with open(file_txt_out, 'a') as f:
    f.write(testo_posterior)

# Grid used to plot the posterior up to 1.9 times the upper limit.
x_max = A_95 * 1.9
A_grid = np.linspace(0.0, x_max, 2000)

# Posterior density, rescaled to a unit maximum (relative probability).
posterior = truncnorm.pdf(A_grid, a=a_trunc, b=b_trunc, loc=A_min, scale=sigma_A)
posterior_display = posterior / posterior.max()


# Express the amplitudes in units of 1e-4 deg^2 for display.
scale_factor = 1e4
A_grid_scaled = A_grid * scale_factor
A_95_scaled = A_95 * scale_factor
x_max_scaled = x_max * scale_factor

# Figure 2: posterior of A_CB and its 95% C.L. upper limit.
fig2, ax2 = plt.subplots(figsize=(7.5, 5.5))

ax2.plot(A_grid_scaled, posterior_display, color="navy", lw=2.5)

# Shade the region below the upper limit (95% of the posterior).
ax2.fill_between(A_grid_scaled, posterior_display,
                 where=(A_grid_scaled <= A_95_scaled),
                 color="navy", alpha=0.15)

# Dashed line at the upper limit. ymax is a fraction of the axes height; it
# keeps the line from crossing the text label.
ax2.axvline(A_95_scaled, color="navy", ls="--", lw=1.2, ymax=0.8)


ax2.set_xlabel(r"$A_{CB}$  [$10^{-4} \, \mathrm{deg}^2$]")
ax2.set_ylabel("Relative Probability")

ax2.set_ylim(0, 1.05)
ax2.set_xlim(0.0, x_max_scaled)


# An empty legend is used as a text box reporting the 95% C.L. result.
ax2.legend(handles=[], frameon=False, loc='upper right',
           title=fr"95% C.L.: $A_{{CB}} <$ {A_95_scaled:.2f} $\times 10^{{-4}}$ deg$^2$")

ax2.minorticks_on()
ax2.tick_params(axis='both', which='major', direction='in', length=6, width=1.5, top=False, right=False)
ax2.tick_params(axis='both', which='minor', direction='in', length=3, width=1.0, top=False, right=False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig2.tight_layout()
fig2.savefig(acb_out_dir + 'acb_posterior.pdf',
             bbox_inches='tight', pad_inches=0.01)
plt.close(fig2)

print(f"\nFigure salvate in: {acb_out_dir}")
print("  - acb_chi2_histogram.pdf")
print("  - acb_posterior.pdf")
print("=" * 70)
