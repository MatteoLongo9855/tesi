import os
import numpy as np
import healpy as hp
import matplotlib
matplotlib.use('Agg') # Forces non-interactive backend for HPC/clusters
import matplotlib.pyplot as plt

import pymaster as nmt
from scipy.optimize import minimize

def D_estimators(all_sims_spectra, ell_arr, nsims, lmin, nsideout, deltal, ipatch, vardl):
    """
    Extracts the optimal cosmic birefringence angle (beta) for a specific 
    sky patch using the D-estimator (Minami-Komatsu approach).
    
    It minimizes the chi-square of the EB cross-spectrum against the 
    difference of EE and BB auto-spectra.
    
    Parameters:
    -----------
    all_sims_spectra : ndarray
        Array containing the pseudo-spectra for all simulations and patches.
    ell_arr : ndarray
        Effective multipoles.
    nsims : int
        Number of simulations.
    lmin : int
        Minimum multipole to consider.
    nsideout : int
        Nside of the original high-resolution maps.
    deltal : int
        Bin width.
    ipatch : int
        Index of the current patch being processed.
    vardl : ndarray
        Variance of the EB spectrum for the current patch (analytical Master covariance).
        
    Returns:
    --------
    mean_beta_func : float
        Mean value of the optimal beta across all simulations (in degrees).
    sigma_beta_func : float
        Standard deviation of beta across simulations.
    var_beta_func : float
        Variance of beta across simulations.
    residui : float
        Residuals defined as mean / (sigma / sqrt(nsims)).
    betas_best : ndarray
        Array of best-fit beta values for each simulation.
    """
    mean_Dest = 0.0
    betamin = np.deg2rad(-1.5)
    betamax = np.deg2rad(1.5)
    
    limit_l = 2 * nsideout
    ell_upper_bound = ell_arr + (deltal / 2.0)
    ell_lower_bound = ell_arr - (deltal / 2.0)
    valid_indices = (ell_upper_bound <= limit_l) & (ell_lower_bound >= lmin)
    
    var_cut = vardl[valid_indices]
    dl_EE_sim = all_sims_spectra[:, ipatch, 1, valid_indices]
    dl_BB_sim = all_sims_spectra[:, ipatch, 2, valid_indices]
    dl_EB_sim = all_sims_spectra[:, ipatch, 4, valid_indices]
    
    betas_best_rad = np.ones(nsims)
    inv_cov = np.diag(1 / var_cut)
    
    # Internal functions for optimization
    def chi2_beta(beta, dl_EB, dl_EE, dl_BB, inv_cov):
        b = beta[0]
        Dest_EB = (dl_EB * np.cos(4 * b)) - 0.5 * (dl_EE - dl_BB) * np.sin(4 * b)
        x_l = Dest_EB - mean_Dest
        return x_l @ inv_cov @ x_l

    def jacobian(beta, dl_EB, dl_EE, dl_BB, inv_cov):
        grad = np.zeros(beta.size)
        for i in range(grad.size):
            betap = np.copy(beta)
            betam = np.copy(beta)
            betap[i] += 1e-10
            betam[i] -= 1e-10
            grad[i] = (chi2_beta(betap, dl_EB, dl_EE, dl_BB, inv_cov) -
                       chi2_beta(betam, dl_EB, dl_EE, dl_BB, inv_cov)) / 2e-10
        return grad

    for i in range(nsims):
        dl_EE = dl_EE_sim[i]
        dl_BB = dl_BB_sim[i]
        dl_EB = dl_EB_sim[i]
       
        res = minimize(chi2_beta, x0=np.array([0.0]), args=(dl_EB, dl_EE, dl_BB, inv_cov),
                       method='L-BFGS-B', jac=jacobian, bounds=[(betamin, betamax)])   
        betas_best_rad[i] = res.x[0]
        
    betas_best = np.rad2deg(betas_best_rad)
    sigma_beta_func = np.std(betas_best, ddof=1)
    var_beta_func = np.var(betas_best, ddof=1)
    mean_beta_func = np.mean(betas_best)
    
    residui = mean_beta_func / (sigma_beta_func / np.sqrt(nsims))
    
    return mean_beta_func, sigma_beta_func, var_beta_func, residui, betas_best

def main():
    """
    Main execution pipeline to construct full-sky maps of the anisotropic 
    cosmic birefringence angle.
    """
    # --- PARAMETERS ---
    root_dir    = '/home/matteo/Scrivania/workspace/'
    mask_in_dir = root_dir + 'simulazione/mask/'
    Cls_out_dir = root_dir + 'simulazione/spectra_out/'
    cov_dir     = root_dir + 'simulazione/cov/'
    nsideout    = 512
    nsidepatch  = 8
    lmax        = 3 * nsideout - 1
    deltal      = 25
    nsims       = 1000 
    lmin        = 50

    output_maps_dir = root_dir + 'simulazione/maps_beta/' 
    plot_maps_dir = output_maps_dir + 'maps_plot/'
    os.makedirs(output_maps_dir, exist_ok=True)
    os.makedirs(plot_maps_dir, exist_ok=True)

    # --- 1. GEOMETRIC MAPPING ---
    print("Creating active patches map...")

    mask_galT = hp.ud_grade(hp.read_map(mask_in_dir + 'COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits', field=0, dtype=np.float64), nsideout, dtype=np.float64)
    mask_galP = hp.ud_grade(hp.read_map(mask_in_dir + 'COM_Mask_CMB-common-Mask-Pol_2048_R3.00.fits', field=0, dtype=np.float64), nsideout, dtype=np.float64)

    mask_galT[mask_galT < 0.5] = 0.0; mask_galT[mask_galT >= 0.5] = 1.0
    mask_galP[mask_galP < 0.5] = 0.0; mask_galP[mask_galP >= 0.5] = 1.0

    npix_patch = hp.nside2npix(nsidepatch)
    active_pixel_indices = [] 

    for i in range(npix_patch):
        mask_patch_LR = np.zeros(npix_patch)
        mask_patch_LR[i] = 1
        mask_patch_HR = hp.ud_grade(mask_patch_LR, nsideout, dtype=np.float64)
        mask_patch_HR[mask_patch_HR < 0.5] = 0.0
        mask_patch_HR[mask_patch_HR >= 0.5] = 1.0
        
        total_patch = np.sum(mask_patch_HR)
        if total_patch == 0: continue 

        fractionT = np.sum(mask_patch_HR * mask_galT) / total_patch
        fractionP = np.sum(mask_patch_HR * mask_galP) / total_patch
        
        if fractionT >= 0.5 and fractionP >= 0.5:
            active_pixel_indices.append(i)

    print(f"Mapping completed: {len(active_pixel_indices)} active patches out of {npix_patch}.")

    # Initialization
    first_file = (Cls_out_dir + 'NamasterCls_lmax' + str(lmax) + '_deltal' + str(deltal) + 
                  '_sim0000_patch0_nsidepatch' + str(nsidepatch) + '.txt')
    n_bins = np.loadtxt(first_file, unpack=True).shape[1]

    all_sims_spectra = np.zeros((nsims, npix_patch, 6, n_bins))
    master_var_EB = np.zeros((npix_patch, n_bins))

    # --- 2. LOADING COVARIANCES & SPECTRA ---
    print("Loading Master Covariances...")
    for k, pixel_idx in enumerate(active_pixel_indices):
        filename_cov = cov_dir + 'cov_ana_EB_patch' + str(k).zfill(4) + '.npy'
        if os.path.exists(filename_cov):
            cov_matrix = np.load(filename_cov)
            master_var_EB[pixel_idx, :] = np.diag(cov_matrix)

    print("Loading Simulation Spectra...")
    for s in range(nsims):
        if s % 100 == 0: 
            print(f"  - Processing simulation {s}/{nsims}...")
        
        for k, pixel_idx in enumerate(active_pixel_indices):
            filename_spectra = (Cls_out_dir + 'NamasterCls_lmax' + str(lmax) + 
                        '_deltal' + str(deltal) + '_sim' + str(s).zfill(4) +   
                        '_patch' + str(k) + '_nsidepatch' + str(nsidepatch) + '.txt')
            if os.path.exists(filename_spectra):
                data = np.loadtxt(filename_spectra, unpack=True, usecols=(1,2,3,4,5,6))
                all_sims_spectra[s, pixel_idx, :, :] = data 
            
    b = nmt.NmtBin.from_lmax_linear(lmax, deltal, is_Dell=True)
    ell_arr = b.get_effective_ells()
    
    maps_beta = np.full((nsims, npix_patch), hp.UNSEEN)
    map_beta_mean = np.full(npix_patch, hp.UNSEEN)
    map_sigma = np.full(npix_patch, hp.UNSEEN)
    map_var = np.full(npix_patch, hp.UNSEEN)
    map_residui = np.full(npix_patch, hp.UNSEEN)

    print("Computing Beta estimators for each patch...")
    for ipatch in range(npix_patch):
        if master_var_EB[ipatch, 0] == 0:
            continue
            
        map_beta_mean[ipatch], map_sigma[ipatch], map_var[ipatch], map_residui[ipatch], maps_beta[:, ipatch] = \
            D_estimators(all_sims_spectra, ell_arr, nsims, lmin, nsideout, deltal, ipatch, master_var_EB[ipatch, :])

    # --- 3. WRITING MAPS TO DISK ---
    hp.write_map(output_maps_dir + 'map_beta_mean.fits', map_beta_mean, overwrite=True)
    hp.write_map(output_maps_dir + 'map_beta_sigma.fits', map_sigma, overwrite=True)
    hp.write_map(output_maps_dir + 'map_beta_var.fits', map_var, overwrite=True)
    hp.write_map(output_maps_dir + 'map_beta_residui.fits', map_residui, overwrite=True)
    for i in range(nsims):
        hp.write_map(output_maps_dir + f'map_beta_sim{i}.fits', maps_beta[i, :], overwrite=True, dtype=np.float64)

    # --- 4. PLOTTING ---
    print("Generating plots...")
    
    # A) MEAN BETA
    plt.figure(figsize=(12, 8))
    hp.mollview(map_beta_mean, title=r'Mean $\beta$ [deg]', unit='deg', cmap='RdBu_r') 
    plt.savefig(plot_maps_dir + 'plot_beta_mean.png')
    plt.close()

    # B) SIGMA
    plt.figure(figsize=(12, 8))
    hp.mollview(map_sigma, title=r'$\sigma_\beta$ (Std Dev) [deg]', unit='deg', cmap='viridis') 
    plt.savefig(plot_maps_dir + 'plot_beta_sigma.png')
    plt.close()

    # C) VARIANCE
    plt.figure(figsize=(12, 8))
    hp.mollview(map_var, title=r'Variance $\beta$ [deg$^2$]', unit='deg^2', cmap='magma') 
    plt.savefig(plot_maps_dir + 'plot_beta_var.png')
    plt.close()

    # D) RESIDUALS
    plt.figure(figsize=(12, 8))
    hp.mollview(map_residui, title=r'Residual ($\mu / \sigma_{mean}$)', unit='sigma', cmap='coolwarm') 
    plt.savefig(plot_maps_dir + 'plot_beta_residui.png')
    plt.close()

    # E) SINGLE SIMULATION PLOT
    plt.figure(figsize=(12, 8))
    hp.mollview(maps_beta[100, :], title=f'Beta [deg] - Simulation 100', unit='deg', cmap='RdBu_r') 
    plt.savefig(plot_maps_dir + 'plot_beta_sim100.png')
    plt.close()

    # F) 1D SCATTER PLOT (Valid patches only)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Filter only active patches for the 1D plot
    valid_mask = map_beta_mean != hp.UNSEEN
    ipatch_arr = np.arange(npix_patch)[valid_mask]
    beta_valid = map_beta_mean[valid_mask]
    sigma_valid = map_sigma[valid_mask]
    residui_valid = map_residui[valid_mask]

    ax1.errorbar(ipatch_arr, beta_valid, yerr=sigma_valid / np.sqrt(nsims), fmt='o', markersize=2, alpha=0.7, label=r'$\beta_{mean}$ [deg]') 
    ax1.set_ylabel(r'$\beta_{mean}$ [deg]') 
    ax1.legend(fontsize=8) 
    ax1.grid(True, ls='--', alpha=0.4)
    ax1.set_title(f"Beta Statistics per Patch (Nside={nsidepatch})") 

    ax2.axhline(0, color='black', lw=0.8) 
    ax2.plot(ipatch_arr, residui_valid, 'o', markersize=2, alpha=0.7) 
    ax2.axhline(3, color='red', ls='--') 
    ax2.axhline(-3, color='red', ls='--') 
    ax2.set_ylabel(r'$\Delta/\sigma$') 
    ax2.set_xlabel('Healpix Patch Index') 
    ax2.grid(True, ls='--', alpha=0.4) 

    plt.tight_layout() 
    plt.savefig(plot_maps_dir + 'plot_beta_classic.png')
    plt.close()

    print(f"All outputs successfully saved in: {plot_maps_dir}")

if __name__ == "__main__":
    main()