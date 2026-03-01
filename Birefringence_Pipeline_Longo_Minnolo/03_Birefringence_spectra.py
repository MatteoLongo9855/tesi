import os
import numpy as np
import healpy as hp
import matplotlib
matplotlib.use('Agg') # Forces non-interactive backend for HPC/clusters
import matplotlib.pyplot as plt

import pymaster as nmt

def create_workspace(lmax, deltal, mask_beta, output_maps_dir, ws_dir, nsideout):
    """
    Computes and saves the NaMaster mode-coupling matrix (workspace) 
    for the pseudo-C_ell inversion of the anisotropic beta maps.
    
    Parameters:
    -----------
    lmax : int
        Maximum multipole.
    deltal : int
        Bin width.
    mask_beta : ndarray
        Apodized mask for the beta maps.
    output_maps_dir : str
        Directory containing the input beta maps.
    ws_dir : str
        Directory to save the computed workspace.
    nsideout : int
        Resolution (Nside) of the beta maps.
    """
    # Load two independent noise realizations to define the masking geometry
    map_beta_a = hp.read_map(output_maps_dir + f'map_beta_sim0.fits', dtype=np.float64)
    map_beta_a[map_beta_a == hp.UNSEEN] = 0.0
    
    map_beta_b = hp.read_map(output_maps_dir + f'map_beta_sim999.fits', dtype=np.float64)
    map_beta_b[map_beta_b == hp.UNSEEN] = 0.0
    
    bpws = np.arange(0, lmax + deltal + 1, deltal)
    b = nmt.NmtBin.from_edges(bpws[:-1], bpws[1:], is_Dell=True)
    
    f_0_a = nmt.NmtField(mask_beta, [map_beta_a], lmax=lmax)
    f_0_b = nmt.NmtField(mask_beta, [map_beta_b], lmax=lmax)
    
    w_00 = nmt.NmtWorkspace()
    w_00.compute_coupling_matrix(f_0_a, f_0_b, b)
    w_00.write_to(ws_dir + 'w_00_beta.fits')

def calculate_spectra_beta(lmax, deltal, ws_dir, nsims, output_maps_dir, mask_beta, Cls_out_dir, nsideout):
    """
    Computes the cross-power spectrum of the anisotropic birefringence angle beta.
    
    Cross-correlating two independent noise realizations (split 1 and split 2)
    mitigates the noise bias that would otherwise dominate the auto-spectrum.
    
    Outputs a scientific plot comparing the recovered variance with 
    Planck baseline sensitivities (Bortolami et al. 2022).
    """
    bpws = np.arange(0, lmax + deltal + 1, deltal)
    b = nmt.NmtBin.from_edges(bpws[:-1], bpws[1:], is_Dell=True)
    ell_arr = b.get_effective_ells()
    
    w_00 = nmt.NmtWorkspace()
    w_00.read_from(ws_dir + 'w_00_beta.fits')
    N_bin = b.get_n_bands()
    
    half_sims = nsims // 2
    Cls_beta = np.zeros((half_sims, N_bin))
    
    print(f"Computing {half_sims} cross-spectra...")
    for i in range(half_sims):
        # Read Split 1 (Realization i)
        map_beta_a = hp.read_map(output_maps_dir + f'map_beta_sim{i}.fits', dtype=np.float64)
        map_beta_a[map_beta_a == hp.UNSEEN] = 0.0
        f_0_a = nmt.NmtField(mask_beta, [map_beta_a], lmax=lmax)
        
        # Read Split 2 (Realization i + half_sims, statistically independent)
        map_beta_b = hp.read_map(output_maps_dir + f'map_beta_sim{i + half_sims}.fits', dtype=np.float64)
        map_beta_b[map_beta_b == hp.UNSEEN] = 0.0
        f_0_b = nmt.NmtField(mask_beta, [map_beta_b], lmax=lmax)

        # Compute decoupled cross-spectrum
        cl_coupled = nmt.compute_coupled_cell(f_0_a, f_0_b)
        cl_00 = w_00.decouple_cell(cl_coupled)
        
        np.savetxt(Cls_out_dir + 'cl_beta_lmax' + str(lmax) + '_sim' + str(i).zfill(4) + '.txt', cl_00[0])
        Cls_beta[i, :] = cl_00[0]

    # Reference standard deviation from Planck (Bortolami et al. 2022)
    std_Planck = np.array([0.006, 0.019, 0.028, 0.049, 0.060, 0.083, 0.090, 0.112, 0.137, 
                           0.135, 0.169, 0.188, 0.224, 0.253, 0.266, 0.307, 0.298, 0.357, 
                           0.403, 0.406, 0.450, 0.497, 0.476])
    
    mean_cl = np.mean(Cls_beta, axis=0)
    std_cl = np.std(Cls_beta, axis=0)        # Standard deviation of a single realization
    err_mean = std_cl / np.sqrt(half_sims)   # Error on the mean
    
    # --- PLOTTING ---
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(8, 10), sharex=True,
        gridspec_kw={'height_ratios': [3, 1, 1.5], 'hspace': 0.05} 
    )

    # 1. Top Panel: Mean Cross-Spectrum D_ell (Null test -> expected to be 0)
    ax1.errorbar(ell_arr, mean_cl, yerr=err_mean, fmt='o', color='navy', ecolor='gray', 
                 markersize=4, capsize=2, label=r'Mean Cross-Spectrum $\langle D_{\ell}^{\beta \beta} \rangle$')
    ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_ylabel(r'$D_{\ell}^{\beta \beta} \, [\mathrm{deg}^2]$', fontsize=12)
    ax1.legend(fontsize=10, loc='upper right', frameon=False)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_title(f'Anisotropic Beta Cross-Spectrum (N={half_sims} simulations)', fontsize=13)

    # 2. Middle Panel: Normalized Residuals (Delta / Sigma)
    residuals = mean_cl / err_mean
    ax2.plot(ell_arr, residuals, 'o', color='crimson', markersize=4)
    ax2.axhline(0, color='black', lw=1)
    ax2.axhline(3, color='gray', ls='--', lw=0.8)  
    ax2.axhline(-3, color='gray', ls='--', lw=0.8) 
    ax2.fill_between(ell_arr, -1, 1, color='gray', alpha=0.1) 
    ax2.set_ylabel(r'$\Delta / \sigma_{\mu}$', fontsize=10)
    ax2.set_ylim(-5, 5) 
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 3. Bottom Panel: Empirical vs Theoretical Standard Deviation
    ax3.plot(ell_arr, std_cl, 's-', color='darkgreen', markersize=4, lw=1.5, label=r'$\sigma(D_{\ell})^{\mathrm{emp}}$')
    
    diff_len = len(ell_arr) - len(std_Planck)
    if diff_len >= 0:
        ell_plot_ref = ell_arr[diff_len:]
        ax3.plot(ell_plot_ref, std_Planck, '^-', color='darkorange', markersize=4, lw=1.5, 
                 label=r'$\sigma(D_{\ell})^{\mathrm{Planck}}$ (Bortolami et al. 2022)')
    else:
        print(f"Warning: std_Planck ({len(std_Planck)}) is longer than ell_arr ({len(ell_arr)}). Truncating Planck data for plot.")
        ax3.plot(ell_arr, std_Planck[-diff_len:], '^-', color='darkorange', markersize=4, lw=1.5, 
                 label=r'$\sigma(D_{\ell})^{\mathrm{Planck}}$ (Bortolami et al. 2022)')

    ax3.set_ylabel(r'$\sigma(D_{\ell}) \, [\mathrm{deg}^2]$', fontsize=12)
    ax3.set_xlabel(r'Multipole $\ell$', fontsize=12)
    ax3.set_yscale('log')
    ax3.legend(fontsize=9, frameon=False, loc='best')
    ax3.grid(True, linestyle=':', alpha=0.6, which='both')

    os.makedirs(output_maps_dir + 'maps_plot/', exist_ok=True)
    output_plot_path = output_maps_dir + 'maps_plot/plot_Dl_beta_cross.png'
    plt.savefig(output_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Main plot saved to: {output_plot_path}")

    # --- Plotting the Ratio ---
    fig2 = plt.figure(figsize=(8, 5))
    if diff_len >= 0:
        ratio = (std_cl[diff_len:] / std_Planck)**(-1)
        plt.plot(ell_plot_ref, ratio, 'o-', color='purple')
    else:
        ratio = (std_cl / std_Planck[-diff_len:])**(-1)
        plt.plot(ell_arr, ratio, 'o-', color='purple')
        
    plt.axhline(1, color='black', linestyle='--')
    plt.ylabel(r'$\sigma_{\mathrm{Planck}} / \sigma_{\mathrm{emp}}$')
    plt.xlabel(r'Multipole $\ell$')
    plt.title('Sensitivity Ratio: Planck vs Empirical Simulation')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_plot_path2 = output_maps_dir + 'maps_plot/plot_Dl_beta_ratio_cross.png'
    plt.savefig(output_plot_path2, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Ratio plot saved to: {output_plot_path2}")

def main():
    """
    Main execution pipeline to compute the anisotropic cosmic birefringence 
    angular cross-power spectrum.
    """
    root_dir        = '/home/matteo/Scrivania/workspace/'
    output_maps_dir = root_dir + 'simulazione/maps_beta/'
    mask_in_dir     = root_dir + 'simulazione/mask/'
    ws_dir          = root_dir + 'simulazione/ws/'
    Cls_out_dir     = root_dir + 'simulazione/spectra_beta/'

    os.makedirs(ws_dir, exist_ok=True)
    os.makedirs(Cls_out_dir, exist_ok=True)

    nsims    = 1000
    nsideout = 8
    lmax     = 3 * nsideout - 1
    deltal   = 1
    
    Spix = hp.nside2pixarea(nsideout, degrees=True)
    fwhm = 2.4 * np.sqrt(Spix)
    aposcale = fwhm
    
    # Load and apodize mask
    mask_beta = hp.ud_grade(hp.read_map(output_maps_dir + 'mask_beta.fits', field=0, dtype=np.float64), nsideout, dtype=np.float64)
    mask_beta = nmt.mask_apodization(mask_beta, aposcale, apotype='C2')
    
    print("Creating Workspace...")
    create_workspace(lmax, deltal, mask_beta, output_maps_dir, ws_dir, nsideout)
    
    print("Calculating Beta Spectra...")
    calculate_spectra_beta(lmax, deltal, ws_dir, nsims, output_maps_dir, mask_beta, Cls_out_dir, nsideout)
    
    print("Processing complete.")

if __name__ == "__main__":
    main()