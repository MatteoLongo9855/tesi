import numpy as np
import healpy as hp
import matplotlib
import os

import matplotlib.pyplot as plt
from time import perf_counter

import pymaster as nmt
from mpi4py import MPI
from scipy.optimize import minimize

def create_workspace (lmax,deltal,mask_beta, output_maps_dir, ws_dir,nsideout):
    map_beta_a= hp.read_map(output_maps_dir + f'map_beta_sim{0}.fits',dtype=np.float64)
    mask_bad_a = (map_beta_a == hp.UNSEEN)
    map_beta_a[mask_bad_a] = 0.0
    map_beta_b= hp.read_map(output_maps_dir + f'map_beta_sim{999}.fits',dtype=np.float64)
    mask_bad_b = (map_beta_b == hp.UNSEEN)
    map_beta_b[mask_bad_b] = 0.0
    wpix = hp.pixwin(nsideout, lmax=lmax)
    bpws = np.arange(0, lmax + deltal + 1, deltal)
    b = nmt.NmtBin.from_edges(bpws[:-1], bpws[1:], is_Dell=True)
    #b = nmt.NmtBin.from_lmax_linear(lmax, deltal, is_Dell=True)
    f_0_a = nmt.NmtField(
        mask_beta,
        [map_beta_a],
        lmax=lmax
    )
    f_0_b = nmt.NmtField(
        mask_beta,
        [map_beta_b],
        lmax=lmax
    )
    w_00 = nmt.NmtWorkspace()
    w_00.compute_coupling_matrix(f_0_a, f_0_b, b)
    w_00.write_to(ws_dir + 'w_00_beta' +'.fits')

def calculate_spectra_beta(lmax,deltal,ws_dir,nsims,output_maps_dir,mask_beta,Cls_out_dir,root_dir,nsideout):
    #b = nmt.NmtBin.from_lmax_linear(lmax, deltal, is_Dell=True)
    bpws = np.arange(0, lmax + deltal + 1, deltal)
    b = nmt.NmtBin.from_edges(bpws[:-1], bpws[1:], is_Dell=True)
    wpix = hp.pixwin(nsideout, lmax=lmax)

    ell_arr = b.get_effective_ells()
    w_00 = nmt.NmtWorkspace()
    w_00.read_from(ws_dir + 'w_00_beta' +'.fits')
    N_bin = b.get_n_bands()
    Cls_beta= np.zeros((nsims//2,N_bin))
    for i in range(nsims//2):
        map_beta_a = hp.read_map(output_maps_dir + f'map_beta_sim{i}.fits',dtype=np.float64)
        mask_bad_a = (map_beta_a == hp.UNSEEN)
        map_beta_a[mask_bad_a] = 0.0
        f_0_a = nmt.NmtField(
        mask_beta,
        [map_beta_a],
        
         lmax=lmax
        )
        map_beta_b = hp.read_map(output_maps_dir + f'map_beta_sim{i+(nsims//2)}.fits',dtype=np.float64)
        mask_bad_b = (map_beta_b == hp.UNSEEN)
        map_beta_b[mask_bad_b] = 0.0
        f_0_b = nmt.NmtField(
        mask_beta,
        [map_beta_b],
        
         lmax=lmax
        )



        cl_coupled = nmt.compute_coupled_cell(f_0_a, f_0_b)
        cl_00 = w_00.decouple_cell(cl_coupled)
        np.savetxt(Cls_out_dir +'cl_beta_lmax' + str(lmax)+ '_sim'+ str(i).zfill(4)+'.txt',cl_00[0])
        Cls_beta[i,:] = cl_00[0]
    mean_cl= np.mean(Cls_beta, axis=0)
    
    std_Planck=[0.006,0.019,0.028,0.049,0.060,0.083,0.090,0.112,0.137,0.135,0.169,0.188,0.224,0.253,0.266,0.307,0.298,0.357,0.403,0.406,0.450,0.497,0.476]
    mean_cl = np.mean(Cls_beta, axis=0)
    std_cl = np.std(Cls_beta, axis=0)        # Deviazione Standard della singola realizzazione
    err_mean = std_cl / np.sqrt(nsims/2) # Errore sulla media
    diff_len = len(ell_arr) - len(std_Planck)
    ell_plot_ref = ell_arr[diff_len:]

    # --- PLOTTING (Look "Paper Scientifico") ---
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(8, 10),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1, 1.5], 'hspace': 0.05} # hspace ridotto per unire i grafici
    )

    # 1. PANNELLO SUPERIORE: Spettro D_ell
    # Essendo un null test, oscillera' intorno allo zero.
    ax1.errorbar(
        ell_arr,
        mean_cl,
        yerr=err_mean,
        fmt='o',
        color='navy',
        ecolor='gray',
        markersize=4,
        capsize=2,
        label=r'Mean Cross-Spectrum $\langle D_{\ell}^{\beta \beta} \rangle$'
    )
    ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_ylabel(r'$D_{\ell}^{\beta \beta} \, [\mathrm{deg}^2]$', fontsize=12)
    ax1.legend(fontsize=10, loc='upper right', frameon=False)
    ax1.grid(True, linestyle=':', alpha=0.6)
    # Titolo informativo
    ax1.set_title(f'Cross-Spectrum Beta (N={nsims/2} simulation)', fontsize=13)

    # 2. PANNELLO CENTRALE: Residui Normalizzati (Delta / Sigma)
    # Qui controlliamo se siamo compatibili con lo zero
    residuals = mean_cl / err_mean
    ax2.plot(ell_arr, residuals, 'o', color='crimson', markersize=4)
    ax2.axhline(0, color='black', lw=1)
    ax2.axhline(3, color='gray', ls='--', lw=0.8)  # Limite 3 sigma
    ax2.axhline(-3, color='gray', ls='--', lw=0.8) # Limite -3 sigma
    ax2.fill_between(ell_arr, -1, 1, color='gray', alpha=0.1) # Banda 1 sigma
    ax2.set_ylabel(r'$\Delta / \sigma_{\mu}$', fontsize=10)
    ax2.set_ylim(-5, 5) # Fisso per vedere subito gli outlier
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 3. Deviazione Standard (Con FIX per lunghezze diverse)
    ax3.plot(ell_arr, std_cl, 's-', color='darkgreen', markersize=4, lw=1.5, label=r'$\sigma(D_{\ell})^{\mathrm{emp}}$')
    
    # --- FIX ALLINEAMENTO ARRAY ---
    # Calcoliamo la differenza di lunghezza tra i tuoi ell e i dati Planck
    diff_len = len(ell_arr) - len(std_Planck)
    
    if diff_len >= 0:
        # Se ell_arr è più lungo, assumiamo che a std_Planck manchino i primi multipoli (es. monopolo)
        # Quindi prendiamo ell_arr partendo dall'indice 'diff_len' in poi.
        ell_plot_ref = ell_arr[diff_len:]
        ax3.plot(ell_plot_ref, std_Planck, '^-', color='darkorange', markersize=4, lw=1.5, 
                 label=r'$\sigma(D_{\ell})^{\mathrm{Planck}}$ (Bortolami et al. 2022)')
    else:
        print(f"ERRORE: std_Planck ({len(std_Planck)}) è più lungo di ell_arr ({len(ell_arr)}). Impossibile plottare.")

    ax3.set_ylabel(r'$\sigma(D_{\ell}) \, [\mathrm{deg}^2]$', fontsize=12)
    ax3.set_xlabel(r'Multipole $\ell$', fontsize=12)
    ax3.set_yscale('log')
    ax3.legend(fontsize=9, frameon=False, loc='best')
    ax3.grid(True, linestyle=':', alpha=0.6, which='both')

    output_plot_path = output_maps_dir + 'maps_plot/plot_Dl_beta_cross.png'
    output_plot_path2 = output_maps_dir + 'maps_plot/plot_Dl_beta_ratio_cross.png'
    plt.savefig(output_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Grafico salvato in: {output_plot_path}")
    plt.plot(ell_arr[1:],((std_cl[1:]/std_Planck)**(-1)))
    plt.savefig(output_plot_path2, bbox_inches='tight', dpi=300)
def main ():
    root_dir='/home/matteo/Scrivania/workspace/'
    output_maps_dir = root_dir + 'simulazione/maps_beta/'
    mask_in_dir = root_dir + 'simulazione/mask/'
    ws_dir      = root_dir + 'simulazione/ws/'
    Cls_out_dir = root_dir + 'simulazione/spectra_beta/'

    nsims= 1000
    nsideout=8
    lmax=3*nsideout-1
    deltal=1
    Spix = hp.nside2pixarea(nsideout, degrees=True)
    fwhm = 2.4 * np.sqrt(Spix)
    aposcale = fwhm
    mask_beta = hp.ud_grade(hp.read_map(output_maps_dir + 'mask_beta.fits', 
                                        field=0, dtype=np.float64),nsideout, dtype= np.float64)
    mask_beta = nmt.mask_apodization(mask_beta, aposcale, apotype='C2')
    create_workspace(lmax,deltal,mask_beta, output_maps_dir, ws_dir,nsideout)
    calculate_spectra_beta(lmax,deltal,ws_dir,nsims,output_maps_dir,mask_beta,Cls_out_dir,root_dir,nsideout)
    print("spettro calcolato")
main()
   