import numpy as np
import healpy as hp
import matplotlib
import os
matplotlib.use('Agg') # Forza il backend "non-interattivo" 
import matplotlib.pyplot as plt
from time import perf_counter

import pymaster as nmt
from mpi4py import MPI
from scipy.optimize import minimize


def D_estimators (all_sims_spectra,ell_arr,
        nsims,lmin,nsideout,deltal,ipatch,vardl
        
    ):
    mean_Dest = 0.0
    betamin = np.deg2rad(-1.5)
    betamax = np.deg2rad(1.5)
    
    limit_l = 2 * nsideout
    ell_upper_bound = ell_arr + (deltal / 2.0)
    ell_lower_bound = ell_arr - (deltal / 2.0)
    valid_indices = (ell_upper_bound <= limit_l) & (ell_lower_bound >= lmin)
    
    
    var_cut = vardl[valid_indices]
    dl_EE_sim = all_sims_spectra[:,ipatch, 1, valid_indices]
    dl_BB_sim = all_sims_spectra[:,ipatch, 2, valid_indices]
    dl_EB_sim = all_sims_spectra[:,ipatch, 4, valid_indices]
    betas_best_rad = np.ones(nsims)
    inv_cov= np.diag(1/var_cut)
    
    
    ####################
    #funzioni utili solo per metodo con minimize
    def chi2_beta(beta, dl_EB, dl_EE,dl_BB, inv_cov):
            
            b = beta[0]
            
            Dest_EB=(dl_EB*np.cos(4*b))-0.5*(dl_EE-dl_BB)*np.sin(4*b)
            x_l = Dest_EB - mean_Dest
            return x_l @ inv_cov @ x_l

    def jacobian (beta, dl_EB, dl_EE,dl_BB,inv_cov):

        grad = np.zeros(beta.size)
        for i in range (grad.size):
            betap=np.copy(beta)
            betam=np.copy(beta)
            betap[i]+=1e-10
            betam[i]-=1e-10
            grad[i]=(chi2_beta(betap, dl_EB, dl_EE,dl_BB, inv_cov)-
                     chi2_beta(betam, dl_EB, dl_EE,dl_BB, inv_cov))/2e-10
        return grad
    ######################

    for i in range(nsims):
        
        dl_EE = dl_EE_sim[i]
        dl_BB = dl_BB_sim[i]
        dl_EB = dl_EB_sim[i]
       
        
        res = minimize(chi2_beta,x0=np.array([0.0]),args=(dl_EB, dl_EE,dl_BB, inv_cov),
                       method='L-BFGS-B',jac=jacobian, bounds=[(betamin, betamax)])   
        betas_best_rad[i] = res.x[0]
        

    betas_best = np.rad2deg(betas_best_rad)
    sigma_beta_func = np.std(betas_best)
    var_beta_func = np.var(betas_best)

    mean_beta_func = np.mean(betas_best)
    residui = mean_beta_func/(sigma_beta_func/np.sqrt(nsims))
    return mean_beta_func,sigma_beta_func,var_beta_func,residui,betas_best

    # fig, ax2 = plt.subplots(1, 1, figsize=(6, 5))

    # nbins = 20


    # ax2.hist(np.rad2deg(betas_best), bins=nbins, edgecolor='black')
    # ax2.set_xlabel(r'$\beta_{\rm best}^{\rm (minimize)}$ [deg]')
    # ax2.set_title(r'Distribuzione di $\beta_{\rm best}$ (minimize) '+varname)
    # ax2.axvline(mean_beta_func, color='red', linestyle='--')
    # ax2.axvline(mean_beta_func + sigma_beta_func, color='red', linestyle=':')
    # ax2.axvline(mean_beta_func - sigma_beta_func, color='red', linestyle=':')
    # ax2.text(0.02, 0.95, rf'$\bar\beta = {mean_beta_func:.2e}$'
    #      '\n' + rf'$\sigma = {sigma_beta_func:.2e}$',
    #      transform=ax2.transAxes, va='top')

    # plt.tight_layout()
    # n_steps_int = int(round((betamax - betamin) / step))
    # plt.savefig(plots_dir + 'Dest_EB_hist_nsims_'+varname
    #         + str(nsims) + '_steps' + str(n_steps_int)
    #         + '_lmax' + str(lmax) +'_index_' + str(ipatch).zfill(4) +'.png')
    # plt.close()







# --- PARAMETRI (Presi dal tuo main) ---
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

# --- 1. MAPPATURA GEOMETRICA ---
print("Creazione mappa indici attivi...")

mask_galT = hp.ud_grade(hp.read_map(mask_in_dir + 'COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits', field=0, dtype=np.float64), nsideout, dtype=np.float64)
mask_galP = hp.ud_grade(hp.read_map(mask_in_dir + 'COM_Mask_CMB-common-Mask-Pol_2048_R3.00.fits', field=0, dtype=np.float64), nsideout, dtype=np.float64)

mask_galT[mask_galT < 0.5] = 0.0; mask_galT[mask_galT >= 0.5] = 1.0
mask_galP[mask_galP < 0.5] = 0.0; mask_galP[mask_galP >= 0.5] = 1.0

npix_patch = hp.nside2npix(nsidepatch)
active_pixel_indices = [] 

for i in range(npix_patch):
    mask_patch_LR = np.zeros(npix_patch); mask_patch_LR[i] = 1
    mask_patch_HR = hp.ud_grade(mask_patch_LR, nsideout, dtype=np.float64)
    mask_patch_HR[mask_patch_HR < 0.5] = 0.0; mask_patch_HR[mask_patch_HR >= 0.5] = 1.0
    
    total_patch = np.sum(mask_patch_HR)
    if total_patch == 0: continue # Check sicurezza

    fractionT = np.sum(mask_patch_HR * mask_galT) / total_patch
    fractionP = np.sum(mask_patch_HR * mask_galP) / total_patch
    
    if fractionT >= 0.5 and fractionP >= 0.5:
        active_pixel_indices.append(i)

print(f"Mappatura completata: {len(active_pixel_indices)} patch attive su {npix_patch}.")

# Inizializzazione Array
first_file = (Cls_out_dir + 'NamasterCls_lmax' + str(lmax) + '_deltal' + str(deltal) + 
              '_sim0000_patch0_nsidepatch' + str(nsidepatch) + '.txt')

n_bins = np.loadtxt(first_file, unpack=True).shape[1]


all_sims_spectra = np.zeros((nsims, npix_patch, 6, n_bins))
master_var_EB = np.zeros((npix_patch, n_bins))


# --- 2. CARICAMENTO COVARIANZE (Loop SOLO sulle patch) ---
# Lo facciamo fuori dal ciclo delle simulazioni per non caricarle 1000 volte
print("Caricamento Covarianze Master...")
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
    
    for k, pixel_idx in enumerate(active_pixel_indices):
        
        filename_spectra = (Cls_out_dir + 'NamasterCls_lmax' + str(lmax) + 
                    '_deltal' + str(deltal) + 
                    '_sim' + str(s).zfill(4) +   
                    '_patch' + str(k) +          
                    '_nsidepatch' + str(nsidepatch) + '.txt')

        data = np.loadtxt(filename_spectra, unpack=True, usecols=(1,2,3,4,5,6))
        all_sims_spectra[s, pixel_idx, :, :] = data #nsims, npix_patch, 6, n_bins
        
b = nmt.NmtBin.from_lmax_linear(lmax, deltal, is_Dell=True)
ell_arr = b.get_effective_ells()
maps_beta=np.zeros((nsims,npix_patch))
map_beta_mean=np.zeros(npix_patch)
map_sigma=np.zeros(npix_patch)
map_var=np.zeros(npix_patch)
map_residui=np.zeros(npix_patch)
maps_beta2=np.zeros((nsims,npix_patch))
map_beta_mean2=np.zeros(npix_patch)
map_sigma2=np.zeros(npix_patch)
map_var2=np.zeros(npix_patch)
map_residui2=np.zeros(npix_patch)
for ipatch in range (npix_patch):
    if master_var_EB[ipatch, 0] == 0:
        map_beta_mean[ipatch] = hp.UNSEEN
        map_sigma[ipatch] = hp.UNSEEN
        map_var[ipatch] = hp.UNSEEN
        map_residui[ipatch] = hp.UNSEEN
        maps_beta[:, ipatch] = hp.UNSEEN
        map_beta_mean2[ipatch] = 0
        map_sigma2[ipatch] = 0
        map_var2[ipatch] = 0
        map_residui2[ipatch] = 0
        maps_beta2[:, ipatch] = 0
        continue
    map_beta_mean[ipatch],map_sigma[ipatch],map_var[ipatch],map_residui[ipatch],maps_beta[:,ipatch]= \
        D_estimators(all_sims_spectra,ell_arr,
        nsims,lmin,nsideout,deltal,ipatch,master_var_EB[ipatch,:])
    map_beta_mean2[ipatch],map_sigma2[ipatch],map_var2[ipatch],map_residui2[ipatch],maps_beta2[:,ipatch]= \
        D_estimators(all_sims_spectra,ell_arr,
        nsims,lmin,nsideout,deltal,ipatch,master_var_EB[ipatch,:])
output_maps_dir = root_dir + 'simulazione/maps_beta/' 
os.makedirs(output_maps_dir, exist_ok=True)
plot_maps_dir = output_maps_dir+ 'maps_plot/'
os.makedirs(plot_maps_dir, exist_ok=True)

hp.write_map(output_maps_dir + 'map_beta_mean.fits', map_beta_mean, overwrite=True)
hp.write_map(output_maps_dir + 'map_beta_sigma.fits', map_sigma, overwrite=True)
hp.write_map(output_maps_dir + 'map_beta_var.fits', map_var, overwrite=True)
hp.write_map(output_maps_dir + 'map_beta_residui.fits', map_residui, overwrite=True)
for i in range (nsims):

    hp.write_map(output_maps_dir + f'map_beta_sim{i}.fits', maps_beta[i, :], overwrite=True, dtype=np.float64)




# A) MEDIA BETA
plt.figure(figsize=(12, 8))
hp.mollview(map_beta_mean, 
            title=r'Mean $\beta$ [deg]', 
            unit='deg', 
            cmap='RdBu_r', 
            ) 
plt.savefig(plot_maps_dir + 'plot_beta_mean.png')
plt.close()

# B) SIGMA
plt.figure(figsize=(12, 8))
hp.mollview(map_sigma, 
            title=r'$\sigma_\beta$ (Std Dev) [deg]', 
            unit='deg', 
            cmap='viridis') 
plt.savefig(plot_maps_dir + 'plot_beta_sigma.png')
plt.close()

# C) VARIANZA
plt.figure(figsize=(12, 8))
hp.mollview(map_var, 
            title=r'Variance $\beta$ [deg$^2$]', 
            unit='deg^2', 
            cmap='magma') 
plt.savefig(plot_maps_dir + 'plot_beta_var.png')
plt.close()

# D) RESIDUI
plt.figure(figsize=(12, 8))
hp.mollview(map_residui, 
            title=r'Residual ($\mu / \sigma_{mean}$)', 
            unit='sigma', 
            cmap='coolwarm', 
            ) 
plt.savefig(plot_maps_dir + 'plot_beta_residui.png')
plt.close()

# ---  Plot Singola Simulazione ---
plt.figure(figsize=(12, 8))
hp.mollview(maps_beta[100, :], 
            title=f'Beta [deg] - Simulation {100}', 
            unit='deg', 
            cmap='RdBu_r', 
            ) # Uso la stessa scala della media per confronto
plt.savefig(plot_maps_dir + f'plot_beta_sim{100}.png')
plt.close()
ipatch_arr = np.arange(npix_patch) 

fig, (ax1, ax2) = plt.subplots( 
         2, 1, 
         figsize=(7, 8), 
         gridspec_kw={'height_ratios': [3, 1]}, 
         sharex=True # È meglio True così zoomando sopra, zoomi anche sotto
     )

# Pannello 1: Valori medi
ax1.errorbar( 
             ipatch_arr, 
             map_beta_mean2, 
             yerr=map_sigma2/(np.sqrt(nsims)), 
             fmt='o', 
             markersize=2, # Ridotto leggermente dato che sono 768 punti
             alpha=0.7,    # Un po' di trasparenza aiuta a vedere la densità
             label=r'$\beta_{mean}$ [deg]' 
         ) 
ax1.set_ylabel(r'$\beta_{mean}$ [deg]') 
ax1.legend(fontsize=8) 
ax1.grid(True, ls='--', alpha=0.4)
ax1.set_title(f"Beta Statistics per Patch (Nside={nsidepatch})") # Titolo aggiunto

# Pannello 2: Residui
ax2.axhline(0, color='black', lw=0.8) 
ax2.plot(ipatch_arr, map_residui2, 'o', markersize=2, alpha=0.7) 
ax2.axhline(3, color='red', ls='--') 
ax2.axhline(-3, color='red', ls='--') 
ax2.set_ylabel(r'$\Delta/\sigma$') 
ax2.grid(True, ls='--', alpha=0.4) 

# 2. Aggiunta Label Asse X fondamentale
ax2.set_xlabel('Healpix Patch Index') 

plt.tight_layout() # Aggiungi sempre questo per evitare sovrapposizioni
plt.savefig(plot_maps_dir + f'plot_beta_classic.png')
plt.close()


print(f"Tutto salvato correttamente in: {plot_maps_dir}")







