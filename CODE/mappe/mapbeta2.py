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
    betas_best_func = np.ones(nsims)
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
        betas_best_func[i] = res.x[0]
        #maps_beta[i,ipatch] = res.x[0]

    
    sigma_beta_func = np.rad2deg(np.std(betas_best_func, ddof=1))
    var_beta_func = np.rad2deg(np.var(betas_best_func, ddof=1))

    mean_beta_func = np.rad2deg(np.mean(betas_best_func))
    residui = mean_beta_func/(sigma_beta_func/np.sqrt(nsims))
    return mean_beta_func,sigma_beta_func,var_beta_func,residui,betas_best_func

    # fig, ax2 = plt.subplots(1, 1, figsize=(6, 5))

    # nbins = 20


    # ax2.hist(np.rad2deg(betas_best_func), bins=nbins, edgecolor='black')
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
output_maps_dir = root_dir + 'simulazione/maps_beta/' # Assicurati che esista
os.makedirs(output_maps_dir, exist_ok=True)
plot_maps_dir = output_maps_dir+ 'maps_plot/'
os.makedirs(plot_maps_dir, exist_ok=True)
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
mask_beta= np.ones(npix_patch)
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
    else:
        mask_beta[i]=0
 
hp.write_map(output_maps_dir + 'mask_beta.fits', mask_beta, overwrite=True)
hp.mollview(mask_beta)
plt.savefig(plot_maps_dir + 'mask_beta.png')

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


# --- 3. CARICAMENTO SPETTRI SIMULAZIONI (Loop Sim + Patch) ---
print("Caricamento Spettri Simulazioni...")
for s in range(nsims):
    if s % 100 == 0: print(f"  - Elaborazione simulazione {s}/{nsims}...")
    
    for k, pixel_idx in enumerate(active_pixel_indices):
        
        filename_spectra = (Cls_out_dir + 'NamasterCls_lmax' + str(lmax) + 
                    '_deltal' + str(deltal) + 
                    '_sim' + str(s).zfill(4) +   
                    '_patch' + str(k) +          # Patch senza zfill qui
                    '_nsidepatch' + str(nsidepatch) + '.txt')
        
        try:
            # Carico colonne 1-6
            data = np.loadtxt(filename_spectra, unpack=True, usecols=(1,2,3,4,5,6))
            all_sims_spectra[s, pixel_idx, :, :] = data #nsims, npix_patch, 6, n_bins
        except:
            pass
b = nmt.NmtBin.from_lmax_linear(lmax, deltal, is_Dell=True)
ell_arr = b.get_effective_ells()
maps_beta=np.zeros((nsims,npix_patch))
map_beta_mean=np.zeros(npix_patch)
map_sigma=np.zeros(npix_patch)
map_var=np.zeros(npix_patch)
map_residui=np.zeros(npix_patch)
for ipatch in range (npix_patch):
    if master_var_EB[ipatch, 0] == 0:
        map_beta_mean[ipatch] = hp.UNSEEN
        map_sigma[ipatch] = hp.UNSEEN
        map_var[ipatch] = hp.UNSEEN
        map_residui[ipatch] = hp.UNSEEN
        maps_beta[:, ipatch] = hp.UNSEEN
        continue
    map_beta_mean[ipatch],map_sigma[ipatch],map_var[ipatch],map_residui[ipatch],maps_beta[:,ipatch]= \
        D_estimators(all_sims_spectra,ell_arr,
        nsims,lmin,nsideout,deltal,ipatch,master_var_EB[ipatch,:])
output_maps_dir = root_dir + 'simulazione/maps_beta/' # Assicurati che esista
os.makedirs(output_maps_dir, exist_ok=True)
plot_maps_dir = output_maps_dir+ 'maps_plot/'
os.makedirs(plot_maps_dir, exist_ok=True)
mask_in_dir + 'COM_Mask_CMB-common-Mask-Pol_2048_R3.00.fits'
hp.write_map(output_maps_dir + 'map_beta_mean.fits', map_beta_mean, overwrite=True)
hp.write_map(output_maps_dir + 'map_beta_sigma.fits', map_sigma, overwrite=True)
hp.write_map(output_maps_dir + 'map_beta_var.fits', map_var, overwrite=True)
hp.write_map(output_maps_dir + 'map_beta_residui.fits', map_residui, overwrite=True)
for i in range (nsims):

    hp.write_map(output_maps_dir + f'map_beta_sim{i}.fits', maps_beta[i, :], overwrite=True, dtype=np.float64)


# 3. Creazione Grafici PNG

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
            title=r'Residual ($\mu / \sigma$)', 
            unit='sigma', 
            cmap='coolwarm', 
            ) 
plt.savefig(plot_maps_dir + 'plot_beta_residui.png')
plt.close()

# --- NUOVO: Plot Singola Simulazione ---
plt.figure(figsize=(12, 8))
hp.mollview(maps_beta[400, :], 
            title=f'Beta [deg] - Simulation {400}', 
            unit='deg', 
            cmap='RdBu_r', 
            ) # Uso la stessa scala della media per confronto
plt.savefig(plot_maps_dir + f'plot_beta_sim{00}.png')
plt.close()

print(f"Tutto salvato correttamente in: {plot_maps_dir}")







