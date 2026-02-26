import numpy as np
import healpy as hp
import pymaster as nmt
from scipy import interpolate
from mpi4py import MPI
import os
import sys

#python covs_namaster.py 1 commander master 22 ana [TT,EE,BB,TE,TB, EB]

#parameters from terminal
comp_sep          = sys.argv[2]      #ex: commander
method            = sys.argv[3]      #ex: master or pure_B
which_spin        = sys.argv[4] 
mode              = sys.argv[5]      #ana, sim or comparison (which covariances to calculate and save)
len_which_spectra = len(sys.argv[6]) 
which_spectra     = sys.argv[6][1:len_which_spectra-1] #the spectra of which we want the covariances, e.g. [TT,TE,TB]
which_spectra     = which_spectra.split(',')
namerun      = 'deltaell'+str(deltal)


if(mode=='ana' or mode=='comparison'):
    #ANALYTIC COVARIANCE FROM NAMASTER
    print("analytic covariance")

    method='master'
    purify_e=False
    purify_b=False
    
    #l binning
    b = nmt.NmtBin.from_lmax_linear(lmax,deltal,is_Dell=True)
    ell_arr = b.get_effective_ells()
    n_ell = len(ell_arr)

    #beam and wpix
    bl = hp.gauss_beam(np.deg2rad(fwhm),lmax=lmax,pol=True)
    wpix = hp.pixwin(nside,pol=True,lmax=lmax)
    sm = np.zeros((2,len(bl)))
    sm[0] = (bl[:,0]*wpix[0])**2
    sm[1] = (bl[:,1]*wpix[1])**2
    
    #read fiducial power spectrum
    l,TT,TE,EE,BB=np.loadtxt(fiducial_path+fiducial+'.txt',unpack=True,usecols=(0,1,2,3,4))
    l  = np.arange(lmax-1)+2 #from 2 onward
    ll = l*(l+1)/2/np.pi
    cl_tt = np.zeros(lmax+1)
    cl_te = np.zeros(lmax+1)
    cl_tb = np.zeros(lmax+1)
    cl_ee = np.zeros(lmax+1)
    cl_bb = np.zeros(lmax+1)
    cl_eb = np.zeros_like(cl_ee)
    cl_tt[2:lmax+1] = TT[0:lmax-1]/ll[0:lmax-1]
    cl_te[2:lmax+1] = TE[0:lmax-1]/ll[0:lmax-1]
    cl_ee[2:lmax+1] = EE[0:lmax-1]/ll[0:lmax-1]
    cl_bb[2:lmax+1] = BB[0:lmax-1]/ll[0:lmax-1]

    cl_N_TT = np.zeros(lmax+1)
    cl_N_EE = np.zeros(lmax+1)
    cl_N_BB = np.zeros(lmax+1)
    cl_N_TT = np.zeros(lmax+1)
    cl_N_EE = np.zeros(lmax+1)
    cl_N_BB = np.zeros(lmax+1)

    patch_mask_T = hp.read_map(mask_path+masktype+'_apo'+apotype+'scale'+str(aposcale)+'_T.fits')
    patch_mask_P = hp.read_map(mask_path+masktype+'_apo'+apotype+'scale'+str(aposcale)+'_P.fits')

    f0 = nmt.NmtField(patch_mask_T,[np.zeros(npix)],purify_e=purify_e,purify_b=purify_b,beam=sm[0],lmax_sht=lmax)
    f2 = nmt.NmtField(patch_mask_P,[np.zeros(npix),np.zeros(npix)],purify_e=purify_e,purify_b=purify_b,beam=sm[0],lmax_sht=lmax)
 
    w00 = nmt.NmtWorkspace()
    w00.compute_coupling_matrix(f0, f0, b)
    cw00 = nmt.NmtCovarianceWorkspace()
    cw00.compute_coupling_coefficients(f0, f0, f0, f0)

    #spin 0 - spin 0
    covar_00_00 = nmt.gaussian_covariance(cw00,
                                      0, 0, 0, 0,  # Spins of the 4 fields
                                      [cl_tt*sm[0]+cl_N_TT],  # TT
                                      [cl_tt*sm[0]],  # TT
                                      [cl_tt*sm[0]],  # TT
                                      [cl_tt*sm[0]+cl_N_TT],  # TT
                                      w00, wb=w00).reshape([n_ell, 1,
                                                            n_ell, 1])
    covar_TT_TT = covar_00_00[:, 0, :, 0]
    np.save(save_path+'cov_ana_TT_'+method+'_'+namerun+'.npy',covar_TT_TT)


###################################### qui

# da namaster web
# covar_00_22 = nmt.gaussian_covariance(cw, 0, 0, 2, 2,  # Spins of the 4 fields
#                                       [cl_te, cl_tb],  # TE, TB
#                                       [cl_te, cl_tb],  # TE, TB
#                                       [cl_te, cl_tb],  # TE, TB
#                                       [cl_te, cl_tb],  # TE, TB
#                                       w00, wb=w22).reshape([n_ell, 1,
#                                                             n_ell, 4])
# covar_TT_EE = covar_00_22[:, 0, :, 0]
# covar_TT_EB = covar_00_22[:, 0, :, 1]
# covar_TT_BE = covar_00_22[:, 0, :, 2]
# covar_TT_BB = covar_00_22[:, 0, :, 3]


###################################################

    w02 = nmt.NmtWorkspace()
    w02.compute_coupling_matrix(f0, f2, b)
    cw02 = nmt.NmtCovarianceWorkspace()
    cw02.compute_coupling_coefficients(f0, f2, f0, f2)

    #spin 0 - spin 2
    covar_02_02 = nmt.gaussian_covariance(cw02, 0, 2, 0, 2,  # Spins of the 4 fields
                                          [cl_tt*sm[0]+cl_N_TT],  # TT
                                          [cl_te*np.sqrt(sm[0])*np.sqrt(sm[1]), cl_tb*np.sqrt(sm[0])*np.sqrt(sm[1])],  # TE, TB
                                          [cl_te*np.sqrt(sm[0])*np.sqrt(sm[1]), cl_tb*np.sqrt(sm[0])*np.sqrt(sm[1])],  # ET, BT
                                          [cl_ee*sm[1]+cl_N_EE, cl_eb*sm[1],
                                           cl_eb*sm[1], cl_bb*sm[1]+cl_N_BB],  # EE, EB, BE, BB
                                          w02, wb=w02).reshape([n_ell, 2,
                                                                    n_ell, 2])
    covar_TE_TE = covar_02_02[:, 0, :, 0]
    covar_TE_TB = covar_02_02[:, 0, :, 1]
    covar_TB_TE = covar_02_02[:, 1, :, 0]
    covar_TB_TB = covar_02_02[:, 1, :, 1]

    covar_02_20 = nmt.gaussian_covariance(cw02, 0, 2, 0, 2,  # Spins of the 4 fields
                                          [cl_tt*sm[0]+cl_N_TT],  # TT
                                          [cl_te*np.sqrt(sm[0])*np.sqrt(sm[1]), cl_tb*np.sqrt(sm[0])*np.sqrt(sm[1])],  # TE, TB
                                          [cl_te*np.sqrt(sm[0])*np.sqrt(sm[1]), cl_tb*np.sqrt(sm[0])*np.sqrt(sm[1])],  # ET, BT
                                          [cl_ee*sm[1], cl_eb*sm[1],
                                           cl_eb*sm[1], cl_bb*sm[1]],  # EE, EB, BE, BB
                                          w02, wb=w02).reshape([n_ell, 2,
                                                            n_ell, 2])
    covar_TE_ET = covar_02_20[:, 0, :, 0]
    covar_TE_BT = covar_02_20[:, 0, :, 1]
    covar_TB_ET = covar_02_20[:, 1, :, 0]
    covar_TB_BT = covar_02_20[:, 1, :, 1]

    covar_20_02 = nmt.gaussian_covariance(cw02, 2, 0, 2, 0,  # Spins of the 4 fields
                                          [cl_ee*sm[1], cl_eb*sm[1],
                                           cl_eb*sm[1], cl_bb*sm[1]],  # EE, EB, BE, BB
                                          [cl_te*np.sqrt(sm[0])*np.sqrt(sm[1]), cl_tb*np.sqrt(sm[0])*np.sqrt(sm[1])],  # TE, TB
                                          [cl_te*np.sqrt(sm[0])*np.sqrt(sm[1]), cl_tb*np.sqrt(sm[0])*np.sqrt(sm[1])],  # ET, BT
                                          [cl_tt*sm[0]+cl_N_TT],  # TT
                                          w02, wb=w02).reshape([n_ell, 2,
                                                                n_ell, 2])
    covar_ET_TE = covar_20_02[:, 0, :, 0]
    covar_ET_TB = covar_20_02[:, 0, :, 1]
    covar_BT_TE = covar_20_02[:, 1, :, 0]
    covar_BT_TB = covar_20_02[:, 1, :, 1]

    covar_20_20 = nmt.gaussian_covariance(cw02, 2, 0, 2, 0,  # Spins of the 4 fields
                                          [cl_ee*sm[1]+cl_N_EE, cl_eb*sm[1],
                                           cl_eb*sm[1], cl_bb*sm[1]+cl_N_BB],  # EE, EB, BE, BB
                                          [cl_te*np.sqrt(sm[0])*np.sqrt(sm[1]), cl_tb*np.sqrt(sm[0])*np.sqrt(sm[1])],  # TE, TB
                                          [cl_te*np.sqrt(sm[0])*np.sqrt(sm[1]), cl_tb*np.sqrt(sm[0])*np.sqrt(sm[1])],  # ET, BT
                                          [cl_tt*sm[0]+cl_N_TT],  # TT
                                          w02, wb=w02).reshape([n_ell, 2,
                                                                n_ell, 2])
    covar_ET_ET = covar_20_20[:, 0, :, 0]
    covar_ET_BT = covar_20_20[:, 0, :, 1]
    covar_BT_ET = covar_20_20[:, 1, :, 0]
    covar_BT_BT = covar_20_20[:, 1, :, 1]

        
            #TE
    np.save(save_path+'cov_ana_TE_'+method+'_'+namerun+'.npy',(covar_TE_TE+covar_TE_ET+covar_ET_TE+covar_ET_ET)/4.)

            
    np.save(save_path+'cov_ana_TB_'+method+'_'+namerun+'.npy',(covar_TB_TB+covar_TB_BT+covar_BT_TB+covar_BT_BT)/4.)

    
    w22 = nmt.NmtWorkspace()
    w22.compute_coupling_matrix(f2, f2, b)
    cw22 = nmt.NmtCovarianceWorkspace()
    cw22.compute_coupling_coefficients(f2, f2, f2, f2)
    #spin 2 - spin 2
    covar_22_22 = nmt.gaussian_covariance(cw22, 2, 2, 2, 2,  # Spins of the 4 fields
                                              [cl_ee*sm[1]+cl_N_EE, cl_eb*sm[1],
                                               cl_eb*sm[1], cl_bb*sm[1]+cl_N_BB],  # EE, EB, BE, BB
                                              [cl_ee*sm[1], cl_eb*sm[1],
                                               cl_eb*sm[1], cl_bb*sm[1]],  # EE, EB, BE, BB
                                              [cl_ee*sm[1], cl_eb*sm[1],
                                               cl_eb*sm[1], cl_bb*sm[1]],  # EE, EB, BE, BB
                                              [cl_ee*sm[1]+cl_N_EE, cl_eb*sm[1],
                                               cl_eb*sm[1], cl_bb*sm[1]+cl_N_BB],  # EE, EB, BE, BB
                                              w22, wb=w22).reshape([n_ell, 4,
                                                                    n_ell, 4])
        covar_EE_EE = covar_22_22[:, 0, :, 0]
        covar_EE_EB = covar_22_22[:, 0, :, 1]
        covar_EE_BE = covar_22_22[:, 0, :, 2]
        covar_EE_BB = covar_22_22[:, 0, :, 3]
        covar_EB_EE = covar_22_22[:, 1, :, 0]
        covar_EB_EB = covar_22_22[:, 1, :, 1]
        covar_EB_BE = covar_22_22[:, 1, :, 2]
        covar_EB_BB = covar_22_22[:, 1, :, 3]
        covar_BE_EE = covar_22_22[:, 2, :, 0]
        covar_BE_EB = covar_22_22[:, 2, :, 1]
        covar_BE_BE = covar_22_22[:, 2, :, 2]
        covar_BE_BB = covar_22_22[:, 2, :, 3]
        covar_BB_EE = covar_22_22[:, 3, :, 0]
        covar_BB_EB = covar_22_22[:, 3, :, 1]
        covar_BB_BE = covar_22_22[:, 3, :, 2]
        covar_BB_BB = covar_22_22[:, 3, :, 3]


        
            #EE
        np.save(save_path+'cov_ana_EE_'+method+'_'+namerun+'.npy',covar_EE_EE)
        
            #BB
        np.save(save_path+'cov_ana_BB_'+method+'_'+namerun+'.npy',covar_BB_BB)
        
        np.save(save_path+'cov_ana_EB_'+method+'_'+namerun+'.npy',(covar_EB_EB+covar_EB_BE+covar_BE_EB+covar_BE_BE)/4.)


#########################################################
#########################################################
# if(mode=='sim' or mode=='comparison'):
#     for i in np.arange(rank*patchperproc, (rank+1)*patchperproc,1):
#         if(i < npatches):
#             #get patch index
#             ipatch = patch_indices[i+istart]

#             #COVARIANCE FROM SIMULATIONS
#             print("covariance from simulations")

#             def covariance_from_loaded_spectrum(path,col):
#                 covar_sample = np.zeros([n_ell, n_ell])
#                 mean_sample = np.zeros(n_ell)

#                 for isim in np.arange(nsims):#(nsims+1):
#                     if(isim!=970):
#                         #cl is just a name, can be also the bandpower Dl
#                         cl = np.load(path+'patch'+str(ipatch).zfill(index_digits)+'/'+method+'_'+namerun+'_sim'+str(isim).zfill(4)+'.npy')[:,col]
#                         covar_sample += cl[None, :]*cl[:, None]
#                         mean_sample += cl
#                 mean_sample /= nsims
#                 covar_sample = covar_sample / nsims
#                 covar_sample -= mean_sample[None, :] * mean_sample[:, None]

#                 #if(comp_sep=='sevem'):
#                 #    for isim in np.arange(200,800,1):
#                 #        #cl is just a name, can be also the bandpower Dl
#                 #        cl = np.load(path+'patch'+str(ipatch).zfill(index_digits)+'/'+method+'_'+namerun+'_spin22_sim'+str(isim).zfill(4)+'.npy')[:,col]
#                 #        covar_sample += cl[None, :]*cl[:, None]
#                 #        mean_sample += cl

#                 #    mean_sample /= 600
#                 #    covar_sample = covar_sample / 600
#                 #    covar_sample -= mean_sample[None, :] * mean_sample[:, None]
                
#                 #if(comp_sep=='commander'):
#                 #    for isim in np.arange(300,600,1):
#                 #        #cl is just a name, can be also the bandpower Dl
#                 #        cl = np.load(path+'patch'+str(ipatch).zfill(index_digits)+'/'+method+'_'+namerun+'_spin22_sim'+str(isim).zfill(4)+'.npy')[:,col]
#                 #        covar_sample += cl[None, :]*cl[:, None]
#                 #        mean_sample += cl
#                 #    for isim in np.arange(700,800,1):
#                 #        #cl is just a name, can be also the bandpower Dl
#                 #        cl = np.load(path+'patch'+str(ipatch).zfill(index_digits)+'/'+method+'_'+namerun+'_spin22_sim'+str(isim).zfill(4)+'.npy')[:,col]
#                 #        covar_sample += cl[None, :]*cl[:, None]
#                 #        mean_sample += cl

#                 #    mean_sample /= 400
#                 #    covar_sample = covar_sample / 400
#                 #    covar_sample -= mean_sample[None, :] * mean_sample[:, None]
                
#                 return covar_sample*conversion_factor**2

#             #l TT EE BB TE TB EB
#             # if(check_TT):
#             #     covar_sample_TT = covariance_from_loaded_spectrum(spectra_path,1)
#             #     np.save(save_path+'cov_sim_TT_'+method+'_'+namerun+'_'+str(nsims)+'sims_patch'+str(ipatch).zfill(index_digits)+'.npy',covar_sample_TT)
#             # if(check_TE):
#             #     covar_sample_TE = covariance_from_loaded_spectrum(spectra_path,4)
#             #     np.save(save_path+'cov_sim_TE_'+method+'_'+namerun+'_'+str(nsims)+'sims_patch'+str(ipatch).zfill(index_digits)+'.npy',covar_sample_TE)
#             # if(check_TB):
#             #     covar_sample_TB = covariance_from_loaded_spectrum(spectra_path,5)
#             #     np.save(save_path+'cov_sim_TB_'+method+'_'+namerun+'_'+str(nsims)+'sims_patch'+str(ipatch).zfill(index_digits)+'.npy',covar_sample_TB)
#             # if(check_EE):
#             #     covar_sample_EE = covariance_from_loaded_spectrum(spectra_path,2)
#             #     np.save(save_path+'cov_sim_EE_'+method+'_'+namerun+'_'+str(nsims)+'sims_patch'+str(ipatch).zfill(index_digits)+'.npy',covar_sample_EE)
#             # if(check_BB):
#             #     covar_sample_BB = covariance_from_loaded_spectrum(spectra_path,3)
#             #     np.save(save_path+'cov_sim_BB_'+method+'_'+namerun+'_'+str(nsims)+'sims_patch'+str(ipatch).zfill(index_digits)+'.npy',covar_sample_BB)
#             # if(check_EB):
#             #     covar_sample_EB = covariance_from_loaded_spectrum(spectra_path,6)
#             #     np.save(save_path+'cov_sim_EB_'+method+'_'+namerun+'_'+str(nsims)+'sims_patch'+str(ipatch).zfill(index_digits)+'.npy',covar_sample_EB)
#             if(which_spin=='22'):
#                 if(check_EE):
#                     covar_sample_EE = covariance_from_loaded_spectrum(spectra_path,1)
#                     np.save(save_path+'cov_sim_EE_'+method+'_'+namerun+'_'+str(nsims)+'sims_patch'+str(ipatch).zfill(index_digits)+'.npy',covar_sample_EE)
#                 if(check_BB):
#                     covar_sample_BB = covariance_from_loaded_spectrum(spectra_path,2)
#                     np.save(save_path+'cov_sim_BB_'+method+'_'+namerun+'_'+str(nsims)+'sims_patch'+str(ipatch).zfill(index_digits)+'.npy',covar_sample_BB)
#                 if(check_EB):
#                     covar_sample_EB = covariance_from_loaded_spectrum(spectra_path,3)
#                     np.save(save_path+'cov_sim_EB_'+method+'_'+namerun+'_'+str(nsims)+'sims_patch'+str(ipatch).zfill(index_digits)+'.npy',covar_sample_EB)
            

# #END
