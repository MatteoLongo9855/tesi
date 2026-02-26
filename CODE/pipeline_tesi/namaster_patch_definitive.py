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

comm  = MPI.COMM_WORLD
nproc = comm.Get_size() #number of processors
rank  = comm.Get_rank() #processor tag (0, 1, 2, ...)

def read_Cls (Cls_dir_in,lmax,ll):
    TT_th,EE_th,BB_th,TE_th=np.loadtxt(Cls_dir_in+'ffp10_lensedCls.dat',unpack=True,usecols=(1,2,3,4))
    Cls_th=np.zeros((6,lmax+1))
    Cls_th[0,2:]=TT_th[0:lmax-1]/ll[2:]
    Cls_th[1,2:]=EE_th[0:lmax-1]/ll[2:]
    Cls_th[2,2:]=BB_th[0:lmax-1]/ll[2:]
    Cls_th[3,2:]=TE_th[0:lmax-1]/ll[2:]
    Cls_th[4, :] = 0.0 
    Cls_th[5, :] = 0.0
    return(Cls_th)

def create_mask_patch(mask_in_dir,mask_patch_dir,nsideout,nsidepatch,npix_patch,aposcale,apotype,plots_dir):
        patch_quality_map = np.full(npix_patch, hp.UNSEEN)
        mask_galT = hp.ud_grade(hp.read_map(mask_in_dir + 'COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits', field=0, dtype=np.float64),nsideout, dtype= np.float64)
        mask_galP = hp.ud_grade(hp.read_map(mask_in_dir + 'COM_Mask_CMB-common-Mask-Pol_2048_R3.00.fits', field=0, dtype=np.float64),nsideout, dtype= np.float64)
        
        mask_galT[mask_galT < 0.5] = 0.0
        mask_galT[mask_galT >= 0.5] = 1.0
        mask_galP[mask_galP < 0.5] = 0.0
        mask_galP[mask_galP >= 0.5] = 1.0
        k = 0

        for i in range(npix_patch):

            print("creazione patch numero   :",i)
            mask_patch_LR = np.zeros(npix_patch)
            mask_patch_LR[i] = 1
    
        
            mask_patch_HR = hp.ud_grade(mask_patch_LR, nsideout, dtype= np.float64)
            mask_patch_HR[mask_patch_HR < 0.5] = 0.0
            mask_patch_HR[mask_patch_HR >= 0.5] = 1.0

            total_patch_pixels = np.sum(mask_patch_HR)
            total_maskT = mask_patch_HR * mask_galT
            total_maskT_pixels = np.sum(total_maskT)
            fractionT = total_maskT_pixels / total_patch_pixels

            total_maskP = mask_patch_HR * mask_galP
            total_maskP_pixels = np.sum(total_maskP)
            fractionP = total_maskP_pixels / total_patch_pixels

            if fractionT >= 0.5 and fractionP >= 0.5:

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

        hp.mollview(patch_quality_map, title="%Average ctive Pixels", unit="%", cmap='viridis', min=50, max=100)
        plt.savefig(plots_dir +'quality_maps/' + 'patch_preview' + '.png')
        plt.close()
        np.savetxt(mask_patch_dir+'active_patches_nside_'+str(nsidepatch)+
                              '.txt', [k], fmt='%d')
              


def create_maps(Cls_th,lmax,nsideout,nsims,maps_dir,fwhm,var,npix_map,sigma_noise):
    
    for i in range (nsims):
        noisesT = np.random.normal(loc=0, scale=np.sqrt(var), size=2 * npix_map)
        noisesP = np.random.normal(loc=0, scale=np.sqrt(2 * var), size=4 * npix_map)
      
        m_sign=hp.synfast(cls=Cls_th ,lmax=lmax,nside=nsideout,pol=True,new=True,fwhm=np.deg2rad(fwhm),
            pixwin=True)
        ma_TQU = np.zeros_like(m_sign)
        mb_TQU = np.zeros_like(m_sign)

        ma_TQU[0] = m_sign[0] + noisesT[0:npix_map]
        ma_TQU[1] = m_sign[1] + noisesP[0:npix_map]
        ma_TQU[2] = m_sign[2] + noisesP[npix_map:2 * npix_map]

        mb_TQU[0] = m_sign[0] + noisesT[npix_map:2 * npix_map]
        mb_TQU[1] = m_sign[1] + noisesP[2 * npix_map:3 * npix_map]
        mb_TQU[2] = m_sign[2] + noisesP[3 * npix_map:4 * npix_map]



        filenameA = (maps_dir +'mapA_nside' + str(nsideout) +'_whitenoise' + str(sigma_noise) +'microK_arcmin' +
                     '_sim' + str(i).zfill(4) + '.fits')

        filenameB = (maps_dir +'mapB_nside' + str(nsideout) +'_whitenoise' + str(sigma_noise) +'microK_arcmin' +
                     '_sim' + str(i).zfill(4) + '.fits')
        hp.write_map(filenameA, ma_TQU, overwrite=True, dtype= np.float64)
        hp.write_map(filenameB, mb_TQU, overwrite=True, dtype= np.float64)
        
        print(f"[Rank 0] Generazione Mappe: {i} / {nsims}", flush=True)
                
          
   
   
def create_workspace (lmax,deltal,is_Dell,fwhm,nsideout,maskT_out,maskP_out,purify_e,purify_b,ws_dir,maps_dir,sigma_noise,ipatch):

    b = nmt.NmtBin.from_lmax_linear(lmax, deltal, is_Dell=is_Dell)

    bl = hp.gauss_beam(
        np.deg2rad(fwhm),
        lmax=lmax,
        pol=True
    )
    wpix = hp.pixwin(nsideout, pol=True, lmax=lmax)
    sm = np.zeros((2,len(bl)))
    sm[0] = (bl[:,0]*wpix[0])**2
    sm[1] = (bl[:,1]*wpix[1])**2
    


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
   
    w_00 = nmt.NmtWorkspace()
    w_02 = nmt.NmtWorkspace()
    w_20 = nmt.NmtWorkspace()
    w_22 = nmt.NmtWorkspace()
    
    w_00.compute_coupling_matrix(f_0_a, f_0_b, b)
    w_02.compute_coupling_matrix(f_0_a, f_2_b, b)
    w_20.compute_coupling_matrix(f_2_a, f_0_b, b)
    w_22.compute_coupling_matrix(f_2_a, f_2_b, b)

    w_00.write_to(ws_dir + 'w_00_deltaell_' + str(deltal).zfill(1) + '_patch'+str(ipatch).zfill(4)+'.fits')
    w_02.write_to(ws_dir + 'w_02_deltaell_' + str(deltal).zfill(1) +'_patch'+str(ipatch).zfill(4)+ '.fits')
    w_20.write_to(ws_dir + 'w_20_deltaell_' + str(deltal).zfill(1) + '_patch'+str(ipatch).zfill(4)+'.fits')
    w_22.write_to(ws_dir + 'w_22_deltaell_' + str(deltal).zfill(1) + '_patch'+str(ipatch).zfill(4)+'.fits')

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

    b = nmt.NmtBin.from_lmax_linear(lmax, deltal, is_Dell=is_Dell)

    bl = hp.gauss_beam(
        np.deg2rad(fwhm),
        lmax=lmax,
        pol=True
    )
    wpix = hp.pixwin(nsideout, pol=True, lmax=lmax)

    sm = np.zeros((2,len(bl)))
    sm[0] = (bl[:,0]*wpix[0])**2
    
    sm[1] = (bl[:,1]*wpix[1])**2

    ell_arr = b.get_effective_ells()
    n_ell = len(ell_arr)

    Cl_noiseT = np.full(lmax+1,(sigma_noise*np.deg2rad(1/60))**2)
    Cl_noiseP = np.full(lmax+1,(np.sqrt(2)*sigma_noise*np.deg2rad(1/60))**2)
    

    w_00_th = nmt.NmtWorkspace()
    w_02_th = nmt.NmtWorkspace()
    w_22_th = nmt.NmtWorkspace()
    
    Cw_00 = nmt.NmtCovarianceWorkspace()
    Cw_02 = nmt.NmtCovarianceWorkspace()
    Cw_22 = nmt.NmtCovarianceWorkspace()
    

    w_00_th.read_from(ws_dir + 'w_00_th_deltaell_' + str(deltal).zfill(1) +'_patch'+str(ipatch).zfill(4)+ '.fits')
    w_02_th.read_from(ws_dir + 'w_02_th_deltaell_' + str(deltal).zfill(1) +'_patch'+str(ipatch).zfill(4)+ '.fits')
    w_22_th.read_from(ws_dir + 'w_22_th_deltaell_' + str(deltal).zfill(1) +'_patch'+str(ipatch).zfill(4)+ '.fits')

    Cw_00.read_from(ws_dir + 'Cw_00_deltaell_' + str(deltal).zfill(1) +'_patch'+str(ipatch).zfill(4)+ '.fits')
    Cw_02.read_from(ws_dir + 'Cw_02_deltaell_' + str(deltal).zfill(1) + '_patch'+str(ipatch).zfill(4)+'.fits')
    Cw_22.read_from(ws_dir + 'Cw_22_deltaell_' + str(deltal).zfill(1) + '_patch'+str(ipatch).zfill(4)+'.fits')

    #spin 0 - spin 0
    covar_00_00 = nmt.gaussian_covariance(Cw_00,
                                      0, 0, 0, 0,  # Spins of the 4 fields
                                      [Cls_th[0,:]*sm[0]+Cl_noiseT],  # TT
                                      [Cls_th[0,:]*sm[0]],  # TT
                                      [Cls_th[0,:]*sm[0]],  # TT
                                      [Cls_th[0,:]*sm[0]+Cl_noiseT],  # TT
                                      w_00_th, wb=w_00_th).reshape([n_ell, 1,
                                                            n_ell, 1])
    covar_TT_TT = covar_00_00[:, 0, :, 0]
    np.save(cov_dir+'cov_ana_TT'+'_patch'+str(ipatch).zfill(4)+'.npy',covar_TT_TT)
     #spin 0 - spin 2
    covar_02_02 = nmt.gaussian_covariance(Cw_02, 0, 2, 0, 2,  # Spins of the 4 fields
                                          [Cls_th[0,:]*sm[0]+Cl_noiseT],  # TT
                                          [Cls_th[3,:]*np.sqrt(sm[0])*np.sqrt(sm[1]), Cls_th[5,:]*np.sqrt(sm[0])*np.sqrt(sm[1])],  # TE, TB
                                          [Cls_th[3,:]*np.sqrt(sm[0])*np.sqrt(sm[1]), Cls_th[5,:]*np.sqrt(sm[0])*np.sqrt(sm[1])],  # ET, BT
                                          [Cls_th[1,:]*sm[1]+Cl_noiseP, Cls_th[4,:]*sm[1],
                                           Cls_th[4,:]*sm[1], Cls_th[2,:]*sm[1]+Cl_noiseP],  # EE, EB, BE, BB
                                          w_02_th, wb=w_02_th).reshape([n_ell, 2,
                                                                    n_ell, 2])
    covar_TE_TE = covar_02_02[:, 0, :, 0]
    covar_TE_TB = covar_02_02[:, 0, :, 1]
    covar_TB_TE = covar_02_02[:, 1, :, 0]
    covar_TB_TB = covar_02_02[:, 1, :, 1]
    

    covar_02_20 = nmt.gaussian_covariance(Cw_02, 0, 2, 0, 2,  # Spins of the 4 fields
                                          [Cls_th[0,:]*sm[0]+Cl_noiseT],  # TT
                                          [Cls_th[3,:]*np.sqrt(sm[0])*np.sqrt(sm[1]), Cls_th[5,:]*np.sqrt(sm[0])*np.sqrt(sm[1])],  # TE, TB
                                          [Cls_th[3,:]*np.sqrt(sm[0])*np.sqrt(sm[1]), Cls_th[5,:]*np.sqrt(sm[0])*np.sqrt(sm[1])],  # ET, BT
                                          [Cls_th[1,:]*sm[1], Cls_th[4,:]*sm[1],
                                           Cls_th[4,:]*sm[1], Cls_th[2,:]*sm[1]],  # EE, EB, BE, BB
                                          w_02_th, wb=w_02_th).reshape([n_ell, 2,
                                                            n_ell, 2])
    covar_TE_ET = covar_02_20[:, 0, :, 0]
    covar_TE_BT = covar_02_20[:, 0, :, 1]
    covar_TB_ET = covar_02_20[:, 1, :, 0]
    covar_TB_BT = covar_02_20[:, 1, :, 1]
    

    covar_20_02 = nmt.gaussian_covariance(Cw_02, 2, 0, 2, 0,  # Spins of the 4 fields
                                          [Cls_th[1,:]*sm[1], Cls_th[4,:]*sm[1],
                                           Cls_th[4,:]*sm[1], Cls_th[2,:]*sm[1]],  # EE, EB, BE, BB
                                          [Cls_th[3,:]*np.sqrt(sm[0])*np.sqrt(sm[1]), Cls_th[5,:]*np.sqrt(sm[0])*np.sqrt(sm[1])],  # TE, TB
                                          [Cls_th[3,:]*np.sqrt(sm[0])*np.sqrt(sm[1]), Cls_th[5,:] *np.sqrt(sm[0])*np.sqrt(sm[1])],  # ET, BT
                                          [Cls_th[0,:]*sm[0]+Cl_noiseT],  # TT
                                          w_02_th, wb=w_02_th).reshape([n_ell, 2,
                                                                n_ell, 2])
    covar_ET_TE = covar_20_02[:, 0, :, 0]
    covar_ET_TB = covar_20_02[:, 0, :, 1]
    covar_BT_TE = covar_20_02[:, 1, :, 0]
    covar_BT_TB = covar_20_02[:, 1, :, 1]
  

    covar_20_20 = nmt.gaussian_covariance(Cw_02, 2, 0, 2, 0,  # Spins of the 4 fields
                                          [Cls_th[1,:]*sm[1]+Cl_noiseP, Cls_th[4,:]*sm[1],
                                           Cls_th[4,:]*sm[1], Cls_th[2,:]*sm[1]+Cl_noiseP],  # EE, EB, BE, BB
                                          [Cls_th[3,:]*np.sqrt(sm[0])*np.sqrt(sm[1]), Cls_th[5,:]*np.sqrt(sm[0])*np.sqrt(sm[1])],  # TE, TB
                                          [Cls_th[3,:]*np.sqrt(sm[0])*np.sqrt(sm[1]), Cls_th[5,:]*np.sqrt(sm[0])*np.sqrt(sm[1])],  # ET, BT
                                          [Cls_th[0,:]*sm[0]+Cl_noiseT],  # TT
                                          w_02_th, wb=w_02_th).reshape([n_ell, 2,
                                                                n_ell, 2])
    covar_ET_ET = covar_20_20[:, 0, :, 0]
    covar_ET_BT = covar_20_20[:, 0, :, 1]
    covar_BT_ET = covar_20_20[:, 1, :, 0]
    covar_BT_BT = covar_20_20[:, 1, :, 1]
    np.save(cov_dir+'cov_ana_TE'+'_patch'+str(ipatch).zfill(4)+'.npy',(covar_TE_TE+covar_TE_ET+covar_ET_TE+covar_ET_ET)/4.)
    np.save(cov_dir+'cov_ana_TB'+'_patch'+str(ipatch).zfill(4)+'.npy',(covar_TB_TB+covar_TB_BT+covar_BT_TB+covar_BT_BT)/4.)
    

    covar_22_22 = nmt.gaussian_covariance(Cw_22, 2, 2, 2, 2,  # Spins of the 4 fields
                                              [Cls_th[1,:]*sm[1]+Cl_noiseP, Cls_th[4,:]*sm[1],
                                               Cls_th[4,:]*sm[1], Cls_th[2,:]*sm[1]+Cl_noiseP],  # EE, EB, BE, BB
                                              [Cls_th[1,:]*sm[1], Cls_th[4,:]*sm[1],
                                               Cls_th[4,:]*sm[1], Cls_th[2,:]*sm[1]],  # EE, EB, BE, BB
                                              [Cls_th[1,:]*sm[1], Cls_th[4,:]*sm[1],
                                               Cls_th[4,:]*sm[1], Cls_th[2,:]*sm[1]],  # EE, EB, BE, BB
                                              [Cls_th[1,:]*sm[1]+Cl_noiseP, Cls_th[4,:]*sm[1],
                                               Cls_th[4,:]*sm[1], Cls_th[2,:]*sm[1]+Cl_noiseP],  # EE, EB, BE, BB
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
    np.save(cov_dir+'cov_ana_EE'+'_patch'+str(ipatch).zfill(4)+'.npy',covar_EE_EE)
    np.save(cov_dir+'cov_ana_BB'+'_patch'+str(ipatch).zfill(4)+'.npy',covar_BB_BB)
    np.save(cov_dir+'cov_ana_EB'+'_patch'+str(ipatch).zfill(4)+'.npy',(covar_EB_EB+covar_EB_BE+covar_BE_EB+covar_BE_BE)/4.)

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

    b = nmt.NmtBin.from_lmax_linear(lmax, deltal, is_Dell=is_Dell)
    ell_arr = b.get_effective_ells()

    bl = hp.gauss_beam(
        np.deg2rad(fwhm),
        lmax=lmax,
        pol=True
    )
    wpix = hp.pixwin(nsideout, pol=True, lmax=lmax)
    beam_T = bl[:, 0] * wpix[0]
    beam_P = bl[:, 1] * wpix[1]
    w_00 = nmt.NmtWorkspace()
    w_02 = nmt.NmtWorkspace()
    w_20 = nmt.NmtWorkspace()
    w_22 = nmt.NmtWorkspace()
   
    w_00.read_from(ws_dir + 'w_00_deltaell_' + str(deltal).zfill(1) +'_patch'+str(ipatch).zfill(4)+ '.fits')
    w_02.read_from(ws_dir + 'w_02_deltaell_' + str(deltal).zfill(1) +'_patch'+str(ipatch).zfill(4)+ '.fits')
    w_20.read_from(ws_dir + 'w_20_deltaell_' + str(deltal).zfill(1) +'_patch'+str(ipatch).zfill(4)+ '.fits')
    w_22.read_from(ws_dir + 'w_22_deltaell_' + str(deltal).zfill(1) +'_patch'+str(ipatch).zfill(4)+ '.fits')
    

    N_bin = b.get_n_bands()

    # 0:TT, 1:EE, 2:BB, 3:TE, 4:EB, 5:TB
    Dls_sims = np.zeros((nsims, 6, N_bin))

  
    if new_spectra:
        
        for j in range(nsims):
                filename_out = (
                    Cls_out_dir
                    + 'NamasterCls_lmax' + str(lmax)
                    + '_deltal' + str(deltal)
                    + '_sim' + str(j).zfill(4) +'_patch'+str(ipatch)+'_nsidepatch'+str(nsidepatch)+'.txt'
                )
                if os.path.exists(filename_out) and os.path.getsize(filename_out) > 0:
                    try:
                        data_loaded = np.loadtxt(filename_out, unpack=True)
                        
                        Dls_sims[j, 0, :] = data_loaded[1] # TT
                        Dls_sims[j, 1, :] = data_loaded[2] # EE
                        Dls_sims[j, 2, :] = data_loaded[3] # BB
                        Dls_sims[j, 3, :] = data_loaded[4] # TE
                        Dls_sims[j, 4, :] = data_loaded[5] # EB
                        Dls_sims[j, 5, :] = data_loaded[6] # TB
                         
                        print(f"[Rank {rank}] Patch {ipatch} Sim {j} skipped (found).", flush=True)
                        continue 
                    except:
                        pass
                
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
                
                dl_coupled_00 = nmt.compute_coupled_cell(f_0_a, f_0_b)
                dl_coupled_02 = nmt.compute_coupled_cell(f_0_a, f_2_b)
                dl_coupled_20 = nmt.compute_coupled_cell(f_2_a, f_0_b)
                dl_coupled_22 = nmt.compute_coupled_cell(f_2_a, f_2_b)
            
                dl_00 = w_00.decouple_cell(dl_coupled_00)
                dl_02 = w_02.decouple_cell(dl_coupled_02)
                dl_20 = w_20.decouple_cell(dl_coupled_20)
                dl_22 = w_22.decouple_cell(dl_coupled_22)

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
                Dls_sims[j, 0, :] = dl_00[0]                      # TT
                Dls_sims[j, 1, :] = dl_22[0]                      # EE
                Dls_sims[j, 2, :] = dl_22[3]                      # BB
                Dls_sims[j, 3, :] = 0.5 * (dl_02[0] + dl_20[0])   # TE-ET
                Dls_sims[j, 4, :] = 0.5 * (dl_22[1] + dl_22[2])   # EB-BE
                Dls_sims[j, 5, :] = 0.5 * (dl_02[1] + dl_20[1])   # TB-BT
                
                print(f"[Rank {rank}] Patch {ipatch} | Sto calcolando sim {j}/{nsims}...", flush=True)
        
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
            
    mean_Dls = np.mean(Dls_sims, axis=0)
    std_Dls = np.std(Dls_sims, axis=0) 
    err_Dls = std_Dls / np.sqrt(nsims)
    var_sim = np.var(Dls_sims,axis=0)

    cl_00_th = np.zeros((1, lmax + 1))
    cl_02_th = np.zeros((2, lmax + 1))
    cl_22_th = np.zeros((4, lmax + 1))

    cl_00_th[0, :] = Cls_th[0, :]
    cl_22_th[0, :] = Cls_th[1, :]
    cl_22_th[3, :] = Cls_th[2, :]
    cl_02_th[0, :] = Cls_th[3, :]
    
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


    delta_TT = (Dl_TT_sim - dl_TT_th) / err_TT
    delta_EE = (Dl_EE_sim - dl_EE_th) / err_EE
    delta_BB = (Dl_BB_sim - dl_BB_th) / err_BB
    delta_TE = (Dl_TE_sim - dl_TE_th) / err_TE
    delta_EB = (Dl_EB_sim ) / err_EB
    delta_TB = (Dl_TB_sim ) / err_TB
    
   

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




    plot_validation (ell_arr,nsideout,deltal,is_Dell, Dl_TT_sim, err_TT, dl_TT_th, delta_TT, 'TT',nsims, lmax,lmin, plots_dir,var_sim[0,:],var_master_TT,ipatch,nsidepatch)
    plot_validation (ell_arr,nsideout,deltal,is_Dell, Dl_EE_sim, err_EE, dl_EE_th, delta_EE, 'EE',nsims, lmax,lmin, plots_dir,var_sim[1,:],var_master_EE,ipatch,nsidepatch)
    plot_validation (ell_arr,nsideout,deltal,is_Dell, Dl_BB_sim, err_BB, dl_BB_th, delta_BB, 'BB',nsims, lmax,lmin, plots_dir,var_sim[2,:],var_master_BB,ipatch,nsidepatch)
    plot_validation (ell_arr,nsideout,deltal,is_Dell, Dl_TE_sim, err_TE, dl_TE_th, delta_TE, 'TE',nsims, lmax,lmin, plots_dir,var_sim[3,:],var_master_TE,ipatch,nsidepatch ,yscale='linear')
    plot_validation (ell_arr,nsideout,deltal,is_Dell, Dl_EB_sim, err_EB, np.zeros(dl_TT_th.shape[0]), delta_EB, 'EB',nsims, lmax,lmin, plots_dir,var_sim[4,:],var_master_EB,ipatch,nsidepatch,yscale='linear')
    plot_validation (ell_arr,nsideout,deltal,is_Dell, Dl_TB_sim, err_TB, np.zeros(dl_TT_th.shape[0]), delta_TB, 'TB',nsims, lmax,lmin, plots_dir,var_sim[5,:],var_master_TB,ipatch,nsidepatch,yscale='linear')
   
#----CALCOLO D-ESTIMATORS E ISTOGRAMMA BETA----
def D_estimators (Dls_sims,ell_arr,
        nsims,lmax,lmin,nsideout,deltal,Cls_out_dir,ipatch,nsidepatch,plots_dir,vardl,varname
        
    ):
    mean_Dest = 0.0
    betamin = np.deg2rad(-1.5)
    betamax = np.deg2rad(1.5)
    step = np.deg2rad(0.001)
    limit_l = 2 * nsideout
    ell_upper_bound = ell_arr + (deltal / 2.0)
    ell_lower_bound = ell_arr - (deltal / 2.0)
    valid_indices = (ell_upper_bound <= limit_l) & (ell_lower_bound >= lmin)
    
    
    var_cut = vardl[valid_indices]
    dl_EE_sim = Dls_sims[:, 1, valid_indices]
    dl_BB_sim = Dls_sims[:, 2, valid_indices]
    dl_EB_sim = Dls_sims[:, 4, valid_indices]
   
    

    betas_best_grid = np.ones(nsims)
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
        betas = np.arange(betamin, betamax + step,step)
        chi2_grid = np.zeros(len(betas))
       

        # for j in range(len(betas)):
        #     beta=betas[j]
            
        #     Dest_EB=(dl_EB*np.cos(4*beta))-0.5*(dl_EE-dl_BB)*np.sin(4*beta)
        #     x_l=Dest_EB-mean_Dest
        #     chi2_grid[j] = x_l @ inv_cov @ x_l

        # betas_best_grid[i]=betas[np.argmin(chi2_grid)]

        res = minimize(chi2_beta,x0=np.array([0.0]),args=(dl_EB, dl_EE,dl_BB, inv_cov),
                       method='L-BFGS-B',jac=jacobian, bounds=[(betamin, betamax)])   
        betas_best_func[i] = res.x[0]
        

    # sigma_beta_grid = np.rad2deg(np.std(betas_best_grid, ddof=1))
    sigma_beta_func = np.rad2deg(np.std(betas_best_func, ddof=1))
    # mean_beta_grid = np.rad2deg(np.mean(betas_best_grid))
    mean_beta_func = np.rad2deg(np.mean(betas_best_func))

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
    n_steps_int = int(round((betamax - betamin) / step))
    plt.savefig(plots_dir + 'Dest_EB_hist_nsims_'+varname
            + str(nsims) + '_steps' + str(n_steps_int)
            + '_lmax' + str(lmax) +'_index_' + str(ipatch).zfill(4) +'.png')
    plt.close()




def main(root_dir,new_spectra,new_patches,new_maps,new_workspace,is_Dell,nsideout,nsidepatch,nsims,lmin,deltal,sigma_noise,purify_e,purify_b,apotype):
    plots_dir   = root_dir + 'simulazione/plots/namaster_patch/nersc_nside512_nsidepatch8/'
    ws_dir      = root_dir + 'simulazione/ws/'
    cov_dir     = root_dir + 'simulazione/cov/'
    Cls_dir_in  = root_dir+'simulazione/'
    maps_dir    = root_dir+'simulazione/'+'maps/'
    Cls_out_dir = root_dir + 'simulazione/spectra_out/'
    mask_in_dir = root_dir + 'simulazione/mask/'
    mask_patch_dir  = root_dir + 'simulazione/mask/mask_patch/'

    npix_map= hp.nside2npix(nsideout)
    npix_patch= hp.nside2npix(nsidepatch)
    Spix = hp.nside2pixarea(nsideout, degrees=True)
    var = sigma_noise * sigma_noise / (Spix * 3600.)
    fwhm = 2.4 * np.sqrt(Spix)
    aposcale = fwhm
    

    if (rank==0):
        
        if new_patches:
            start1=perf_counter()
            create_mask_patch(mask_in_dir,mask_patch_dir,nsideout,nsidepatch,npix_patch,aposcale,apotype,plots_dir)
            end1=perf_counter()
            print("tempo creazione patches:   ",(end1-start1))
        npatches = np.loadtxt(mask_patch_dir+'active_patches_nside_'+str(nsidepatch)+
                              '.txt')
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
         npatches = None
         lmax     = None
         ell      = None
         ll       = None
         Cls_th   = None

    lmax     = comm.bcast(lmax, root=0)
    ell      = comm.bcast(ell, root=0)
    ll       = comm.bcast(ll, root=0)
    Cls_th   = comm.bcast(Cls_th, root=0)
    npatches = comm.bcast(npatches, root=0)

    simperproc = int(np.ceil(npatches/nproc))
    

    start3=perf_counter()    
    for ipatch in np.arange(rank*simperproc, (rank+1)*simperproc,1):
        
        if (ipatch<npatches):
            print(f"[Rank {rank}] >>> INIZIO Patch {ipatch}", flush=True)
       
            
            maskT_out=hp.read_map(mask_patch_dir+'patch_maskT_nside_'+str(nsidepatch)+'_index_'+str(ipatch).zfill(4)+'.fits', dtype=np.float64)
            maskP_out=hp.read_map(mask_patch_dir+'patch_maskP_nside_'+str(nsidepatch)+'_index_'+str(ipatch).zfill(4)+'.fits', dtype=np.float64)
            if new_workspace:
                create_workspace (lmax,deltal,is_Dell,fwhm,nsideout,maskT_out,maskP_out,purify_e,purify_b,ws_dir,maps_dir,sigma_noise,ipatch)

                calculate_cov_th(lmax, fwhm, nsideout, 
                        ws_dir,deltal,
                        is_Dell, Cls_th, cov_dir,sigma_noise,ipatch)
            

            Dls_sims,mean_Dls,std_Dls,err_Dls,var_sim,dl_TT_th,dl_EE_th,dl_BB_th,dl_TE_th,ell_arr=\
                    calculate_spectrum_pol(lmax, fwhm, nsideout, 
                    ws_dir, Cls_th, nsims,
                    deltal, maps_dir, sigma_noise,
                    purify_e, purify_b, maskT_out, 
                    maskP_out,Cls_out_dir,
                    is_Dell,Cls_dir_in,ipatch,nsidepatch,new_spectra
            )
        
            validation_spectra(
                    mean_Dls, std_Dls, err_Dls,
                    ell_arr, plots_dir, 
                    nsims, lmax,lmin, Cls_dir_in,var_sim,cov_dir,sigma_noise,ll,fwhm,nsideout,
                    dl_TT_th,dl_EE_th,dl_BB_th,dl_TE_th,ipatch,nsidepatch,deltal,is_Dell
            )
            
            D_estimators (Dls_sims,ell_arr,
                nsims,lmax,lmin,nsideout,deltal,Cls_out_dir,ipatch,nsidepatch,plots_dir,var_sim[4],'var_sim'
                )
            
            C_master_EB = np.load(cov_dir + 'cov_ana_EB'+'_patch'+str(ipatch).zfill(4) +'.npy')
            var_master_EB = np.diag(C_master_EB)

            D_estimators (Dls_sims,ell_arr,
                nsims,lmax,lmin,nsideout,deltal,Cls_out_dir,ipatch,nsidepatch,plots_dir,var_master_EB,'var_master'
                )
            
    end3=perf_counter()
    print("tempo elaborazione dati:   ",(end3-start3))
        
 

main(root_dir='/pscratch/sd/m/matteol/',  #per Nersc '/pscratch/sd/m/matteol/', per pc '/home/matteo/Scrivania/workspace/'
    new_spectra = True,
    new_patches = False,
    new_maps = False,
    new_workspace = True,
    is_Dell = True,
    nsideout = 512,
    nsidepatch = 8,
    nsims = 1000,
    lmin = 50,
    deltal = 25,
    sigma_noise = 2.0, #microK * arcmin
    purify_e = False,
    purify_b = True,
    apotype = "C2"
    )