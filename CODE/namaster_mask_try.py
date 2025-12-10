import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt

def create_workspace(lmax,fwhm,nside,Cls_dir,ll,var,npix,ws_dir,mask_dir,deltal):
    purify_e=False
    purify_b=False
    
    b = nmt.NmtBin.from_lmax_linear(lmax, deltal, is_Dell=False)
    ell_arr = b.get_effective_ells()
    bl = hp.gauss_beam(np.deg2rad(fwhm/60.0), lmax=lmax, pol=True)
    wpix = hp.pixwin(nside, pol=True, lmax=lmax)

    TT_th,EE_th,BB_th,TE_th=np.loadtxt(Cls_dir+'ffp10_lensedCls.dat',unpack=True,usecols=(1,2,3,4))
    Cls_th=np.zeros((4,lmax+1))
    Cls_th[0,2:]=TT_th[0:lmax-1]/ll[2:]
    Cls_th[1,2:]=EE_th[0:lmax-1]/ll[2:]
    Cls_th[2,2:]=BB_th[0:lmax-1]/ll[2:]
    Cls_th[3,2:]=TE_th[0:lmax-1]/ll[2:]

    noisesT = np.random.normal(loc=0,scale=np.sqrt(var),size=2*npix)
    noisesP = np.random.normal(loc=0,scale=np.sqrt(2*var),size=4*npix)
    m_sign=hp.synfast(cls=Cls_th ,lmax=lmax,nside=nside,pol=True,new=True,fwhm=np.deg2rad(fwhm/60.0), pixwin=True)
    ma_TQU=np.zeros_like(m_sign)
    mb_TQU=np.zeros_like(m_sign)

    ma_TQU[0]=m_sign[0]+noisesT[0:npix]
    ma_TQU[1]=m_sign[1]+noisesP[0:npix]
    ma_TQU[2]=m_sign[2]+noisesP[npix:2*npix]
    mb_TQU[0]=m_sign[0]+noisesT[npix:2*npix]
    mb_TQU[1]=m_sign[1]+noisesP[2*npix:3*npix]
    mb_TQU[2]=m_sign[2]+noisesP[3*npix:4*npix]
  
    maskT_in = hp.read_map(mask_dir+'COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits')
    maskP_in = hp.read_map(mask_dir+'COM_Mask_CMB-common-Mask-Pol_2048_R3.00.fits')
    maskT_out = nmt.mask_apodization(hp.ud_grade(maskT_in, nside),0.8,apotype="C1")
    maskP_out = nmt.mask_apodization(hp.ud_grade(maskP_in, nside),0.8,apotype="C1")  

    f_0_oe1 = nmt.NmtField(maskT_out,[ma_TQU[0]] ,beam=bl[:,0]*wpix[0],lmax=lmax)
    f_0_oe2 = nmt.NmtField(maskT_out,[mb_TQU[0]] ,beam=bl[:,0]*wpix[0],lmax=lmax)
    f_2_oe1 = nmt.NmtField(maskP_out,[ma_TQU[1], ma_TQU[2]],purify_e=purify_e,purify_b=purify_b,
                           beam=bl[:,1]*wpix[1],lmax=lmax)
    f_2_oe2 = nmt.NmtField(maskP_out,[mb_TQU[1], mb_TQU[2]],purify_e=purify_e,purify_b=purify_b,
                           beam=bl[:,1]*wpix[1],lmax=lmax)
    w_00 = nmt.NmtWorkspace()
    w_02 = nmt.NmtWorkspace()
    w_20 = nmt.NmtWorkspace()
    w_22 = nmt.NmtWorkspace()

    w_00.compute_coupling_matrix(f_0_oe1, f_0_oe2, b)
    w_02.compute_coupling_matrix(f_0_oe1, f_2_oe2, b)
    w_20.compute_coupling_matrix(f_2_oe1, f_0_oe2, b)
    w_22.compute_coupling_matrix(f_2_oe1, f_2_oe2, b)
    w_00.write_to(ws_dir+'w_00_deltaell_'+str(deltal).zfill(1)+'.fits')
    w_02.write_to(ws_dir+'w_02_deltaell_'+str(deltal).zfill(1)+'.fits')
    w_20.write_to(ws_dir+'w_20_deltaell_'+str(deltal).zfill(1)+'.fits')
    w_22.write_to(ws_dir+'w_22_deltaell_'+str(deltal).zfill(1)+'.fits')

def calculate_spectrum_pol(lmax,fwhm,nside,ws_dir,Cls_dir,ll,var,npix,nsims,mask_dir,deltal):
    
    purify_e=False
    purify_b=False
    b = nmt.NmtBin.from_lmax_linear(lmax, deltal, is_Dell=False)
    ell_arr = b.get_effective_ells()
    bl = hp.gauss_beam(np.deg2rad(fwhm/60.0), lmax=lmax, pol=True)
    wpix = hp.pixwin(nside, pol=True, lmax=lmax)

    w_00 = nmt.NmtWorkspace()
    w_02 = nmt.NmtWorkspace()
    w_20 = nmt.NmtWorkspace()
    w_22 = nmt.NmtWorkspace()
    w_00.read_from(ws_dir+'w_00_deltaell_'+str(deltal).zfill(1)+'.fits')
    w_02.read_from(ws_dir+'w_02_deltaell_'+str(deltal).zfill(1)+'.fits')
    w_20.read_from(ws_dir+'w_20_deltaell_'+str(deltal).zfill(1)+'.fits')
    w_22.read_from(ws_dir+'w_22_deltaell_'+str(deltal).zfill(1)+'.fits')

    TT_th,EE_th,BB_th,TE_th=np.loadtxt(Cls_dir+'ffp10_lensedCls.dat',unpack=True,usecols=(1,2,3,4))
    Cls_th=np.zeros((4,lmax+1))
    Cls_th[0,2:]=TT_th[0:lmax-1]/ll[2:]
    Cls_th[1,2:]=EE_th[0:lmax-1]/ll[2:]
    Cls_th[2,2:]=BB_th[0:lmax-1]/ll[2:]
    Cls_th[3,2:]=TE_th[0:lmax-1]/ll[2:]

  
    
    maskT_in = hp.read_map(mask_dir+'COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits')
    maskP_in = hp.read_map(mask_dir+'COM_Mask_CMB-common-Mask-Pol_2048_R3.00.fits')
    
    maskT_out = nmt.mask_apodization(hp.ud_grade(maskT_in, nside),0.8,apotype="C1")
    maskP_out = nmt.mask_apodization(hp.ud_grade(maskP_in, nside),0.8,apotype="C1")
    
    N_bin = b.get_n_bands()
    
    Cls_sims = np.zeros((6, N_bin, nsims))  # 0:TT, 1:EE, 2:BB, 3:TE, 4:TB, 5:EB

    for i in range (nsims):
        noisesT = np.random.normal(loc=0,scale=np.sqrt(var),size=2*npix)
        noisesP = np.random.normal(loc=0,scale=np.sqrt(2*var),size=4*npix)

        m_sign=hp.synfast(cls=Cls_th ,lmax=lmax,nside=nside,pol=True,new=True,fwhm=np.deg2rad(fwhm/60.0), 
                          pixwin=True)
        ma_TQU=np.zeros_like(m_sign)
        mb_TQU=np.zeros_like(m_sign)
        ma_TQU[0]=m_sign[0]+noisesT[0:npix]
        ma_TQU[1]=m_sign[1]+noisesP[0:npix]
        ma_TQU[2]=m_sign[2]+noisesP[npix:2*npix]
        mb_TQU[0]=m_sign[0]+noisesT[npix:2*npix]
        mb_TQU[1]=m_sign[1]+noisesP[2*npix:3*npix]
        mb_TQU[2]=m_sign[2]+noisesP[3*npix:4*npix]

        f_0_oe1 = nmt.NmtField(maskT_out,[ma_TQU[0]] ,beam=bl[:,0]*wpix[0],lmax=lmax)
        f_0_oe2 = nmt.NmtField(maskT_out,[mb_TQU[0]] ,beam=bl[:,0]*wpix[0],lmax=lmax)
        f_2_oe1 = nmt.NmtField(maskP_out,[ma_TQU[1], ma_TQU[2]],purify_e=purify_e,purify_b=purify_b,
                               beam=bl[:,1]*wpix[1],lmax=lmax)
        f_2_oe2 = nmt.NmtField(maskP_out,[mb_TQU[1], mb_TQU[2]],purify_e=purify_e,purify_b=purify_b,
                               beam=bl[:,1]*wpix[1],lmax=lmax)

        cl_coupled_00 = nmt.compute_coupled_cell(f_0_oe1, f_0_oe2)
        cl_coupled_02 = nmt.compute_coupled_cell(f_0_oe1, f_2_oe2)
        cl_coupled_20 = nmt.compute_coupled_cell(f_2_oe1, f_0_oe2)        
        cl_coupled_22 = nmt.compute_coupled_cell(f_2_oe1, f_2_oe2) 
        cl_00 = w_00.decouple_cell(cl_coupled_00)
        cl_02 = w_02.decouple_cell(cl_coupled_02)
        cl_20 = w_20.decouple_cell(cl_coupled_20)
        cl_22 = w_22.decouple_cell(cl_coupled_22)

        Cls_sims[0,:,i] = cl_00[0]  # TT
        Cls_sims[1,:,i] = cl_22[0]  # EE
        Cls_sims[2,:,i] = cl_22[3]  # BB
        Cls_sims[3,:,i] = 0.5*(cl_02[0] + cl_20[0])  # TE-ET
        Cls_sims[4,:,i] = 0.5*(cl_02[1] + cl_20[1])  # TB-BT
        Cls_sims[5,:,i] = 0.5*(cl_22[1] + cl_22[2])  # EB-BE
        print(i+1)

    mean_Cls=np.mean(Cls_sims, axis=2)

    cl_00_th = np.zeros((1,lmax+1))
    cl_02_th = np.zeros((2,lmax+1))
    cl_22_th = np.zeros((4,lmax+1))
    cl_00_th[0,:] = Cls_th[0,:]
    cl_02_th[0,:] = Cls_th[3,:]
    cl_22_th[0,:] = Cls_th[1,:]
    cl_22_th[3,:] = Cls_th[2,:]

    cl_00_th_dec = w_00.decouple_cell(w_00.couple_cell(cl_00_th))
    cl_02_th_dec = w_02.decouple_cell(w_02.couple_cell(cl_02_th))
    cl_22_th_dec = w_22.decouple_cell(w_22.couple_cell(cl_22_th))

    cl_TT_th = cl_00_th_dec[0]
    cl_TT_master = mean_Cls[0]
    cl_EE_th = cl_22_th_dec[0]
    cl_EE_master = mean_Cls[1]
    cl_BB_th = cl_22_th_dec[3]
    cl_BB_master = mean_Cls[2]
    cl_TE_th = cl_02_th_dec[0]
    cl_TE_master = mean_Cls[3]

    plt.figure(figsize=(10,8))
    plt.subplot(2,2,1)
    plt.loglog(ell_arr, cl_TT_th, label='TT_th')
    plt.loglog(ell_arr, cl_TT_master, label='TT_MASTER')
    plt.legend()
    plt.subplot(2,2,2)
    plt.loglog(ell_arr, cl_EE_th, label='EE_th')
    plt.loglog(ell_arr, cl_EE_master, label='EE_MASTER')
    plt.legend()
    plt.subplot(2,2,3)
    plt.loglog(ell_arr, cl_BB_th, label='BB_th')
    plt.loglog(ell_arr, cl_BB_master, label='BB_MASTER')
    plt.legend()
    plt.subplot(2,2,4)
    plt.loglog(ell_arr, cl_TE_th, label='TE_th')
    plt.loglog(ell_arr, cl_TE_master, label='TE_MASTER')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    root_dir='/home/matteo/Scrivania/workspace/'
    Cls_dir=root_dir+'simulazione/'
    ws_dir=root_dir+'simulazione/'+'ws/'
    mask_dir=root_dir+'simulazione/'+'mask/'
    maps_dir=root_dir+'simulazione/'+'maps/'
    nside=64
    nsims=100
    deltal=5
    lmax=2*nside
    sigma=2.0
    Spix=hp.nside2pixarea(nside, degrees=True)
    var=sigma*sigma/(Spix*3600.) 
    npix=hp.nside2npix(nside)
    fwhm=2*np.sqrt(Spix)
    ell=np.arange(0,lmax+1) 
    ll=ell*(ell+1)/2/np.pi
    create_workspace(lmax,fwhm,nside,Cls_dir,ll,var,npix,ws_dir,mask_dir,deltal)
    calculate_spectrum_pol(lmax,fwhm,nside,ws_dir,Cls_dir,ll,var,npix,nsims,mask_dir,deltal)

main()
