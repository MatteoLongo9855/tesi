import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt
from scipy.optimize import minimize

#----LETTURA SPETTRI----
def read_Cls(Cls_dir, lmax, ll, lmin):
    TT_th, EE_th, BB_th, TE_th = np.loadtxt(
        Cls_dir + 'ffp10_lensedCls.dat',
        unpack=True,
        usecols=(1, 2, 3, 4)
    )

    Cls_th = np.zeros((4, lmax + 1))
    #il file input contiene i Dl anzichè i Cl
    Cls_th[0, lmin:] = TT_th[0:lmax - 1] / ll[lmin:]
    Cls_th[1, lmin:] = EE_th[0:lmax - 1] / ll[lmin:]
    Cls_th[2, lmin:] = BB_th[0:lmax - 1] / ll[lmin:]
    Cls_th[3, lmin:] = TE_th[0:lmax - 1] / ll[lmin:]

    return Cls_th

#----LETTURA, RESCALE E APODIZZAZIONE MASCHERE----
def read_mask(mask_dir, real_mask, nside, npix, aposize):
    if real_mask:
        maskT_in = hp.read_map(
            mask_dir + 'COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits'
        )
        maskP_in = hp.read_map(
            mask_dir + 'COM_Mask_CMB-common-Mask-Pol_2048_R3.00.fits'
        )

        maskT_out = nmt.mask_apodization(
            hp.ud_grade(maskT_in, nside),
            aposize,
            apotype="C1"
        )
        maskP_out = nmt.mask_apodization(
            hp.ud_grade(maskP_in, nside),
            aposize,
            apotype="C1"
        )
    else:
        maskT_out = np.ones(npix)
        maskP_out = np.ones(npix)

    return maskT_out, maskP_out

#----CREAZIONE MAPPE SYNFAST+WHITENOISE----
def create_maps(
        Cls_th, lmax, nsims, 
        nside, npix, var, 
        fwhm, sigma_noise, 
        maps_dir
    ):
    for i in range(nsims):
        noisesT = np.random.normal(loc=0, scale=np.sqrt(var), size=2 * npix)
        noisesP = np.random.normal(loc=0, scale=np.sqrt(2 * var), size=4 * npix)

        m_sign = hp.synfast(
            cls=Cls_th,
            lmax=lmax,
            nside=nside,
            pol=True,
            new=True,
            fwhm=np.deg2rad(fwhm / 60.0),
            pixwin=True
        )

        ma_TQU = np.zeros_like(m_sign)
        mb_TQU = np.zeros_like(m_sign)

        ma_TQU[0] = m_sign[0] + noisesT[0:npix]
        ma_TQU[1] = m_sign[1] + noisesP[0:npix]
        ma_TQU[2] = m_sign[2] + noisesP[npix:2 * npix]

        mb_TQU[0] = m_sign[0] + noisesT[npix:2 * npix]
        mb_TQU[1] = m_sign[1] + noisesP[2 * npix:3 * npix]
        mb_TQU[2] = m_sign[2] + noisesP[3 * npix:4 * npix]

        filenameA = (
            maps_dir
            + 'mapA_nside' + str(nside)
            + '_whitenoise' + str(sigma_noise) + 'microK*arcmin'
            + '_sim' + str(i).zfill(4) + '.fits'
        )

        filenameB = (
            maps_dir
            + 'mapB_nside' + str(nside)
            + '_whitenoise' + str(sigma_noise) + 'microK*arcmin'
            + '_sim' + str(i).zfill(4) + '.fits'
        )

        hp.write_map(filenameA, ma_TQU, overwrite=True)
        hp.write_map(filenameB, mb_TQU, overwrite=True)

#----CREAZIONE WORKSPACE NAMASTER----
def create_workspace(
        lmax, fwhm, nside, 
        ws_dir,deltal, maps_dir, 
        sigma_noise,purify_e, purify_b, 
        maskT_out,maskP_out, is_Dell
    ):

    b = nmt.NmtBin.from_lmax_linear(lmax, deltal, is_Dell=is_Dell)

    bl = hp.gauss_beam(
        np.deg2rad(fwhm / 60.0),
        lmax=lmax,
        pol=True
    )
    wpix = hp.pixwin(nside, pol=True, lmax=lmax)

    filenameA = (
        maps_dir
        + 'mapA_nside' + str(nside)
        + '_whitenoise' + str(sigma_noise) + 'microK*arcmin'
        + '_sim' + str(0).zfill(4) + '.fits'
    )

    filenameB = (
        maps_dir
        + 'mapB_nside' + str(nside)
        + '_whitenoise' + str(sigma_noise) + 'microK*arcmin'
        + '_sim' + str(0).zfill(4) + '.fits'
    )

    ma_TQU = hp.read_map(filenameA, [0, 1, 2])
    mb_TQU = hp.read_map(filenameB, [0, 1, 2])

    f_0_oe1 = nmt.NmtField(
        maskT_out,
        [ma_TQU[0]],
        beam=bl[:, 0] * wpix[0],
        lmax=lmax
    )
    f_0_oe2 = nmt.NmtField(
        maskT_out,
        [mb_TQU[0]],
        beam=bl[:, 0] * wpix[0],
        lmax=lmax
    )
    f_2_oe1 = nmt.NmtField(
        maskP_out,
        [ma_TQU[1], ma_TQU[2]],
        purify_e=purify_e,
        purify_b=purify_b,
        beam=bl[:, 1] * wpix[1],
        lmax=lmax
    )
    f_2_oe2 = nmt.NmtField(
        maskP_out,
        [mb_TQU[1], mb_TQU[2]],
        purify_e=purify_e,
        purify_b=purify_b,
        beam=bl[:, 1] * wpix[1],
        lmax=lmax
    )

    w_00 = nmt.NmtWorkspace()
    w_02 = nmt.NmtWorkspace()
    w_20 = nmt.NmtWorkspace()
    w_22 = nmt.NmtWorkspace()

    w_00.compute_coupling_matrix(f_0_oe1, f_0_oe2, b)
    w_02.compute_coupling_matrix(f_0_oe1, f_2_oe2, b)
    w_20.compute_coupling_matrix(f_2_oe1, f_0_oe2, b)
    w_22.compute_coupling_matrix(f_2_oe1, f_2_oe2, b)

    w_00.write_to(ws_dir + 'w_00_deltaell_' + str(deltal).zfill(1) + '.fits')
    w_02.write_to(ws_dir + 'w_02_deltaell_' + str(deltal).zfill(1) + '.fits')
    w_20.write_to(ws_dir + 'w_20_deltaell_' + str(deltal).zfill(1) + '.fits')
    w_22.write_to(ws_dir + 'w_22_deltaell_' + str(deltal).zfill(1) + '.fits')

#----CALCOLO SPETTRI,MEDIA E DEVIAZIONE----
def calculate_spectrum_pol(
        lmax, fwhm, nside, 
        ws_dir, Cls_th, nsims,
        deltal, maps_dir, sigma_noise,
        purify_e, purify_b, maskT_out, 
        maskP_out,Cls_out_dir, new_spectra, 
        is_Dell
    ):

    b = nmt.NmtBin.from_lmax_linear(lmax, deltal, is_Dell=is_Dell)
    ell_arr = b.get_effective_ells()

    bl = hp.gauss_beam(
        np.deg2rad(fwhm / 60.0),
        lmax=lmax,
        pol=True
    )
    wpix = hp.pixwin(nside, pol=True, lmax=lmax)

    w_00 = nmt.NmtWorkspace()
    w_02 = nmt.NmtWorkspace()
    w_20 = nmt.NmtWorkspace()
    w_22 = nmt.NmtWorkspace()

    w_00.read_from(ws_dir + 'w_00_deltaell_' + str(deltal).zfill(1) + '.fits')
    w_02.read_from(ws_dir + 'w_02_deltaell_' + str(deltal).zfill(1) + '.fits')
    w_20.read_from(ws_dir + 'w_20_deltaell_' + str(deltal).zfill(1) + '.fits')
    w_22.read_from(ws_dir + 'w_22_deltaell_' + str(deltal).zfill(1) + '.fits')

    N_bin = b.get_n_bands()

    # 0:TT, 1:EE, 2:BB, 3:TE, 4:TB, 5:EB
    Cls_sims = np.zeros((6, N_bin, nsims))

    if new_spectra:
        for i in range(nsims):
            filenameA = (
                maps_dir
                + 'mapA_nside' + str(nside)
                + '_whitenoise' + str(sigma_noise) + 'microK*arcmin'
                + '_sim' + str(i).zfill(4) + '.fits'
            )

            filenameB = (
                maps_dir
                + 'mapB_nside' + str(nside)
                + '_whitenoise' + str(sigma_noise) + 'microK*arcmin'
                + '_sim' + str(i).zfill(4) + '.fits'
            )

            ma_TQU = hp.read_map(filenameA, [0, 1, 2])
            mb_TQU = hp.read_map(filenameB, [0, 1, 2])

            f_0_oe1 = nmt.NmtField(
                maskT_out,
                [ma_TQU[0]],
                beam=bl[:, 0] * wpix[0],
                lmax=lmax
            )
            f_0_oe2 = nmt.NmtField(
                maskT_out,
                [mb_TQU[0]],
                beam=bl[:, 0] * wpix[0],
                lmax=lmax
            )
            f_2_oe1 = nmt.NmtField(
                maskP_out,
                [ma_TQU[1], ma_TQU[2]],
                purify_e=purify_e,
                purify_b=purify_b,
                beam=bl[:, 1] * wpix[1],
                lmax=lmax
            )
            f_2_oe2 = nmt.NmtField(
                maskP_out,
                [mb_TQU[1], mb_TQU[2]],
                purify_e=purify_e,
                purify_b=purify_b,
                beam=bl[:, 1] * wpix[1],
                lmax=lmax
            )

            cl_coupled_00 = nmt.compute_coupled_cell(f_0_oe1, f_0_oe2)
            cl_coupled_02 = nmt.compute_coupled_cell(f_0_oe1, f_2_oe2)
            cl_coupled_20 = nmt.compute_coupled_cell(f_2_oe1, f_0_oe2)
            cl_coupled_22 = nmt.compute_coupled_cell(f_2_oe1, f_2_oe2)

            cl_00 = w_00.decouple_cell(cl_coupled_00)
            cl_02 = w_02.decouple_cell(cl_coupled_02)
            cl_20 = w_20.decouple_cell(cl_coupled_20)
            cl_22 = w_22.decouple_cell(cl_coupled_22)

            Cls_sims[0, :, i] = cl_00[0]                      # TT
            Cls_sims[1, :, i] = cl_22[0]                      # EE
            Cls_sims[2, :, i] = cl_22[3]                      # BB
            Cls_sims[3, :, i] = 0.5 * (cl_02[0] + cl_20[0])   # TE-ET
            Cls_sims[4, :, i] = 0.5 * (cl_02[1] + cl_20[1])   # TB-BT
            Cls_sims[5, :, i] = 0.5 * (cl_22[1] + cl_22[2])   # EB-BE

            np.savetxt(
                Cls_out_dir
                + 'NamasterCls_lmax' + str(lmax)
                + '_deltal' + str(deltal)
                + '_sim' + str(i).zfill(4) + '.txt',
                np.column_stack([
                    ell_arr,
                    cl_00[0],
                    cl_22[0],
                    cl_22[3],
                    (cl_02[0] + cl_20[0]) / 2,
                    (cl_02[1] + cl_20[1]) / 2,
                    (cl_22[1] + cl_22[2]) / 2
                ]),
                header='#l TT EE BB TE TB EB'
            )

            print(i + 1)

    else:
        for i in range(nsims):
            ell_arr, cl_TT, cl_EE, cl_BB, cl_TE, cl_TB, cl_EB = np.loadtxt(
                Cls_out_dir
                + 'NamasterCls_lmax' + str(lmax)
                + '_deltal' + str(deltal)
                + '_sim' + str(i).zfill(4) + '.txt',
                unpack=True
            )

            Cls_sims[0, :, i] = cl_TT  # TT
            Cls_sims[1, :, i] = cl_EE  # EE
            Cls_sims[2, :, i] = cl_BB  # BB
            Cls_sims[3, :, i] = cl_TE  # TE
            Cls_sims[4, :, i] = cl_TB  # TB
            Cls_sims[5, :, i] = cl_EB  # EB

    mean_Cls = np.mean(Cls_sims, axis=2)
    std_Cls = np.std(Cls_sims, axis=2, ddof=1)
    err_Cls = std_Cls / np.sqrt(nsims)

    cl_00_th = np.zeros((1, lmax + 1))
    cl_02_th = np.zeros((2, lmax + 1))
    cl_22_th = np.zeros((4, lmax + 1))

    cl_00_th[0, :] = Cls_th[0, :]
    cl_02_th[0, :] = Cls_th[3, :]
    cl_22_th[0, :] = Cls_th[1, :]
    cl_22_th[3, :] = Cls_th[2, :]

    cl_00_th_dec = w_00.decouple_cell(w_00.couple_cell(cl_00_th))
    cl_02_th_dec = w_02.decouple_cell(w_02.couple_cell(cl_02_th))
    cl_22_th_dec = w_22.decouple_cell(w_22.couple_cell(cl_22_th))

    cl_TT_th = cl_00_th_dec[0]
    cl_EE_th = cl_22_th_dec[0]
    cl_BB_th = cl_22_th_dec[3]
    cl_TE_th = cl_02_th_dec[0]

    return (
        mean_Cls,
        std_Cls,
        err_Cls,
        cl_TT_th,
        cl_EE_th,
        cl_BB_th,
        cl_TE_th,
        ell_arr
    )
#----MATRICI PER BINNING----
def PQ_matrix(lmin,lmax,deltal):

    nbin = int(np.floor((lmax-lmin+1)/deltal))
    P  = np.zeros((nbin,lmax-lmin+1))
    Qt = np.zeros((nbin,lmax-lmin+1))
    j = 0
    k = 0
    for i in range(nbin):
        j = k
        k += deltal
        P[i,j:k]  = 1./deltal
        Qt[i,j:k] = 1.
    Q  = Qt.T
    Pt = P.T
    return (P, Pt, Q, Qt)
#----GRAFICI SPETTRI----
def validation_plot(
        mean_Cls, std_Cls, err_Cls,
        cl_TT_th, cl_EE_th, cl_BB_th, 
        cl_TE_th, ell_arr, plots_dir, 
        nsims, lmax, real_mask, 
        sigma_noise):

    cl_TT_master = mean_Cls[0]
    cl_EE_master = mean_Cls[1]
    cl_BB_master = mean_Cls[2]
    cl_TE_master = mean_Cls[3]
    cl_TB_master = mean_Cls[4]
    cl_EB_master = mean_Cls[5]

    err_TT = err_Cls[0]
    err_EE = err_Cls[1]
    err_BB = err_Cls[2]
    err_TE = err_Cls[3]

    delta_TT = (cl_TT_master - cl_TT_th) / err_TT
    delta_EE = (cl_EE_master - cl_EE_th) / err_EE
    delta_BB = (cl_BB_master - cl_BB_th) / err_BB
    delta_TE = (cl_TE_master - cl_TE_th) / err_TE

    # TT
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(7, 6),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True
    )

    ax1.errorbar(
        ell_arr,
        cl_TT_master,
        yerr=err_TT,
        fmt='o',
        markersize=3,
        label=r'$\langle C_\ell^{TT}\rangle \pm \sigma_{\bar C_\ell}$'
    )
    ax1.plot(ell_arr, cl_TT_th, label=r'$C_{\ell,\mathrm{th}}^{TT}$')
    ax1.set_xscale('linear')
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$C_\ell^{TT}$')
    ax1.legend(fontsize=8)
    ax1.grid(True, ls='--', alpha=0.4)

    ax2.axhline(0, color='black', lw=0.8)
    ax2.plot(ell_arr, delta_TT, 'o', markersize=2)
    ax2.axhline(3, color='red', ls='--')
    ax2.axhline(-3, color='red', ls='--')
    ax2.set_xscale('linear')
    ax2.set_ylim(-5, 5)
    ax2.set_xlabel(r'$\ell$')
    ax2.set_ylabel(r'$\Delta/\sigma_noise$')
    ax2.grid(True, ls='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(plots_dir+'cl_TT_nsims' + str(nsims)+ '_lmax' + str(lmax)+ '_mask' + str(real_mask) + 
                '_whitenoise' + str(sigma_noise) + 'microK*arcmin'+ '.png')

    plt.show()
   
    # EE
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(7, 6),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True
    )

    ax1.errorbar(
        ell_arr,
        cl_EE_master,
        yerr=err_EE,
        fmt='o',
        markersize=3,
        label=r'$\langle C_\ell^{EE}\rangle \pm \sigma_{\bar C_\ell}$'
    )
    ax1.plot(ell_arr, cl_EE_th, label=r'$C_{\ell,\mathrm{th}}^{EE}$')
    ax1.set_xscale('linear')
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$C_\ell^{EE}$')
    ax1.legend(fontsize=8)
    ax1.grid(True, ls='--', alpha=0.4)

    ax2.axhline(0, color='black', lw=0.8)
    ax2.plot(ell_arr, delta_EE, 'o', markersize=2)
    ax2.axhline(3, color='red', ls='--')
    ax2.axhline(-3, color='red', ls='--')
    ax2.set_xscale('linear')
    ax2.set_ylim(-5, 5)
    ax2.set_xlabel(r'$\ell$')
    ax2.set_ylabel(r'$\Delta/\sigma_noise$')
    ax2.grid(True, ls='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(plots_dir+'cl_EE_nsims' + str(nsims)+ '_lmax' + str(lmax)+ '_mask' + str(real_mask) + 
                '_whitenoise' + str(sigma_noise) + 'microK*arcmin'+ '.png')
    plt.show()
    

    # BB
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(7, 6),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True
    )

    ax1.errorbar(
        ell_arr,
        cl_BB_master,
        yerr=err_BB,
        fmt='o',
        markersize=3,
        label=r'$\langle C_\ell^{BB}\rangle \pm \sigma_{\bar C_\ell}$'
    )
    ax1.plot(ell_arr, cl_BB_th, label=r'$C_{\ell,\mathrm{th}}^{BB}$')
    ax1.set_xscale('linear')
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$C_\ell^{BB}$')
    ax1.legend(fontsize=8)
    ax1.grid(True, ls='--', alpha=0.4)

    ax2.axhline(0, color='black', lw=0.8)
    ax2.plot(ell_arr, delta_BB, 'o', markersize=2)
    ax2.axhline(3, color='red', ls='--')
    ax2.axhline(-3, color='red', ls='--')
    ax2.set_xscale('linear')
    ax2.set_ylim(-5, 5)
    ax2.set_xlabel(r'$\ell$')
    ax2.set_ylabel(r'$\Delta/\sigma_noise$')
    ax2.grid(True, ls='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(plots_dir+'cl_BB_nsims' + str(nsims)+ '_lmax' + str(lmax)+ '_mask' + str(real_mask) + 
                '_whitenoise' + str(sigma_noise) + 'microK*arcmin'+ '.png')
    plt.show()

    # TE
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(7, 6),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True
    )

    ax1.errorbar(
        ell_arr,
        cl_TE_master,
        yerr=err_TE,
        fmt='o',
        markersize=3,
        label=r'$\langle C_\ell^{TE}\rangle \pm \sigma_{\bar C_\ell}$'
    )
    ax1.plot(ell_arr, cl_TE_th, label=r'$C_{\ell,\mathrm{th}}^{TE}$')
    ax1.set_xscale('linear')
    ax1.set_yscale('linear')
    ax1.set_ylabel(r'$C_\ell^{TE}$')
    ax1.legend(fontsize=8)
    ax1.grid(True, ls='--', alpha=0.4)

    ax2.axhline(0, color='black', lw=0.8)
    ax2.plot(ell_arr, delta_TE, 'o', markersize=2)
    ax2.axhline(3, color='red', ls='--')
    ax2.axhline(-3, color='red', ls='--')
    ax2.set_xscale('log')
    ax2.set_ylim(-5, 5)
    ax2.set_xlabel(r'$\ell$')
    ax2.set_ylabel(r'$\Delta/\sigma_noise$')
    ax2.grid(True, ls='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(plots_dir+'cl_TE_nsims' + str(nsims)+ '_lmax' + str(lmax)+ '_mask' + str(real_mask) + 
                '_whitenoise' + str(sigma_noise) + 'microK*arcmin'+ '.png')

    plt.show()
    
#----CALCOLO D-ESTIMATORS E ISTOGRAMMA BETA----
def D_estimators (
        nsims,Cls_out_dir,lmax,
        lmin, deltal,std_Cls,
        mean_Cls, plots_dir, P,
        Pt, sigma_noise, var, 
        Cls_th, Cls_dir
    ):
    
    mean_Dest = 0.0
    betamin = -0.5
    betamax = 0.5
    step = 0.001
    ell = np.arange(lmin, lmax + 1)
    betas_best_grid = np.ones(nsims)
    betas_best_func = np.ones(nsims)
    Cl_BB_sim = np.zeros((nsims,mean_Cls.shape[1]))
    Cl_EE_sim = np.zeros((nsims,mean_Cls.shape[1]))
    Cl_EB_sim = np.zeros((nsims,mean_Cls.shape[1]))
    Cl_noiseT = np.full(lmax-lmin+1,(sigma_noise*np.deg2rad(1/60))**2)*(ell*(ell+1))/(2*np.pi)
    Cl_noiseP = np.full(lmax-lmin+1,(2*sigma_noise*np.deg2rad(1/60))**2)*(ell*(ell+1))/(2*np.pi)
    #Cl_noiseT = np.full(lmax-lmin+1,var)
    #Cl_noiseP = np.full(lmax-lmin+1,4 * var)

    for i in range(nsims):
        ell_arr,cl_TTi,cl_EEi,cl_BBi,cl_TEi,cl_TBi,cl_EBi=\
            np.loadtxt(
                Cls_out_dir+'NamasterCls_lmax'+str(lmax)+
                '_deltal'+str(deltal)+'_sim'+str(i).zfill(4) +
                '.txt', unpack=True
        )
        TT_th, EE_th, BB_th, TE_th = np.loadtxt(
        Cls_dir + 'ffp10_lensedCls.dat',
        unpack=True,
        usecols=(1, 2, 3, 4)
        )
        #Cl_TE_sim[i, :]= cl_TEi
        #Cl_TB_sim[i, :]= cl_TBi
        Cl_EB_sim[i, :]= cl_EBi
        Cl_EE_sim[i, :]= cl_EEi
        Cl_BB_sim[i, :]= cl_BBi
    
    var_EB= np.var(Cl_EB_sim, axis=0)
    var_Dest_th = (EE_th[:lmax-1] + Cl_noiseP) * (BB_th[:lmax-1] + Cl_noiseP) * (1/(2*ell + 1))
    var_Dest_sim  = var_EB

    matriceDest=np.diag(var_Dest_th)
    matricebin= P @ matriceDest @ Pt
    
    bin_cov= np.diag(1/np.diag(matricebin))
    ####################
    #funzioni utili solo per metodo con minimize
    def chi2_beta(beta, cl_EB, cl_EE,cl_BB, bin_cov):
            
            b = beta[0]
            #Dest_TB = cl_TB * np.cos(2 * b) - cl_TE * np.sin(2 * b)
            Dest_EB=(cl_EB*np.cos(4*b))-0.5*(cl_EE-cl_BB)*np.sin(4*b)
            x_l = Dest_EB - mean_Dest
            return x_l @ bin_cov @ x_l
    
    def jacobian (beta, cl_EB, cl_EE,cl_BB,bin_cov):

        grad = np.zeros(beta.size)
        for i in range (grad.size):
            betap=np.copy(beta)
            betam=np.copy(beta)
            betap[i]+=1e-10
            betam[i]-=1e-10
            grad[i]=(chi2_beta(betap, cl_EB, cl_EE,cl_BB, bin_cov)-
                     chi2_beta(betam, cl_EB, cl_EE,cl_BB, bin_cov))/2e-10
        return grad
    ######################

    for i in range(nsims):
        
        cl_EE = Cl_EE_sim[i]
        cl_BB = Cl_BB_sim[i]
        cl_EB = Cl_EB_sim[i]
        betas = np.arange(betamin, betamax + step,step)
        chi2_grid = np.zeros(len(betas))
       

        for j in range(len(betas)):
            beta=betas[j]
            #Dest_TB=cl_TB*np.cos(2*beta)-cl_TE*np.sin(2*beta)
            Dest_EB=(cl_EB*np.cos(4*beta))-0.5*(cl_EE-cl_BB)*np.sin(4*beta)
            x_l=Dest_EB-mean_Dest
            
            #var_Dest_sim  = (var_TB * (np.cos(2*beta))**2) + (var_TE * (np.sin(2*beta))**2) - 2 * np.cos(2*beta) * np.sin(2*beta) * std_TB * std_TE
            
            #Prec = np.diag(1.0 / var_Dest_sim)
            
            chi2_grid[j] = x_l @ bin_cov @ x_l

        betas_best_grid[i]=betas[np.argmin(chi2_grid)]

        res = minimize(chi2_beta,x0=np.array([0.0]),args=(cl_EB, cl_EE,cl_BB, bin_cov),
                       method='L-BFGS-B',jac=jacobian, bounds=[(-0.5, 0.5)])   
        betas_best_func[i] = res.x[0]
        print("beta_",i)

    sigma_beta_grid = np.std(betas_best_grid, ddof=1)
    sigma_beta_func = np.std(betas_best_func, ddof=1)
    mean_beta_grid = np.mean(betas_best_grid)
    mean_beta_func = np.mean(betas_best_func)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    nbins = 8
    ax1.hist(betas_best_grid, bins=nbins, edgecolor='black')
    ax1.set_xlabel(r'$\beta_{\rm best}^{\rm (grid)}$')
    ax1.set_ylabel('Counts')
    ax1.set_title(r'Distribuzione di $\beta_{\rm best}$ (grid)')
    ax1.axvline(mean_beta_grid, color='red', linestyle='--')
    ax1.axvline(mean_beta_grid + sigma_beta_grid, color='red', linestyle=':')
    ax1.axvline(mean_beta_grid - sigma_beta_grid, color='red', linestyle=':')
    ax1.text(0.02, 0.95, rf'$\bar\beta = {mean_beta_grid:.2e}$'
         '\n' + rf'$\ = {sigma_beta_grid:.2e}$',
         transform=ax1.transAxes, va='top')

    ax2.hist(betas_best_func, bins=nbins, edgecolor='black')
    ax2.set_xlabel(r'$\beta_{\rm best}^{\rm (minimize)}$')
    ax2.set_title(r'Distribuzione di $\beta_{\rm best}$ (minimize)')
    ax2.axvline(mean_beta_func, color='red', linestyle='--')
    ax2.axvline(mean_beta_func + sigma_beta_func, color='red', linestyle=':')
    ax2.axvline(mean_beta_func - sigma_beta_func, color='red', linestyle=':')
    ax2.text(0.02, 0.95, rf'$\bar\beta = {mean_beta_func:.2e}$'
         '\n' + rf'$\ = {sigma_beta_func:.2e}$',
         transform=ax2.transAxes, va='top')

    plt.tight_layout()
    plt.savefig(plots_dir + 'Dest_EB_hist_nsims'
            + str(nsims) + '_steps' + str(step)
            + '_lmax' + str(lmax) + '.png')
    plt.show()
    var_matr= np.diag(var_Dest_th)
    bin_var = P @ var_matr @ Pt
    sigma_var_sim = np.sqrt(var_Dest_sim)
    sigma_var_th = np.sqrt(np.diag(bin_var))
         
   

    fig2, ax3 = plt.subplots(figsize=(7, 4))

    ax3.plot(ell_arr, sigma_var_sim, 'o-', markersize=2,
         label=r'$\sigma_{\bar D_\ell}^{\mathrm{emp}}$')

    ax3.plot(ell_arr, sigma_var_th, 'o-', markersize=2,
         label=r'$\sigma_{\bar D_\ell}^{\mathrm{th,\,binned}}$')

    ax3.legend()
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.set_xlabel(r'Multipole $\ell$')
    ax3.set_ylabel(r'$\sigma_{\bar D_\ell}$')
    ax3.grid(True, ls='--', alpha=0.2)

    plt.tight_layout()
    plt.savefig(plots_dir + 'Dest_EB_varcomparison_nsims'
            + str(nsims) + '_lmax' + str(lmax) + '.png')
    plt.show()

def main():
    root_dir = '/home/matteo/Scrivania/workspace/'
    Cls_dir = root_dir + 'simulazione/'
    ws_dir = root_dir + 'simulazione/ws/'
    mask_dir = root_dir + 'simulazione/mask/'
    maps_dir = root_dir + 'simulazione/maps/'
    Cls_out_dir = root_dir + 'simulazione/spectra_out/'
    plots_dir = root_dir + 'simulazione/plots/'

    newmaps = True # se True, genera nuove mappe altrimenti carica quelle esistenti
    new_spectra = True # se True, genera nuovi spettri, altrimenti carica quelli già esistenti
    real_mask = True # se True, importa una maschera altrimenti genera una maschera identità
    is_Dell= True
    
    purify_e = False
    purify_b = False

    nside = 64
    nsims = 1000
    deltal = 5
    lmax = 2 * nside # usare 3*nside-1 se si usa purify_e e purify_b
    lmin = 2
    sigma_noise = 2.0 #microK * arcmin

    Spix = hp.nside2pixarea(nside, degrees=True)
    var = sigma_noise * sigma_noise / (Spix * 3600.)

    npix = hp.nside2npix(nside)
    fwhm = 2.4 * np.sqrt(Spix)
    aposize = fwhm  # per nside=64, aposize>=1.05 altrimenti segm fault

    ell = np.arange(0, lmax + 1)
    ll = ell * (ell + 1) / (2 * np.pi)

    Cls_th = read_Cls(Cls_dir, lmax, ll, lmin)

    maskT_out, maskP_out = read_mask(mask_dir, real_mask, nside, 
                                     npix, aposize)

    if newmaps:
        create_maps(Cls_th, lmax, nsims, 
                    nside, npix, var, 
                    fwhm, sigma_noise, maps_dir)

    create_workspace(
        lmax, fwhm, nside,
        ws_dir, deltal,
        maps_dir, sigma_noise,
        purify_e, purify_b,
        maskT_out, maskP_out, is_Dell
    )

    mean_Cls, std_Cls, err_Cls, cl_TT_th, cl_EE_th, cl_BB_th, cl_TE_th, ell_arr = \
        calculate_spectrum_pol(
            lmax, fwhm, nside, 
            ws_dir, Cls_th, nsims,
            deltal, maps_dir, sigma_noise,
            purify_e, purify_b,
            maskT_out, maskP_out,
            Cls_out_dir, new_spectra, is_Dell
        )
    
    P, Pt, Q, Qt= PQ_matrix(lmin,lmax,deltal)

    if new_spectra:
        validation_plot(mean_Cls, std_Cls, err_Cls,
                        cl_TT_th, cl_EE_th, cl_BB_th, cl_TE_th,
                        ell_arr, plots_dir, nsims, lmax, real_mask, 
                        sigma_noise)
    

    D_estimators(nsims, Cls_out_dir, lmax, 
                 lmin, deltal, std_Cls, 
                 mean_Cls, plots_dir, P,
                 Pt,sigma_noise,var,
                 Cls_th,Cls_dir)


main()
