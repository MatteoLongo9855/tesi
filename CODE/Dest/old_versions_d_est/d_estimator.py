import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt
from scipy.optimize import minimize


def read_Cls(Cls_dir, lmax, ll):
    TT_th, EE_th, BB_th, TE_th = np.loadtxt(
        Cls_dir + 'ffp10_lensedCls.dat',
        unpack=True,
        usecols=(1, 2, 3, 4)
    )

    Cls_th = np.zeros((4, lmax + 1))
    Cls_th[0, 2:] = TT_th[0:lmax - 1] / ll[2:]
    Cls_th[1, 2:] = EE_th[0:lmax - 1] / ll[2:]
    Cls_th[2, 2:] = BB_th[0:lmax - 1] / ll[2:]
    Cls_th[3, 2:] = TE_th[0:lmax - 1] / ll[2:]

    return Cls_th


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


def create_maps(Cls_th, lmax, nsims, nside, npix, var, fwhm, sigma, maps_dir):
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
            + '_whitenoise' + str(sigma) + 'microK*arcmin'
            + '_sim' + str(i).zfill(4) + '.fits'
        )

        filenameB = (
            maps_dir
            + 'mapB_nside' + str(nside)
            + '_whitenoise' + str(sigma) + 'microK*arcmin'
            + '_sim' + str(i).zfill(4) + '.fits'
        )

        hp.write_map(filenameA, ma_TQU, overwrite=True)
        hp.write_map(filenameB, mb_TQU, overwrite=True)


def create_workspace(lmax, fwhm, nside, ws_dir,
                     deltal, maps_dir, sigma,
                     purify_e, purify_b, maskT_out, maskP_out):

    b = nmt.NmtBin.from_lmax_linear(lmax, deltal, is_Dell=False)

    bl = hp.gauss_beam(
        np.deg2rad(fwhm / 60.0),
        lmax=lmax,
        pol=True
    )
    wpix = hp.pixwin(nside, pol=True, lmax=lmax)

    filenameA = (
        maps_dir
        + 'mapA_nside' + str(nside)
        + '_whitenoise' + str(sigma) + 'microK*arcmin'
        + '_sim' + str(0).zfill(4) + '.fits'
    )

    filenameB = (
        maps_dir
        + 'mapB_nside' + str(nside)
        + '_whitenoise' + str(sigma) + 'microK*arcmin'
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


def calculate_spectrum_pol(lmax, fwhm, nside, ws_dir, Cls_th, nsims,
                           deltal, maps_dir, sigma,
                           purify_e, purify_b, maskT_out, maskP_out,
                           Cls_out_dir, new_spectra):

    b = nmt.NmtBin.from_lmax_linear(lmax, deltal, is_Dell=False)
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
                + '_whitenoise' + str(sigma) + 'microK*arcmin'
                + '_sim' + str(i).zfill(4) + '.fits'
            )

            filenameB = (
                maps_dir
                + 'mapB_nside' + str(nside)
                + '_whitenoise' + str(sigma) + 'microK*arcmin'
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


def validation_plot(mean_Cls, std_Cls, err_Cls,
                    cl_TT_th, cl_EE_th, cl_BB_th, cl_TE_th, ell_arr):

    cl_TT_master = mean_Cls[0]
    cl_EE_master = mean_Cls[1]
    cl_BB_master = mean_Cls[2]
    cl_TE_master = mean_Cls[3]

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
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$C_\ell^{TT}$')
    ax1.legend(fontsize=8)
    ax1.grid(True, ls='--', alpha=0.4)

    ax2.axhline(0, color='black', lw=0.8)
    ax2.plot(ell_arr, delta_TT, 'o', markersize=2)
    ax2.axhline(3, color='red', ls='--')
    ax2.axhline(-3, color='red', ls='--')
    ax2.set_xscale('log')
    ax2.set_ylim(-5, 5)
    ax2.set_xlabel(r'$\ell$')
    ax2.set_ylabel(r'$\Delta/\sigma$')
    ax2.grid(True, ls='--', alpha=0.4)

    plt.tight_layout()
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
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$C_\ell^{EE}$')
    ax1.legend(fontsize=8)
    ax1.grid(True, ls='--', alpha=0.4)

    ax2.axhline(0, color='black', lw=0.8)
    ax2.plot(ell_arr, delta_EE, 'o', markersize=2)
    ax2.axhline(3, color='red', ls='--')
    ax2.axhline(-3, color='red', ls='--')
    ax2.set_xscale('log')
    ax2.set_ylim(-5, 5)
    ax2.set_xlabel(r'$\ell$')
    ax2.set_ylabel(r'$\Delta/\sigma$')
    ax2.grid(True, ls='--', alpha=0.4)

    plt.tight_layout()
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
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$C_\ell^{BB}$')
    ax1.legend(fontsize=8)
    ax1.grid(True, ls='--', alpha=0.4)

    ax2.axhline(0, color='black', lw=0.8)
    ax2.plot(ell_arr, delta_BB, 'o', markersize=2)
    ax2.axhline(3, color='red', ls='--')
    ax2.axhline(-3, color='red', ls='--')
    ax2.set_xscale('log')
    ax2.set_ylim(-5, 5)
    ax2.set_xlabel(r'$\ell$')
    ax2.set_ylabel(r'$\Delta/\sigma$')
    ax2.grid(True, ls='--', alpha=0.4)

    plt.tight_layout()
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
    ax1.set_xscale('log')
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
    ax2.set_ylabel(r'$\Delta/\sigma$')
    ax2.grid(True, ls='--', alpha=0.4)

    plt.tight_layout()
    plt.show()


def D_estimators (nsims,Cls_out_dir,lmax,deltal,std_Cls):

    mean_Dest=0.0
    betamin=-0.5
    betamax=0.5
    step=0.001
    betas_bestA=np.ones(nsims)
    betas_bestB=np.ones(nsims)
    ####################
    #funzione utile solo per metodo con minimize
    def chi2_beta(beta, cl_TB, cl_TE, std_TB, std_TE):
            
            b = beta[0]
            Dest_TB = cl_TB * np.cos(2 * b) - cl_TE * np.sin(2 * b)
            x_l = Dest_TB - mean_Dest
            var_Dest  = (np.cos(2 * b) * std_TB)**2 + (np.sin(2 * b) * std_TE)**2
            Prec = np.diag(1.0 / var_Dest)
            return x_l @ Prec @ x_l
    #######################

    for i in range(nsims):
        ell_arr,cl_TT,cl_EE,cl_BB,cl_TE,cl_TB,cl_EB=np.loadtxt(Cls_out_dir+'NamasterCls_lmax'+str(lmax)+
                                                               '_deltal'+str(deltal)+'_sim'+str(i).zfill(4) + 
                                                               '.txt', unpack=True)
        betas = np.arange(betamin, betamax + step,step)
        chi2A = np.zeros(len(betas))
     

        for j in range(len(betas)):
            beta=betas[j]
            Dest_TB=cl_TB*np.cos(2*beta)-cl_TE*np.sin(2*beta)
            x_l=Dest_TB-mean_Dest
            #Dest_EB=(cl_EB*np.cos(4*beta))-0.5*(cl_EE-cl_BB)*np.sin(4*beta)
            var_Dest=(np.cos(2*beta)*std_Cls[4,:])**2+(np.sin(2*beta)*std_Cls[3,:])**2
            Prec = np.diag(1.0 / var_Dest)
            chi2A[j] = x_l @ Prec @ x_l

        betas_bestA[i]=betas[np.argmin(chi2A)]

        res = minimize(chi2_beta,x0=np.array([0.0]),args=(cl_TB, cl_TE, std_Cls[4,:], std_Cls[3,:]),
                       method='L-BFGS-B',bounds=[(-0.5, 0.5)])   
        betas_bestB[i] = res.x[0]
        print("beta_",i)

    sigma_betaA = np.std(betas_bestA, ddof=1)
    sigma_betaB = np.std(betas_bestB, ddof=1)
    mean_betaA = np.mean(betas_bestA)
    mean_betaB = np.mean(betas_bestB)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    nbins=8
    ax1.hist(betas_bestA, bins=nbins, edgecolor='black')
    ax1.set_xlabel(r'$\beta_{\rm best}^{\rm (grid)}$')
    ax1.set_ylabel('Counts')
    ax1.set_title(r'Distribuzione di $\beta_{\rm best}$ (grid)')
    ax1.axvline(mean_betaA, color='red', linestyle='--')
    ax1.axvline(mean_betaA + sigma_betaA, color='red', linestyle=':')
    ax1.axvline(mean_betaA - sigma_betaA, color='red', linestyle=':')
    ax1.text(0.02, 0.95,rf'$\bar\beta = {mean_betaA:.2e}$'+'\n'+rf'$\sigma = {sigma_betaA:.2e}$',
             transform=ax1.transAxes, fontsize=10, va='top')

    ax2.hist(betas_bestB, bins='sqrt', edgecolor='black')
    ax2.set_xlabel(r'$\beta_{\rm best}^{\rm (minimize)}$')
    ax2.set_title(r'Distribuzione di $\beta_{\rm best}$ (minimize)')
    ax2.axvline(mean_betaB, color='red', linestyle='--')
    ax2.axvline(mean_betaB + sigma_betaB, color='red', linestyle=':')
    ax2.axvline(mean_betaB - sigma_betaB, color='red', linestyle=':')
    ax2.text(0.02, 0.95,rf'$\bar\beta = {mean_betaB:.2e}$'+'\n'+rf'$\sigma = {sigma_betaB:.2e}$',
             transform=ax2.transAxes, fontsize=10, va='top')

    plt.tight_layout()
    plt.show()


def main():
    root_dir = '/home/matteo/Scrivania/workspace/'
    Cls_dir = root_dir + 'simulazione/'
    ws_dir = root_dir + 'simulazione/ws/'
    mask_dir = root_dir + 'simulazione/mask/'
    maps_dir = root_dir + 'simulazione/maps/'
    Cls_out_dir = root_dir + 'simulazione/spectra_out/'

    newmaps = False      # se True, genera nuove mappe altrimenti carica quelle esistenti
    new_spectra = False  # se True, genera nuovi spettri, altrimenti carica quelli già esistenti
    real_mask = True   # se True, importa una maschera altrimenti genera una maschera identità

     # per nside=64, aposize>=1.05 altrimenti segm fault
    purify_e = False
    purify_b = False

    nside = 64
    nsims = 1000
    deltal = 5
    lmax = 2 * nside      # usare 3*nside-1 se si usa purify_e e purify_b
    sigma = 2.0

    Spix = hp.nside2pixarea(nside, degrees=True)
    var = sigma * sigma / (Spix * 3600.)

    npix = hp.nside2npix(nside)
    fwhm = 2.4 * np.sqrt(Spix)
    aposize = fwhm  # per nside=64, aposize>=1.05 altrimenti segm fault

    ell = np.arange(0, lmax + 1)
    ll = ell * (ell + 1) / (2 * np.pi)

    Cls_th = read_Cls(Cls_dir, lmax, ll)
    maskT_out, maskP_out = read_mask(mask_dir, real_mask, nside, npix, aposize)

    if newmaps:
        create_maps(Cls_th, lmax, nsims, nside, npix, var, fwhm, sigma, maps_dir)

    create_workspace(
        lmax, fwhm, nside,
        ws_dir, deltal,
        maps_dir, sigma,
        purify_e, purify_b,
        maskT_out, maskP_out
    )

    mean_Cls, std_Cls, err_Cls, cl_TT_th, cl_EE_th, cl_BB_th, cl_TE_th, ell_arr = \
        calculate_spectrum_pol(
            lmax, fwhm, nside, ws_dir, Cls_th, nsims,
            deltal, maps_dir, sigma,
            purify_e, purify_b,
            maskT_out, maskP_out,
            Cls_out_dir, new_spectra
        )
    
    validation_plot(
        mean_Cls, std_Cls, err_Cls,
        cl_TT_th, cl_EE_th, cl_BB_th, cl_TE_th,
        ell_arr
    )

    D_estimators(nsims, Cls_out_dir, lmax, deltal, std_Cls)


main()
