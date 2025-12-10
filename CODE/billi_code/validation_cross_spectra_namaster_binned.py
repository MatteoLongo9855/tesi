import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
opj = os.path.join


'''
==================================
STATISTICAl ANALYSIS OF SPETTRA
==================================
'''

def valid_sims (l, dl_th, mean, std_mean, deltaCl, a_title, b_title, type_spectra,i,output_path):
    f = plt.figure (figsize =(10,5))
    gs = gridspec.GridSpec (3,1)
    ax1 = f.add_subplot(gs[0:2,0])
    ax4 = f.add_subplot(gs[2,0], sharex=ax1)
    plt.rcParams['axes.labelsize'] = 17

    miny=mean[0:220].min()
    maxy=mean[0:220].max()


    ax1.plot (l, dl_th, 'r--', label='Fiducial spectrum')
    ax1.plot (l, mean,'b--')
    ax1.errorbar(l, mean, yerr = std_mean, fmt=".", color='b', ecolor="b", capsize=3, lw=1, label='MC Average 'r'$\pm \sigma_{\mu}$')
    ax1.legend(loc="upper right")
    ax1.set_xlim(0,1000)
    ax1.set_ylim(miny,maxy)

    ax1.set_ylabel (a_title[i])

    ax4.plot(l,deltaCl, 'b.')
    dminy=deltaCl[0:220].min()
    dmaxy=deltaCl[0:220].max()
    ax4.axhline(-3., linestyle ='--', color = 'b')
    ax4.axhline(3., linestyle ='--', color = 'b')
    ax4.axhline(0., linestyle ='--', color = 'k')
    ax4.set_xlim(0,1000)
    ax4.set_ylim(dminy,dmaxy)
    ax4.set_ylabel (b_title[i])
    ax4.set_xlabel (r'$\ell$')

    xticklabel = ax1.get_xticklabels()
    plt.setp(xticklabel, visible = False)

    f.subplots_adjust (hspace =0.1)
    f.tight_layout()
    plt.savefig(output_path+'Validation/spectra/valid_'+type_spectra[i]+'.pdf')
    #show()
    plt.close (fig = f) 

def plots_covs(Cls, covth, ell, dir_out, aps_name):
    l=np.arange(ell.shape[0])
    covmatemp=np.cov(Cls)
    varemp=covmatemp[l,l]
    varth=covth[l,l]

    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(ell,varemp, 'r-', lw = 1, label= aps_name+' Variance from sims')
    ax1.plot(ell,varth, 'k-', lw = 1, label= aps_name+' Variance analitical from fid')
    ax1.set_ylabel ('Variance')
    ax1.set_xlabel (r'$\ell$')
    ax1.legend()
    fig.tight_layout()
    plt.savefig(opj(dir_out,'Validation/Covariance/'+aps_name+'_variances.png'))
    #show()
    plt.close (fig = fig)

def plots_covs_ratio(Cls, covth,ell, dir_out, aps_name):
    l=np.arange(ell.shape[0])
    covmatemp=np.cov(Cls)
    varemp=covmatemp[l,l]
    varth =covth[l,l]

    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(ell,varth/varemp, 'k-', lw = 1, label= aps_name+' Var analitic / Var sims')
    ax1.set_ylabel ('Variance')
    ax1.set_xlabel (r'$\ell$')
    ax1.legend()
    fig.tight_layout()
    plt.savefig(opj(dir_out,'Validation/Covariance/'+aps_name+'_variances_ratio.png'))
    #show()
    plt.close (fig = fig)

def analitic_var(clTT, clEE, clBB, clTE, nT, nP, lmin,lmax,auto=False):
    l=np.arange(lmin,lmax+1)
    if(auto):
        varthTT=2./((2*l)+1)*((clTT+nT)**2)
        varthEE=2./((2*l)+1)*((clEE+nP)**2)
        varthBB=2./((2*l)+1)*((clBB+nP)**2)
    else:
        varthTT=2./((2*l)+1)*(clTT**2+clTT*nT+nT**2/2)
        varthEE=2./((2*l)+1)*(clEE**2+clEE*nP+nP**2/2)
        varthBB=2./((2*l)+1)*(clBB**2+clBB*nP+nP**2/2)
    varthTE=1./((2*l)+1)*(((clTT+nT)*(clEE+nP))+(clTE**2))
    varthEB=1./((2*l)+1)*((clEE+nP)*(clBB+nP))
    varthTB=1./((2*l)+1)*((clTT+nT)*(clBB+nP))
    return(np.diag(varthTT),np.diag(varthEE),np.diag(varthBB),np.diag(varthTE),np.diag(varthEB),np.diag(varthTB))

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


##################################################################################
## Tutti gli spettri sono dati in forma l(l+1)/2pi e nel range armonico lmin lmax

def plots_stat_spectra(deltal, nsims, nside, sigmaT,rootdir_path, read_all_spectra = False):
    spectra_path = rootdir_path+'spectra_out/'
    output_path  = rootdir_path+'validation_namaster_spectra/'

    a_title =[  r'$ D_{\ell}^{TT}$'r'$(\mu K^2)$', \
            r'$ D_{\ell}^{EE}$'r'$(\mu K^2)$',\
            r'$ D_{\ell}^{TE}$'r'$(\mu K^2)$',\
            r'$ D_{\ell}^{BB}$'r'$(\mu K^2)$',\
            r'$ D_{\ell}^{EB}$'r'$(\mu K^2)$',\
            r'$ D_{\ell}^{TB}$'r'$(\mu K^2)$']
    b_title =[  r'$ \frac{\Delta D_{\ell}^{TT}}{\sigma_{\mu}}$', \
            r'$ \frac{\Delta D_{\ell}^{EE}}{\sigma_{\mu}}$', \
            r'$ \frac{\Delta D_{\ell}^{TE}}{\sigma_{\mu}}$', \
            r'$ \frac{\Delta D_{\ell}^{BB}}{\sigma_{\mu}}$', \
            r'$ \frac{\Delta D_{\ell}^{EB}}{\sigma_{\mu}}$', \
            r'$ \frac{\Delta D_{\ell}^{TB}}{\sigma_{\mu}}$']
    type_spectra = ['TT','EE','TE','BB','EB','TB']

    lmin         = 2
    lmax         = 2*nside
    P,Pt,Q,Qt = PQ_matrix(lmin,lmax,deltal)
    ell_bin = P@(np.arange(lmin, lmax+1,1))
    ell_size = ell_bin.shape[0]

    if read_all_spectra:
        cl_TT = np.zeros((ell_size,nsims))
        cl_EE = np.zeros((ell_size,nsims))
        cl_BB = np.zeros((ell_size,nsims))
        cl_TE = np.zeros((ell_size,nsims))
        cl_TB = np.zeros((ell_size,nsims))
        cl_EB = np.zeros((ell_size,nsims))
        for isim in np.arange(nsims):
            Cls = np.loadtxt(spectra_path+'NamasterCls_lmax128_deltal5_sim'+str(isim).zfill(4)+'.txt', unpack=True, usecols=(1,2,3,4,5,6))
            cl_TT[:,isim] = Cls[0]
            cl_EE[:,isim] = Cls[1]
            cl_BB[:,isim] = Cls[2]
            cl_TE[:,isim] = Cls[3]
            cl_EB[:,isim] = Cls[4]
            cl_TB[:,isim] = Cls[5]
        np.savetxt(spectra_path+'cl_TT_cross.txt',cl_TT)
        np.savetxt(spectra_path+'cl_EE_cross.txt',cl_EE)
        np.savetxt(spectra_path+'cl_BB_cross.txt',cl_BB)
        np.savetxt(spectra_path+'cl_TE_cross.txt',cl_TE)
        np.savetxt(spectra_path+'cl_EB_cross.txt',cl_EB)
        np.savetxt(spectra_path+'cl_TB_cross.txt',cl_TB)

    dl_TT_sim = np.loadtxt(spectra_path+'cl_TT_cross.txt')
    dl_EE_sim = np.loadtxt(spectra_path+'cl_EE_cross.txt')
    dl_BB_sim = np.loadtxt(spectra_path+'cl_BB_cross.txt')
    dl_TE_sim = np.loadtxt(spectra_path+'cl_TE_cross.txt')
    dl_EB_sim = np.loadtxt(spectra_path+'cl_EB_cross.txt')
    dl_TB_sim = np.loadtxt(spectra_path+'cl_TB_cross.txt')

    TT_th,EE_th,BB_th,TE_th = np.loadtxt(output_path+'/ffp10_lensedCls.dat',unpack=True,usecols=(1,2,3,4))
    
    cl_TT_th = (TT_th[lmin-2:lmax-lmin+1])
    cl_EE_th = (EE_th[lmin-2:lmax-lmin+1])  
    cl_BB_th = (BB_th[lmin-2:lmax-lmin+1])  
    cl_TE_th = (TE_th[lmin-2:lmax-lmin+1])  

    ## READING THE NOISE
    NoiseT = np.full(lmax-2+1,(sigmaT*np.deg2rad(1/60))**2)
    NoiseP = 2*NoiseT
    l=np.arange(lmin,lmax+1)
    ll=l*(l+1)/2/np.pi
    NoiseT *=ll 
    NoiseP *=ll

    ### Covariance validation
    covthTT,covthEE,covthBB,covthTE,covthEB,covthTB = analitic_var(cl_TT_th, cl_EE_th, cl_BB_th, cl_TE_th,NoiseT, NoiseP,lmin,lmax)
    plots_covs(dl_TT_sim, P@covthTT@Pt, ell_bin, output_path, 'TT')
    plots_covs(dl_EE_sim, P@covthEE@Pt, ell_bin, output_path, 'EE')
    plots_covs(dl_BB_sim, P@covthBB@Pt, ell_bin, output_path, 'BB')
    plots_covs(dl_TE_sim, P@covthTE@Pt, ell_bin, output_path, 'TE')
    plots_covs(dl_EB_sim, P@covthEB@Pt, ell_bin, output_path, 'EB')
    plots_covs(dl_TB_sim, P@covthTB@Pt, ell_bin, output_path, 'TB')

    plots_covs_ratio(dl_TT_sim, P@covthTT@Pt, ell_bin, output_path, 'TT')
    plots_covs_ratio(dl_EE_sim, P@covthEE@Pt, ell_bin, output_path, 'EE')
    plots_covs_ratio(dl_BB_sim, P@covthBB@Pt, ell_bin, output_path, 'BB')
    plots_covs_ratio(dl_TE_sim, P@covthTE@Pt, ell_bin, output_path, 'TE')
    plots_covs_ratio(dl_EB_sim, P@covthEB@Pt, ell_bin, output_path, 'EB')
    plots_covs_ratio(dl_TB_sim, P@covthTB@Pt, ell_bin, output_path, 'TB')
   

    dl_TT_th = P@cl_TT_th
    dl_EE_th = P@cl_EE_th  
    dl_BB_th = P@cl_BB_th  
    dl_TE_th = P@cl_TE_th  
    dl_TB_th = np.zeros_like(dl_TT_th)
    dl_EB_th = np.zeros_like(dl_TT_th) 



      
    ## MC validations

    valid_sims (ell_bin,                                                                   ## Harmonic range 
            dl_TT_th,                                                                ## Fiducial        
            dl_TT_sim.mean(axis=1),                                                  ## MC Average
            np.sqrt(dl_TT_sim.var(axis=1)/nsims),                                    ## sigma mean        
            (dl_TT_sim.mean(axis=1)-dl_TT_th)/np.sqrt(dl_TT_sim.var(axis=1)/nsims),  ## variation wrt the fiducial in unit of sigma mean
            a_title,b_title,type_spectra,0,output_path)                             ## graphical parameters
    
    valid_sims (ell_bin, 
            dl_EE_th,
            dl_EE_sim.mean(axis=1), 
            np.sqrt(dl_EE_sim.var(axis=1)/nsims), 
            (dl_EE_sim.mean(axis=1)-dl_EE_th)/np.sqrt(dl_EE_sim.var(axis=1)/nsims),
            a_title,b_title,type_spectra,1,output_path)

    valid_sims (ell_bin, 
            dl_TE_th, 
            dl_TE_sim.mean(axis=1), 
            np.sqrt(dl_TE_sim.var(axis=1)/nsims), 
            (dl_TE_sim.mean(axis=1)-dl_TE_th)/np.sqrt(dl_TE_sim.var(axis=1)/nsims), 
            a_title,b_title,type_spectra,2,output_path)

    valid_sims (ell_bin,
            dl_BB_th,
            dl_BB_sim.mean(axis=1),
            np.sqrt(dl_BB_sim.var(axis=1)/nsims), 
            (dl_BB_sim.mean(axis=1)-dl_BB_th)/np.sqrt(dl_BB_sim.var(axis=1)/nsims), 
            a_title,b_title,type_spectra,3,output_path)

    valid_sims (ell_bin, 
            dl_TB_th, 
            dl_TB_sim.mean(axis=1), 
            np.sqrt(dl_TB_sim.var(axis=1)/nsims), 
            (dl_TB_sim.mean(axis=1)-dl_TB_th)/np.sqrt(dl_TB_sim.var(axis=1)/nsims), 
            a_title,b_title,type_spectra,5,output_path)
    
    valid_sims (ell_bin, 
            dl_EB_th, 
            dl_EB_sim.mean(axis=1), 
            np.sqrt(dl_EB_sim.var(axis=1)/nsims),
            (dl_EB_sim.mean(axis=1)-dl_EB_th)/np.sqrt(dl_EB_sim.var(axis=1)/nsims), 
            a_title,b_title,type_spectra,4,output_path)


plots_stat_spectra(deltal=5, 
                nsims = 1000, 
                nside = 64, 
                sigmaT=2.0,
                rootdir_path = '',
                read_all_spectra = True)