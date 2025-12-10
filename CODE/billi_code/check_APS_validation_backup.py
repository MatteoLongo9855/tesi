from matplotlib.pylab import *
import matplotlib.gridspec as gridspec
import numpy as np
import healpy as hp
import os
opj = os.path.join

def load_spectra(path,lmin,lmax,Nsky):
    cls_tot=np.zeros((6,lmax+1,Nsky))

    for isim in np.arange(Nsky):
        # cls_tot[:,:,isim]=np.load(path+'cl_SpN_isim'+str(isim).zfill(4)+'.npy')
        cls_tot[:,:,isim]=np.load(path+'cl_isim'+str(isim).zfill(4)+'.npy')

    
    return cls_tot

def rescaling_spectra(lmin,lmax,Nsky,TT,EE,BB,TE,EB,TB):
    l=np.arange(lmin,lmax+1)
    ll=l*(l+1)/2/np.pi
    for i in range (Nsky):
        TT [:,i] *= ll
        EE [:,i] *= ll
        BB [:,i] *= ll
        TE [:,i] *= ll
        EB [:,i] *= ll
        TB [:,i] *= ll
    return(TT,EE,BB,TE,EB,TB)


def valid_sims (lmin,lmax, cl_th, mean, std_mean, s, a,b,i, c,j,d, type_spectra,dir_out):
    f = figure (figsize =(10,5))
    gs = gridspec.GridSpec (3,1)
    ax1 = f.add_subplot(gs[0:2,0])
    ax4 = f.add_subplot(gs[2,0], sharex=ax1)
    x = arange (lmin,lmax+1)

    ax1.plot (x, cl_th, 'r--', label=c[j])
    ax1.plot (x, mean,'b--')
    ax1.errorbar(x, mean, yerr = std_mean, fmt=".", color='b', ecolor="b", capsize=3, lw=1, label='MC')
    ax1.legend(loc="upper right")
    #ax4.set_ylim(-3,+3)
    ax1.set_ylabel (a[i])

    ax4.plot(x,s, 'b.')
    l1=np.array([lmin,lmax])
    m1=np.array([0,0])
    ax4.axhline(-3., linestyle ='--', color = 'b')
    ax4.axhline(3., linestyle ='--', color = 'b')
    ax4.plot(l1,m1,'--', color='k')
    ax1.set_xlabel (r'$\ell$')
    ax4.set_ylabel (b[i])
    ax4.set_xlabel (r'$\ell$')

    xticklabel = ax1.get_xticklabels()
    setp(xticklabel, visible = False)

    f.subplots_adjust (hspace =0.1)
    f.tight_layout()
    savefig(opj(dir_out,'Validation/spectra/valid_'+str(d[i])+'_'+type_spectra+'.pdf'))
    #show()
    close (fig = f) 

def analitic_var(clTT, clEE, clBB, clTE, nT, nP, lmin,lmax,auto):
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
    return(varthTT,varthEE,varthBB,varthTE,varthEB,varthTB)


def plots_covs(Cls, varth, lmin, lmax, dir_out, aps_name):
    l=np.arange(lmax-lmin+1)
    covmatemp=np.cov(Cls)
    varemp=covmatemp[l,l]
    
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(l,varemp, 'r-', lw = 1, label= aps_name+' Variance from sims')
    ax1.plot(l,varth, 'k-', lw = 1, label= aps_name+' Variance analitical from fid')
    ax1.set_ylabel ('Variance')
    ax1.set_xlabel (r'$\ell$')
    ax1.legend()
    fig.tight_layout()
    plt.savefig(opj(dir_out,'Validation/Covariance/'+aps_name+'_variances.png'))
    #show()
    close (fig = fig)

def plots_covs_ratio(Cls, varth,lmin, lmax, dir_out, aps_name):
    l=np.arange(lmax-lmin+1)
    covmatemp=np.cov(Cls)
    varemp=covmatemp[l,l]
    
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(l,varth/varemp, 'k-', lw = 1, label= aps_name+' Var analitic / Var sims')
    ax1.set_ylabel ('Variance')
    ax1.set_xlabel (r'$\ell$')
    ax1.legend()
    fig.tight_layout()
    plt.savefig(opj(dir_out,'Validation/Covariance/'+aps_name+'_variances_ratio.png'))
    #show()
    close (fig = fig)

def plot_fiducial_auto(cl_TT_th, cl_EE_th, cl_BB_th, lmin,lmax, dir_out):
    l=np.arange(lmin,lmax+1)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(l,cl_TT_th, 'k-', lw = 1, label= 'TT')
    ax1.plot(l,cl_EE_th, 'r-', lw = 1, label= 'EE')
    ax1.plot(l,cl_BB_th, 'b-', lw = 1, label= 'BB')

    ax1.set_ylabel (r'$\frac{\ell(\ell+1)}{2\pi} C_{\ell}(\mu K^2)$')
    ax1.set_xlabel (r'$\ell$')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid()
    fig.tight_layout()
    plt.savefig(opj(dir_out,'Validation/Fiducial/auto_spectra.png'))
    #show()
    close (fig = fig)

def plot_fiducial_cross(cl_TE_th, lmin,lmax, dir_out):
    l=np.arange(lmin,lmax+1)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(l,cl_TE_th, 'g-', lw = 1, label= 'TE')
    ax1.set_ylabel (r'$\frac{\ell(\ell+1)}{2\pi} C_{\ell}^{TE}(\mu K^2)$')
    ax1.set_xlabel (r'$\ell$')
    ax1.legend()
    ax1.grid()
    fig.tight_layout()
    plt.savefig(opj(dir_out,'Validation/Fiducial/cross_spectraTE.png'))
    #show()
    close (fig = fig)

## Tutti gli spettri sono dati in forma l(l+1)/2pi e nel range armonico lmin lmax
def main_valid_plot(lmin,lmax, Nsky,
    cl_TT_sim, cl_EE_sim, cl_BB_sim, cl_TE_sim, cl_EB_sim, cl_TB_sim,
    cl_TT_th, cl_EE_th, cl_BB_th, cl_TE_th, cl_EB_th, cl_TB_th,
    NoiseT, NoiseP, auto, dir_out):

    a_title =[  r'$ C_{\ell}^{TT}$'r'$(\mu K^2)$', \
            r'$ C_{\ell}^{EE}$'r'$(\mu K^2)$',\
            r'$ C_{\ell}^{TE}$'r'$(\mu K^2)$',\
            r'$ C_{\ell}^{BB}$'r'$(\mu K^2)$',\
            r'$ C_{\ell}^{EB}$'r'$(\mu K^2)$',\
            r'$ C_{\ell}^{TB}$'r'$(\mu K^2)$']
    b_title =[  r'$ \frac{\Delta D_{\ell}^{TT}}{\sigma_{\mu}}$', \
            r'$ \frac{\Delta D_{\ell}^{EE}}{\sigma_{\mu}}$', \
            r'$ \frac{\Delta D_{\ell}^{TE}}{\sigma_{\mu}}$', \
            r'$ \frac{\Delta D_{\ell}^{BB}}{\sigma_{\mu}}$', \
            r'$ \frac{\Delta D_{\ell}^{EB}}{\sigma_{\mu}}$', \
            r'$ \frac{\Delta D_{\ell}^{TB}}{\sigma_{\mu}}$']
    c = ['Planck Data', 'Fiduciale']
    d = ['tt','ee','te','bb','eb','tb']

    ##FIDUCIAL PLOT
    plot_fiducial_auto(cl_TT_th, cl_EE_th, cl_BB_th, lmin,lmax, dir_out)
    plot_fiducial_cross(cl_TE_th, lmin,lmax, dir_out)

    ## Noise validation (only for the maps)
    # valid_sims (lmin,lmax,  ## range asse x
    #             NoiseT,     ## modello        
    #             cl_TT_noise.mean(axis=1), ## media sims
    #             np.sqrt(cl_TT_noise.var(axis=1)/Nsky), ## sigma media        
    #             (cl_TT_noise.mean(axis=1)-NoiseT)/np.sqrt(cl_TT_noise.var(axis=1)/Nsky),  ## vairazione rispetto modello in sigma media
    #             a_title,b_title,0, c,1,d,'noise',dir_out) ## parametri per grafico

    # valid_sims (lmin,lmax, 
    #             NoiseP,
    #             cl_EE_noise.mean(axis=1), 
    #             np.sqrt(cl_EE_noise.var(axis=1)/Nsky), 
    #             (cl_EE_noise.mean(axis=1)-NoiseP)/np.sqrt(cl_EE_noise.var(axis=1)/Nsky),
    #             a_title,b_title,1, c,1,d,'noise',dir_out)

    # valid_sims (lmin,lmax, 
    #             np.zeros(lmax-lmin), 
    #             cl_TE_noise.mean(axis=1), 
    #             np.sqrt(cl_TE_noise.var(axis=1)/Nsky), 
    #             (cl_TE_noise.mean(axis=1)-np.zeros(lmax-lmin))/np.sqrt(cl_TE_noise.var(axis=1)/Nsky), 
    #             a_title,b_title,2, c,1,d,'noise',dir_out)

    # valid_sims (lmin,lmax, NoiseP,
    #             cl_BB_noise.mean(axis=1),
    #             np.sqrt(cl_BB_noise.var(axis=1)/Nsky), 
    #             (cl_BB_noise.mean(axis=1)-NoiseP)/np.sqrt(cl_BB_noise.var(axis=1)/Nsky), 
    #             a_title,b_title,3, c,1,d,'noise',dir_out)

    # valid_sims (lmin,lmax, 
    #             np.zeros(lmax-lmin), 
    #             cl_EB_noise.mean(axis=1), 
    #             np.sqrt(cl_EB_noise.var(axis=1)/Nsky),
    #             (cl_EB_noise.mean(axis=1)-np.zeros(lmax-lmin))/np.sqrt(cl_EB_noise.var(axis=1)/Nsky), 
    #             a_title,b_title,4, c,1,d,'noise',dir_out)

    # valid_sims (lmin,lmax, 
    #             np.zeros(lmax-lmin), 
    #             cl_TB_noise.mean(axis=1), 
    #             np.sqrt(cl_TB_noise.var(axis=1)/Nsky), 
    #             (cl_TB_noise.mean(axis=1)-np.zeros(lmax-lmin))/np.sqrt(cl_TB_noise.var(axis=1)/Nsky), 
    #             a_title,b_title,5, c,1,d,'noise',dir_out)

    ## Spectra validation
    if(auto):
        valid_sims (lmin,lmax,  ## range asse x
                    cl_TT_th,     ## modello        
                    cl_TT_sim.mean(axis=1)-NoiseT, ## media sims
                    np.sqrt(cl_TT_sim.var(axis=1)/Nsky), ## sigma media        
                    (cl_TT_sim.mean(axis=1)-NoiseT-cl_TT_th)/np.sqrt(cl_TT_sim.var(axis=1)/Nsky),  ## vairazione rispetto modello in sigma media
                    a_title,b_title,0, c,1,d,'spectra',dir_out) ## parametri per grafico

        valid_sims (lmin,lmax, 
                    cl_EE_th,
                    cl_EE_sim.mean(axis=1)-NoiseP, 
                    np.sqrt(cl_EE_sim.var(axis=1)/Nsky), 
                    (cl_EE_sim.mean(axis=1)-NoiseP-cl_EE_th)/np.sqrt(cl_EE_sim.var(axis=1)/Nsky),
                    a_title,b_title,1, c,1,d,'spectra',dir_out)

        valid_sims (lmin,lmax, 
                    cl_TE_th, 
                    cl_TE_sim.mean(axis=1), 
                    np.sqrt(cl_TE_sim.var(axis=1)/Nsky), 
                    (cl_TE_sim.mean(axis=1)-cl_TE_th)/np.sqrt(cl_TE_sim.var(axis=1)/Nsky), 
                    a_title,b_title,2, c,1,d,'spectra',dir_out)

        valid_sims (lmin,lmax,
                    cl_BB_th,
                    cl_BB_sim.mean(axis=1)-NoiseP,
                    np.sqrt(cl_BB_sim.var(axis=1)/Nsky), 
                    (cl_BB_sim.mean(axis=1)-NoiseP-cl_BB_th)/np.sqrt(cl_BB_sim.var(axis=1)/Nsky), 
                    a_title,b_title,3, c,1,d,'spectra',dir_out)

        valid_sims (lmin,lmax, 
                    cl_EB_th, 
                    cl_EB_sim.mean(axis=1), 
                    np.sqrt(cl_EB_sim.var(axis=1)/Nsky),
                    (cl_EB_sim.mean(axis=1)-cl_EB_th)/np.sqrt(cl_EB_sim.var(axis=1)/Nsky), 
                    a_title,b_title,4, c,1,d,'spectra',dir_out)

        valid_sims (lmin,lmax, 
                    cl_TB_th, 
                    cl_TB_sim.mean(axis=1), 
                    np.sqrt(cl_TB_sim.var(axis=1)/Nsky), 
                    (cl_TB_sim.mean(axis=1)-cl_TB_th)/np.sqrt(cl_TB_sim.var(axis=1)/Nsky), 
                    a_title,b_title,5, c,1,d,'spectra',dir_out)
    else:
        valid_sims (lmin,lmax,  ## range asse x
                    cl_TT_th,     ## modello        
                    cl_TT_sim.mean(axis=1), ## media sims
                    np.sqrt(cl_TT_sim.var(axis=1)/Nsky), ## sigma media        
                    (cl_TT_sim.mean(axis=1)-cl_TT_th)/np.sqrt(cl_TT_sim.var(axis=1)/Nsky),  ## vairazione rispetto modello in sigma media
                    a_title,b_title,0, c,1,d,'spectra',dir_out) ## parametri per grafico

        valid_sims (lmin,lmax, 
                    cl_EE_th,
                    cl_EE_sim.mean(axis=1), 
                    np.sqrt(cl_EE_sim.var(axis=1)/Nsky), 
                    (cl_EE_sim.mean(axis=1)-cl_EE_th)/np.sqrt(cl_EE_sim.var(axis=1)/Nsky),
                    a_title,b_title,1, c,1,d,'spectra',dir_out)

        valid_sims (lmin,lmax, 
                    cl_TE_th, 
                    cl_TE_sim.mean(axis=1), 
                    np.sqrt(cl_TE_sim.var(axis=1)/Nsky), 
                    (cl_TE_sim.mean(axis=1)-cl_TE_th)/np.sqrt(cl_TE_sim.var(axis=1)/Nsky), 
                    a_title,b_title,2, c,1,d,'spectra',dir_out)

        valid_sims (lmin,lmax,
                    cl_BB_th,
                    cl_BB_sim.mean(axis=1),
                    np.sqrt(cl_BB_sim.var(axis=1)/Nsky), 
                    (cl_BB_sim.mean(axis=1)-cl_BB_th)/np.sqrt(cl_BB_sim.var(axis=1)/Nsky), 
                    a_title,b_title,3, c,1,d,'spectra',dir_out)

        valid_sims (lmin,lmax, 
                    cl_EB_th, 
                    cl_EB_sim.mean(axis=1), 
                    np.sqrt(cl_EB_sim.var(axis=1)/Nsky),
                    (cl_EB_sim.mean(axis=1)-cl_EB_th)/np.sqrt(cl_EB_sim.var(axis=1)/Nsky), 
                    a_title,b_title,4, c,1,d,'spectra',dir_out)

        valid_sims (lmin,lmax, 
                    cl_TB_th, 
                    cl_TB_sim.mean(axis=1), 
                    np.sqrt(cl_TB_sim.var(axis=1)/Nsky), 
                    (cl_TB_sim.mean(axis=1)-cl_TB_th)/np.sqrt(cl_TB_sim.var(axis=1)/Nsky), 
                    a_title,b_title,5, c,1,d,'spectra',dir_out)


    ### Covariance validation
    varthTT,varthEE,varthBB,varthTE,varthEB,varthTB = analitic_var(cl_TT_th, cl_EE_th, cl_BB_th, cl_TE_th,NoiseT, NoiseP,lmin,lmax, auto)
    plots_covs(cl_TT_sim, varthTT, lmin, lmax, dir_out, 'TT')
    plots_covs(cl_EE_sim, varthEE, lmin, lmax, dir_out, 'EE')
    plots_covs(cl_BB_sim, varthBB, lmin, lmax, dir_out, 'BB')
    plots_covs(cl_TE_sim, varthTE, lmin, lmax, dir_out, 'TE')
    plots_covs(cl_EB_sim, varthEB, lmin, lmax, dir_out, 'EB')
    plots_covs(cl_TB_sim, varthTB, lmin, lmax, dir_out, 'TB')

    plots_covs_ratio(cl_TT_sim, varthTT, lmin, lmax, dir_out, 'TT')
    plots_covs_ratio(cl_EE_sim, varthEE, lmin, lmax, dir_out, 'EE')
    plots_covs_ratio(cl_BB_sim, varthBB, lmin, lmax, dir_out, 'BB')
    plots_covs_ratio(cl_TE_sim, varthTE, lmin, lmax, dir_out, 'TE')
    plots_covs_ratio(cl_EB_sim, varthEB, lmin, lmax, dir_out, 'EB')
    plots_covs_ratio(cl_TB_sim, varthTB, lmin, lmax, dir_out, 'TB')


def main(lmin,
        lmax, 
        Nsky,
        sigmaT,
        auto,
        dir_inp,
        dir_out):
    
    ## READING THE SIMS
    # Cls =np.zeros((6,2*Nside+1,Nsky))  
    # for i in range (Nsky):
    #     icls=np.load(opj(dir_out,'Cls_from_alm_SpN'+str(i)+'.npy'))
    #     Cls[:,:,i]= icls
    # np.save(opj(dir_out,'Cls_from_alm_SpN_ACTpolNoise_lmax4097.npy'),Cls)

    Cls = load_spectra(dir_inp,lmin,lmax,Nsky)#np.load(opj(dir_out,'Cls_from_alm_SpN_lownoise_lmax4000.npy'))
    cl_TT_sim, cl_EE_sim, cl_BB_sim, cl_TE_sim, cl_EB_sim, cl_TB_sim = \
    Cls[0,lmin:lmax+1],Cls[1,lmin:lmax+1],Cls[2,lmin:lmax+1],\
    Cls[3,lmin:lmax+1],Cls[4,lmin:lmax+1],Cls[5,lmin:lmax+1]
    cl_TT_sim, cl_EE_sim, cl_BB_sim, cl_TE_sim, cl_EB_sim, cl_TB_sim = \
    rescaling_spectra(lmin,lmax,Nsky, cl_TT_sim, cl_EE_sim, cl_BB_sim, cl_TE_sim, cl_EB_sim, cl_TB_sim)
    

    ## READING THE NOISE
    # cl_TT_noise, cl_EE_noise, cl_BB_noise, cl_TE_noise, cl_EB_noise, cl_TB_noise = \
    # Cls_noise_map[0,lmin:lmax],Cls_noise_map[1,lmin:lmax],Cls_noise_map[2,lmin:lmax],\
    # Cls_noise_map[3,lmin:lmax],Cls_noise_map[4,lmin:lmax],Cls_noise_map[5,lmin:lmax]
    # cl_TT_noise, cl_EE_noise, cl_BB_noise, cl_TE_noise, cl_EB_noise, cl_TB_noise = \
    # rescaling_spectra(lmin,lmax,Nsky, cl_TT_noise, cl_EE_noise, cl_BB_noise, cl_TE_noise, cl_EB_noise, cl_TB_noise)
    NoiseT = np.full(lmax-2+1,(sigmaT*np.deg2rad(1/60))**2)
    NoiseP = 2*NoiseT
    l=np.arange(lmin,lmax+1)
    ll=l*(l+1)/2/np.pi
    NoiseT *=ll 
    NoiseP *=ll
   
    ## Reading the fiducial    
    #cl_TT_th, cl_EE_th, cl_BB_th, cl_TE_th, cl_EB_th, cl_TB_th =\
    #reading_camb_spectra(lmin, lmax, dir_inp, 'camb_spectra_5000.txt')
    fid_dir = ''
    l,TT,EE,BB,TE = np.loadtxt(fid_dir+'ffp10_lensedCls.dat',unpack=True,usecols=(0,1,2,3,4))

    l = l[(lmin-2):(lmax-2+1)]
    ll = l*(l+1)/2/np.pi
    TT = TT[(lmin-2):(lmax-2+1)]
    EE = EE[(lmin-2):(lmax-2+1)]
    BB = BB[(lmin-2):(lmax-2+1)]
    TE = TE[(lmin-2):(lmax-2+1)]

    # cl_TT_th = TT/ll
    # cl_EE_th = EE/ll
    # cl_BB_th = BB/ll
    # cl_TE_th = TE/ll
    cl_TT_th = TT
    cl_EE_th = EE
    cl_BB_th = BB
    cl_TE_th = TE
    cl_EB_th = np.zeros_like(TT)
    cl_TB_th = np.zeros_like(TT)

    main_valid_plot(lmin,lmax, Nsky,
    cl_TT_sim, cl_EE_sim, cl_BB_sim, cl_TE_sim, cl_EB_sim, cl_TB_sim,
    cl_TT_th, cl_EE_th, cl_BB_th, cl_TE_th, cl_EB_th, cl_TB_th,
    NoiseT[0:lmax-2+1], NoiseP[0:lmax-2+1], auto, dir_out)
    

if __name__ == '__main__':
    main(lmin= 2,
        lmax = 4096, 
        Nsky = 1000,
        sigmaT=4.0,
        auto = False, # auto or cross spectrum
        dir_inp = 'APS_from_maps_cross_fiducialffp10_lensedCls_nside2048_lmax4096_noiseT1.0uKarcmin_nobeam/',
        dir_out = '')
        # dir_inp = '/marconi_scratch/userexternal/mbortola/CB_QML_alpha/estimator_CB_EE_bilinear/APS_from_maps_cross_lmax2048/',
        # dir_out = '/marconi_scratch/userexternal/mbilli00/CB_QML_alpha/estimator_CB_EE_bilinear/Vanilla_Plancklike/Validation_sims/')

