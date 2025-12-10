import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


def read_Cls (Cls_dir,lmax,ll):
    TT_th,EE_th,BB_th,TE_th=np.loadtxt(Cls_dir+'ffp10_lensedCls.dat',unpack=True,usecols=(1,2,3,4))
    Cls_th=np.zeros((4,lmax+1))
    
    Cls_th[0,2:]=TT_th[0:lmax-1]/ll[2:]
    Cls_th[1,2:]=EE_th[0:lmax-1]/ll[2:]
    Cls_th[2,2:]=BB_th[0:lmax-1]/ll[2:]
    Cls_th[3,2:]=TE_th[0:lmax-1]/ll[2:]
    return(Cls_th)

def read_msk(msk_dir,nside):
    maskT_in = hp.read_map(msk_dir+'COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits')
    maskP_in = hp.read_map(msk_dir+'COM_Mask_CMB-common-Mask-Pol_2048_R3.00.fits')
    maskT_out = hp.ud_grade(maskT_in, nside)
    maskP_out = hp.ud_grade(maskP_in, nside)
    
    
    #si può usare direttamente il valore corrispondete a field=, ma per sicurezza lo calcolo
    f_skyT=(np.count_nonzero(maskT_in)/np.size(maskT_in))
    #print(f_skyT)
    f_skyP=(np.count_nonzero(maskP_in)/np.size(maskP_in))
    #print(f_skyP)

    return(maskT_out,maskP_out,f_skyT,f_skyP)



def maps_noise_and_Cls(Cls_th,lmax,nside,nsims,npix,var,maskT_out,maskP_out):
   
    Cls_sims=np.zeros((6,lmax+1,nsims))
    
    
    for i in range (nsims):
        m_sign=hp.synfast(cls=Cls_th ,lmax=lmax,nside=nside,pol=True,new=True)
        noisesT = np.random.normal(loc=0,scale=np.sqrt(var),size=2*npix)
        noisesP = np.random.normal(loc=0,scale=np.sqrt(2*var),size=4*npix)
        ma_TQU=np.zeros_like(m_sign)
        mb_TQU=np.zeros_like(m_sign)
        ma_TQU[0]=(m_sign[0]+noisesT[0:npix])*maskT_out
        ma_TQU[1]=(m_sign[1]+noisesP[0:npix])*maskP_out
        ma_TQU[2]=(m_sign[2]+noisesP[npix:2*npix])*maskP_out

        
        mb_TQU[0]=(m_sign[0]+noisesT[npix:2*npix])*maskT_out
        mb_TQU[1]=(m_sign[1]+noisesP[2*npix:3*npix])*maskP_out
        mb_TQU[2]=(m_sign[2]+noisesP[3*npix:4*npix])*maskP_out
        Cls_sims[:,:,i]=hp.anafast(ma_TQU,mb_TQU,lmax=lmax,pol=True)                  
        print(i)  # per vedere da terminale l'avanzamento 
    return(Cls_sims)

def validation_spectra(Cls_th,Cls_sims,ll,ell,nsims,name,f_skyT,f_skyP):
    mean_Cls=np.mean(Cls_sims, axis=2)
    std_Cls=np.std(Cls_sims,axis=2, ddof=1)#ddof=1 calcola la deviazione standrd con 1\sqrt(n-1)
    if name=="TT":
        j=0
        f_sky=f_skyT
    elif name=="EE":
        j=1
        f_sky=f_skyP
    elif name=="BB":
        j=2
        f_sky=f_skyP
    elif name=="TE":
        j=3
        f_sky=f_skyP
    else: 
        print("ERROR")

    Dl_sim=mean_Cls*ll
    Dl_err=(std_Cls*ll)/np.sqrt(nsims)
    Dl_th=Cls_th[j]*ll*f_sky
    cosm_var= Dl_th[2:]*np.sqrt(2/((2*ell[2:])+1))
    delta=(Dl_sim[j,2:]-Dl_th[2:])/Dl_err[j, 2:]                         

   
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7,8), gridspec_kw={'height_ratios': [3,1,3]}, sharex=True)

    ax1.errorbar(ell[2:], Dl_sim[j,2:], yerr=Dl_err[j,2:], fmt='o', markersize=3, color='tab:blue', 
                 ecolor='gray', elinewidth=1, capsize=2, 
                 label=rf'$\langle D_\ell^{{{name}}} \rangle \pm \sigma_{{\bar D_\ell}}$')
    ax1.plot(ell[2:], Dl_th[2:], color='tab:red', label=rf'$D_{{\ell,\ \mathrm{{theory}}}}^{{{name}}}$')
    ax1.set_ylabel(rf'$D_\ell^{{{name}}}$')
    ax1.set_title(f'{name} spectrum from {nsims} realizations')
    ax1.legend(fontsize=9)
    ax1.grid(True, ls='--', alpha=0.5)

    ax2.axhline(0, color='black', lw=0.8)
    ax2.plot(ell[2:], delta, 'o', markersize=2, color='tab:blue')

    
    ax2.axhline(3, color='red', ls='--', lw=0.8)
    ax2.axhline(-3, color='red', ls='--', lw=0.8)

    
    ax2.text(ell[-1]*0.98, 3.2, r'$+3\sigma$', color='red', fontsize=8,
         ha='right', va='bottom')
    ax2.text(ell[-1]*0.98, -3.4, r'$-3\sigma$', color='red', fontsize=8,
         ha='right', va='top')


    ax2.set_xscale('linear')
    ax2.set_ylim(-5, 5)
    ax2.set_ylabel(r'$\Delta/\sigma$')
    ax2.grid(True, ls='--', alpha=0.4)

    ax3.plot(ell[2:], Dl_err[j,2:], 'o-', markersize=2,
         label=r'$\sigma_{\bar D_\ell}^{\mathrm{emp}}$')
    ax3.plot(ell[2:], cosm_var/np.sqrt(nsims), 'o-', markersize=2,
         label=r'$\sigma_{\bar D_\ell}^{\mathrm{CV}}$')
    ax3.legend(fontsize=9)
    ax3.set_xscale('linear')
    ax3.set_xlabel(r'Multipole $\ell$')
    ax3.set_ylabel(r'$\sigma_{\bar D_\ell}$')
    ax3.grid(True, ls='--', alpha=0.1)

    plt.tight_layout()
    plt.show()

def main():
    root_dir='/home/matteo/Scrivania/workspace/'
    Cls_dir=root_dir+'simulazione/'
    msk_dir=root_dir+'simulazione/'+'mask/'
    nside=64 #minimo 256 altrimenti la maschera copre tutta la mappa
    nsims=100
    lmax=2*nside
    sigma=2.0 #microK*arcmin 
    Spix=hp.nside2pixarea(nside, degrees=True)
    var=sigma*sigma/(Spix*3600.) #microK^2
    npix=hp.nside2npix(nside)
    ell=np.arange(0,lmax+1) 
    ll=ell*(ell+1)/2/np.pi
    maskT_out,maskP_out,f_skyT,f_skyP=read_msk(msk_dir,nside)
    Cls_th=read_Cls(Cls_dir,lmax,ll)
    Cls_sims=maps_noise_and_Cls(Cls_th,lmax,nside,nsims,npix,var,maskT_out,maskP_out)
    validation_spectra(Cls_th,Cls_sims,ll,ell,nsims,"BB",f_skyT,f_skyP) #inserire dentro apici TT,EE,BB, o TE a seconda dello spettro desiderato
     
main()