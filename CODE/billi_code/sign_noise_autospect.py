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

def maps_noise_and_Cls(Cls_th,lmax,nside,nsims,npix,var):
    Cls_sims=np.zeros((6,lmax+1,nsims))
    m_TQU = np.zeros((3,npix),dtype=float)
    for i in range (nsims):
        m_sign=hp.synfast(cls=Cls_th ,lmax=lmax,nside=nside,pol=True,new=True)
        noisesT = np.random.normal(loc=0,scale=np.sqrt(var),size=npix)
        noisesP = np.random.normal(loc=0,scale=np.sqrt(2*var),size=2*npix)
        m_TQU=np.zeros_like(m_sign)
        m_TQU[0]=m_sign[0]+noisesT[0:npix]
        m_TQU[1]=m_sign[1]+noisesP[0:npix]
        m_TQU[2]=m_sign[2]+noisesP[npix:2*npix]

        Cls_sims[:,:,i]=hp.anafast(m_TQU,lmax=lmax,pol=True)                  
        print(i)  # per vedere da terminale l'avanzamento 
    return(Cls_sims)

def main(root_dir, nside, nsims, sigma):
   
    lmax=2*nside

    Spix=hp.nside2pixarea(nside, degrees=True)
    var=sigma*sigma/(Spix*3600.) #microK^2

    npix=hp.nside2npix(nside)

    ell=np.arange(0,lmax+1) 
    ll=ell*(ell+1)/2/np.pi
    Cls_th=read_Cls(root_dir,lmax,ll)
    Cls_sims=maps_noise_and_Cls(Cls_th,lmax,nside,nsims,npix,var)
    np.save(root_dir+'cls_sim.npy',Cls_sims)
    
main(root_dir='valid_auto_spectra/',
    nside=128,
    nsims=1000,
    sigma=2.0)