import numpy as np
import healpy as hp
import matplotlib
matplotlib.use('Agg') # Forza il backend "non-interattivo" (salva solo su file)
import matplotlib.pyplot as plt
import pymaster as nmt

def create_mask_patch(mask_in_dir,mask_patch_dir,nsideout,nsidepatch,npix_patch,aposcale,apotype,plots_dir,output_maps_dir):
        patch_quality_map = np.full(npix_patch, hp.UNSEEN)
        mask_galT = hp.ud_grade(hp.read_map(mask_in_dir + 'COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits', field=0, dtype=np.float64),nsideout, dtype= np.float64)
        mask_galP = hp.ud_grade(hp.read_map(mask_in_dir + 'COM_Mask_CMB-common-Mask-Pol_2048_R3.00.fits', field=0, dtype=np.float64),nsideout, dtype= np.float64)
        
        mask_galT[mask_galT < 0.5] = 0.0
        mask_galT[mask_galT >= 0.5] = 1.0
        mask_galP[mask_galP < 0.5] = 0.0
        mask_galP[mask_galP >= 0.5] = 1.0
        k = 0
        mask_beta = hp.ud_grade(hp.read_map(output_maps_dir + 'mask_beta.fits', 
                                        field=0, dtype=np.float64),nsideout, dtype= np.float64)
        mask_beta = nmt.mask_apodization(mask_beta, aposcale, apotype='C2')
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
                #hp.write_map(mask_patch_dir+'patch_maskT_nside_'+str(nsidepatch)+
                #              '_index_'+str(k).zfill(4)+'.fits', maskT, overwrite=True, dtype= np.float64)
                
                maskP = total_maskP
                patch_quality_map[i] = (fractionP + fractionT)/2 * 100
                maskP = nmt.mask_apodization(maskP, aposcale, apotype=apotype)
                hp.mollview(maskP*mask_beta, 
                            title=f"Active Patch k={k} with {((fractionP+fractionT)/2)*100:.2f} average active pixel", 
                            cbar=False, 
                            hold=True)
                plt.savefig(plots_dir +'quality_maps/patches/' + 'patch_index'+str(k).zfill(4) + '.png')
                plt.close()
                #hp.write_map(mask_patch_dir+'patch_maskP_nside_'+str(nsidepatch)+
                #'_index_'+str(k).zfill(4)+'.fits', maskP, overwrite=True, dtype= np.float64)
                k = k + 1

            else:
                 patch_quality_map[i] = hp.UNSEEN 

        #hp.mollview(patch_quality_map, title="%Average ctive Pixels", unit="%", cmap='viridis', min=50, max=100)
        #plt.savefig(plots_dir +'quality_maps/' + 'patch_preview' + '.png')
        #plt.close()
        #np.savetxt(mask_patch_dir+'active_patches_nside_'+str(nsidepatch)+
        #                      '.txt', [k], fmt='%d')

def main ():
    root_dir='/home/matteo/Scrivania/workspace/'
    mask_in_dir = root_dir + 'simulazione/mask/'
    plots_dir   = root_dir + 'simulazione/plots/namaster_patch/'
    mask_patch_dir  = root_dir + 'simulazione/mask/mask_patch/'
    output_maps_dir = root_dir + 'simulazione/maps_beta/'
    nsideout= 512
    nsidepatch=8
    npix_map= hp.nside2npix(nsideout)
    npix_patch= hp.nside2npix(nsidepatch)
    Spix = hp.nside2pixarea(nsideout, degrees=True)
    fwhm = 2.4 * np.sqrt(Spix)
    aposcale = fwhm
    apotype = "C2"
    create_mask_patch(mask_in_dir,mask_patch_dir,nsideout,nsidepatch,npix_patch,aposcale,apotype,plots_dir,output_maps_dir)


main()
    

