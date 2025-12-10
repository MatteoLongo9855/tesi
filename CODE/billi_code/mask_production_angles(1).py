import numpy as np
import healpy as hp
import pymaster as nmt
from astropy.io import fits

#parameters
nsidepatch = 16   #resolution of "big pixel" left by the patch mask
nsideout   = 256  #resolution of the output mask
index_digits = 4  #how many digits the map index should have
apotype    = 'C1'
aposcale   = 1.
theta_values = 8 #how many theta values to test

#useful paths
rootdir             = '/marconi_work/INF21_indark/mbortola/CB/'
mask_path_in        = rootdir+'masks/COM_Mask_CMB-common-Mask-Pol_2048_R3.00.fits'
mask_path_out       = rootdir+'masks/'


#MAIN PROGRAM

#calculate number of pixels for patch resolution
npixpatch = hp.nside2npix(nsidepatch)
print("number of big pixels: ", npixpatch)

#load mask and eventually reduce resolution
mask_loaded = hp.ud_grade(hp.read_map(mask_path_in,[0] ,verbose=False),nsideout)

#create pixel masks, merge with the loaded one, apodize and save
for theta_index in range(theta_values+1):
    #create map of 0s of resolution nsidepatch
    mask_patch = np.zeros(npixpatch)

    #calculate theta and corresponding pixel index
    theta = np.pi * theta_index/theta_values
    i = hp.ang2pix(nsidepatch, theta, 0.)
    print("theta = ",theta,", i = ",i)

    #set that pixel to 1
    mask_patch[i] = 1
    
    #increase resolution to the desired value
    mask_patch = hp.ud_grade(mask_patch, nsideout)
    
    #merge patch and loaded masks
    mask = np.copy(mask_loaded)
    mask[np.logical_not(mask_patch)] = 0.
    
    #apodize final mask
    mask = nmt.mask_apodization(mask,aposcale,apotype=apotype)
    
    #save them
    hp.write_map(mask_path_out+'patch_mask_nsidepatch_'+str(nsidepatch)+'_nside_'+str(nsideout)+'_index_'+str(i).zfill(index_digits)+'.fits', mask, overwrite=True)

#END
