import numpy as np
import healpy as hp
import pymaster as nmt
import argparse as ap
import os
import sys

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

opj = os.path.join

# MAIN PROGRAM

def load_masks(maskT=None, 
    maskT_path=None,
    maskP=None, 
    maskP_path=None,
    aposcale=0.,
    apotype=None,
    apodization=False,
    apo_tag=None):

    # load mask and apodize the mask with processor 0, then broadcast it
    if rank == 0:
        if apodization:
            maskT = nmt.mask_apodization(hp.read_map(maskT_path,[0]), aposcale, apotype=apotype)
            maskP = nmt.mask_apodization(hp.read_map(maskP_path,[0]), aposcale, apotype=apotype)
        else:
            maskT = hp.read_map(maskT_path+'.fits', [0])
            maskP = hp.read_map(maskP_path+'.fits', [0])
    maskT = comm.bcast(maskT, root=0)
    maskP = comm.bcast(maskP, root=0)
    return maskT,maskP


def create_workspace(lmax=2047, 
    deltal=5, 
    purify_e=False,
    purify_b=True,
    fwhm=5.,
    nside=2048,
    maskT=None,
    maskP=None,
    workspace_path=None,
    tag_map_signal=None,
    tag_map_noise_oe1=None,
    tag_map_noise_oe2=None,
    isimstart=300,
    MC_signal_path=None,
    MC_noise_path=None,
    gtag=None,
    **kwargs):

    # Initialize binning scheme with "deltal" ells per bandpower
    b = nmt.NmtBin.from_lmax_linear(lmax, deltal, is_Dell=True)
    ell_arr = b.get_effective_ells()

    # Beam
    bl = hp.gauss_beam(np.deg2rad(fwhm/60.0), lmax=lmax, pol=True)
    wpix = hp.pixwin(nside, pol=True, lmax=lmax)

    # Load simulated map
    signal     = hp.read_map(opj(MC_signal_path, tag_map_signal+'_00000_raw.fits'), [0,1,2])
    noise_oe1  = hp.read_map(opj(MC_noise_path,  tag_map_noise_oe1+'_0000_raw.fits'), [0,1,2])
    noise_oe2  = hp.read_map(opj(MC_noise_path,  tag_map_noise_oe2+'_00000_raw.fits'), [0,1,2])
    cmb_oe1    = signal + noise_oe1
    cmb_oe2    = signal + noise_oe2

    # Create fields
    f_0_oe1 = nmt.NmtField(maskT,[cmb_oe1[0]] ,purify_e=purify_e,purify_b=purify_b,beam=bl[:,0]*wpix[0],lmax=lmax)
    f_0_oe2 = nmt.NmtField(maskT,[cmb_oe2[0]] ,purify_e=purify_e,purify_b=purify_b,beam=bl[:,0]*wpix[0],lmax=lmax)
    f_2_oe1 = nmt.NmtField(maskP,[cmb_oe1[1], cmb_oe1[2]],purify_e=purify_e,purify_b=purify_b,beam=bl[:,1]*wpix[1],lmax=lmax)
    f_2_oe2 = nmt.NmtField(maskP,[cmb_oe2[1], cmb_oe2[2]],purify_e=purify_e,purify_b=purify_b,beam=bl[:,1]*wpix[1],lmax=lmax)

    # Create Workspace
    w_00 = nmt.NmtWorkspace()
    w_02 = nmt.NmtWorkspace()
    w_20 = nmt.NmtWorkspace()
    w_22 = nmt.NmtWorkspace()

    # Compute matrices
    w_00.compute_coupling_matrix(f_0_oe1, f_0_oe2, b)
    w_02.compute_coupling_matrix(f_0_oe1, f_2_oe2, b)
    w_20.compute_coupling_matrix(f_2_oe1, f_0_oe2, b)
    w_22.compute_coupling_matrix(f_2_oe1, f_2_oe2, b)

    # Save workspace
    w_00.write_to(workspace_path+'w_00_deltaell_'+str(deltal).zfill(1)+'.fits')
    w_02.write_to(workspace_path+'w_02_deltaell_'+str(deltal).zfill(1)+'.fits')
    w_20.write_to(workspace_path+'w_20_deltaell_'+str(deltal).zfill(1)+'.fits')
    w_22.write_to(workspace_path+'w_22_deltaell_'+str(deltal).zfill(1)+'.fits')


def calculate_spectrum_pol(lmax=2047, 
    deltal=5, 
    purify_e=False,
    purify_b=True,
    fwhm = 5.,
    nside = 2048,
    nsims = 10,
    maskT=None,
    maskP=None,
    workspace_path=None,
    tag_map_signal=None,
    tag_map_noise_oe1=None,
    tag_map_noise_oe2=None,
    isimstart = 300,
    outdir = None,
    MC_signal_path=None,
    MC_noise_path=None,
    tag_type_spectra=None,
    gtag=None,
    **kwargs):

    # Initialize binning scheme with "deltal" ells per bandpower
    b = nmt.NmtBin.from_lmax_linear(lmax, deltal, is_Dell=True)
    ell_arr = b.get_effective_ells()

    # Gaussian beam
    bl = hp.gauss_beam(np.deg2rad(fwhm/60.0), lmax=lmax, pol=True)
    wpix = hp.pixwin(nside, pol=True, lmax=lmax)

    # Compute spectra and save them
    if(rank==0):
        print("Number of spectra calculated:")

    # Sim per processor
    simperproc = int(np.ceil(nsims/nproc))

    # Read workspace
    w_00 = nmt.NmtWorkspace()
    w_02 = nmt.NmtWorkspace()
    w_20 = nmt.NmtWorkspace()
    w_22 = nmt.NmtWorkspace()
    w_00.read_from(workspace_path+'w_00_deltaell_'+str(deltal).zfill(1)+'.fits')
    w_02.read_from(workspace_path+'w_02_deltaell_'+str(deltal).zfill(1)+'.fits')
    w_20.read_from(workspace_path+'w_20_deltaell_'+str(deltal).zfill(1)+'.fits')
    w_22.read_from(workspace_path+'w_22_deltaell_'+str(deltal).zfill(1)+'.fits')


    for isim in np.arange(rank*simperproc, (rank+1)*simperproc,1):
        print("nproc,rank,simperproc",nproc,rank,simperproc)
        if(isim < nsims):
            if(not(isim+1)%10):
                print(isim+1,"/",nsims)

            # Load simulated map
            isign = isimstart+isim
            if isign < 300:
                inoise = isign
            else:
                if isign < 600:
                    inoise = isign - 300
                else:
                    inoise = isign - 600

            signal     = hp.read_map(opj(MC_signal_path, tag_map_signal+str(isign).zfill(5)+'_raw.fits'), [0,1,2])
            noise_oe1  = hp.read_map(opj(MC_noise_path,  tag_map_noise_oe1+str(inoise).zfill(5)+'_raw.fits'), [0,1,2])
            noise_oe2  = hp.read_map(opj(MC_noise_path,  tag_map_noise_oe2+str(inoise).zfill(5)+'_raw.fits'), [0,1,2])
            cmb_oe1    = signal + noise_oe1
            cmb_oe2    = signal + noise_oe2

            # Create fields
            f_0_oe1 = nmt.NmtField(maskT,[cmb_oe1[0]] ,purify_e=purify_e,purify_b=purify_b,beam=bl[:,0]*wpix[0],lmax=lmax)
            f_0_oe2 = nmt.NmtField(maskT,[cmb_oe2[0]] ,purify_e=purify_e,purify_b=purify_b,beam=bl[:,0]*wpix[0],lmax=lmax)
            f_2_oe1 = nmt.NmtField(maskP,[cmb_oe1[1], cmb_oe1[2]],purify_e=purify_e,purify_b=purify_b,beam=bl[:,1]*wpix[1],lmax=lmax)
            f_2_oe2 = nmt.NmtField(maskP,[cmb_oe2[1], cmb_oe2[2]],purify_e=purify_e,purify_b=purify_b,beam=bl[:,1]*wpix[1],lmax=lmax)

            # Compute spectrum
            cl_coupled_00 = nmt.compute_coupled_cell(f_0_oe1, f_0_oe2)
            cl_coupled_02 = nmt.compute_coupled_cell(f_0_oe1, f_2_oe2)
            cl_coupled_20 = nmt.compute_coupled_cell(f_2_oe1, f_0_oe2)
            cl_coupled_22 = nmt.compute_coupled_cell(f_2_oe1, f_2_oe2)           

            # Decouple spectra
            cl_00 = w_00.decouple_cell(cl_coupled_00)
            cl_02 = w_02.decouple_cell(cl_coupled_02)
            cl_20 = w_02.decouple_cell(cl_coupled_20)
            cl_22 = w_22.decouple_cell(cl_coupled_22)

            np.savetxt(opj(outdir, tag_type_spectra+'spectra_isim'+str(isimstart+isim).zfill(4)+'.txt'),
                   np.column_stack([ell_arr,cl_00[0],cl_22[0],cl_22[3],(cl_02[0]+cl_20[0])/2,(cl_02[1]+cl_20[1])/2,(cl_22[1]+cl_22[2])/2]),
                   header='#l TT EE BB TE TB EB')
    

def main():
    # Process commandline input
    parser = ap.ArgumentParser(formatter_class=ap.ArgumentDefaultsHelpFormatter)

    # I/O:
    parser.add_argument('--system_name', action='store', dest='system_name', type=str, default='nersc_mb')

    # Run options:
    parser.add_argument('--nside', action='store', dest='nside', default=2048, type=int, help='Nside map')
    parser.add_argument('--lmax', action='store', dest='lmax', default=2048, type=int, help='lamx map')
    parser.add_argument('--fwhm', action='store', dest='fwhm', default=5., type=float, help="Beam FWHM in arcmin")
    parser.add_argument('--deltal', action='store', dest='deltal', default=5, type=int, help="Multipoles for bin")
    parser.add_argument('--nsims', action='store', dest='nsims', default=10, type=int, help="Number of simulations")
    parser.add_argument('--apotype', action='store', dest='apotype', default='C1', type=str, help="Type of apodization")
    parser.add_argument('--aposcale', action='store', dest='aposcale', default=5.0, type=float, help="Scale of apodization in degree")
    parser.add_argument('--apodization', action='store', dest='apodization', default=False, type=bool, help="Apodization of the mask")

    # General options:
    parser.add_argument('--isimstart', action='store', dest='isimstart', default=0, type=int, help='tag in the name of sims which identifies the starting number')
    
    # Choosing operation:
    parser.add_argument('--load_masks', action='store_true', dest='load_masks', default=True, help='Function which loads the masks')
    parser.add_argument('--create_workspace', action='store_true', dest='create_workspace', default=True, help='Function which creates the workspace')
    parser.add_argument('--calculate_spectrum_pol', action='store_true', dest='calculate_spectrum_pol', default=True, help='Function which compute the APS')

    # Analysis arguments: 
    parser.add_argument('--maskT', action='store', dest='maskT', default=None, type=float, help="Type of mask")
    parser.add_argument('--maskP', action='store', dest='maskP', default=None, type=float, help="Type of mask")
    parser.add_argument('--purify_e', action='store', dest='purify_e', default=False, type=bool, help="Purification of E-mode")
    parser.add_argument('--purify_b', action='store', dest='purify_b', default=False, type=bool, help="Purification of B-mode")
    
    # Tags arguments: S
    parser.add_argument('--maskT_tag', action='store', dest='maskT_tag', default='COM_Mask_CMB-common-Mask-Int_2048_R3.00', type=str, help="Tag of mask")
    parser.add_argument('--maskP_tag', action='store', dest='maskP_tag', default='COM_Mask_CMB-common-Mask-Pol_2048_R3.00', type=str, help="Tag of mask")
    parser.add_argument('--pipeline',  action='store', dest='pipeline',  default='sevem', type=str, help="Planck pipeline")
    
    args = parser.parse_args()
    tag_map_signal    = 'dx12_v3_'+args.pipeline+'_cmb_mc_'
    tag_map_noise_oe1 = 'dx12_v3_'+args.pipeline+'_noise_oe1_mc_'
    tag_map_noise_oe2 = 'dx12_v3_'+args.pipeline+'_noise_oe2_mc_'

    apo_tag='_apo'+str(args.apotype)+'scale'+str(args.aposcale)
    gtag= 'nside_'+str(args.nside).zfill(3)
    unify_tag = str(gtag+'/')

    if args.system_name == 'nersc_mb':
        base_dir  = '/global/homes/m/mbilli00/'
        MC_path   = opj(base_dir,'maps/nside_'+str(args.nside))
        MC_signal_path = '/global/cfs/projectdirs/cmb/data/planck2018/ffp10/compsep/mc_cmb/',
        MC_noise_path  = '/pscratch/sd/m/mbilli00/Planck_maps/PR3/MC_composep_noise_oe/'+args.pipeline+'/'
        mask_path = opj(base_dir,'input_maps_ECLIPSE/input_mask/')
        outdir    = opj(base_dir,'Planck_power_spectra/namaster_spectra/')

    elif args.system_name == 'local_mb':
        base_dir  = '/home/matteobilli/Documenti/Santander_Project/work_project/CB_IPPEI/CODE_and_RESULTS/CODE/Planck-like_sims_spectra'
        MC_path   = opj(base_dir,'maps/nside_'+str(args.nside))
        MC_signal_path=MC_path,
        MC_noise_path=MC_path,
        mask_path = opj(base_dir,'input_mask/')
        outdir    = opj(base_dir,'namaster_spectra/')

    if(args.purify_e and args.purify_b):
        tag_type_spectra = unify_tag+'spectra_pure_EB/'
        workspace_path = opj(base_dir,unify_tag+'workspaces_pure_EB/')
    else:
        if(args.purify_e):
            tag_type_spectra = unify_tag+'spectra_pure_E/'
            workspace_path = opj(base_dir,'namaster_spectra/'+unify_tag+'workspaces_pure_E/')
        else:
            if(args.purify_b):
                tag_type_spectra = unify_tag+'spectra_pure_B/'
                workspace_path = opj(base_dir,'namaster_spectra/'+unify_tag+'workspaces_pure_B/')
            else:
                tag_type_spectra = unify_tag+'spectra_master/'
                workspace_path = opj(base_dir,'namaster_spectra/'+unify_tag+'workspaces_master/')

    if (rank ==0):
        print('Begin program')

    if args.load_masks:
        args.maskT, args.maskP = load_masks(maskT = args.maskT, maskT_path=mask_path+args.maskT_tag, 
                                            maskP = args.maskP, maskP_path=mask_path+args.maskP_tag, 
                                            aposcale=args.aposcale, apotype=args.apotype, 
                                            apodization=args.apodization, apo_tag=apo_tag)
    if (rank ==0):
        print ('load_masks ok')

        if args.create_workspace:
            workspace_opts = dict(lmax=args.lmax, 
                                  deltal=args.deltal, 
                                  purify_e=args.purify_e,
                                  purify_b=args.purify_b,
                                  fwhm =args.fwhm,
                                  nside =args.nside,
                                  maskT=args.maskT,
                                  maskP=args.maskP,
                                  workspace_path=workspace_path,
                                  tag_map_signal=tag_map_signal,
                                  tag_map_noise_oe1=tag_map_noise_oe1,
                                  tag_map_noise_oe2=tag_map_noise_oe2,
                                  isimstart =args.isimstart,
                                  MC_signal_path=MC_signal_path, 
                                  MC_noise_path=MC_noise_path,                                  
                                  gtag=gtag)

            create_workspace(**workspace_opts)
            print('create_workspace ok')

    comm.Barrier()

    if args.calculate_spectrum_pol:
        calcAPS_opts = dict(lmax=args.lmax, 
                            deltal=args.deltal, 
                            purify_e=args.purify_e,
                            purify_b=args.purify_b,
                            fwhm =args.fwhm,
                            nside =args.nside,
                            nsims =args.nsims,
                            maskT=args.maskT,
                            maskP=args.maskP,
                            workspace_path=workspace_path,
                            tag_map_signal=tag_map_signal,
                            tag_map_noise_oe1=tag_map_noise_oe1,
                            tag_map_noise_oe2=tag_map_noise_oe2,
                            isimstart =args.isimstart,
                            outdir =outdir,
                            MC_signal_path=None,
                            MC_noise_path=None,                            
                            tag_type_spectra=tag_type_spectra,
                            gtag=gtag)

        calculate_spectrum_pol(**calcAPS_opts)

        
        if (rank ==0):
            print ('end')

        return

if __name__ == '__main__':
    main()