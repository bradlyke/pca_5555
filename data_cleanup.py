from astropy.io import fits
import numpy as np
import cat_tools as ct
import sys

def dclean(infile,seq_file):
    tdata = fits.open(infile)[1].data
    sdata = fits.open(seq_file)[1].data
    # The data selection and column cleanup goes here.
    colnames_out = np.array(['PLATE','MJD','FIBERID','RA','DEC','OBJTYPE','Z','Z_ERR',
                            'SN_MEDIAN_U','SN_MEDIAN_G','SN_MEDIAN_R','SN_MEDIAN_I',
                            'SN_MEDIAN_Z','SN_MEDIAN_ALL','OBJC_TYPE','OBJC_PROB_PSF','FRACNSIGMA_3',
                            'FRACNSIGHI_3','FRACNSIGLO_3','SPECTROFLUX_U','SPECTROFLUX_G','SPECTROFLUX_R',
                            'SPECTROFLUX_I','SPECTROFLUX_Z','SPECTROFLUX_IVAR_U','SPECTROFLUX_IVAR_G',
                            'SPECTROFLUX_IVAR_R','SPECTROFLUX_IVAR_I','SPECTROFLUX_IVAR_Z',
                            'SPECTROSYNFLUX_U','SPECTROSYNFLUX_G','SPECTROSYNFLUX_R','SPECTROSYNFLUX_I',
                            'SPECTROSYNFLUX_Z','SPECTROSYNFLUX_IVAR_U','SPECTROSYNFLUX_IVAR_G',
                            'SPECTROSYNFLUX_IVAR_R','SPECTROSYNFLUX_IVAR_I','SPECTROSYNFLUX_IVAR_Z',
                            'EXTINCTION_U','EXTINCTION_G','EXTINCTION_R','EXTINCTION_I','EXTINCTION_Z',
                            'PSFFLUX_U','PSFFLUX_G','PSFFLUX_R','PSFFLUX_I','PSFFLUX_Z',
                            'PSFFLUX_IVAR_U','PSFFLUX_IVAR_G','PSFFLUX_IVAR_R',
                            'PSFFLUX_IVAR_I','PSFFLUX_IVAR_Z','PSFMAG_U','PSFMAG_G',
                            'PSFMAG_R','PSFMAG_I','PSFMAG_Z','PSFMAGERR_U','PSFMAGERR_G',
                            'PSFMAGERR_R','PSFMAGERR_I','PSFMAGERR_Z','FIBERFLUX_U',
                            'FIBERFLUX_G','FIBERFLUX_R','FIBERFLUX_I','FIBERFLUX_Z',
                            'FIBERFLUX_IVAR_U','FIBERFLUX_IVAR_G','FIBERFLUX_IVAR_R',
                            'FIBERFLUX_IVAR_I','FIBERFLUX_IVAR_Z','FIBERMAG_U','FIBERMAG_G',
                            'FIBERMAG_R','FIBERMAG_I','FIBERMAG_Z','FIBERMAGERR_U',
                            'FIBERMAGERR_G','FIBERMAGERR_R','FIBERMAGERR_I','FIBERMAGERR_Z',
                            'PETROTHETA_U','PETROTHETA_G','PETROTHETA_R',
                            'PETROTHETA_I','PETROTHETA_Z','PETROTHETAERR_U',
                            'PETROTHETAERR_G','PETROTHETAERR_R','PETROTHETAERR_I',
                            'PETROTHETAERR_Z','PETROTH50_U','PETROTH50_G','PETROTH50_R',
                            'PETROTH50_I','PETROTH50_Z','PETROTH50ERR_U','PETROTH50ERR_G',
                            'PETROTH50ERR_R','PETROTH50ERR_I','PETROTH50ERR_Z','Q_U',
                            'Q_G','Q_R','Q_I','Q_Z','QERR_U','QERR_G','QERR_R','QERR_I',
                            'QERR_Z','U_U','U_G','U_R','U_I','U_Z','UERR_U','UERR_G',
                            'UERR_R','UERR_I','UERR_Z','M_E1_U','M_E1_G','M_E1_R','M_E1_I',
                            'M_E1_Z','M_E2_U','M_E2_G','M_E2_R','M_E2_I','M_E2_Z',
                            'M_E1E1ERR_U','M_E1E1ERR_G','M_E1E1ERR_R','M_E1E1ERR_I',
                            'M_E1E1ERR_Z','M_E1E2ERR_U','M_E1E2ERR_G','M_E1E2ERR_R',
                            'M_E1E2ERR_I','M_E1E2ERR_Z','M_E2E2ERR_U','M_E2E2ERR_G',
                            'M_E2E2ERR_R','M_E2E2ERR_I','M_E2E2ERR_Z','UMG','GMR','RMI',
                            'IMZ','CLASS_PERSON','ZCONF'])
    colforms_out = np.array(['i4','i4','i4','f8','f8','U16','f8','f8','f8','f8',
                            'f8','f8','i4','f8','f8','f8','f8',
                            'f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8',
                            'f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8',
                            'f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8',
                            'f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8',
                            'f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8',
                            'f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8',
                            'f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8',
                            'f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8',
                            'f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8',
                            'f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8',
                            'f8','f8','f8','f8','f8','f8','i2','i2'])
    print(len(colnames_out))
    print(len(colforms_out))

    cols_to_copy = np.array(['PLATE','MJD','FIBERID','RA','DEC','OBJTYPE','Z','Z_ERR',
                            'SN_MEDIAN_ALL','OBJC_TYPE','OBJC_PROB_PSF'])
    arr_to_break = np.array(['SN_MEDIAN','SPECTROFLUX','SPECTROFLUX_IVAR','SPECTROSYNFLUX',
                            'SPECTROSYNFLUX_IVAR','EXTINCTION','PSFFLUX','PSFFLUX_IVAR',
                            'PSFMAG','PSFMAGERR','FIBERFLUX','FIBERFLUX_IVAR',
                            'FIBERMAG','FIBERMAGERR','PETROTHETA','PETROTHETAERR',
                            'PETROTH50','PETROTH50ERR','Q','QERR','U','UERR','M_E1',
                            'M_E2','M_E1E1ERR','M_E1E2ERR','M_E2E2ERR'])
    color_arr = np.array(['U','G','R','I','Z'])
    data_out = np.zeros(len(tdata),dtype={'names':colnames_out,'formats':colforms_out})

    # Some columns get copied straight across.
    for cname in cols_to_copy:
        data_out[cname] = tdata[cname]

    # Many columns are stored as 5-element vectors for UGRIZ colors. We need to
    # break these into separate columns
    for cname in arr_to_break:
        for i in range(5):
            out_col = cname + '_{}'.format(color_arr[i])
            data_out[out_col] = tdata[cname][:,i]


    data_out['FRACNSIGMA_3'] = tdata['FRACNSIGMA'][:,2]
    data_out['FRACNSIGHI_3'] = tdata['FRACNSIGHI'][:,2]
    data_out['FRACNSIGLO_3'] = tdata['FRACNSIGLO'][:,2]
    data_out['UMG'] = (data_out['PSFMAG_U'] - data_out['EXTINCTION_U']) - (data_out['PSFMAG_G'] - data_out['EXTINCTION_G'])
    data_out['GMR'] = (data_out['PSFMAG_G'] - data_out['EXTINCTION_G']) - (data_out['PSFMAG_R'] - data_out['EXTINCTION_R'])
    data_out['RMI'] = (data_out['PSFMAG_R'] - data_out['EXTINCTION_R']) - (data_out['PSFMAG_I'] - data_out['EXTINCTION_I'])
    data_out['IMZ'] = (data_out['PSFMAG_I'] - data_out['EXTINCTION_I']) - (data_out['PSFMAG_Z'] - data_out['EXTINCTION_Z'])

    dout_hash,seq_hash = ct.mk_hash(data_out),ct.mk_hash(sdata)
    dout_args,seq_args = ct.rec_match_srt(dout_hash,seq_hash)
    data_out['CLASS_PERSON'][dout_args] = sdata['CLASS_PERSON'][seq_args]
    data_out['ZCONF'][dout_args] = sdata['Z_CONF_PERSON'][seq_args]

    outfile_name = '../data/spA_breaker_590'
    dof = fits.BinTableHDU.from_columns(data_out)
    ct.fet(dof,outfile_name,quiet=True)


if __name__=='__main__':
    spAll_file = sys.argv[1]
    sequels_file = sys.argv[2]
    dclean(spAll_file,sequels_file)
