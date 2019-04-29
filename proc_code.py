from astropy.io import fits
import numpy as np
from sklearn.tree import DecisionTreeClassifier as dct


def proc_main():
    colnames_out = np.array(['SN_MEDIAN_U','SN_MEDIAN_G','SN_MEDIAN_R','SN_MEDIAN_I',
                            'SN_MEDIAN_Z','SN_MEDIAN_ALL','FRACNSIGMA_3',
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
                            'IMZ'])
    data = fits.open('spA_goodrec.fits')[1].data
    sample = np.zeros((len(data),len(colnames_out)),dtype='f8')
    for i in range(le(colnames_out)):
        sample[:,i] = data[colnames_out[i]]
    label[:] = data['CLASS_PERSON'][:]
    n,p = len(data),len(colnames_out)

    sample_train = sample[0:int(0.75*n),:]
    label_train = label[0:int(0.75*n)]
    sample_test = sample[int(0.75*n):,:]
    label_test = label[int(0.75*n):]

    '''
    bigsig = np.cov(sample_train)
    lam,w = np.linalg.eig(bigsig)
    lams = -1*np.sort(-lam)
    lsort = np.argsort(-lam)
    eigarr = np.zeros((10,p))
    '''
    
if __name__=='__main__':
    proc_main()
