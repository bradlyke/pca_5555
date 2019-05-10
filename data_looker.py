from astropy.io import fits
import numpy as np
import sys
import time
import cat_tools as ct
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
matplotlib.rc('text',usetex=True)
matplotlib.rc('font',size=15)

# This program is just a way to look at the results of my data, how data is
# distributed, etc.
def data_look():
    # This set of columns includes everything other than identifying information
    # from the data set. Notably it includes the Legendre parameters, as well as
    # magnitude difference colors (and the my computed SNR Mean, min, and max).
    datacols1 = np.array(['LGD_PRM0','LGD_PRM1','LGD_PRM2','LGD_PRM3',
                        'SNR_MEAN','SNR_MAX','SNR_MIN',
                        'SN_MEDIAN_U','SN_MEDIAN_G','SN_MEDIAN_R','SN_MEDIAN_I',
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
    # This set of columns is the same as above, but removes the PSFMAG for the
    # five different bands. Because the color differences are made from these,
    # I have dependent columns above. The color differences are kept (rather than
    # the individual bands) as there are fewer columns that way. I also removed
    # the extinctions.
    datacols0 = np.array(['LGD_PRM0','LGD_PRM1','LGD_PRM2','LGD_PRM3',
                        'SNR_MEAN','SNR_MAX','SNR_MIN',
                        'SN_MEDIAN_U','SN_MEDIAN_G','SN_MEDIAN_R','SN_MEDIAN_I',
                        'SN_MEDIAN_Z','SN_MEDIAN_ALL','FRACNSIGMA_3',
                        'FRACNSIGHI_3','FRACNSIGLO_3','SPECTROFLUX_U','SPECTROFLUX_G','SPECTROFLUX_R',
                        'SPECTROFLUX_I','SPECTROFLUX_Z','SPECTROFLUX_IVAR_U','SPECTROFLUX_IVAR_G',
                        'SPECTROFLUX_IVAR_R','SPECTROFLUX_IVAR_I','SPECTROFLUX_IVAR_Z',
                        'SPECTROSYNFLUX_U','SPECTROSYNFLUX_G','SPECTROSYNFLUX_R','SPECTROSYNFLUX_I',
                        'SPECTROSYNFLUX_Z','SPECTROSYNFLUX_IVAR_U','SPECTROSYNFLUX_IVAR_G',
                        'SPECTROSYNFLUX_IVAR_R','SPECTROSYNFLUX_IVAR_I','SPECTROSYNFLUX_IVAR_Z',
                        'PSFFLUX_U','PSFFLUX_G','PSFFLUX_R','PSFFLUX_I','PSFFLUX_Z',
                        'PSFFLUX_IVAR_U','PSFFLUX_IVAR_G','PSFFLUX_IVAR_R',
                        'PSFFLUX_IVAR_I','PSFFLUX_IVAR_Z','FIBERFLUX_U',
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
    # This pulls in the data columns that I generated from ICA project and from
    # the color magnitude differences. Ultimately this set is just the parameters
    # to make a color-color plot and the legendre parameters encode the shape of
    # the continuum.
    datacols2 = np.array(['LGD_PRM0','LGD_PRM1','LGD_PRM2','LGD_PRM3',
                        'SNR_MEAN','SNR_MAX','SNR_MIN','UMG','GMR',
                        'RMI','IMZ'])
    print(len(datacols1),len(datacols0),len(datacols2))
    # Load the data from the fits file. It needs to be loaded into a simple
    # numpy array, however, so this block will do that based on which columns
    # are to be kept. We sample1 refers to the first set of data columns, sample0
    # refers to the set without the individual magnitudes/extinctions, and sample2
    # refers to the set that I generated from the ICA project (based on the
    # actual spectra files).
    data = fits.open('../data/spA_goodSpec_ica.fits')[1].data
    n,p1,p0,p2 = len(data),len(datacols1),len(datacols0),len(datacols2)
    sample1 = np.zeros((n,p1),dtype='f8')
    sample0 = np.zeros((n,p0),dtype='f8')
    sample2 = np.zeros((n,p2),dtype='f8')
    label = np.zeros(n,dtype='i2')
    for i in range(p1):
        sample1[:,i] = data[datacols1[i]]
    for i in range(p0):
        sample0[:,i] = data[datacols0[i]]
    for i in range(p2):
        sample2[:,i] = data[datacols2[i]]
    label[:] = data['CLASS_PERSON'][:]

    # This will break it up into training and testing sets. Since I have 3 different
    # "sets" of data, I need three separate training and testing samples.
    sample_train1 = sample1[0:int(0.75*n),:]
    sample_train0 = sample0[0:int(0.75*n),:]
    sample_train2 = sample2[0:int(0.75*n),:]
    sample_test1 = sample1[int(0.75*n):,:]
    sample_test0 = sample0[int(0.75*n):,:]
    sample_test2 = sample2[int(0.75*n):,:]
    label_train = label[0:int(0.75*n)]
    label_test = label[int(0.75*n):]
    label_train_plotter = np.chararray(len(label_train),itemsize=6,unicode=True)
    label_test_plotter = np.chararray(len(label_test),itemsize=6,unicode=True)

    # Where are the stars, galaxies, and quasars in the training set.
    wtr1 = np.where(label_train==1)[0]
    wtr3 = np.where(label_train==3)[0]
    wtr4 = np.where(label_train==4)[0]
    label_train_plotter[wtr1] = 'STAR'
    label_train_plotter[wtr3] = 'QUASAR'
    label_train_plotter[wtr4] = 'GALAXY'

    # Where are the stars, galaxies, and quasars in the testing set.
    wte1 = np.where(label_test==1)[0]
    wte3 = np.where(label_test==3)[0]
    wte4 = np.where(label_test==4)[0]
    label_test_plotter[wte1] = 'STAR'
    label_test_plotter[wte3] = 'QUASAR'
    label_test_plotter[wte4] = 'GALAXY'

    # Make da histo
    fig,ax = plt.subplots(figsize=(5,4.5))
    ax.hist(label_train_plotter,bins=3,density=True,histtype='stepfilled',color='black',alpha=0.75,label='Training')
    ax.hist(label_test_plotter,bins=3,density=True,histtype='stepfilled',color='red',alpha=0.5,label='Testing')
    ax.set_xlabel('Class Label')
    ax.set_ylabel('Frequency')
    ax.xaxis.set_tick_params('center')
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    data_look()
