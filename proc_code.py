from astropy.io import fits
import numpy as np
from sklearn.tree import DecisionTreeClassifier as dct
from sklearn.ensemble import RandomForestClassifier as rfc
import sys
import time
import cat_tools as ct

# This defines the classification error. For final reporting I'll also generate
# completeness and contamination (which are differently defined).
def cls_err(f,y):
    n = len(f)
    wicor = np.where(f!=y)[0]
    num_icor = len(wicor)
    return num_icor / n

def comp_cont(f,y):
    wq = np.where(y==3)[0]
    num_true_q = len(wq)
    wq_corr = np.where((f==3)&(y==3))[0]
    comp = len(wq_corr) / num_true_q

    wqnq = np.where((y!=3)&(f==3))[0]
    cont = len(wqnq) / (len(wq_corr) + len(wqnq))

    return comp,cont


def proc_main(k,m,class_type='dt'):
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

    # I want to test whether PCA will improve my classifications for the two
    # larger data sets. The smaller one only has 11 columns, so PCA is wasted there.
    # I calculate the covariance matrix myself below and solve the eigen problem
    # for each of the data sets 1 and 0.
    # NOTE: I chose to keep 10 parameters. This is hardcoded as PCA always
    # underperforms against the raw data right now (no point in tuning this
    # parameter).
    bigsig1 = np.cov(sample_train1,rowvar=False)
    bigsig0 = np.cov(sample_train0,rowvar=False)
    lam1,w1 = np.linalg.eig(bigsig1)
    lam0,w0 = np.linalg.eig(bigsig0)
    lams1 = -1*np.sort(-lam1)
    lsort1 = np.argsort(-lam1)
    eigarr1 = np.zeros((p1,10))
    lams0 = -1*np.sort(-lam0)
    lsort0 = np.argsort(-lam0)
    eigarr0 = np.zeros((p0,10))
    w1 = w1[:,lsort1]
    w0 = w0[:,lsort0]
    # This actually makes the projection matrix.
    for i in range(10):
        eigarr1[:,i] = w1[:,i]
        eigarr0[:,i] = w0[:,i]

    # And now we project the data sets for 1 and 0 onto the PCA-reduced
    # features.
    sample_train_trial1 = sample_train1 @ eigarr1
    sample_train_trial0 = sample_train0 @ eigarr0
    sample_test_trial1 = sample_test1 @ eigarr1
    sample_test_trial0 = sample_test0 @ eigarr0

    # I wanted to test if random forest or decision tree was better. This is
    # selected using the command line argument. k defines the maximum number
    # of levels to the tree. m defines the number of models to use in random
    # forest. Both of these are also input at the command line.
    if class_type == 'rf':
        clf1 = rfc(random_state=0,max_depth=k,bootstrap=True,n_estimators=m).fit(sample_train1,label_train)
        clf0 = rfc(random_state=0,max_depth=k,bootstrap=True,n_estimators=m).fit(sample_train0,label_train)
        clf2 = rfc(random_state=0,max_depth=k,bootstrap=True,n_estimators=m).fit(sample_train2,label_train)
        clfp1 = rfc(random_state=0,max_depth=k,bootstrap=True,n_estimators=m).fit(sample_train_trial1,label_train)
        clfp0 = rfc(random_state=0,max_depth=k,bootstrap=True,n_estimators=m).fit(sample_train_trial0,label_train)
    else:
        clf1 = dct(random_state=0,max_depth=k).fit(sample_train1,label_train)
        clf0 = dct(random_state=0,max_depth=k).fit(sample_train0,label_train)
        clf2 = dct(random_state=0,max_depth=k).fit(sample_train2,label_train)
        clfp1 = dct(random_state=0,max_depth=k).fit(sample_train_trial1,label_train)
        clfp0 = dct(random_state=0,max_depth=k).fit(sample_train_trial0,label_train)

    # And now we have to do predictions and generate classification errors for
    # the various training and testing sets.
    label_train_pred1 = clf1.predict(sample_train1)
    label_train_pred0 = clf0.predict(sample_train0)
    label_train_pred2 = clf2.predict(sample_train2)
    label_train_predP1 = clfp1.predict(sample_train_trial1)
    label_train_predP0 = clfp0.predict(sample_train_trial0)

    label_test_pred1 = clf1.predict(sample_test1)
    label_test_pred0 = clf0.predict(sample_test0)
    label_test_pred2 = clf2.predict(sample_test2)
    label_test_predP1 = clfp1.predict(sample_test_trial1)
    label_test_predP0 = clfp1.predict(sample_test_trial0)

    tr_err1 = cls_err(label_train_pred1,label_train)
    tr_err0 = cls_err(label_train_pred0,label_train)
    tr_err2 = cls_err(label_train_pred2,label_train)
    tr_errp1 = cls_err(label_train_predP1,label_train)
    tr_errp0 = cls_err(label_train_predP0,label_train)

    ts_err1 = cls_err(label_test_pred1,label_test)
    ts_err0 = cls_err(label_test_pred0,label_test)
    ts_err2 = cls_err(label_test_pred2,label_test)
    ts_errp1 = cls_err(label_test_predP1,label_test)
    ts_errp0 = cls_err(label_test_predP0,label_test)

    # And report these errors to the user, along with the input parameters (in
    # case I choose to vary this with a loop).
    print('\n')
    if class_type=='rf':
        print('Random Forest Classifier, m={}, max_depth={}'.format(m,k))
    else:
        print('Decision Tree Classifier, max_depth={}'.format(k))
    print('-----------------------------------------------------------------------')
    print('|  Columns 1  |  Columns 0  |  Columns 2  |  PCA Col 1  |  PCA Col 0  |')
    print('-----------------------------------------------------------------------')
    tr_str = '|   {:.5f}   |   {:.5f}   |   {:.5f}   |   {:.5f}   |   {:.5f}   | Training Error'.format(tr_err1,tr_err0,tr_err2,tr_errp1,tr_errp0)
    ts_str = '|   {:.5f}   |   {:.5f}   |   {:.5f}   |   {:.5f}   |   {:.5f}   | Testing Error'.format(ts_err1,ts_err0,ts_err2,ts_errp1,ts_errp0)
    print(tr_str)
    print(ts_str)

    tr_err_arr = np.zeros(5,dtype='f8')
    ts_err_arr = np.zeros(5,dtype='f8')
    tr_err_arr = np.array([tr_err1,tr_err0,tr_err2,tr_errp1,tr_errp0])
    ts_err_arr = np.array([ts_err1,ts_err0,ts_err2,ts_errp1,ts_errp0])

    bst_comp,bst_cont = comp_cont(label_test_pred2,label_test)
    print('Completeness: {:.5f} | Contamination: {:.5f}'.format(bst_comp,bst_cont))

    return tr_err_arr,ts_err_arr,bst_comp,bst_cont

if __name__=='__main__':
    #cls_in = sys.argv[1] #Do you want Decision Tree (dt) or Random Forest (rf)
    depth_arr = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    mod_arr = np.array([1,5,10,15,20,25,30,35,40,45,50])
    num_depth = len(depth_arr)
    num_mod = len(mod_arr)
    #num_depth = int(sys.argv[2]) # How many layers should the trees have.
    #num_models = int(sys.argv[3]) # How many models should the forest use.

    dt_arr = np.zeros(num_depth,dtype=[('NUM_DEPTH','int16'),('TR_ERR','5float64'),
                                    ('TS_ERR','5float64'),('COMPLETENESS','float64'),
                                    ('CONTAMINATION','float64')])
    rf_arr = np.zeros(num_mod,dtype=[('NUM_DEPTH','int16'),('NUM_MODELS','i2'),
                                    ('TR_ERR','5float64'),('TS_ERR','5float64'),
                                    ('COMPLETENESS','float64'),
                                    ('CONTAMINATION','float64')])

    date_str = time.strftime('%Y%m%dT%H%M%S')
    for i in range(num_depth):
        dt_arr['NUM_DEPTH'][i] = depth_arr[i]
        dt_arr['TR_ERR'][i],dt_arr['TS_ERR'][i],dt_arr['COMPLETENESS'][i],dt_arr['CONTAMINATION'][i]=proc_main(depth_arr[i],1,class_type='dt') # And call it.

    dof_dt = fits.BinTableHDU.from_columns(dt_arr)
    dtoutname = 'dt_results_{}'.format(date_str)
    dt_write_name = ct.fet(dof_dt,dtoutname)

    for i in range(num_mod):
        rf_arr['NUM_DEPTH'][i] = 15
        rf_arr['NUM_MODELS'][i] = mod_arr[i]
        rf_arr['TR_ERR'][i],rf_arr['TS_ERR'][i],rf_arr['COMPLETENESS'][i],rf_arr['CONTAMINATION'][i]=proc_main(15,mod_arr[i],class_type='rf') # And call it.

    dof_rf = fits.BinTableHDU.from_columns(rf_arr)
    rfoutname = 'rf_results_{}'.format(date_str)
    rf_write_name = ct.fet(dof_rf,rfoutname)
    # Considerations for plotting this:
    # Vary the number of models for random forest,
    # Compare decision tree vs random forest
    # Compare completeness and contamination for num models in RF
    # Vary the max depth of the trees.
