import cat_tools as ct
import numpy as np
from astropy.io import fits
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
matplotlib.rc('text',usetex=True)
matplotlib.rc('font',size=15)

def plotter(ycol,pca):
    datadt = fits.open('../data/dt_results_20190501T122018.fits')[1].data
    datarf = fits.open('../data/rf_results_20190501T122018.fits')[1].data
    label_arr = np.array(['A','B','C','D','E'])
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4.5))
    if pca==1:
        num_cols = 5
    else:
        num_cols = 3

    for i in range(num_cols):
        ax[0].plot(datadt['NUM_DEPTH'],datadt[ycol][:,i],label='Set {}'.format(label_arr[i]))
        ax[1].plot(datarf['NUM_MODELS'],datarf[ycol][:,i],label='Set {}'.format(label_arr[i]))
    ax[0].set_xlabel('Maximum Depth')
    ax[1].set_xlabel('Number of Models')
    ax[0].set_ylabel('Classification Error')
    ax[1].set_ylabel('Classification Error')
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    plt.show()

def plot2():
    datarf = fits.open('../data/rf_results_20190501T122018.fits')[1].data
    fig2,ax2 = plt.subplots(nrows=1,ncols=2,figsize=(10,4.5))
    ax2[0].plot(datarf['NUM_MODELS'],datarf['COMPLETENESS'],color='green')
    ax2[1].plot(datarf['NUM_MODELS'],datarf['CONTAMINATION'],color='green')
    ax2[0].set_xlabel('Number of Models')
    ax2[1].set_xlabel('Number of Models')
    ax2[0].set_ylabel('Completeness')
    ax2[1].set_ylabel('Contamination')
    plt.tight_layout()
    plt.show()

def plot3():
    datarf = fits.open('../data/rf_results_20190501T122018.fits')[1].data
    fig,ax = plt.subplots(figsize=(5,4.5))
    ax.plot(datarf['NUM_MODELS'],datarf['TR_ERR'][:,2],label='Training',color='black')
    ax.plot(datarf['NUM_MODELS'],datarf['TS_ERR'][:,2],label='Testing',color='red')

    ax.set_xlabel('Number of Models')

    ax.set_ylabel('Classification Error')

    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    # This will generate side-by-side plots for Decision Tree vs. Random Forest
    # with four plots made, one with training error, one with testing error,
    # and for each of these one with and without the PCA plots (different scale).
    for i in range(2):
        plotter('TR_ERR',i)
        plotter('TS_ERR',i)

    # This generates plots for completeness vs. contamination for set C Random Forest
    plot2()
    # This generates a plot for training and testing error for set C only in Random Forest
    plot3()
