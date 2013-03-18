import matplotlib.pyplot as plt
from flipper import liteMap
import numpy as np
import sys, pickle, os, time
try:
    from utils.utilities import initializeProgressBar
except:
    pass
 
def __stringToFunction(astr):
    """
    @brief given a string containing the name of a function, convert 
    it to a function

    @param astr: string of function name
    """
    module, _, function = astr.rpartition('.')
    if module:
        __import__(module)
        mod = sys.modules[module]
    else:
        mod = sys.modules['__main__']

    return getattr(mod, function)
    
def estimateBinCorrelations(dataDir, tag='', maxBinSep=5):
    """
    @brief estimate the correlations between bins using all data files that
    satisfy tag
    """
    
    # store the individual statistics and errors
    stats = []
    errs  = []

    # read in all the individual statistics 
    for f in os.listdir(dataDir):

        # consider only the data files
        if "dat" not in f:
           continue

        # consider only those files containing 'tag'
        if tag not in f:
           continue

        x, stat, err = np.loadtxt('%s/%s' %(dataDir, f), unpack=True)
        stats.append(stat*1e10)
        errs.append(err*1e10)

    stats      = np.array(stats)
    errs = np.array(errs)
    stats_mean = np.mean(stats, axis=0)
    errs_mean = np.mean(errs, axis=0)
    
    binSeps = range(0, maxBinSep+1)
    nBins = len(stats_mean)
    nStats = len(stats)

    # generate simulated data
    sim_data = []
    for i in range(nStats):
        
        sim_data.append(np.array([np.random.normal(loc=stats_mean[i], scale=errs_mean[i]) for i in range(len(errs_mean)) ]))
    
    bin_corrs = []


    #stats = np.array(sim_data)
    #stats_mean = np.mean(stats, axis=0)
    
    # loop over all bin separations
    for binSep in binSeps:

        # compute average of (p_i - <p_i>)(p_i+binSep - <p_i+binSep>)
        corrs_i = []
        for i in range(nBins-binSep):
            N = 0
            sumStat = 0.
            for j in range(nStats):   
                sumStat += (stats[j, i] - stats_mean[i])*(stats[j, i+binSep] - stats_mean[i+binSep]) 
                N += 1

            corrs_i.append(sumStat/N)

        # compute average of (p_i - <p_i>)(p_i - <p_i>)
        vars0_i = []    
        for i in range(nBins-binSep):
            N = 0
            sumStat = 0.
            for j in range(nStats):    

                sumStat += (stats[j, i] - stats_mean[i])*(stats[j, i] - stats_mean[i]) 
                N += 1

            vars0_i.append(sumStat/N)

        # compute average of (p_i+binSep - <p_i+binSep>)(p_i+binSep - <p_i+binSep>)
        vars1_i = []
        for i in range(nBins-binSep):
            N = 0
            sumStat = 0.
            for j in range(nStats):

                sumStat += (stats[j, i] - stats_mean[i])*(stats[j, i] - stats_mean[i]) 
                N += 1

            vars1_i.append(sumStat/N)    

        corr = np.mean(corrs_i)
        norm = 1. / (np.sqrt(np.mean(vars0_i))*np.sqrt(np.mean(vars1_i)))
        corr *= norm
        bin_corrs.append(corr)

    bin_corrs = np.array(bin_corrs)

    return bin_corrs
    
def updateParams(paramsToUpdate, params, num):
    """
    @brief function to update the parameters dictionary
    
    @param paramsToUpdate: dictionary telling us which key/val pairs
    in params to update (dict)
    @param params: original dictionary of parameters (dict)
    @param num: the number to insert into the new values (int)
    """
    
    # loop over each parameter name and new value in the dictionary
    for key, val in paramsToUpdate.items():
        # update each element of a list
        if type(val) == list:
            for i in range(len(val)):
                params[key][i] = val[i] %num
        else:
            params[key] = val %num

    return params

    
def topHatFilter(ell, ell_lcut=75., ell_hcut=500., exp_width=50.):
    """
    @brief return a top hat ell-space filter, which falls off exponentially
    at high ell end and rises as sin^2 at low ell end
    
    @param ell: the multipole numbers (array)
    @param ell_lcut: ell value to fall off at lo-ell end (float)
    @param ell_hcut: ell value to fall off at hi-ell end (float)
    @param exp_width: the width of the exponential cutoff (float)
    
    @return Fell: the filter in ell space 
    """
    # make sure ell is an array
    if type(ell) == list:
        ell = np.array(ell)
        
    Fell = ell*0. + 1.0
    
    # make the high ell exponential cutoff
    inds = np.where(ell >= ell_hcut)
    Fell[inds] *= np.exp(-(ell[inds] - ell_hcut)**2/(2*exp_width**2))
    
    # make the low ell fall off as sin^2
    inds2 = np.where(ell < ell_lcut)
    Fell[inds2] = (np.sin(ell[inds2]/(ell_lcut-1)*np.pi/2.))**2

  
    return Fell 
    
def gaussianFilter(ell, scale=4.0):
    """
    @brief return the ell-space version of a real-space Gaussian 
    filter of FWHM scale (in arcminutes)
    
    @param ell: the multipole numbers (array)
    @param scale: the FWHM of the real space filter in arcminutes (float)
    """
    # make sure ell is an array
    if type(ell) == list:
        ell = np.array(ell)
        
    theta_fwhm = scale/60.*np.pi/180.0
    Fell = np.exp(- ell**2 * theta_fwhm**2 / (16. * np.log(2.) ))
    
    return Fell
    
    
def plot1DRealCorr(pickleFile, currFig=None, ylim=None, data=None, xshift=0., rerrs=None):
    """
    @brief bin and plot the 1D realspace correlation from a 2D array
    
    @param pickleFile: name of pickle file containing tuple (theta, corr)
    @param currFig: the figure to plot to
    @param ylim: set the y limits to ylim=(y_min, y_max)
    @param data: data tuple of (theta, corr)
    @param xshift: shift the x axis by xshift (float)
    @param rerrs: errors from random correlations to use (list)
    
    @return: current figure, thetas, correlation, error
    """
    # annular averaging 
    bins = [[0.,10.],[10.,20],[20.,30.],[30.,40.],[40.,50.],[50.,60.],[60.,80.]] #arcmin   

    # read in the data
    if data is not None:
        theta = data[0]
        corr = data[1]
        
    else:
        d = pickle.load(open(pickleFile))
        theta = d[0]
        corr = d[1]
    
    r = []
    mean = []
    std = []
    # bin the 2D correlation array
    for bin in bins:
        inds = np.where((theta < bin[1]) & (theta > bin[0]))
        mean +=[ np.mean(corr[inds])]
        std += [np.std(corr[inds])/np.sqrt(len(inds[0])*1.0)]
        r += [theta[inds].mean()]

    mean = np.array(mean)
    std = np.array(std)
    n = mean.max()
    
    # check that rerrs is correct length, if provided
    if rerrs is not None:
        assert(len(mean) == len(rerrs))
        std = rerrs
    
    # set up the matplotlib axes
    plt.rcParams['figure.subplot.left'] = 0.17
    if currFig is not None:
        ax = currFig.get_axes()[0]
    else:
        ax = plt.gca()
    
    # plot the correlation with std dev within the bin as the error
    ax.errorbar(r+xshift, mean, std)
    ax.axhline(y=0, c='k')
    
    # label the axes
    ax.set_xlabel(r"$\mathrm{\theta \ (arcminutes)}$", fontsize=16)
    ax.set_ylabel(r"$\mathrm{\xi(\theta)}$", fontsize=16)

    if ylim is not None:
        plt.set_ylim(*ylim)
    
    return plt.gcf(), r, mean, std
    
def plot1DFourierCorr(dataFile, currFig=None, ylim=None, xshift=0., rerrs=None):
    """
    @brief bin and plot the 1D fourier space correlation from a data file

    @param dataFile: name of data file (str)
    @param currFig: the figure to plot to
    @param ylim: set the y limits to ylim=(y_min, y_max)
    @param xshift: shift the x axis by xshift (float)
    @param rerrs: errors from random correlations to use (list)
    
    @return: current figure, ell, cell, cell_err
    """
    
    # get the data spectrum
    l, cl, clerr = np.loadtxt(dataFile, unpack=True)
    
    # check that rerrs is correct length, if provided
    if rerrs is not None:
        assert(len(cl) == len(rerrs))
        clerr = rerrs
        
    # get the current axes
    if currFig is not None:
        ax = currFig.get_axes()[0]
    else:
        ax = plt.gca()
        
    # make the x axis log
    ax.set_xscale('log')

    # plot the data
    ax.errorbar(l+xshift, cl, clerr, ls='', marker='.', markersize=2)
    ax.axhline(y=0, c='k', ls='--')
  
    ax.set_xlabel(r"$\mathrm{\ell}$", fontsize=16)
    ax.set_ylabel(r"$\mathrm{C_{\ell}}$", fontsize=16)
    
    # se the limits
    ax.set_xlim(left=10.0)

    if ylim is not None:
        plt.set_ylim(*ylim)

    return plt.gcf(), l, cl, clerr

    
   
def summarizeRealSpaceResults(baseDir, baseTime=0., fileTag=''):
    """
    @brief summarize real space correlation plots by 
    plotting the 1D correlation for a directory of *.pkl files
    and making a gif file of the correlations
    
    @param baseDir: directory of the output files (str)
    @param baseTime: the system time when the program started (float)
    @param fileTag: only consider files with fileTag (str)
    """
    data = os.listdir(dataDir)

    rs = []
    corrs = []
    errs = []

    fig = None
    tags = []
    for d in data:

        # only consider files that have been recently edited
        if os.path.getmtime('%s/data/%s' %(baseDir, d)) < baseTime:
            continue
        # only consider pkl files
        if not '.pkl' in d:
            continue

        # only consider files that have fileTag
        if not fileTag in d:
            continue

        tags.append(d.split('corr')[1].split('.pkl')[0])
        fig, r, mean, err = plot1DRealCorr(d, currFig=fig)
        rs.append(r)
        corrs.append(mean)
        errs.append(err)  

    fig.savefig('%s/summary.png' %baseDir)
    mincorr, maxcorr = fig.get_axes()[0].get_ylim()
    plt.cla()

    # make a temporary directory to store figures
    if not os.path.exists('.tmp/'):
        os.makedirs('.tmp')
        
    for i in range(len(tags)):

        plt.errorbar(rs[i], corrs[i], errs[i], marker='.', markersize=2)
        plt.ylim(mincorr, maxcorr)
        plt.axhline(y=0, c='k', ls='--')
        plt.xlabel(r"$\mathrm{\theta \ (arcminutes)}$", fontsize=16)
        plt.ylabel(r"$\mathrm{\xi(\theta)}$", fontsize=16)
        plt.savefig('.tmp/corr_1d_%s.png' %tags[i])
        
        plt.cla()

    # make the gif file
    os.system('convert -loop 0 -delay 60  .tmp/*.png %s/summary.gif' %baseDir)
    
    # delete the temporary directory
    os.system('rm -rf .tmp')
    
    return 0
    
def summarizeFourierSpaceResults(baseDir, baseTime=0., fileTag=''):
    """
    @brief summarize fourier space correlation plots by 
    plotting the 1D correlation for a directory of data files
    and making a gif file of the spectra
    
    @param baseDir: directory of the output files (str)
    @param baseTime: the system time when the program started (float)
    @param fileTag: only consider files with fileTag (str)
    """
    
    data = os.listdir('%s/data/' %baseDir)

    # lists for storing the spectra
    ells = []
    cls = []
    clerrs = []
    tags = []
    
    maxcl = 0.0
    mincl = np.inf
    plt.gca().set_xscale('log')
    
    for d in data:
        
        # only consider files that have been recently edited
        if os.path.getmtime('%s/data/%s' %(baseDir, d)) < baseTime:
            continue
            
        # only consider pkl files
        if not '.dat' in d:
            continue

        # only consider files that have fileTag
        if not fileTag in d:
            continue
        
        # load the spectra
        l, cl, clerr = np.loadtxt('%s/data/%s' %(baseDir, d), unpack=True)
        ells.append(l)
        cls.append(cl)
        clerrs.append(clerr)
        tags.append(d.split('.dat')[0])

        plt.errorbar(l, cl, clerr, ls='', marker='.', markersize=2)

    plt.axhline(y=0, c='k', ls='--')
    plt.xlabel(r"$\mathrm{\ell}$", fontsize=16)
    plt.ylabel(r"$\mathrm{C_{\ell}}$", fontsize=16)
    plt.savefig('%s/summary.png' %baseDir)
    plt.cla()
    mincl, maxcl = plt.gca().get_ylim()

    # make a temporary directory to store figures
    if not os.path.exists('.tmp/'):
        os.makedirs('.tmp')

    i = 0
    for i in range(len(tags)):
        
        plt.gca().set_xscale('log')
        plt.errorbar(ells[i], cls[i], clerrs[i], ls='', marker='.', markersize=2)
        plt.ylim(mincl, maxcl)
        plt.axhline(y=0, c='k', ls='--')
        plt.xlabel(r"$\mathrm{\ell}$", fontsize=16)
        plt.ylabel(r"$\mathrm{C_{\ell}}$", fontsize=16)
        plt.savefig('.tmp/%s.png' %tags[i])
        plt.cla()
        i += 1

    # make the gif file
    os.system('convert -loop 0 -delay 60 .tmp/*.png %s/summary.gif' %baseDir)

    # delete the temporary directory
    os.system('rm -rf .tmp')
    
    return 0
    
