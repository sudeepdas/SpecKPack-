import matplotlib.pyplot as plt
from flipper import liteMap, flipperDict
import numpy as np
import pyfits, pickle, scipy, sys, os
import xcorrUtils, xcorrTheory, xcorrParallel
from scipy.interpolate import InterpolatedUnivariateSpline
try:
    from utils.utilities import initializeProgressBar
except:
    pass
import multiprocessing as mp
    

def __ArrayHorizontalReverse(array2D):
    """
    @brief return a copy of the input 2D array
    but with the rows reversed
    """
    array2D = array2D.transpose()
    array2D = array2D[::-1]
    array2D = array2D.transpose()
    
    return array2D

def __ArrayVerticalReverse(array2D):
    """
    @brief return a copy of the input 2D array
    but with the columns reversed
    """
    
    array2D = array2D[::-1]
    return array2D

def MapShiftFunc(mapArray, n, m):
    """
    @brief given a 2D data array shift the array n units to the right
    and m units up
    
    @param mapArray: 2D data array to shift (array)
    @param n: positive n shifts Map to the right (int)
    @param m: positive m shifts Map up (int)
    
    @return shifted: the shifted 2D array
    """
                                
    shifted = np.roll(mapArray, n, axis = 1)
    # shift the input array left by n units
    if n < 0:
        shifted = __ArrayHorizontalReverse(shifted)
        shifted = np.delete(shifted, np.s_[0:abs(n):1], 1)
        shifted = np.insert(shifted, np.zeros( abs(n)), 0.0, axis = 1)
        shifted = __ArrayHorizontalReverse(shifted)
    # shift the input array right by n units
    else:
        shifted = np.delete(shifted, np.s_[0:abs(n):1], 1)
        shifted = np.insert(shifted, np.zeros(abs(n)), 0.0, axis = 1)

    shifted = np.roll(shifted, m, axis = 0)
    # shift the input array down by m units
    if m<0:
        shifted = __ArrayVerticalReverse(shifted)
        shifted = np.delete(shifted, np.s_[0:abs(m):1], 0 )
        shifted = np.insert(shifted, np.zeros(abs(m)), 0.0, axis = 0)
        shifted = __ArrayVerticalReverse(shifted)
    # shift the input array up by m units
    else:
        shifted = np.delete(shifted, np.s_[0:abs(m):1], 0)
        shifted = np.insert(shifted, np.zeros(abs(m)), 0.0, axis = 0)
    
    return shifted
    
def realSpaceCorr(params):
    """
    @brief compute the real space correlation between two maps
           and output the relevant files
    @param params: dictionary containing the following:

           totalMapPairs: the total pairs of maps to compute correlation over (int)
           root_A: the file tag for map #1 (str)
           root_B: the file tag for map #2 (str)
           thetaExtent: the angular extent to compute correlation to, in degrees (float)
           files_A: array containing filenames of map A (can be more than 1 non overlapping regions)
           files_B: array containing filenames of map B (can be more than 1 non overlapping regions)
           makePlots: whether to plot the results (bool)     
    """

    totalMapPairs = params['totalMapPairs']

    tag_A = params['root_A']
    tag_B = params['root_B']


    print "cross correlating %s and %s..." %(tag_A, tag_B)

    # extent of correlation to calculate
    dimDegree = params['thetaExtent']
    dimDegree *= np.pi/180.

    # loop over all map pairs
    for i in range(totalMapPairs):

        print 'correlating map pairs #%d...' %i

        # read in maps to cross correlate
        map_A = liteMap.liteMapFromFits(params['files_A'][i])
        map_B = liteMap.liteMapFromFits(params['files_B'][i])

        map_A.data = np.nan_to_num(map_A.data)
        map_B.data = np.nan_to_num(map_B.data)

        # set outliers to the median value
        inds = np.where(map_A.data != 0.0)
        dev = np.std(map_A.data[inds])
        inds2 = np.where(abs(map_A.data) > 10.*dev)
        map_A.data[inds2] = np.median(map_A.data[inds])

        inds = np.where(map_B.data != 0.0)
        dev = np.std(map_B.data[inds])
        inds2 = np.where(abs(map_B.data) > 10.*dev)
        map_B.data[inds2] = np.median(map_B.data[inds])

        # this map will store mapA_shifted*mapB values
        map_AB = map_A.copy()
        map_AB.data[:] = 0.

        # make map that will be use for shifting
        map_A_shifted = map_A.copy()
        map_A_shifted.data[:] = 0.

        # only do this first time around
        if i == 0:
            # num of pixels in this theta extent
            Ndim = np.int(dimDegree/map_A.pixScaleX)
            if np.mod(Ndim,2) != 0:
                Ndim += 1

            # initialize correlation and theta matrices
            corr  = np.zeros((Ndim+1,Ndim+1), dtype=float)
            theta  = np.zeros((Ndim+1,Ndim+1), dtype=float) # this will be in arcminutes
            weight = np.zeros((Ndim+1,Ndim+1), dtype=float) # weight array for combining multiple maps

    
        # might not have this package
        try:
            bar = initializeProgressBar((Ndim+1.0)**2)
        except:
            pass

        # n shifts map in x-direction, m shifts map in y directions
        iter = 0
        for m in xrange(-Ndim/2, Ndim/2+1, 1):
            for n in xrange(-Ndim/2, Ndim/2+1, 1):


                try:
                    bar.update(iter + 1)
                except:
                    pass

                iter += 1

                # shift map A and then multiply shifted map A by map B
                map_A_shifted.data = MapShiftFunc(map_A.data, n, m)
                map_AB.data[:] = map_A_shifted.data[:]*map_B.data[:]
                inds = np.where(map_AB.data != 0.)

                w = 1.0*len(inds[0]) # number of nonzero values in mean

                # due weighted sum of this corr value and any past corr values
                corr[m+Ndim/2,n+Ndim/2] = corr[m+Ndim/2,n+Ndim/2]*weight[m+Ndim/2,n+Ndim/2] + w*(map_AB.data[inds]).mean()

                # update the nonzero elements at this array element
                weight[m+Ndim/2,n+Ndim/2] += w

                # divide by the total weight
                corr[m+Ndim/2,n+Ndim/2] /= weight[m+Ndim/2,n+Ndim/2]

                # store the theta value
                theta[m+Ndim/2,n+Ndim/2] = np.sqrt(n**2*map_A.pixScaleX**2+m**2*map_A.pixScaleY**2)*180.*60/np.pi


    # plot and save the 2D correlation figure
    arcmin = 180.*60/np.pi

    # make sure output directories exist
    if not os.path.exists('./output'):
        os.makedirs('./output')
        
    if params['makePlots']:
        
        # make sure figs directory exists
        if not os.path.exists('./output/figs'):
            os.makedirs('./output/figs')
        
        plt.imshow(corr,origin='down',cmap = cm.gray,extent=[-Ndim/2*map_A.pixScaleX*arcmin,(Ndim/2+1)*map_A.pixScaleX*arcmin,\
                                                                -Ndim/2*map_A.pixScaleY*arcmin,(Ndim/2+1)*map_A.pixScaleY*arcmin])
        plt.colorbar()
        
        plt.savefig('./output/figs/corr_%s_%s.png'%(tag_A,tag_B))


        # plot the 1D correlation and save
        fig, r, mean, std = plot1DCorr(None, data=(theta, corr))
        fig.savefig('./output/figs/corr_1d_%s_%s.png'%(tag_A,tag_B))


    # make sure data directory exists
    if not os.path.exists('./output/data'):
        os.makedirs('./output/data')

    # save the 2D correlation and theta as a pickle
    pickle.dump([theta, corr],open('./output/data/corr_%s_%s.pkl' %(tag_A,tag_B), 'w'))
    plt.close()

    # save the 2D correlation as a fits file    
    hdu = pyfits.PrimaryHDU(corr)
    hdu.writeto('./output/data/corr_%s_%s.fits'%(tag_A,tag_B),clobber=True)

    return 0

   

def plot(params):
    """
    @brief plot a real space correlation measured from data and the corresponding theory
    prediction

    @param params: dictionary containing the following:
           powerSpectrum: name of file containing spectrum to use for theory curve (str)
           dataCorr: name of pickle file containing 2D correlation from data (str)
           filter1_path: path of file containing filter #1 (str)
           filter1FromFunction: tuple containing ('function_name', functionKWArgs)
           filter2_path: path of file containing filter #2 (str)
           filter2FromFunction: tuple containing ('function_name', functionKWArgs)       
    """

    # load the data spectrum to plot
    ell, cl = np.loadtxt(params['powerSpectrum'], usecols=[0,1], unpack=True)

    # make the first filter, if provided
    F1 = np.ones(len(ell))

    # first, read from a path, if given 
    if params['filter1_path'] is not None:
            x, F1 = np.loadtxt(params['filter1_path'], unpack=True)
            f = InterpolatedUnivariateSpline(x, F1)
            F1 = f(ell)
    # then try to make the filter from a function  
    elif params['filter1FromFunction'] is not None:

        func = xcorrUtils.__stringToFunction(params['filter1FromFunction'][0])
        F1 = func(ell, **params['filter1FromFunction'][1])

    # make the second filter, if provided
    F2 = np.ones(len(ell))

     # first, read from a path, if given 
    if params['filter2_path'] is not None:

        x, F2 = np.loadtxt(params['filter2_path'], unpack=True)
        f = InterpolatedUnivariateSpline(x, F2)
        F2 = f(ell)
    # then try to make the filter from a function   
    elif params['filter2FromFunction'] is not None:
        func = xcorrUtils.__stringToFunction(params['filter2FromFunction'][0])
        F1 = func(ell, **params['filter2FromFunction'][1])

    # compute the expected correlation given the filter and power spectrum
    theta_th, corr_th = xcorrTheory.realSpaceCorrFromTheory(ell, cl, Fell_1=F1, Fell_2=F2)

    # plot the binned 1D correlation from the 2D data
    fig, theta, corr, corr_errs = xcorrUtils.plot1DRealCorr(params['dataCorr'])

    # also plot the expected curve from theory
    curr_ax = fig_gca()
    curr_ax.plot(theta_th*60, corr_th, ls='--')

    # add a horizontal axis line
    plt.axhline(y=0, c='k')
    plt.show()

    return 0
    

def realSpaceCorrFromRandoms(paramsToUpdate, valsToUpdate):
    """
    @brief compute the real space correlation between a true data map
    and random map, for the nMaps pairs of maps
    
    @param paramsToUpdate: the parameters to change for each pair of maps 
           that we are correlating (dict)
    @param valsToUpdate: the numbers to insert into the paramsToUpdate values (list)
    """
    
    # establish communication queues for different processes
    tasks = mp.JoinableQueue()
    results = mp.Queue()
    
    # store the system time
    sys_time = time.time()
    
    # start a worker for each cpu available
    num_workers = mp.cpu_count()
    print 'Creating %d workers' % num_workers
    workers = [ xcorrParallel.worker(tasks, results) for i in xrange(num_workers) ]
    for w in workers:
        w.start()

    num_jobs = len(valsToUpdate) # the number of correlations to do
    
    # read the base parameter file
    params = flipperDict.flipperDict()
    try:
        params.readFromFile("RealSpaceCorr.dict")
    except:
        raise
    
    # enqueue the tasks, which is equal to number of maps we are correlating
    for i in xrange(num_jobs):
        tasks.put(xcorrParallel.realCorrTask(params, valsToUpdate[i], paramsToUpdate))
    
    # Add a poison pill for each worker
    for i in xrange(num_workers):
        tasks.put(None)
    
    # wait for all of the tasks to finish
    tasks.join()
    
    # summarize the results
    dataDir = './output/'
    xcorrUtils.summarizeRealSpaceResults(dataDir, baseTime=sys_time)
    
    return 0
    
def estimateErrorFromRandoms(randomsDir, tag=''):
    """
    @brief estimate the errors on a given bin from correlations
    with random maps
    
    @param randomsDir: the directory storing the results of the random correlations
    """   
    corr_total = []

    for f in os.listdir(randomsDir):

        # consider only the pickle files
        if "pkl" not in f:
            continue

        # consider only those files containing 'tag'
        if tag not in f:
            continue

        fig, theta, corr, corr_errs = xcorrUtils.plot1DRealCorr("%s/%s" %(randomsDir, f))

    corr_total = np.array(corr_total)

    corr_mean = np.mean(corr_total, axis=0)
    corr_err = np.std(corr_total, axis = 0)
    
    return corr_err
    