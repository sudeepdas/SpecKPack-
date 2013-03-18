from speck import *
from speckCross import *
import os, sys, time
import multiprocessing as mp
from flipper import flipperDict
try:
    from utils.utilities import initializeProgressBar
except:
    pass
import xcorrParallel, xcorrUtils
import numpy as np


def runSpeck(params):
    """
    @brief run each step of the speck pipeline to compute the autospectrum
    @param params: dictionary containing the relevant parameters (dict)
    """
    
    # cut the necessary patches
    err = cutPatches.cutPatches(params)
    
    # compute the mode coupling matrix
    err = computeMCM.computeMCM(params)
    
    # compile the spectra
    err = spectraUtils.compileSpectra(params)
    
    # combine and calibrate the various spectra
    err = spectraUtils.combineAndCalibrate(params)
    
    # compute the analytic error
    err = analyticError.computeAnalyticError(params)
    
    # plot the auto spectrum
    err = spectraUtils.plotSpectrum(params)
    
    # package the results
    err = spectraUtils.packageResults(params)
    
    return 0
    
def runSpeckCross(params):
    """
    @brief run each step of the speckCross pipeline to compute the cross spectrum
    @param params: dictionary containing the relevant parameters (dict)
    """

    # cut the necessary patches
    err = scCutPatches.cutPatches(params)
    
    # compute the mode coupling matrix
    err = scComputeMCM.computeMCM(params)
    
    # compile the spectra
    err = scSpectraUtils.compileSpectra(params)
    
    # combine and calibrate the various spectra
    err = scSpectraUtils.combineAndCalibrate(params)
    
    # compute the analytic error
    err = scSpectraUtils.computeAnalyticError(params)

    # plot the auto spectrum
    err = scSpectraUtils.plotSpectrum(params)

    # package the results
    err = scSpectraUtils.packageResults(params)
    
    return 0

def fourierSpaceCorrFromRandoms(paramsToUpdate, valsToUpdate, nProcs, corrType='cross'):
    """
    @brief compute the real space correlation between a true data map
    and random map, for the nMaps pairs of maps

    @param paramsToUpdate: the parameters to change for each pair of maps 
       that we are correlating (dict)
    @param valsToUpdate: the numbers to insert into the paramsToUpdate values (list)
    @param nProcs: the number of processes to use (int)
    @keyword corrType: type of spectrum to compute ('cross' or 'auto')
    """

    num_jobs = len(valsToUpdate) # the number of correlations to do

    # read the base parameter file
    params = flipperDict.flipperDict()
    try:
        params.readFromFile("global.dict")
    except:
        raise

    originalStdOut = sys.stdout
        
    # might not have this package
    try:
        bar = initializeProgressBar(num_jobs)
    except:
        pass
        
    def worker(job_queue, results_queue):
        """
        @brief worker function for multiprocessing
        """
           
        # pull tasks until there are none left
        while True:
            
            # dequeue the next job 
            next_task = job_queue.get()

            # task=None means this worker is finished
            if next_task is None:
                # make sure we tell the queue we finished the task
                job_queue.task_done()
                break
            else:
                # tasks are tuples of params
                oldParams, num = next_task 

            # try to update the progress bar
            try: 
                bar.update(num+1)
            except:
                pass
            
            initialPath = os.getcwd()
            
            # do the work 
            # make new directory and cd there
            if not os.path.exists('tmp_%d' %num):
                os.makedirs('tmp_%d' %num)
            os.chdir('tmp_%d' %num)
            
            # update the parameters
            newParams = xcorrUtils.updateParams(paramsToUpdate, oldParams, valsToUpdate[num])

            # hide output from the speck(Cross) output
            sys.stdout = open(os.devnull, "w")    

            # compute the fourier space correlation, given the parameters
            if corrType == 'cross':
                err = runSpeckCross(newParams)
            if corrType == 'auto':
                err = runSpeck(newParams)

            # go back to old directory and delete temporary directory
            os.chdir(initialPath)
            if os.path.exists('tmp_%d' %num):
                os.system('rm -rf ./tmp_%d' %num)
                
            # restore stdout to original value
            sys.stdout = originalStdOut

            # store the results
            results_queue.put( (num) )

            # make sure we tell the queue we finished the task
            job_queue.task_done()
            

        return 0
    
    # store the system time
    sys_time = time.time()
    
    # establish communication queues that contain all jobs
    job_numbers = mp.JoinableQueue()
    results     = mp.Queue()
    
    # create a process for each cpu available or up to the limit specified by user
    if nProcs <= mp.cpu_count():
        num_workers = nProcs
    else:
        num_workers = mp.cpu_count()
        
    print 'Creating %d workers' % num_workers
    procs = [ mp.Process(target=worker, args=(job_numbers, results, )) for i in xrange(num_workers) ]
    
    # start the processes
    for proc in procs:
        proc.start()
        
    # enqueue the positions 
    for i in xrange(num_jobs):
        job_numbers.put( (params, i) )

    # Add a poison pill (=None) for each worker
    for i in xrange(num_workers):
        job_numbers.put(None)

    # wait for all of the jobs to finish
    job_numbers.join()
    
    return 0
    
def estimateErrorFromRandoms(randomsDir, tag=''):
    """
    @brief estimate the errors on a given bin from the std dev of the
    correlations with random maps

    @param randomsDir: the directory storing the results of the random correlations
    """   
    
    cls    = []
    clerrs = []

    for f in os.listdir(randomsDir):
        
        # consider only the data files
        if "dat" not in f:
            continue
        
        # consider only those files containing 'tag'
        if tag not in f:
            continue
        
        l, cl, clerr = np.loadtxt('%s/%s' %(randomsDir, f), unpack=True)
        cls.append(cl)
        clerrs.append(clerr)

    cls = np.array(cls)
    rerrs = np.std(cls, axis=0)

    return rerrs
