from flipper import * 
import modeCoupling as mcm
import pickle
import scipy
import multiprocessing as mp


def getNoiseAndWeightMapsPerPatch(p, patchDir, labels, iPatch,\
                                  nDivs, taper, theoryFile,\
                                  binningFile, verticalkMaskLimits=None):
    twoDNoise = []
    pwDict = p['prewhitener']

    # --- precompute the pewhitening transfer
    ell, f_ell = numpy.transpose(numpy.loadtxt\
                                 (os.environ['SPECK_DIR']+'/data/'+p['beamFile_%s'%labels[0]]))
    tf = ell.copy()*0. + 1.0
    if pwDict != None and pwDict['apply']:
        
        pw = prewhitener.prewhitener(pwDict['radius'],\
                                     addBackFraction= pwDict['addBackFraction'],\
                                     smoothingFWHM=pwDict['gaussFWHM'])
        tf = pw.correctSpectrum(ell,tf)
        tf[0] = tf[1]
        
    for label in labels:
        countAuto = 0
        countCross = 0
        if nDivs[label] > 0:
            for i in xrange(nDivs[label]):
                m0 = liteMap.liteMapFromFits(patchDir+'/patch_%s_%03d_%d'%(label,iPatch,i))
                for j in xrange(i+1):
                    m1 = liteMap.liteMapFromFits(patchDir+'/patch_%s_%03d_%d'%(label,iPatch,j))
                    if i == j:
                        if countAuto == 0:
                            p2dAuto = fftTools.powerFromLiteMap(m0,m1,applySlepianTaper=True,nresForSlepian=1.0)
                        else:
                            p2d =  fftTools.powerFromLiteMap(m0,m1,applySlepianTaper=True,nresForSlepian=1.0)
                            p2dAuto.powerMap[:] += p2d.powerMap[:]
                        countAuto += 1
                    else:
                        if countCross == 0:
                            p2dCross = fftTools.powerFromLiteMap(m0,m1,applySlepianTaper=True,nresForSlepian=1.0)
                        else:
                            p2d =  fftTools.powerFromLiteMap(m0,m1,applySlepianTaper=True,nresForSlepian=1.0)
                            p2dCross.powerMap[:] += p2d.powerMap[:]
                        countCross += 1
                
            print "%s GHz nCross %d nAuto %d"%(label,countCross, countAuto)
            p2dCross.powerMap[:] /= countCross #-- mean cross
            p2dAuto.powerMap[:] /= countAuto #-- mean auto
            # --- Noise = auto - cross --- #
            p2dNoise = p2dCross.copy()
            p2dNoise.powerMap[:] = p2dAuto.powerMap[:] - p2dCross.powerMap[:]
            assert(p2dNoise.powerMap.min() >0.)
        
            # --- This should be replaced by seasonNoise
            # --- A good approximation is a  1/nDiv factor
            p2dNoise.powerMap[:] /= float(nDivs[label])
            print "Noise Map generated for %s GHz"%label
            # --- Take out the beam, prewhitening and calibrate
            try:
                ell, f_ell = numpy.transpose(numpy.loadtxt(os.environ['SPECK_DIR']+'/data/'+p['beamFile_%s'%label]))
            except:
                ell, f_ell = numpy.transpose(numpy.loadtxt(p['beamFile_%s'%label]))
            
            tck = scipy.interpolate.splrep(ell,f_ell**2/tf)
            twoDBeam = scipy.interpolate.splev(p2dNoise.modLMap.ravel(),tck)
            twoDBeam = numpy.reshape(twoDBeam,[p2dNoise.Ny,p2dNoise.Nx])
            p2dNoise.powerMap[:] /= twoDBeam[:]*(p['calibration_%s'%label])**2
        else:
            print "for %s, assuming flat noise"%label
            m0 = liteMap.liteMapFromFits(patchDir+'/patch_%s_%03d_season'%(label,iPatch))
            p2dNoise = fftTools.powerFromLiteMap(m0)
            p2dNoise.powerMap[:] = 1.0
            
        twoDNoise += [p2dNoise]
        # p2dNoiseOut = p2dNoise.copy()
        # p2dNoiseOut.powerMap[:] *= p2d.modLMap[:]**2
        p2dNoise.writeFits('noiseAndWeights/noisePower_%s_%03d.fits'%(label,iPatch),overWrite=True)
        
    
    p2dWeight = p2dNoise.copy()
    p2dTheory = p2dNoise.copy()
    #  --- add the theory spectrum to this (should have a different th spec for the non-CMB map)
    X = numpy.loadtxt(theoryFile)
    ell = X[:,0]
    Cell = X[:,1]/(ell*(ell+1))*(2*numpy.pi)
    tck = scipy.interpolate.splrep(ell,Cell,k=3)
    ll = numpy.ravel(p2dTheory.modLMap)
    cll = scipy.interpolate.splev(ll,tck)
    p2dTheory.powerMap = numpy.reshape(cll,[p2dTheory.Ny,p2dTheory.Nx] ) #calibration?
    id = numpy.where( p2dTheory.modLMap > ell.max()) #avoid weird spline extrapolations
    p2dTheory.powerMap[id] = 0.
    
    # --- if single label
    if len(labels) == 1:
        p2dWeight.powerMap[:] = 1./(p2dNoise.powerMap[:] + p2dTheory.powerMap[:])**2
    # --- if two labels then variance (Cl+Nl00)*(Cl+Nl11)
    elif len(labels) == 2:
        p2dWeight.powerMap[:] = 1./((((twoDNoise[0]).powerMap[:] + p2dTheory.powerMap[:])*\
                                               ((twoDNoise[1]).powerMap[:] + 0.*p2dTheory.powerMap[:])))
        
    # ---- throw out outliers (do this in bins because median of the whole map
    # does not make much sense )
    binLo, binHi, BinCe = fftTools.readBinningFile(binningFile)
    modIntLMap = numpy.array(p2dWeight.modLMap + 0.5,dtype='int64')
    for ibin in xrange(len(binLo)):
        loc = numpy.where((modIntLMap >= binLo[ibin]) & (modIntLMap <= binHi[ibin]))
        weightInRing  =  p2dWeight.powerMap[loc]
        med = numpy.median(weightInRing)
        #med = numpy.std(weightInRing)
        weightInRing[numpy.where(weightInRing > 5*med)] = med
        # --- flatten by mean  --- #
        p2dWeight.powerMap[loc] = weightInRing/weightInRing.mean()
        
    loc2 = numpy.where(modIntLMap>binHi.max())
    p2dWeight.powerMap[loc2] = 0.0
    # --- smooth the maps to further iron out noisy pixels (except for the lowest few bins)
    loc = numpy.where(p2dWeight.modLMap <500)
    wt = p2dWeight.powerMap[loc]
    
    
    kernel_width = (3,3)
    p2dWeight.powerMap[:] = scipy.ndimage.gaussian_filter(p2dWeight.powerMap, kernel_width)
    p2dWeight.powerMap[loc] = wt
    
    
    # --- put in the vertical k mask if any ---
    if verticalkMaskLimits != None:
        p2dWeight.createKspaceMask(verticalStripe=verticalkMaskLimits)
        p2dWeight.powerMap[:] *= p2dWeight.kMask[:]
        
    # --- write out the weight maps ---
    p2dWeight.writeFits('noiseAndWeights/weightMap%03d.fits'%iPatch,overWrite=True)
    pickle.dump(p2dWeight,open('noiseAndWeights/weightMap%03d.pkl'%iPatch,mode="w"))
    
    del p2dTheory, p2dNoise
    return p2dWeight
    
def computeMCM(p):
    """
    @brief generates the mode-coupling matrices per patch and stores them as pickles
    """
    
    patchDir = "patches"
    labels = p["labels"]
    
    taper = p['taper']
    gaussApod = p['gaussianApodization']
    applyMask = p['applyMask']
    
    pwDict = p['prewhitener']
    try:
        os.makedirs("noiseAndWeights")
    except:
        pass
    
    try:
        os.makedirs("mcm")
    except:
        pass
     
    if taper['apply'] and gaussApod['apply']:
        raise ValueError, "Both taper and Gaussian Apodization cannot be applied."+\
              "Use one or the other"

    l = os.listdir(patchDir)
    nDivs = 0
    
    # find the number of patches 
    nPatches = 0
    for il in l:
        if 'patch_%s_00'%labels[0] in il and '_season' in il[-7:]:
            nPatches += 1
    
    print "Found %d season patch(es) ..."%nPatches
    nDivs  = {}
    for label in labels:
        nn = 0
        for il in l:
            if 'patch_%s_000'%label in il and (not('season' in il)):
                nn += 1
        nDivs['%s'%label] = nn

    for label in labels: 
        print "Found %d sub-season splits for %s"%(nDivs[label],label)
        

    if (not(taper['apply']) and not(applyMask) and not(gaussApod['apply'])):
        raise ValueError, 'I cannot yet deal with the no apodization, no mask situation ... sorry!'
    
    binLower,binUpper,binCenter = fftTools.readBinningFile(p['binningFile'])
    id = numpy.where(binUpper < p['trimAtL'])
    
    # calculate the window           
    windows = []
    for j in range(nPatches):
    
        # if apply taper, read in a patch and create the order zero taper taper
        m = liteMap.liteMapFromFits(patchDir+os.path.sep+'patch_%s_%03d_season'%(labels[0], j))

        window0 = m.copy()  
        window0.data[:] = 1.0

        if taper['apply']:
            t = utils.slepianTaper00(m.Nx,m.Ny,taper['nres'])
            window0.data[:] *= t[:]
        if gaussApod['apply']:
            gp = gaussApod.copy()
            gp.pop('apply')
            apod  = m.createGaussianApodization(**gp)
            window0.data[:] = apod.data[:]

        windows.append(window0.copy())    
        
    #if pixel space weighting is to be applied, make weightmaps by co-adding all weightMaps
    if p['applyPixelWeights']:
        
        pixWeightMaps = []
        for iPatch in xrange(nPatches):
            wt = liteMap.liteMapFromFits(patchDir+os.path.sep+'patch_%s_%03d_season'%(labels[0],iPatch))
            wt.data[:] = 1.
            for label in labels:
                try:
                    wtmap = liteMap.liteMapFromFits(patchDir+'/weight_%s_%03d_season'%(label,iPatch))
                    wt.data[:] *= wtmap.data[:]
                except:
                    pass
            
            if wt.data.sum() == 0:
                print "no pixel weights found ..defaulting to  unity "
                wt.data[:] = 1.0
            
            wt.writeFits(patchDir+"/totalWeightMap_%03d"%iPatch,overWrite=True)
            pixWeightMaps += [wt]

            
    # beam transfer
    print "Reading beam ..."
    for label in labels:
        try:
            ell, f_ell = numpy.transpose(numpy.loadtxt(os.environ['SPECK_DIR']+'/data/'+p['beamFile_%s'%label]))
        except:
            ell, f_ell = numpy.transpose(numpy.loadtxt(p['beamFile_%s'%label]))
            
        if label == labels[0]:
            f_ell0 = f_ell
        else:
            f_ell0 *= f_ell
            f_ell0 = numpy.sqrt(f_ell0)
        
    transfer = [ell,f_ell0**2] 
    print "...done"
    if pwDict != None and pwDict['apply']:
        pw = prewhitener.prewhitener(pwDict['radius'],\
                                     addBackFraction= pwDict['addBackFraction'],\
                                     smoothingFWHM=pwDict['gaussFWHM'])
        tf = ell.copy()*0. + 1.0
        tf = pw.correctSpectrum(ell,tf)
        # tf = numpy.nan_to_num()
        tf[0] = tf[1]
        transfer = [ell,f_ell0**2/tf]
        
        print "Done generating prewhitener transfer ..."

            
    def workerFunc(i):
        
        if p['useAzWeights']:
            print "Generating Az weights ..."
            p2dWeight = getNoiseAndWeightMapsPerPatch(p, patchDir,labels,i,nDivs,taper,p['theoryFile'],\
                        p['binningFile'],verticalkMaskLimits=p['verticalkMaskLimits'])
            print " ... done"
        else:
            print "Will use flat weights"
            p2dWeight = fftTools.powerFromLiteMap(windows[i])
            p2dWeight.powerMap[:] = 1.
            if p['verticalkMaskLimits'] != None:
                p2dWeight.createKspaceMask(verticalStripe=p['verticalkMaskLimits'])
                p2dWeight.powerMap[:] *= p2dWeight.kMask[:]
            p2dWeight.writeFits('noiseAndWeights/weightMap%03d.fits'%i,overWrite=True)
            pickle.dump(p2dWeight,open('noiseAndWeights/weightMap%03d.pkl'%i,mode="w"))
            print " ... done"

        window = windows[i]
        if applyMask:
            m = liteMap.liteMapFromFits(patchDir+os.path.sep+'mask%03d'%i)
            window.data[:] *= m.data[:]
        if p['applyPixelWeights']:
            window.data[:] *= pixWeightMaps[i].data[:]
            window.writeFits("auxMaps/finalWindow%03d.fits"%i,overWrite=True)
        print "In patch %03d"%i

        mbb = mcm.generateMCM(window,binLower[id],\
                              binUpper[id],p['trimAtL'],\
                              mbbFilename="mcm/"+p['mcmFileRoot']+'_%03d.pkl'%i,\
                              transfer = transfer,\
                              binningWeightMap = p2dWeight.powerMap,\
                              powerOfL=p['powerOfL'])
        return 0

    patchesDone = 0
    # make sure we compute MCM for each map patch
    while patchesDone < nPatches:
        procs = []
        nProcs = 0
        
        # only spawn a number of new processes up to amount specified by user
        while nProcs < p['nfork']:
            
            # break if we've done all the patches
            if patchesDone == nPatches:
                break
                
            # make the new process
            proc = mp.Process(target=workerFunc, args=(patchesDone,))
            
            # start the new process and save it
            proc.start()
            procs.append(proc)
            
            # update the number of running processes and patches done
            nProcs += 1
            patchesDone += 1
            
        # wait for current number of processes to finish
        for proc in procs:
            proc.join()
    
    return 0 # success!