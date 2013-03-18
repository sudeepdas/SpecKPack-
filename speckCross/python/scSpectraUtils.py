from flipper import *
from scipy.interpolate import splrep,splev
import scipy
import os
import pickle
import time
import speckMisc


def weightedBinInAnnuli(p2d,weightMap,binningFile,trimAtL,powerOfL):
    binLo,binHi,binCent = fftTools.readBinningFile(binningFile)
    id = numpy.where(binHi<trimAtL)
    binHi = binHi[id]
    binLo = binLo[id]
    binCent = binCent[id]
    binnedPower = binCent.copy()*0.
    binCount = binCent.copy()*0. 
    modIntLMap = numpy.array(p2d.modLMap + 0.5,dtype='int64')
    for ibin in xrange(len(binHi)):
        loc = numpy.where((modIntLMap >= binLo[ibin]) & (modIntLMap <= binHi[ibin]))
        binMap = p2d.powerMap.copy()*0.
        binMap[loc] = weightMap[loc]
        binnedPower[ibin] = numpy.sum(p2d.powerMap*binMap*p2d.modLMap**powerOfL)/numpy.sum(binMap)
        binCount[ibin] = len(loc[0])
        
    return binLo,binHi,binCent,binnedPower, binCount/2


def get2DSpectrum(ma,mb,taper,gaussApod,mask=None,pixelWeight=None):

    if taper['apply'] and gaussApod['apply']:
        raise ValueError, "Both taper and Gaussian Apodization cannot be applied."+\
              "Use one or the other"
    
    m0 = ma.copy()
    m1 = mb.copy()
    
    if mask!=None:
        m0.data[:] *= mask.data[:]
        m1.data[:] *= mask.data[:]
    
    if pixelWeight != None:
        m0.data[:] *= pixelWeight.data[:]
        m1.data[:] *= pixelWeight.data[:]
    if taper['apply']:
        
        p2d = fftTools.powerFromLiteMap(m0,m1,applySlepianTaper=taper['apply'],\
                                        nresForSlepian=taper['nres'])
        
    elif gaussApod['apply']:
        gp = gaussApod.copy()
        gp.pop('apply')
        apod = m0.createGaussianApodization(**gp)
        m0.data[:] *= apod.data[:]
        m1.data[:] *= apod.data[:]
        p2d = fftTools.powerFromLiteMap(m0,m1)
        
    
    return p2d
    
def compileSpectra(p):
    """
    @brief compile the various auto/cross spectra
    """
    
    patchDir = "patches"
    labels = p['labels']
    taper = p['taper']
    gaussApod = p['gaussianApodization']
    applyMask = p['applyMask']
    
    nPatches = 0
    l = os.listdir(patchDir)
    for il in l:
        if 'patch_%s_00'%labels[0] in il and '_season' in il[-7:]:
            nPatches += 1
    print "Found %d season patch(es) ..."%nPatches
    
    
    
    
    if taper['apply'] and gaussApod['apply']:
        raise ValueError, "Both taper and Gaussian Apodization cannot be applied."+\
              "Use one or the other"
    
    
    trimAtL = p['trimAtL']
    specDir = 'spectra/'
    
    try:
        os.makedirs(specDir)
    except:
        pass
    
    lU,lL,lCen = fftTools.readBinningFile(p['binningFile'])
    ii = numpy.where(lU<p['trimAtL'])
    
    #beam transfer ()
    binnedBeamWindow = []
    for label in labels:
        Bb = speckMisc.getBinnedBeamTransfer(p['beamFile_%s'%label],p['binningFile'],trimAtL)
        binnedBeamWindow += [Bb]
    
    hpfDict = p['highPassCosSqFilter']
    filter = 1.0
    if hpfDict['apply']:
        print "Will take off the cos^2 high pass filter"
        filter = speckMisc.getBinnedInvCosSqFilter(hpfDict['lMin'],hpfDict['lMax'],p['binningFile'],trimAtL)
        
        
    
    for iPatch in xrange(nPatches):
        
        mbb = pickle.load(open('mcm/'+p['mcmFileRoot']+'_%03d.pkl'%iPatch,mode="r"))
        mbbInv = scipy.linalg.inv(mbb)

        if applyMask:
            mask = liteMap.liteMapFromFits("%s/mask%03d"%(patchDir,iPatch))
        else:
            mask = None
        pixW = None
        if p['applyPixelWeights']:
            pixW = (liteMap.liteMapFromFits("%s/totalWeightMap_%03d"%(patchDir,iPatch)))


        binWeightMap = pickle.load(open('noiseAndWeights/weightMap%03d.pkl'%iPatch,mode="r"))

        # Get the auto spectra first
        ilabel = 0 
        for label in labels:

            print "In patch: %03d, computing %sx%s season spectrum "%(iPatch,label,label)
            m0 = liteMap.liteMapFromFits("%s/patch_%s_%03d_season" %(patchDir,label,iPatch))
            area = m0.Nx*m0.Ny*m0.pixScaleX*m0.pixScaleY
            p2d = get2DSpectrum(m0,m0,taper,gaussApod,mask=mask,pixelWeight=pixW)
            lL,lU,lbin,clbin,binCount = weightedBinInAnnuli(p2d,\
                                                            binWeightMap.powerMap,\
                                                            p['binningFile'],p['trimAtL'],\
                                                            p['powerOfL'])
            
            clbinDecoup = numpy.dot(mbbInv,clbin)*area*filter**2
            # There is an additional correction for the autos as MCM had a transfer
            # function B_l_AR1*B_l_AR_2
            clbinDecoup *= binnedBeamWindow[ilabel-1]/binnedBeamWindow[ilabel]

            inds = numpy.where(clbinDecoup < 0.0)[0]
            clbinDecoup[inds] = 0.0

            fName = "%s/clBinAutoSeason_%sX%s_%03d.dat"%(specDir,label,label,iPatch)
            speckMisc.writeBinnedSpectrum(lbin,clbinDecoup,binCount,fName)

            ilabel += 1


        # Now do the cross- spectra
        m0 = liteMap.liteMapFromFits("%s/patch_%s_%03d_season"\
                                     %(patchDir,labels[0],iPatch))
        area = m0.Nx*m0.Ny*m0.pixScaleX*m0.pixScaleY

        m1 = liteMap.liteMapFromFits("%s/patch_%s_%03d_season"\
                                     %(patchDir,labels[1],iPatch))

        print "In patch: %03d, computing %sx%s season spectrum "%(iPatch,labels[0],labels[-1])



        p2d = get2DSpectrum(m0,m1,taper,gaussApod,mask=mask,pixelWeight=pixW)
        lL,lU,lbin,clbin,binCount = weightedBinInAnnuli(p2d,\
                                                        binWeightMap.powerMap,\
                                                        p['binningFile'],p['trimAtL'],\
                                                        p['powerOfL'])
        clbinDecoup = numpy.dot(mbbInv,clbin)*area*filter**2

        fName = "%s/clBinCrossSeason_%sX%s_%03d.dat"%(specDir,labels[0],labels[-1],iPatch)
        speckMisc.writeBinnedSpectrum(lbin,clbinDecoup,binCount,fName)
        
    return 0 #success
        
def combineAndCalibrate(p):
    """
    @brief combine and calibrate the various auto/cross spectra
    """
    patchDir = "patches"
    labels = p['labels']
    specDir = 'spectra/'

    nPatches = 0
    l = os.listdir(patchDir)
    for il in l:
        if 'patch_%s_00'%labels[0] in il and '_season' in il[-7:]:
            nPatches += 1
    print "Found %d season patch(es) ..."%nPatches

    #Combine Autos
    for label in labels:
        clAuto = []
        for iPatch in xrange(nPatches):
            lbin,clbin,binWeight = numpy.loadtxt("%s/clBinAutoSeason_%sX%s_%03d.dat"\
                                                 %(specDir,label,label,iPatch),unpack=True)
            clAuto += [clbin]
        clAutoMean = numpy.mean(clAuto, axis=0)*p['calibration_%s'%label]**2
        fName = "%s/clBinAutoGlobalMean_%sX%s.dat"%(specDir,label,label)
        speckMisc.writeBinnedSpectrum(lbin,clAutoMean,binWeight,fName)

    clCross = []
    clAuto = [] #Cross- label power
    for iPatch in xrange(nPatches):
        lbin,clbin,binWeight = numpy.loadtxt("%s/clBinCrossSeason_%sX%s_%03d.dat"\
                                             %(specDir,labels[0],labels[1],iPatch),\
                                             unpack=True)
        clCross += [clbin]


    clCrossMean = numpy.mean(clCross,axis=0)*p['calibration_%s'%labels[0]]*p['calibration_%s'%labels[1]]
    fName = "%s/clBinCrossGlobalMean_%sX%s.dat"%(specDir,labels[0],labels[1])
    speckMisc.writeBinnedSpectrum(lbin,clCrossMean,binWeight,fName)
    
    return 0 #sucess
    
def computeAnalyticError(p):
    """
    @brief compute the analytic error for cross spectrum
    """
    
    specDir = 'spectra/'
    patchDir = 'patches'
    
    labels = p['labels']
    powerOfL = p['powerOfL']

    if p['fileTag'] is None:
        fileTag = ""
    else:
        fileTag = p['fileTag']

    nPatches = 0
    l = os.listdir(patchDir)
    for il in l:
        if 'patch_%s_00'%labels[0] in il and '_season' in il[-7:]:
            nPatches += 1
    print "Found %d season patch(es) ..."%nPatches

    #Read in the single label auto spectra first
    #(C_b + N_b)^AA
    lbin,clbin_aa,binWeight = numpy.loadtxt("%s/clBinAutoGlobalMean_%sX%s.dat"\
                                            %(specDir,labels[0],labels[0]),unpack=True)

    #(C_b + N_b)^BB
    lbin,clbin_bb,binWeight = numpy.loadtxt("%s/clBinAutoGlobalMean_%sX%s.dat"\
                                            %(specDir,labels[1],labels[1]),unpack=True)

    #Cross spectrum
    lbin,clbin_cross,binWeight = numpy.loadtxt("%s/clBinCrossGlobalMean_%sX%s.dat"\
                                            %(specDir,labels[0],labels[1]),unpack=True)

    clbin_aa /= lbin**powerOfL
    clbin_bb /= lbin**powerOfL
    clbin_cross /= lbin**powerOfL

    fac = 2.0 
    #if labels[0]== labels[-1]: fac = 2.0

    
    Theta = (clbin_aa*clbin_bb+clbin_cross**2)/(fac*binWeight)

    C_b_err = numpy.sqrt(Theta/nPatches)
    C_b = clbin_cross

    g = open('%s/spectrum_%sx%s_%s.dat'%(specDir,labels[0],labels[1], fileTag),"w")
    for k  in xrange(len(lbin)):
        g.write("%d %e %e\n"%(lbin[k],lbin[k]**powerOfL*C_b[k],lbin[k]**powerOfL*C_b_err[k]))
    g.close()

    return 0 # success
    
def packageResults(p):
    """
    @brief group the results nicely for easy interpretation
    """
    
    labels = p['labels']

    if p['fileTag'] is None:
        fileTag = ""
    else:
        fileTag = p['fileTag']

    
    assert(os.path.exists("spectra/spectrum_%sx%s_%s.dat"%(labels[0],labels[1], fileTag )))

    resultsDir = p['resultsDir']
    if not os.path.exists(resultsDir):
        os.makedirs(resultsDir)

    if not os.path.exists("%s/data/" %resultsDir):
        os.makedirs("%s/data/" %resultsDir)

    if not os.path.exists("%s/plots/" %resultsDir):
        os.makedirs("%s/plots/" %resultsDir)

    os.system("mv spectra/*%s*png %s/plots/" %(p['fileTag'], resultsDir))
    os.system("mv spectra/spectrum*%s*dat %s/data/" %(p['fileTag'], resultsDir))
    
    return 0 # success
    
def plotSpectrum(p):
    """
    @brief plot the cross spectrum obtained
    """
    
    labels = p['labels']
    theoryFile = p['theoryFile']
    
    if p['fileTag'] is None:
        fileTag = ""
    else:
        fileTag = p['fileTag']
        
    specDir ='spectra/'

    # get the data spectrum
    l,cl,clerr = numpy.loadtxt("%s/spectrum_%sx%s_%s.dat"%(specDir,labels[0], labels[-1], fileTag),unpack=True)

    pylab.cla()
    
    # get the theory spectrum
    if theoryFile is not None:
        X = numpy.loadtxt(theoryFile)
        lTh = X[:,0]
        clTh = X[:,1]
        
        # plot the theory
        pylab.plot(lTh, clTh, c='k')
    
    # make the x axis log
    pylab.gca().set_xscale('log')
    
    # plot the data
    pylab.errorbar(l, cl, clerr, ls='', marker='.', markersize=2)
    pylab.axhline(y=0, c='k', ls='--')
    
    pylab.xlabel(r"$\mathrm{\ell}$", fontsize=16)
    pylab.ylabel(r"$\mathrm{C_{\ell}}$", fontsize=16)
    
    pylab.gca().set_xlim(left=10.0)
    pylab.savefig("%s/spectrum_%sx%s_%s.png"%(specDir,labels[0],labels[-1], fileTag))
    pylab.close()

    return 0
    
