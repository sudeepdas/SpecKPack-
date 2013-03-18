from flipper import *
from scipy.interpolate import splrep,splev
import scipy
import os, sys
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
    weightedBincount = binCent.copy()
    modIntLMap = numpy.array(p2d.modLMap + 0.5,dtype='int64')
    for ibin in xrange(len(binHi)):
        loc = numpy.where((modIntLMap >= binLo[ibin]) & (modIntLMap <= binHi[ibin]))
        binMap = p2d.powerMap.copy()*0.
        binMap[loc] = weightMap[loc]
        binnedPower[ibin] = numpy.sum(p2d.powerMap*binMap*p2d.modLMap**powerOfL)/numpy.sum(binMap)
        binCount[ibin] = len(loc[0])
        weightedBincount[ibin] = 1./(numpy.sum(weightMap[loc]**2)/(numpy.sum(weightMap[loc]))**2)
        #print binCount[ibin]/weightedBincount
    return binLo,binHi,binCent,binnedPower, weightedBincount/2.


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
    else:
        p2d = fftTools.powerFromLiteMap(m0,m1,applySlepianTaper=False)
        
    return p2d
    
def compileSpectra(p):
    """
    @brief compile the various auto/cross spectra
    """
    specDir = 'spectra/'
    patchDir = "patches"
        
    
    freqs = p['frequencies']
    taper = p['taper']
    gaussApod = p['gaussianApodization']
    applyMask = p['applyMask']
    
        
    nDivs, nPatches = speckMisc.getPatchStats(patchDir,freqs[0])
    
    print "Found %d patches with %d sub-season divisions in each"%(nPatches, nDivs)
    
    if taper['apply'] and gaussApod['apply']:
        raise ValueError, "Both taper and Gaussian Apodization cannot be applied."+\
              "Use one or the other"
    
    
    trimAtL = p['trimAtL']
    
    
    try:
        os.makedirs(specDir)
    except:
        pass
    
    lU,lL,lCen = fftTools.readBinningFile(p['binningFile'])
    ii = numpy.where(lU<p['trimAtL'])
    
    #beam transfer (used in slaved spec only)
    binnedBeamWindow = []
    for freq in freqs:
        Bb = speckMisc.getBinnedBeamTransfer(p['beamFile_%d'%freq],p['binningFile'],trimAtL)
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
        ifreq = 0 
        for freq in freqs:
            clAutoPatch = []
            for i in xrange(nDivs):
                print "In patch: %03d, computing %dx%d spectrum: %d%d "%(iPatch,freq,freq,i,i)
                m0 = liteMap.liteMapFromFits("%s/patch_%d_%03d_%d" %(patchDir,freq,iPatch,i))
                area = m0.Nx*m0.Ny*m0.pixScaleX*m0.pixScaleY
                p2d = get2DSpectrum(m0,m0,taper,gaussApod,mask=mask,pixelWeight=pixW)
                lL,lU,lbin,clbin,binCount = weightedBinInAnnuli(p2d,\
                                                                binWeightMap.powerMap,\
                                                                p['binningFile'],p['trimAtL'],\
                                                                p['powerOfL'])
                
                clbinDecoup = numpy.dot(mbbInv,clbin)*area*filter**2
                # There is an additional correction for the autos as MCM had a transfer
                # function B_l_AR1*B_l_AR_2
                clbinDecoup *= binnedBeamWindow[ifreq-1]/binnedBeamWindow[ifreq]
                
                fName = "%s/clBinDecoup_%dX%d_%03d_%d%d.dat"%(specDir,freq,freq,iPatch,i,i)
                speckMisc.writeBinnedSpectrum(lbin,clbinDecoup,binCount,fName)

                clAutoPatch += [clbinDecoup]

            clAutoPatchMean = numpy.mean(clAutoPatch,axis=0)
            fName = "%s/clBinAutoMean_%dX%d_%03d.dat"%(specDir,freq,freq,iPatch)
            speckMisc.writeBinnedSpectrum(lbin,clAutoPatchMean,binCount,fName)
            
            ifreq += 1
        
        # Now do the cross-frequency spectra
        clCrossPatch = []
        clAutoPatch = [] #cross-freq auto spec
        for i in xrange(nDivs):
            m0 = liteMap.liteMapFromFits("%s/patch_%d_%03d_%d"\
                                         %(patchDir,freqs[0],iPatch,i))
            area = m0.Nx*m0.Ny*m0.pixScaleX*m0.pixScaleY
            for j in xrange(nDivs):
                #if i == j: continue
                if freqs[0] == freqs[-1] and i<=j: continue
                print "In patch: %03d, computing %dx%d spectrum: %d%d "%(iPatch,freqs[0],freqs[-1],i,j)
                m1 = liteMap.liteMapFromFits("%s/patch_%d_%03d_%d"\
                                             %(patchDir,freqs[-1],iPatch,j))
                
                p2d = get2DSpectrum(m0,m1,taper,gaussApod,mask=mask,pixelWeight=pixW)
                lL,lU,lbin,clbin,binCount = weightedBinInAnnuli(p2d,\
                                                                binWeightMap.powerMap,\
                                                                p['binningFile'],p['trimAtL'],\
                                                                p['powerOfL'])
                clbinDecoup = numpy.dot(mbbInv,clbin)*area*filter**2
                fName = "%s/clBinDecoup_%dX%d_%03d_%d%d.dat"%(specDir,freqs[0],freqs[-1],iPatch,i,j)
                speckMisc.writeBinnedSpectrum(lbin,clbinDecoup,binCount,fName)
                if i != j :
                    clCrossPatch += [clbinDecoup]
                else:
                    clAutoPatch += [clbinDecoup]
        #print clCrossPatch
        if nDivs > 1:
            clCrossPatchMean = numpy.mean(clCrossPatch,axis=0)
        else:
            clCrossPatchMean = clbinDecoup
        #print clCrossPatchMean
        fName = "%s/clBinCrossMean_%dX%d_%03d.dat"%(specDir,freqs[0],freqs[-1],iPatch)
        speckMisc.writeBinnedSpectrum(lbin,clCrossPatchMean,binCount,fName)

        if len(clAutoPatch)>0:
            clAutoPatchMean = numpy.mean(clAutoPatch,axis=0)
            fName = "%s/clBinAutoMean_%dX%d_%03d.dat"%(specDir,freqs[0],freqs[-1],iPatch)
            speckMisc.writeBinnedSpectrum(lbin,clAutoPatchMean,binCount,fName)
            
            

            
    return 0 #success!
    
def combineAndCalibrate(p):
    """
    @brief combine and calibrate the various auto/cross spectra
    """
    
    freqs = p['frequencies']
    patchDir = "patches"
    specDir = 'spectra/'

    nDivs, nPatches = speckMisc.getPatchStats(patchDir,freqs[0])
    print "Found %d patches with %d sub-season divisions in each"%(nPatches, nDivs)


    #Combine Autos
    clAutos = {freqs[0]:[],freqs[-1]:[]}

    for freq in freqs:
        clAutoPatch = []
        for iPatch in xrange(nPatches):
            lbin,clbin,binWeight = numpy.loadtxt("%s/clBinAutoMean_%dX%d_%03d.dat"\
                                                 %(specDir,freq,freq,iPatch),unpack=True)
            clAutoPatch += [clbin*p['calibration_%d'%freq]**2]
            clAutos[freq] += [clbin*p['calibration_%d'%freq]**2]
        clAutoMean = numpy.mean(clAutoPatch, axis=0)
        fName = "%s/clBinAutoGlobalMean_%dX%d.dat"%(specDir,freq,freq)
        speckMisc.writeBinnedSpectrum(lbin,clAutoMean,binWeight,fName)

    print clAutos

    clCross = []
    clAutoCF = [] #Cross- freq autos
    for iPatch in xrange(nPatches):
        lbin,clbin,binWeight = numpy.loadtxt("%s/clBinCrossMean_%dX%d_%03d.dat"\
                                             %(specDir,freqs[0],freqs[-1],iPatch),\
                                             unpack=True)
        clCross += [clbin*p['calibration_%d'%freqs[0]]*p['calibration_%d'%freqs[-1]]]

        lbin,clbin,binWeight = numpy.loadtxt("%s/clBinAutoMean_%dX%d_%03d.dat"\
                                             %(specDir,freqs[0],freqs[-1],iPatch),\
                                             unpack=True)
        clAutoCF += [clbin*p['calibration_%d'%freqs[0]]*p['calibration_%d'%freqs[-1]]]

    clCrossMean = numpy.mean(clCross,axis=0)
    fName = "%s/clBinCrossGlobalMean_%dX%d.dat"%(specDir,freqs[0],freqs[-1])
    speckMisc.writeBinnedSpectrum(lbin,clCrossMean,binWeight,fName)



    clAutoMean = numpy.mean(clAutoCF,axis=0)
    fName = "%s/clBinAutoGlobalMean_%dX%d.dat"%(specDir,freqs[0],freqs[-1])
    speckMisc.writeBinnedSpectrum(lbin,clAutoMean,binWeight,fName)


    # Now the  cross-spectrum weighted by patch weights

    clPatchWeights =[]

    X = numpy.loadtxt(p["theoryFile"])
    lTh = X[:,0]
    clTh = X[:,1]*(2*numpy.pi)/(lTh*(lTh+1.))

    # binned theory

    lbTh, cbTh = fftTools.binTheoryPower(lTh,clTh,p['binningFile'])
    lL,lU,lC = fftTools.readBinningFile(p['binningFile'])

    print len(cbTh), len(lbin)

    id = numpy.where(lU < p['trimAtL'])

    cbTh = cbTh[id]
    print len(cbTh), len(lbin)
    print lbTh[id]
    print lbin
    if len(cbTh) < len(lbin):
        cbTh.append(numpy.zeros(len(lbin)-len(cbTh)))
    print len(cbTh), len(lbin)
    clCrossWeighted = []
    wls = []
    for iPatch in xrange(nPatches):

        Nl_aa = clAutos[freqs[0]][iPatch] - clCross[iPatch]
        Nl_bb = clAutos[freqs[-1]][iPatch] - clCross[iPatch]
        Nl_ab = clAutoCF[iPatch] - clCross[iPatch]


        if nDivs == 1:  
            theta = 2*cbTh**2 + cbTh*(Nl_aa+Nl_bb) + 2*cbTh*Nl_ab + \
                    (Nl_aa*Nl_bb+Nl_ab*Nl_ab)
        else:
            theta = 2*cbTh**2 + cbTh*(Nl_aa+Nl_bb)/nDivs + 2/nDivs*cbTh*Nl_ab + \
                    (Nl_aa*Nl_bb+Nl_ab*Nl_ab)/(nDivs*(nDivs-1.0))
        theta /= (2*binWeight)
        fName = "%s/sigmaSqBin_%dX%d_%03d.dat"%(specDir,freqs[0],freqs[-1],iPatch)
        speckMisc.writeBinnedSpectrum(lbin,theta,binWeight,fName)
        clCrossWeighted += [clCross[iPatch]/theta]
        wls += [1./theta]

    clCrossWeightedMean = numpy.sum(clCrossWeighted,axis=0)/numpy.sum(wls,axis=0)

    fName = "%s/clBinCrossGlobalWeightedMean_%dX%d.dat"%(specDir,freqs[0],freqs[-1])
    speckMisc.writeBinnedSpectrum(lbin,clCrossWeightedMean,binWeight,fName)
    
    return 0 # success
    
    
def packageResults(p):
    """
    @brief group the results nicely for easy interpretation
    """
    
    freqs = p['frequencies']
    
    if p['fileTag'] is None:
        fileTag = ""
    else:
        fileTag = p['fileTag']

    assert(os.path.exists("spectra/spectrum_%dx%d_%s.dat"%(freqs[0], freqs[-1], fileTag )))

    resultsDir = p['resultsDir']
    if not os.path.exists(resultsDir):
        os.makedirs(resultsDir)

    if not os.path.exists("%s/data/" %resultsDir):
        os.makedirs("%s/data/" %resultsDir)

    if not os.path.exists("%s/plots/" %resultsDir):
        os.makedirs("%s/plots/" %resultsDir)

    os.system("mv spectra/*%s*png %s/plots/" %(p['fileTag'], resultsDir))
    os.system("mv spectra/spectrum*%s*dat %s/data/" %(p['fileTag'], resultsDir))
    
    return 0 #success
    
def plotSpectrum(p):
    """
    @brief plot the spectrum
    """
    
    freqs = p['frequencies']
    theoryFile = p['theoryFile']
    
    if p['fileTag'] is None:
        fileTag = ""
    else:
        fileTag = p['fileTag']
    
    specDir ='spectra/'

    # get the data spectrum
    l,cl,clerr = numpy.loadtxt("%s/spectrum_%dx%d_%s.dat"%(specDir,freqs[0],freqs[-1], fileTag),unpack=True)

    pylab.cla()
    # get the theory spectrum, assumed to be of form clTh*lTh*(lTh+1)/2pi
    if theoryFile is not None:
        X = numpy.loadtxt(theoryFile)
        lTh = X[:,0]
        clTh = X[:,1]
    
    # make a log-log plot
    pylab.gca().set_xscale('log')
    pylab.gca().set_yscale('log')

    # plot the data
    pylab.errorbar(l, cl*l*(l+1)/2./numpy.pi, clerr, ls='', marker='.', markersize=2)
    
    # plot the theory
    if theoryFile is not None:
        pylab.plot(lTh, clTh*p['modelCalib_148x148'], c='k')
    
    pylab.xlabel(r"$\mathrm{\ell}$", fontsize=16)
    pylab.ylabel(r"$\mathrm{C_{\ell}}$", fontsize=16)
    
    pylab.gca().set_xlim(left=10.0)
    pylab.savefig("%s/spectrum_%dx%d_%s.png"%(specDir,freqs[0],freqs[-1], fileTag))
    pylab.close()

    
    return 0 # success
    