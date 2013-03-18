from flipper import *
import speckMisc

def apoCorrection(specDir,patchDir,p):
    """
    Correction due to apodization and pixel weights
    """
    freqs = p['frequencies']
    facs = []
    dir = specDir
    q = p.copy()
    for i in xrange(nPatches):
        m = liteMap.liteMapFromFits("%s/patch_%s_%03d_0"%\
                                    (patchDir,freqs[0],i))
        m.data[:] = 1.0
        
        
        gA = p['gaussianApodization']
        g = m.createGaussianApodization(pad = gA['pad'], kern = gA['kern'])
        if gA['apply']:
            print 'apply gauss apo'
            m.data[:] *= g.data[:]
        if p['applyMask']:
            print 'apply mask'
            mask = liteMap.liteMapFromFits("%s/mask%03d"%(patchDir,i))
            m.data[:] *= mask.data[:]
            
            
        if p['applyPixelWeights']:
            print 'apply pixel weights'
            pixw = liteMap.liteMapFromFits("%s/totalWeightMap_%03d"\
                                           %(patchDir,i))
            m.data[:] *= pixw.data[:]
            
        factor= numpy.mean(m.data**4)/(numpy.mean(m.data**2))**2/numpy.mean(g.data[:])#*numpy.max(m.data[:])
        facs += [factor]
        print i,  numpy.sqrt(factor)
        
    return numpy.sqrt(numpy.mean(facs))  


def poissonError(specDir,patchDir,freq,p):
    """
    The poisson error budget 
    """
    # Estimate fsky
    map = liteMap.liteMapFromFits("%s/patch_%s_000_0"%\
                                (patchDir,freqs[0]))
    radToDeg = 180./numpy.pi
    area = map.pixScaleX*map.pixScaleY*map.Nx*map.Ny*(radToDeg)**2
    fsky  = area/(4*numpy.pi*radToDeg**2)
    # read in beam
    beamFile =  os.environ['SPECK_DIR']+'/data/'+p['beamFile_%s'%freq]
    l, bl = numpy.loadtxt(beamFile,unpack=True)
    pi = numpy.pi
    pfac = sum(2.0*bl*bl*l)/(4.0*pi)
    pAmps = p['poissonAmps_%s'%freq]
    poissonPiece = ((pAmps['P4']-3*pAmps['P2']**2)/pfac**3)/(4.0*pi*fsky)
    return poissonPiece
    
def computeAnalyticError(p):
    """
    @brief compute the analytic error for cross spectrum
    """

    specDir = 'spectra/'
    patchDir = 'patches'

    freqs = p['frequencies']
    powerOfL = p['powerOfL']
    global nPatches
    nDivs, nPatches = speckMisc.getPatchStats(patchDir,freqs[0])
    print "Found %d patches with %d sub-season divisions in each"%(nPatches, nDivs)

    if p['fileTag'] is None:
        fileTag = ""
    else:
        fileTag = p['fileTag']

    # Read in the single freq auto spectra first
    
    # (C_b + N_b)^AA

    lbin,clbin_aa,binWeight = numpy.loadtxt("%s/clBinAutoGlobalMean_%dX%d.dat"\
                                            %(specDir,freqs[0],freqs[0]),unpack=True)

    # (C_b + N_b)^BB

    lbin,clbin_bb,binWeight = numpy.loadtxt("%s/clBinAutoGlobalMean_%dX%d.dat"\
                                            %(specDir,freqs[-1],freqs[-1]),unpack=True)

    # (C_b+N_b)^{AB}

    lbin,clbin_ab,binWeight = numpy.loadtxt("%s/clBinAutoGlobalMean_%dX%d.dat"\
                                            %(specDir,freqs[0],freqs[-1]),unpack=True)

    # Cross spectrum

    lbin,clbin_cross,binWeight = numpy.loadtxt("%s/clBinCrossGlobalWeightedMean_%dX%d.dat"\
                                            %(specDir,freqs[0],freqs[-1]),unpack=True)


    print clbin_cross
    clbin_aa /= lbin**powerOfL
    clbin_bb /= lbin**powerOfL
    clbin_ab /= lbin**powerOfL
    clbin_cross /= lbin**powerOfL

    # Uncertainty in a patch
    # noise spectra
    N_b_aa = (clbin_aa - clbin_cross)
    N_b_bb = (clbin_bb - clbin_cross)
    N_b_ab = (clbin_ab - clbin_cross)

    print "N_b_ab",N_b_ab
    print "N_b_aa",N_b_aa
    print "N_b_bb",N_b_bb

    if len(freqs) == 1:
        g = open("%s/Nlbin_season_%sx%s.dat"%(specDir,freqs[0],freqs[0]),"w")
        for i in xrange(len(lbin)):
            g.write("%d %e\n"%(lbin[i],N_b_aa[i]/nDivs))
        g.close()


    C_b = clbin_cross

    fac = 2.0
    apoCorr = 1.
    
    # For auto spectra, implement the apodization correction
    
    if freqs[0]== freqs[-1]: apoCorr = apoCorrection(specDir,patchDir,p)
    print "Apodization correction = %s"%apoCorr


    poissonPiece = 0.
    # Add in the poisson piece (for 148 only now)
    if freqs[0] == freqs[-1] and freqs[0] == 148 and p['poissonAmps_148'] != None:
        poissonPiece = poissonError(specDir,patchDir,freqs[0],p)

    
    print "Poisson Estimate = %s"%poissonPiece
    
    if nDivs == 1:
        Theta = (clbin_aa*clbin_bb)/(fac*binWeight)
    else:
        Theta = (2*C_b**2 + C_b*(N_b_aa + N_b_bb)/nDivs + \
                 2./nDivs*C_b*N_b_ab \
                 + (N_b_aa*N_b_bb + N_b_ab*N_b_ab)/(nDivs*(nDivs-1)))\
                /(fac*binWeight) + poissonPiece

    C_b_err = numpy.sqrt(Theta/nPatches)*apoCorr
    
    
    g = open('%s/spectrum_%dx%d_%s.dat'%(specDir,freqs[0],freqs[-1], fileTag),"w")
    for k  in xrange(len(lbin)):
        g.write("%d %e %e\n"%(lbin[k],lbin[k]**powerOfL*C_b[k],lbin[k]**powerOfL*C_b_err[k]))
    g.close()
    
    return 0 # success
    
    