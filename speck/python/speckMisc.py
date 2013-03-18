from flipper import *

def getBinnedInvCosSqFilter(lMin,lMax,binningFile, trimAtL = None):
    """
    Cosine Sq Filter binned

    """
    lL,lU,lbin = fftTools.readBinningFile(binningFile)
    
    if trimAtL != None: 
        idx = numpy.where(lU<trimAtL)
        lbin = lbin[idx]
    filter = lbin*0.+1.0
    id = numpy.where((lbin>lMin) &(lbin<lMax))
    
    filter[id] /=(numpy.cos((lMax - lbin[id])/(lMax-lMin)*numpy.pi/2.))**2 
    idLow =  numpy.where((lbin<lMin))
    filter[idLow] = 0.

    return filter
    
    
def getBinnedBeamTransfer(beamFile,binningFile,trimAtL=None):
    """
    reurns a binned version of the beam transfer function.
    Note: Beam Window = (beam TF)**2
    
    """
    X = numpy.loadtxt(os.environ['SPECK_DIR']+'/data/'+beamFile)
    ell = X[:,0]
    Bell = X[:,1]
    lb, Bb = fftTools.binTheoryPower(ell,Bell,binningFile) 
    if trimAtL != None:
        lL,lU,lCen = fftTools.readBinningFile(binningFile)
        id = numpy.where(lU<trimAtL)
        Bb = Bb[id]
    return Bb


def getPatchStats(patchDir,freq,season=None):
    """
    returns how many patches and how many subseason divisions there are
    """
    nDivs = 0
    nPatches = 0
    if season == None:
        base = ''
    else:
        base = season+'_'
    l = os.listdir(patchDir)
    for il in l:
        if 'all' in il:
            continue
        if 'patch_%d_%s000'%(freq,base) in il:
            nDivs += 1
        if 'patch_%d_%s0'%(freq,base) in il and '_0' in il[-2:]:
            print il
            nPatches += 1

    return nDivs, nPatches

def writeBinnedSpectrum(lbin,clbin,binWeight,fileName):
    
    g = open(fileName,mode="w")
    
    for k in xrange(len(lbin)):
        g.write("%f %e %f\n"%(lbin[k],clbin[k],binWeight[k]))
        
    g.close()
                
        
