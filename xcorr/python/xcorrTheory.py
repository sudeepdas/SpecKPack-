import numpy as np
import scipy
from flipper import liteMap
from scipy.interpolate import interp1d


def realSpaceCorrFromTheory(ell, c_ell, Fell_1=None, Fell_2=None):
    """
    @brief compute the angular correlation from a power spectrum
    @param ell: the multipoles to sum over (array)
    @param c_ell: the power spectrum (array)
    @param Fell_1: filter #1 in ell space (array)
    @param Fell_2: filter #2 in ell space (array)
    
    @return theta, corr: theta (in degrees), the correlation (array)
    """
    
    thetaDeg = np.arange(100)/99.*1.0 # theta extent in degrees
    theta = thetaDeg*np.pi/180. # now in radians

    corr = []
    # corr = int (j_0(ell*theta)*ell*C_ell*F_ell^2)
    for thet in theta:
       bes = scipy.special.jn(0, thet*ell)
       corr += [np.sum(ell*bes*Fell_1*Fell_2*c_ell)]

    corr = np.array(corr)
    corr /= (2*np.pi)
    
    return thetaDeg, corr
    
def computeXCorrSNR(params):
    """
    @brief return the expected signal to noise ratio for a cross-correlation, 
    given data estimates of the two auto spectra and a theory estimate for the 
    cross correlation power spectrum

    @param params: dictionary containing the following:
            maps: list of names of map files (list)
            dataAuto_1: data file containing auto spectrum of map #1 (str)
            dataAuto_2: data file containing auto spectrum of map #2 (str)
            theoryCross: data file containng the theory cross spectrum (str) 
            
    @return s_to_n: the signal to noise ratio (float)        
    """

    # read in map coverage and compute f_sky
    f_sky = 0.0
    for mapFile in params['maps']:

        m = liteMap.liteMapFromFits(mapFile)
        inds = np.where(m.data != 0.0)[0]
        f_sky += m.pixScaleX*m.pixScaleY*len(inds)

    f_sky /= (4.*np.pi)

    # load the 1st data auto spectrum
    ell_1, cl_1 = np.loadtxt(params['dataAuto_1'], useCols=[0,1], unpack=True)

    # get the bin width in ell space, assuming constant bins
    dl = ell_1[1] - ell_1[0]

    # load the 2nd data auto spectrum
    ell_2, cl_2 = np.loadtxt(params['dataAuto_2'], useCols=[0,1], unpack=True)

    # load the theory cross spectrum
    ell_12, cl_12 = np.loadtxt(params['theoryCross'], useCols=[0,1], unpack=True)

    # put theory curve into same binning
    f = interp1d(ell_12, cl_12)
    cl_12 = f(ell_1)

    # loop over each ell bin to get S/N
    s_to_n = 0.0
    for i, ell in enumerate(ell_1):
        x = (2*ell*dl) * cl_12[i]**2 / (cl_1[i]*cl_2[i] + cl_12[i]**2)
        s_to_n += x

    s_to_n *= f_sky

    s_to_n = np.sqrt(s_to_n)

    return s_to_n