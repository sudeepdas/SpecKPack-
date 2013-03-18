#!/usr/bin/env python


from flipper import *
#from scipy.integrate import 
from scipy.interpolate import splev,splrep
from flipper import *
from numpy.fft import fftshift,fftfreq,fft2,ifft2, ifftshift
from scipy import interpolate
from scipy import *
import scipy
import os
import random
import sys
import pickle
import time
import pylab



def makeTemplate(m, wl, ell, maxEll, outputFile = None):
    """
    Yanked from Toby's csFilter
    For a given map (m) return a 2D k-space template from a 1D specification wl
    ell = 2pi * i / deltaX
    (m is not overwritten)
    """

    ell = numpy.array(ell)
    wl  = numpy.array(wl)
    
    p2d = fftTools.powerFromLiteMap(m)
    p2d.powerMap[:] = 0.
    # print "max_lx, max_ly", p2d.lx.max(), p2d.ly.max()
    # print "m_dx, m_dy", m.pixScaleX, m.pixScaleY
    # print "m_nx, m_ny", m.Nx, m.Ny
    l_f = numpy.floor(p2d.modLMap)
    l_c = numpy.ceil(p2d.modLMap)
    
    for i in xrange(numpy.shape(p2d.powerMap)[0]):
        for j in xrange(numpy.shape(p2d.powerMap)[1]):
            if l_f[i,j] > maxEll or l_c[i,j] > maxEll:
                continue
            w_lo = wl[l_f[i,j]]
            w_hi = wl[l_c[i,j]]
            trueL = p2d.modLMap[i,j]
            w = (w_hi-w_lo)*(trueL - l_f[i,j]) + w_lo
            p2d.powerMap[i,j] = w



            
    if outputFile != None:
        p2d.writeFits(outputFile, overWrite = True)
    return p2d



def trimShiftKMap(kMap,elTrim,Lx,Ly,lx,ly):
    """
    @brief Trims a 2-D powerMap at a certain Lx, Ly, and returns the trimmed kMap object
    trimmed at elTrim and shifted by Lx and Ly
    """
    # assert(elTrim>0.)
    
    # assert((lx[0] != 0.) and (ly[0] != 0.))
    
    #kM = kMap.copy()
    ta = time.time()
    idx = numpy.where((lx < elTrim+Lx) & (lx > -elTrim+Lx))
    idy = numpy.where((ly < elTrim+Ly) & (ly > -elTrim+Ly))
    # print "numpy.where takes %f secs"%(time.time()-ta)
    trimA = kMap[idy[0],:]
    trimB = trimA[:,idx[0]]
    kTrimmed = trimB
    del trimA,trimB
    # print "plus trim  takes %f secs"%(time.time()-ta)
    return kTrimmed


def generateMCM(window,binLo,binHi,\
                trimAtL=10000,\
                mbbFilename = None,\
                transfer = None,\
                binningWeightMap = None,\
                powerOfL = 0):
    """
    window: data window
    
    """
    powerMaskHolder = fftTools.powerFromLiteMap(window)
    powerMask = powerMaskHolder.powerMap.copy()
    powerMaskShifted = fftshift(powerMask)
    phlx = fftshift(powerMaskHolder.lx)
    phly = fftshift(powerMaskHolder.ly)
    if transfer != None:
        ell, f_ell = transfer
        t = makeTemplate( window.copy(), f_ell, ell, trimAtL )
        pixW = powerMaskHolder.pixelWindow()
        t.powerMap *= pixW
        transferTrim = t.trimAtL(trimAtL)
        
        
    if binningWeightMap ==None:
        binningWeightMap = powerMask.copy()*0.+1.0
    else:
        assert(binningWeightMap.shape == powerMask.shape)
        
        
    powerMaskHolderTrim = powerMaskHolder.trimAtL(trimAtL)
    
    powerMaskTrim = powerMaskHolderTrim.powerMap.copy()
    print "Trimmed kMap dims",powerMaskHolderTrim.Ny,powerMaskHolderTrim.Nx
    assert(binHi.max() <= trimAtL)
    
    pMMaps = []
    #bins = [[0.,200.],[200.,400.],[400.,700.],[700.,1200.],[1200,1800],[1800,2500],[2500,3500]]

    mArray = numpy.zeros(shape=(len(binLo),len(binLo)))
    Bbl = numpy.zeros(shape=(len(binLo),numpy.int(trimAtL)))
    
    modIntLMap = numpy.array(powerMaskHolder.modLMap + 0.5,dtype='int64')
    t00 = time.time()
    cumTerms = 0
    
    for ibin in xrange(len(binLo)):
        t0 = time.time()
            
        location = numpy.where((modIntLMap >= binLo[ibin]) & (modIntLMap <= binHi[ibin]))
        print "where on modIntLMap took %f secs"%(time.time()-t0)
        binMap = powerMask.copy()*0.
        binMap[location] = binningWeightMap[location]
        sumBin = binMap.sum()
        binMap[location] *= powerMaskHolder.modLMap[location]**powerOfL
        #print binningWeightMap[location]
        if sumBin == 0:
            print window.x0, window.x1
        assert(sumBin>0.)
        
        binMap0 = (trimShiftKMap(powerMaskShifted,trimAtL,0,0,\
                                               phlx,phly))*0.

        
        t0 = time.time()
        deltaT = 0.
        cumTerms += len(location[0])
        for i in xrange(len(location[0])):
            ly = powerMaskHolder.ly[location[0][i]]
            lx = powerMaskHolder.lx[location[1][i]]
            #  print ly, lx, trimAtL
            t000 = time.time()
            binMap0 += binMap[location[0][i],location[1][i]]*\
                       (trimShiftKMap(powerMaskShifted,trimAtL,lx,ly,\
                                      phlx,phly))
            deltaT += (time.time() -t000)
            # ifftshift(trimShiftKMap(fftshift(powerMask),trimAtL,lx,ly,\
            #                    fftshift(powerMaskHolder.lx),\
            #                    fftshift(powerMaskHolder.ly)))
            # print "Term %d takes %f secs"%(i,time.time() - t000)
            # print a.shape,(binMap[location[0][i],location[1][i]])
            # binMap[location[0][i],location[1][i]]*a
            # print  "trimShift done in %f secs"%(time.time() -t00)
        print "Total time in trimshift calls %f secs"%deltaT
        print "Time to do sum  of %d terms %f secs"%(len(location[0]),time.time() - t0)
        print "Best estimate of time per step %f"%((time.time() - t00)/cumTerms)
        binMap0 = ifftshift(binMap0)
        if transfer != None:
            pMMaps.append(binMap0[:]*transferTrim.powerMap[:]/sumBin)
        else:
            pMMaps.append(binMap0/sumBin)
        print 'Bin %d (%f,%f) of %d bins took %f secs to compute'\
              %(ibin, binLo[ibin],binHi[ibin],len(binLo),(time.time() - t0))
    print "PM maps done in %f secs"%(time.time() - t00)
        
    larray = numpy.arange(numpy.int(trimAtL))
    deltaLx = numpy.abs(powerMaskHolderTrim.modLMap[0,1] - powerMaskHolderTrim.modLMap[0,0])
    deltaLy = numpy.abs(powerMaskHolderTrim.modLMap[1,0] - powerMaskHolderTrim.modLMap[0,0])
    delta = numpy.min([deltaLx,deltaLy])/2.0
    gaussFactor = 1./numpy.sqrt(2.*numpy.pi*delta**2.)

    modLMapInt = numpy.array(powerMaskHolderTrim.modLMap+0.5, dtype='int64')

    t0 = time.time()
    
    for j in xrange(len(binHi)):
        location = numpy.where((modLMapInt >= binLo[j]) &\
                               (modLMapInt <= binHi[j]))
        binMapTrim = powerMaskTrim.copy()*0.
        binMapTrim[location] = 1.
        binMapTrim[location] *= numpy.nan_to_num(1./(powerMaskHolderTrim.modLMap[location])**powerOfL)
        for i in xrange(len(pMMaps)):
            newMap = pMMaps[i].copy()
            result = (newMap*binMapTrim).sum()
            mArray[i,j] = result

    print "MArray done in %f secs"%(time.time()-t0)
        
    modMap = numpy.ravel(powerMaskHolderTrim.modLMap)    
    deltaLx = numpy.abs(powerMaskHolderTrim.lx[1]-powerMaskHolderTrim.lx[0])
    deltaLy = numpy.abs(powerMaskHolderTrim.ly[1]-powerMaskHolderTrim.ly[0])
    

    t0 = time.time()
    for k in xrange(len(larray)):
        
        gauss = numpy.exp(-(larray-larray[k])**2./(2.*delta**2.))
        sum = gauss.sum()
        # ta = time.time()
        # cSpl = splrep(larray,gauss/gauss.sum())
        
        # gauss /= gauss.sum()
        # cSP = splev(modMap,cSpl)
        # binMapTrim = numpy.reshape(gauss,[powerMaskHolderTrim.Ny,powerMaskHolderTrim.Nx])
        # binMapTrim = numpy.reshape(cSP,[powerMaskHolderTrim.Ny,powerMaskHolderTrim.Nx])
        # print "time taken in Bbl spline %f"%(time.time()-ta)
        # ta = time.time()
        # gauss = numpy.exp(-(larray-larray[k])**2./(2.*delta**2.))
        # sum = gauss.sum()
        # print sum
        gauss = numpy.exp(-(modMap-larray[k])**2./(2.*delta**2.))
        gauss /= sum 
        binMapTrim = numpy.reshape(gauss,[powerMaskHolderTrim.Ny,powerMaskHolderTrim.Nx])
        
        # print "time taken in Bbl NO spline %f"%(time.time()-ta)
        
        
        
        binMapTrim *= powerMaskHolderTrim.modLMap**powerOfL
        for i in xrange(len(pMMaps)):
            #newMap = pMMaps[i].copy()
            result = (pMMaps[i]*binMapTrim).sum()
            Bbl[i,k] = result

    print "Bbl done in %f secs"%(time.time()-t0)
        
    Bbl = numpy.dot(scipy.linalg.inv(mArray), Bbl)
    
    if mbbFilename != None:
        mbbDir = (mbbFilename.split("/"))[0]+"/"
        mbbFilename = (mbbFilename.split("/"))[-1]
        pickle.dump(mArray,open('%s%s'%(mbbDir,mbbFilename),'w'))
        pickle.dump(Bbl,open('%sBbl.%s'%(mbbDir,mbbFilename),'w'))
    return mArray
