"""
Process a map + weight map pair, to make Figure 2 for the first Power Spectrum
paper.  This means the class can:

* Load a full-season map, hit map, and two half-season maps.
* Trim and filter the maps.
* Find a map-wide average for the single-hit variance.
* Draw the results with patches indicated.

J. Fowler, Princeton
December 1-2, 2009
"""

import os
import liteMap, utils
import numpy, pylab, scipy, astLib
import matplotlib.patches as patches

class dummy(object): pass

DEFAULT = dummy()
DEFAULT.map_root = "/scr/queequeg1/shared/maps"
prefix = "ACT_148_v1.4"
suffix = "noprior_gainly_15Dec09"
DEFAULT.map_dir = os.path.join(DEFAULT.map_root, "ACT_148_v1.4_all_noprior_gainly_15Dec09")

DEFAULT.map = os.path.join(DEFAULT.map_dir, "ACT_148_v1.3_all_15Dec09_downweight_0.25_find_10_modes_detau_noise_dedark_debutter_noprior_no_carpets_gainly_1000.fits")
DEFAULT.weight = DEFAULT.map.replace("1000","weights")

DEFAULT.twoway = [os.path.join(DEFAULT.map_dir.replace("all","2way"), "ACT_148_v1.4_2way_bigset_0_myset_%d_15Dec09_downweight_0.25_find_10_modes_detau_noise_dedark_debutter_noprior_no_carpets_gainly_150.fits"%i) for i in range(2)]

DEFAULT.fourway = [os.path.join(DEFAULT.map_dir.replace("all","4way"), "ACT_148_v1.4_4way_bigset_0_myset_%d_15Dec09_downweight_0.25_find_10_modes_detau_noise_dedark_debutter_noprior_no_carpets_gainly_1000.fits"%i) for i in range(4)]

# Kludge: use 4way as 2way
#DEFAULT.twoway = DEFAULT.fourway

CAL_CORRECTION=0.87

class ImagePlot(astLib.astPlots.ImagePlot):
    """Improve on Matt Hilton's astLib.astPlots.ImagePlot object by storing possible
    rectangular boxes for drawing on top of the image."""
    
    def __init__(self, *args, **kwargs):
        self.plotBoxes = []
        astLib.astPlots.ImagePlot.__init__(self, *args, **kwargs)

    def addPlotBoxes(self, RAs, Decs, tag, width=1.0, color="yellow"):
        """Add boxes with the given RA and Dec ranges to the plot.
        RAs and Decs should be (n,2) arrays for n boxes, or (2) arrays for a single box.
        The set is given the name tag, so they can be deleted later.
        """
        
        xMax=self.data.shape[1]
        yMax=self.data.shape[0]
        
        xInPlot=[]
        yInPlot=[]
        RAInPlot=[]
        decInPlot=[]
        for r, d in zip(RAs, Decs):
            corners = self.wcs.wcs2pix(r,d)
            #p2 = self.wcs.wcs2pix(r[-1::-1], d[-1::-1])
            #corners = numpy.vstack((p1,p2))
            valid = True
            for p in corners:
                if p[0] >= 0 and p[0] < xMax and p[1] >= 0 and p[1] < yMax:
                    pass
                else:
                    valid = False
                    print "box dims not valid"
                    break
            if valid:
                for p in corners:
                    xInPlot.append(p[0])
                    yInPlot.append(p[1])
                    ra,dec = self.wcs.pix2wcs( p[0], p[1])
                    RAInPlot.append(ra)
                    decInPlot.append(dec)
        
        xInPlot=numpy.array(xInPlot)
        yInPlot=numpy.array(yInPlot)
        RAInPlot=numpy.array(RAInPlot)
        decInPlot=numpy.array(decInPlot)
        
        alreadyGot=False
        for p in self.plotBoxes:
            if p['tag'] == tag:
                p['x']=xInPlot
                p['y']=yInPlot
                p['RA']=RAInPlot
                p['dec']=decInPlot
                p['tag']=tag
                p['width']=width
                p['color']=color
                alreadyGot=True
                break
        if alreadyGot == False:
            self.plotBoxes.append({'x': xInPlot, 'y': yInPlot, 'RA': RAInPlot, 'dec': decInPlot,
                                'tag': tag, 'width': width, 'color': color})
        self.draw()


    def draw(self):
        astLib.astPlots.ImagePlot.draw(self)
        for b in self.plotBoxes:
            for i in range(0,len(b['x']), 2):
                x = b['x'][i:i+2]
                y = b['y'][i:i+2]
                c = patches.Rectangle((x.min(), y.min()), numpy.abs(x[1]-x[0]), numpy.abs(y[1]-y[0]),
                        fill=False, edgecolor=b['color'], linewidth=b['width'])
                self.axes.add_patch(c)


def filterFromList(map,lFl,setMeanToZero=False, minKx=90.):
    """
    @brief Given an l-space filter as seq [\f$\ell,F_\ell\f$], return filtered map
    @param lFl A tuple [\f$\ell,F_\ell\f$], where \f$\ell\f$ and \f$ F_\ell \f$
    are 1-D arrays representing the filter.
    @param minKx Set all modes with |k_x|<minKx to zero.
    @return The filtered liteMap

    Replaces the liteMap _method_ filterFromList so we can also have the
    minKx parameter
    """
    ft = liteMap.fftFromLiteMap(map)
    if minKx>0.:
        ft.kMap[:,numpy.abs(ft.lx) < minKx] = 0.0
    filtData = ft.mapFromFFT(kFilterFromList=lFl,setMeanToZero=setMeanToZero)
    filtMap = map.copy()
    filtMap.data[:] = filtData[:]
    del ft
    del filtData
    return filtMap




class SkyMap(object):

    def __init__(self, existingSkyMap=None, 
                 mapfile=DEFAULT.map, weightfile=DEFAULT.weight,
                 twowayfiles = DEFAULT.twoway, trim=True,
                 raRange=[0,105], decRange=[-56,-50.33],cal_correction= 1.0):

        if existingSkyMap:
            for a in ("map","fmap","wt","twoway","ftwoway","sensmap","diff"):
                try:
                    self.__dict__[a] = existingSkyMap.__dict__[a]
                except KeyError:
                    pass
            return

        map = liteMap.liteMapFromFits(mapfile)
        wt = liteMap.liteMapFromFits(weightfile)

        if len(twowayfiles) == 4:
            twoway = self.use4way_for_2way(twowayfiles)
        else:
            twoway = [liteMap.liteMapFromFits(f) for f in twowayfiles]
            
        if trim:
            map = map.selectSubMap( raRange[0], raRange[1], decRange[0], decRange[1] )
            wt = wt.selectSubMap( raRange[0], raRange[1], decRange[0], decRange[1] )
            twoway = [tw.selectSubMap( raRange[0], raRange[1], decRange[0], decRange[1] )
                      for tw in twoway]
        print 'Applying calibration factor',cal_correction
        map.data *= cal_correction
        for t in twoway: t.data *= cal_correction

        self.map = map
        self.wt = wt
        self.twoway = twoway
        self.fmap = self.ftwoway = self.sensmap = self.diff = None # will computer later

    def use4way_for_2way(self, fourwayfiles):
        """A kludge to average 4way maps 0+1 and 2+3 to emulate a 2-way split."""
        print "Kludge: using 4-ways as if 2-ways"
        assert len(fourwayfiles)==4
        maps = [liteMap.liteMapFromFits(f) for f in fourwayfiles]
        weightfiles = [f.replace("1000","weights") for f in fourwayfiles]
        wts = [liteMap.liteMapFromFits(f) for f in weightfiles]

        m1 = maps[0].copy()
        m2 = maps[2].copy()

        m1.data = (maps[0].data + maps[2].data)*0.5
        m2.data = (maps[1].data + maps[3].data)*0.5
        
        # Do a weighted sum
        w = [count.data for count in wts]

        #m1.data[::] = ((maps[1].data*w[1] + maps[2].data*w[2])/(w[1]+w[2]))[::]
        #m2.data[::] = ((maps[0].data*w[0] + maps[3].data*w[3])/(w[0]+w[3]))[::]
        return [m1,m2]
    

    def filter(self, minEll=300, deltaEll=400, maxEll=12000,
               plotFilter=False, plotMap= False,
               valueRange=[-250,250]):
        """Apply a filter that rises as sin^2 in the given ell band.
        The 0.50 point is at minEll.  The full width is deltaEll.
        A separate Gaussian low-pass filter falls to 60% at maxEll.
        """

        ell = numpy.arange(0,20000,10.)
        lowestEll = (minEll-0.5*deltaEll)
        Fell = numpy.sin(numpy.pi/2*(ell-lowestEll)/(deltaEll))**2
        Fell[ell<minEll-0.5*deltaEll] = 0.0
        Fell[ell>minEll+0.5*deltaEll] = 1.0
        Fell *= numpy.exp(-0.5*(ell/maxEll)**2)

        if plotFilter:
            pylab.clf()
            pylab.plot(ell,Fell)
            pylab.xlim([10,20000])
            pylab.semilogx()
            utils.saveAndShow()
            print "The filters are plotted"

        self.fmap = filterFromList(self.map, [ell, Fell])
        print "The map is filtered now"

        if plotMap:
            plotSky(self.fmap, boxes=True)
        self.ftwoway = [filterFromList(m, [ell, Fell]) for m in self.twoway]

        if plotMap:
            pylab.clf()
            self.ftwoway[0].plot(valueRange=valueRange,
                                 colBarOrient='horizontal', colorMapName='copper',
                                 axesLabels='sexagesimal', RATickSteps='auto')
            utils.saveAndShow()
            pylab.clf()
            self.ftwoway[1].plot(valueRange=valueRange,
                                 colBarOrient='horizontal', colorMapName='copper',
                                 axesLabels='sexagesimal', RATickSteps='auto')
            utils.saveAndShow()

    def compute_variance(self, minHits=2000, plot=False):
        """Compute the average variance across the map of temperature times sqrt(N)
        for N hits in each pixel.  Given this number V, we can convert the hit map
        (of N) to a sensitivity map (of sqrt(V/N))."""

        # Make 1d vectors of map, weights
        if self.diff:
            mv = self.diff.data.ravel()
            wv = self.wt.data.ravel()
        else:
            mv = self.fmap.data.ravel()
            wv = self.wt.data.ravel()
        mv = mv[wv>minHits]
        wv = wv[wv>minHits]

        tn = numpy.sqrt(wv) * mv
        contents,bins,_ignore = pylab.hist(tn, 150, [-2e5, 2e5], fc='gold')

        # Estimate a gaussian by the moments
        mu = tn.mean()
        sig = tn.std()
        y0 = contents.max()
        binctr = 0.5*(bins[1:] + bins[:-1])

        print "Moments    mu, sig: ",mu,sig
        if plot: pylab.plot( binctr, y0*numpy.exp(-.5*((binctr-mu)/sig)**2), 'red')

        # Estimate a different gaussian by the 16, 84 %ile points
        h= numpy.cumsum(contents)*1.0
        h /= h[-1]
        f = scipy.interpolate.interp1d(h, binctr)
        p16 = f(0.1587)  # The -1 sigma point on a Gaussian
        p84 = f(0.8413)  # The +1 sigma point on a Gaussian
        sig = 0.5*(p84-p16)
        median = f(0.5)
        print "Percentile mu, sig: ",median,sig
        if plot: pylab.plot( binctr, y0*numpy.exp(-.5*((binctr-median)/sig)**2), 'blue')

        print "The raw one-hit variance is %.5g, or sqrt(V)=%.2f"%(
            sig**2, sig)
        size_arcmin = numpy.sqrt(self.fmap.pixScaleX*self.fmap.pixScaleY)* \
                      180/numpy.pi*60.
        sig *= size_arcmin
        self.StdDevHit = sig
        print "The calibrated one-hit std dev is sqrt(V)=%.2f uK-arcmin"%sig
        if plot: utils.saveAndShow()
    
        self.sensmap = self.wt.copy()
        self.sensmap.data[self.sensmap.data<100] = 100
        self.sensmap.data = self.StdDevHit / numpy.sqrt(self.sensmap.data)

        if plot:
            pylab.clf()
            self.sensmap.plot(valueRange=[25,75],
                              colBarOrient='horizontal', colorMapName='hot',
                              axesLabels='sexagesimal', RATickSteps='auto')
            utils.saveAndShow()


    def compute_diff(self):
        self.diff = self.ftwoway[0].copy()
        self.diff.data = 0.5*(self.ftwoway[0].data - self.ftwoway[1].data)

    def plot(self, mapname="fmap", **kwargs):
        mapdict = {"map"    :self.map,
                   "fmap"   :self.fmap,
                   "diff"   :self.diff,
                   "hit"    :self.wt,
                   "sens"   :self.sensmap,
                   "split0" :self.ftwoway[0],
                   "split1" :self.ftwoway[1],
                   "rawsplit0" :self.twoway[0],
                   "rawsplit1" :self.twoway[1]}
        try:
            map = mapdict[mapname]
        except KeyError:
            raise KeyError("The plot() method allows mapnames: %s"%mapdict.keys())
        plotSky(map, **kwargs)


def goodPlots(sm=None):
    if sm is None:
        sm = SkyMap()
        sm.filter()
        sm.compute_variance()

    plotSensitivity(sm)

def plotSensitivity(sm):
    """Histogram the sensitivity in all pixels"""

    data = sm.sensmap.data.ravel()
    dummy = pylab.hist(data, 75, [25,100], fc='red')
    pylab.xlabel("Sensitivity in $\mathrm{\mu}$K$_{\mathrm{cmb}}$-arcmin")
    pylab.ylabel("Pixels")
    pylab.title("Pixel sensitivity distribution derived from hit map")
    pylab.text(90, 400000, "Mean: %.3f uK"%data[data<100].mean(), ha='right')
    pylab.text(90, 370000, "Median: %.3f uK"%numpy.median(data[data<100]), ha='right')
    pylab.text(90, 340000, "RMS: %.3f uK"%data[data<100].std(), ha='right')
    utils.saveAndShow("sensitivity_histogram.png")

def plotValues(sm):
    """Histogram the data (temperature) values in the filtered map and difference map"""

    ax = pylab.subplot(111)

    x0,y0 = sm.map.skyToPix(102.70, -55.263)
    x1,y1 = sm.map.skyToPix(12.00, -51.0854)
    print x0,x1,y0,y1

    data = sm.diff.data[y0:y1,x0:x1].ravel()
    hist = pylab.hist(data, 200, [-400,400],
                      histtype='step', ec='red', axes=ax)
    pylab.text(100,55000,'Values in CMB microK:')
    pylab.text(350,50000,'Diff mean = %6.2f'%data.mean(), ha='right',color='red')
    pylab.text(350,45000,'Diff rms  = %6.2f'%data.std(), ha='right',color='red')

    data = sm.fmap.data[y0:y1,x0:x1].ravel()
    hist = pylab.hist(data, 200, [-400,400],
                      histtype='step', ec='green', axes=ax)
    pylab.text(350,40000,'Data mean = %6.2f'%data.mean(), ha='right',color='green')
    pylab.text(350,35000,'Data rms  = %6.2f'%data.std(), ha='right',color='green')

    utils.saveAndShow("pixel_histogram.png")
    

def plotSky(map, boxes=True, colorMapName='jet', show=True, colorBar=True,
            cutLevels=[-200,200], **kwargs):

    ip = ImagePlot( map.data, map.wcs, colorMapName=colorMapName,
                    cutLevels=cutLevels, axesFontFamily='sans-serif',
                    **kwargs)

    if boxes:
        RAs=[]
        Decs=[]
        dra = (102.700630-12.007338)/13.
        for i in range(13):
            RAs.append([12.07338+dra*i, 12.07338+dra*(i+1)])
            Decs.append([-55.263018, -51.085420])
        ip.addPlotBoxes(RAs, Decs, 'patches', color='black', width=1.5)

    pylab.ylabel("Dec.")

    if colorBar: cb = pylab.colorbar(orientation='horizontal', shrink=.8,
                                     aspect=80, pad=.375)
    if show: utils.saveAndShow()
    return ip


def plotSky_das(map, boxRAs= None, boxDecs = None, colorMapName='jet', show=True, colorBar=True,
            cutLevels=[-200,200], **kwargs):

    ip = ImagePlot( map.data, map.wcs, colorMapName=colorMapName,
                    cutLevels=cutLevels, axesFontFamily='sans-serif',
                    **kwargs)

    if boxRAs != None:
        print "boxRAs", boxRAs
        assert(len(boxRAs) == len(boxDecs))
        ip.addPlotBoxes(boxRAs, boxDecs, 'patches', color='black', width=1.5)

    pylab.ylabel("Dec.")

    if colorBar: cb = pylab.colorbar(orientation='horizontal', shrink=.8,
                                     aspect=80, pad=.375)
    if show: utils.saveAndShow()
    return ip

def figForPaper_das(sm, filename=".tmp.png",boxRAs=None,boxDecs=None):
    fig = pylab.figure(figsize=(8,6.))
    
    ax1 = fig.add_axes([.08,.70,.9, .3])
    ax2 = fig.add_axes([.08,.45,.9, .3])
    ax3 = fig.add_axes([.08,.05,.9, .3])
    plotSky_das(sm.fmap, axes=ax1, colorMapName='hot', colorBar=False, show=False,boxRAs=boxRAs,\
            boxDecs = boxDecs)
    ax1.set_xlabel("")
    #ax1.set_xlabel("xxxxx",position=(.1,.1))
    ax1.set_xticks(())
    ax1.set_title("Sum and difference maps ($\mu$K)", size='medium')

    plotSky_das(sm.diff, axes=ax2, colorMapName='hot', show=False,boxRAs = boxRAs,boxDecs = boxDecs)
    ax2.set_xlabel("R.A. (J2000)",ha='center')

    plotSky_das(sm.sensmap, axes=ax3, cutLevels=[10,50], colorMapName='gray', show=False,boxRAs=boxRAs,\
            boxDecs = boxDecs)
    ax3.set_title("Sensitivity ($\mu$K-arcmin)", size='medium')
    pylab.savefig(filename)
    # utils.saveAndShow(filename)

def figForPaper_das_v2(sm, filename=".tmp.png",boxRAs=None,boxDecs=None,cutLevels=[10,60]):
    fig = pylab.figure(figsize=(8,2.5))
    
    ax1 = fig.add_axes([.08,.05,.9, .9])
    #ax2 = fig.add_axes([.08,.45,.9, .3])
    #ax3 = fig.add_axes([.08,.05,.9, .3])
    ax1.set_xlabel("R.A. (J2000)",ha='center')

    plotSky_das(sm.sensmap, axes=ax1, cutLevels=cutLevels, colorMapName='gray', show=False,boxRAs=boxRAs,\
                boxDecs = boxDecs)
    ax1.set_title("Sensitivity ($\mu$K-arcmin)", size='medium')
    pylab.savefig(filename)
    # utils.saveAndShow(filename)






"""
sm3 = skyMap.SkyMap()
sm3.filter(minEll=200, deltaEll=200, maxEll=12000)
sm3.compute_diff()
sm3.compute_variance()
skyMap.figForPaper(sm3, "maps_filter100_300.png")

reload(skyMap)
sm5 = skyMap.SkyMap(twowayfiles=skyMap.DEFAULT.fourway)
sm5.filter(minEll=300, deltaEll=400, maxEll=12000)
sm5.compute_diff()
sm5.compute_variance()
skyMap.figForPaper(sm5, "maps_filter100_500.png")

sm7 = skyMap.SkyMap()
sm7.filter(minEll=400, deltaEll=600, maxEll=12000)
sm7.compute_diff()
sm7.compute_variance()
skyMap.figForPaper(sm7, "maps_filter100_700.png")
"""
