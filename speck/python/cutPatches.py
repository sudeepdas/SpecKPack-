from flipper import *
from scipy.interpolate import splrep,splev

def cosineSqFilter(map,lMin,lMax,vkMaskLimits=None):
    filteredMap = map.copy()
    ft = fftTools.fftFromLiteMap(map)
    ell = ft.modLMap
    idSub = numpy.where((ell> lMin) & (ell <lMax))
    idLow = numpy.where(ell<lMin)  
    filter = (ell*0.+1.0)
    filter[idLow] = 0.
    filter[idSub] *= (numpy.cos((lMax - ell[idSub])/(lMax-lMin)*numpy.pi/2.))**2
    ft.kMap[:] *= filter[:]
    if vkMaskLimits != None:
        #Yank the k-mode
        idvk = numpy.where((ft.lx >vkMaskLimits[0]) & (ft.lx<vkMaskLimits[1]))
        ft.kMap[:,idvk] = 0.
    
    filteredMap.data[:] = ft.mapFromFFT()
    return filteredMap
    
def cutPatches(p):
    """
    @brief cut patches of a map, corresponding to params
    
    @param params: dictionary of relevant parameters (dict)
    """
    
    freqs = p['frequencies']
    convertTemplateToMicroK = p['convertTemplateToMicroK']
    if convertTemplateToMicroK==None:
        convertTemplateToMicroK = True
    
    if p['separateMapRegions'] > 1:
        for freq in freqs:
            for i in range(1, p['separateMapRegions']):
                assert( len(p['mapFiles_%d_1'%freq]) == len(p['mapFiles_%d_%d'%(freq, i)]))
    
    
    # loop over all divided map regions and freqs
    for freq in freqs:

        for mapRegion in range(1, p['separateMapRegions']+1):
            print freq
            mapFiles = p['mapFiles_%d_%d'%(freq, mapRegion)]
            map0 = liteMap.liteMapFromFits(mapFiles[0])
            map0.info(showHeader=True)
        
            # continue
            srcFile  = p['sourceTemplateFile_%d_%d' %(freq, mapRegion)]
            if srcFile != None:
                print "Reading in the source template file: %s"%srcFile
                srcMap = liteMap.liteMapFromFits(srcFile)
                if convertTemplateToMicroK:
                    srcMap.convertToMicroKFromJyPerSr(freq*1.0)
            
        
            if (p['maskFile_%d' %mapRegion] != None) and (freq == freqs[0]):
                print "reading mask file" 
                ptSrcMask = liteMap.liteMapFromFits(p['maskFile_%d' %mapRegion])
                maskFileBase = (p['maskFile_%d' %mapRegion].split("/"))[-1]
            
            weightFiles = p['weightFiles_%d_%d' %(freq, mapRegion)]
            if weightFiles != None:
                assert(len(weightFiles) == len(mapFiles))
            
            
            if p['bufferedMapBounds'] is not None:
                ra0, ra1, dec0, dec1 = p['bufferedMapBounds']
                
            patchCoords = sorted(p['patchCoords_%d' %mapRegion]) # sort by ra0
            
            vkMaskLimits = p['verticalkMaskLimits']
            nPatches = len(patchCoords)

            mapDir = 'patches'
            auxMapDir = 'auxMaps'
            try:
                os.makedirs(mapDir)
            except:
                pass
            try:
                os.makedirs(mapDir+'/patchMovies')
            except:
                pass
            try:
                os.mkdir(auxMapDir)
            except:
                pass

            print "Will create %d maps"%(nPatches*len(mapFiles))

        
            iMap = 0
            if (p['maskFile_%d' %mapRegion] != None) & (freq == freqs[0]):
                f = open(auxMapDir+os.path.sep+'percAreaLostToMask.dat',mode="w")
                f.write("Mask: %s\n"%maskFileBase)


            filter = p['highPassCosSqFilter']
            pwDict = p['prewhitener']
            print mapFiles
            for mapFile in mapFiles:
                count = 0
                map0 = liteMap.liteMapFromFits(mapFile)
            
                # cut out big enough map to hold all patches
                if p['bufferedMapBounds'] is None:
                    sMap = map0
                else:
                    sMap = map0.selectSubMap(ra0,ra1,dec0,dec1)
            
                if srcFile != None:
                    try:
                        if p['bufferedMapBounds'] is None:
                            subSrcMap = srcMap
                        else:
                            subSrcMap = srcMap.selectSubMap(ra0,ra1,dec0,dec1)
                    except:
                        subSrcMap = srcMap.copy()
                    
                    print "subtracting temaplate"
                    try:
                        sMap.data[:] -= subSrcMap.data[:]
                    except:
                        raise ValueError,"Check template size"
                
                if weightFiles !=None:
                    weightMap = liteMap.liteMapFromFits(weightFiles[iMap])
                    if p['bufferedMapBounds'] is None:
                        weightSubMap = weightMap
                    else:
                        weightSubMap = weightMap.selectSubMap(ra0,ra1,dec0,dec1)
                
           
                sMap.writeFits(auxMapDir+os.path.sep+'subMap_%d_%d_%d.fits'%(freq,mapRegion, iMap),overWrite=True)
                pylab.clf()

                if filter['apply']:
                    print "filtering subMap..."
                    sMap = cosineSqFilter(sMap,filter['lMin'],filter['lMax'],\
                                          vkMaskLimits=vkMaskLimits)
                    #sMap.plot(valueRange=[-200,200],colBarLabel=r'$\delta T_{\mathrm{CMB}}$'+' '+ '$(\mu \mathrm{K}) $',\
                    #          saveFig=auxMapDir+os.path.sep+'subMapFiltered_%d_%d.png'%(freq,iMap),\
                    #          colBarShrink=0.4,axesLabels='decimal',title='mapHighPass_%d_%d_%d'%(freq, mapRegion, iMap))
                    sMap.writeFits(auxMapDir+os.path.sep+'subMapFiltered_%d_%d_%d.fits'%(freq, mapRegion,iMap),overWrite=True)
                    #pylab.clf()
                    print "done"
                    # print sMap.info()
            
                if pwDict['apply']:
                    print "prewhitening subMap ..."
                    if iMap == 0 and freq == freqs[0]:
                        pw = prewhitener.prewhitener(pwDict['radius'],\
                                                     addBackFraction=pwDict['addBackFraction'],\
                                                     smoothingFWHM=pwDict['gaussFWHM'],\
                                                     map = sMap)
                    # print sMap.info()
                    sMap = pw.apply(sMap)
                    # print pw.radius, pw.addBackFraction,pw.smoothingFWHM
                    # print sMap.info()
                    sMap.writeFits(auxMapDir+os.path.sep+'subMapPW_%d_%d_%d.fits'%(freq,mapRegion, iMap),overWrite=True)
                
                map0 = sMap.copy()

                # print "... done .. proceeding to cut patches"
                for i in xrange(nPatches):
                    sRa0 = patchCoords[i][0]
                    sRa1 = patchCoords[i][1]
                
                    sDec0 = patchCoords[i][2]
                    sDec1 = patchCoords[i][3]
                    print sDec0, sDec1, map0.y0, map0.y1
                    patch0 = map0.selectSubMap(sRa0,sRa1,sDec0,sDec1)
                    print "********** Patch %d ***********\n"%count
                    print " "

                    
                    
                    
                    patch0.info()
                    patch0.plot(valueRange=[-500,500],show=False)
                    pylab.title('patch_%d_%03d_%d'%(freq,count,iMap))
                    pylab.savefig(mapDir+os.path.sep+'patchMovies/patch_%s_%03d_%d.png'%(freq,count,iMap))
                    pylab.clf()
                    pylab.close()
                    
                    if ((iMap == 0)&(p['maskFile_%d' %mapRegion] !=None)&(freq == freqs[0])):
                        print "cutting mask"
                        maskPatch = ptSrcMask.selectSubMap(sRa0,sRa1,sDec0,sDec1)
                        maskPatch.writeFits(mapDir+os.path.sep+'mask%03d'%count,overWrite=True)
                        f.write("%03d     %5.3f\n"%(count,(1.-maskPatch.data.mean())*100))
                                                    
                    if weightFiles != None:
                        wtPatch = weightSubMap.selectSubMap(sRa0,sRa1,sDec0,sDec1)
                        print "cutting patch from weightMap"
                        wtPatch.writeFits(mapDir+os.path.sep+'weight_%d_%03d_%d'%(freq,count,iMap),overWrite=True)
                        patch0.writeFits(mapDir+os.path.sep+'patch_%d_%03d_%d'%(freq,count,iMap),overWrite=True)
                    
                    count += 1
            
                del map0
                iMap += 1
            

            if (p['maskFile_%d' %mapRegion] != None) & (freq == freqs[0]) & mapRegion == p['separateMapRegions']: f.close()
    
    return 0 #success