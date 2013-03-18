from flipper import *
from scipy.interpolate import splrep,splev

def cosineSqFilter(map,lMin,lMax,vkMaskLimits=None):
    """
    @brief apply a cosine squared filter to the input map
    """
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
    
    labels = p['labels']
    buffer = p['bufferForLargeMap']
    convertTemplateToMicroK = p['convertTemplateToMicroK']
    if convertTemplateToMicroK==None:
        convertTemplateToMicroK = True
    
    if buffer == None:
        buffer = 2.0
    print "buffer =%s"%buffer
    
    # loop over all divided map regions and both labels    
    for label in labels:
        count = 0
        
        for mapRegion in range(1, p['separateMapRegions']+1):
            
            print label
            mapFiles = [p['seasonMapFile_%s_%d'%(label, mapRegion)]]
            lab = ['season']
            splitMapFiles = p['splitMapFiles_%s_%d'%(label, mapRegion)]
            if splitMapFiles != None:
                mapFiles += (splitMapFiles)
                lab += range(len(splitMapFiles))
                    
            
            print mapFiles
            print lab
            
            map0 = liteMap.liteMapFromFits(mapFiles[0])
            map0.info(showHeader=True)
        
            # continue
            srcFile  = p['sourceTemplateFile_%s_%d' %(label, mapRegion)]
            if srcFile != None:
                print "Reading in the source template file: %s"%srcFile
                srcMap = liteMap.liteMapFromFits(srcFile)
                if convertTemplateToMicroK:
                    srcMap.convertToMicroKFromJyPerSr(label*1.0)
            
        
            if (p['maskFile_%d' %mapRegion] != None) and (label == labels[0]):
                print "reading mask file" 
                ptSrcMask = liteMap.liteMapFromFits(p['maskFile_%d' %mapRegion])
                maskFileBase = (p['maskFile_%d' %mapRegion].split("/"))[-1]
            
            weightFile = p['seasonWeightFile_%s_%d'%(label, mapRegion)]
            if weightFile != None:
                print "reading season weight map"
                weightMap = liteMap.liteMapFromFits(weightFile)
             
            patchCoords = sorted(p['patchCoords_%d' %mapRegion]) # sort coords by ra0
            ra0 = patchCoords[0][0]
            ra1 = patchCoords[-1][1]
        
            dec0 = patchCoords[0][2]
            dec1 = patchCoords[-1][3]
        
            vkMaskLimits = p['verticalkMaskLimits']
            nMaps = len(patchCoords)

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

            print "Will create %d maps"%(nMaps)

        
            iMap = 0
            if (p['maskFile_%d' %mapRegion] != None) & (label == labels[0]):
                f = open(auxMapDir+os.path.sep+'percAreaLostToMask.dat',mode="w")
                f.write("Mask: %s\n"%maskFileBase)


            filter = p['highPassCosSqFilter']
            pwDict = p['prewhitener']
            print mapFiles
            for mapFile in mapFiles:
                map0 = liteMap.liteMapFromFits(mapFile)
            
                # cut out big enough map to hold all patches
                if buffer == 0.0:
                    sMap = map0
                else:
                    sMap = map0.selectSubMap(ra0-buffer,ra1+buffer,dec0-buffer,dec1+buffer)
            
                if srcFile != None:
                    try:
                        if buffer == 0.0:
                            subSrcMap = srcMap
                        else:
                            subSrcMap = srcMap.selectSubMap(ra0-buffer,ra1+buffer,dec0-buffer,dec1+buffer)
                    except:
                        subSrcMap = srcMap.copy()
                    
                    print "subtracting template"
                    try:
                        sMap.data[:] -= subSrcMap.data[:]
                    except:
                        raise ValueError,"Check template size"
                
                if weightFile !=None:
                    weightMap = liteMap.liteMapFromFits(weightFile)
                    if buffer == 0.0:
                        weightSubMap = weightMap
                    else:
                        weightSubMap = weightMap.selectSubMap(ra0-buffer,ra1+buffer,dec0-buffer,dec1+buffer)
                
           
                sMap.writeFits(auxMapDir+os.path.sep+'subMap_%s_%d_%s.fits'%(label,mapRegion,lab[iMap]),overWrite=True)

                if filter['apply']:
                    print "filtering subMap..."
                    sMap = cosineSqFilter(sMap,filter['lMin'],filter['lMax'],\
                                          vkMaskLimits=vkMaskLimits)
                    sMap.plot(valueRange=[-200,200],colBarLabel=r'$\delta T_{\mathrm{CMB}}$'+' '+ '$(\mu \mathrm{K}) $',\
                              saveFig=auxMapDir+os.path.sep+'subMapFiltered_%s_%s.png'%(label,lab[iMap]),\
                              colBarShrink=0.4,axesLabels='decimal',title='mapHighPass_%s_%d_%s'%(label, mapRegion, lab[iMap]))
                    sMap.writeFits(auxMapDir+os.path.sep+'subMapFiltered_%d_%s_%s.fits'%(label, mapRegion, lab[iMap]),overWrite=True)
                    pylab.clf()
                    print "done"
            
                if pwDict['apply']:
                    print "prewhitening subMap ..."
                    pw = prewhitener.prewhitener(pwDict['radius'],pwDict['addBackFraction'],pwDict['gaussFWHM'])
                    sMap = pw.apply(sMap)
                
                map0 = sMap.copy()

                # proceeding to cut patches
                for i in xrange(nMaps):
                    sRa0 = patchCoords[i][0]
                    sRa1 = patchCoords[i][1]
                
                    sDec0 = patchCoords[i][2]
                    sDec1 = patchCoords[i][3]
                    patch0 = map0.selectSubMap(sRa0,sRa1,sDec0,sDec1)

                    print sDec0, sDec1, map0.y0, map0.y1                    
                    print "********** Patch %d ***********\n"%count
                    print " "

                    patch0.writeFits(mapDir+os.path.sep+'patch_%s_%03d_%s'%(label,count,lab[iMap]),overWrite=True)
                    
                    patch0.info()
                    
                    if ((iMap == 0)&(p['maskFile_%d' %mapRegion] !=None)&(label == labels[0])):
                        print "cutting mask"
                        maskPatch = ptSrcMask.selectSubMap(sRa0,sRa1,sDec0,sDec1)
                        maskPatch.writeFits(mapDir+os.path.sep+'mask%03d'%(count),overWrite=True)
                        f.write("%03d     %5.3f\n"%(count,(1.-maskPatch.data.mean())*100))
                                                    
                    if ((iMap == 0)&(weightFile !=None)):
                        wtPatch = weightSubMap.selectSubMap(sRa0,sRa1,sDec0,sDec1)
                        print "cutting patch from weightMap"
                        wtPatch.writeFits(mapDir+os.path.sep+'weight_%s_%03d_%s'%(label,count,lab[iMap]),overWrite=True)
                    
                    count += 1
            
                del map0
                iMap += 1
            

            if (p['maskFile_%d' %mapRegion] != None) & (label == labels[0]) & mapRegion == p['separateMapRegions']: f.close()
    
    return 0 # success
    