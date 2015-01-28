# -*- coding: utf-8 -*-
"""
#==============================================================================
The following script downloads aerial photographs of a certain area, and saves
it as a numpy array. The area for which the photograph will be created is the
area of a ahn2 lidar data file. The filelist.txt file is used to check for
which ahn2  part the aerial photographs are needed, the file...\filelist.txt is
used. Files for which already a numpy array with colors exists are skipped.

Only input required is the right foldername (line 28)

#==============================================================================
"""

# Import packages
#------------------------------------------------------------------------------
import numpy as np
import owslib.wfs
import owslib.wms
import json
import shapely.geometry
import matplotlib.pyplot as plt
import glob

#==============================================================================
# ENTER YOUR FOLDERNAME:
#==============================================================================
foldername = r'D:\rongen\Documents\External node data\\'

# Select tiles from downloadlist and check which tiles are already done
#------------------------------------------------------------------------------
filenames  = np.loadtxt(foldername+'filelist.txt', dtype = 'str')
filenames  = filenames[np.arange(0, len(filenames), 2)]
for i in xrange(len(filenames)):
    filenames[i] = filenames[i][-13:-8]

filesinfolder = []
for name in glob.glob(foldername+'\aerial_photographs\*.npy'):
        tag = name.split('\\')[-1]
        filesinfolder.append(tag)

filesinfolder = np.array(filesinfolder)

# Get properties from WFS
#------------------------------------------------------------------------------

# Enter url for Web Feature Service (boxes)
wfsurl = 'http://geodata.nationaalgeoregister.nl/ahn2/wfs'
wfs = owslib.wfs.WebFeatureService(wfsurl, version="2.0.0")
wfslayer = wfs.contents['ahn2:ahn2_bladindex']

# Get boxes from WFS
f = wfs.getfeature(typename=[wfslayer.id], outputFormat="json")
data = json.load(f)


# Go through tiles
#------------------------------------------------------------------------------
for tile in filenames:

    if sum(('rgb_'+tile+'.npy') == filesinfolder) > 0:
        print tile,'already processed!'
        continue

    for feature in data['features']:
        if feature['properties']['bladnr'] == tile:
            shape = shapely.geometry.asShape(feature['geometry'])[0]

    # get the boundaries of the tiles
    north = int(shape.exterior.coords.xy[1][0])
    south = int(shape.exterior.coords.xy[1][2])
    west  = int(shape.exterior.coords.xy[0][0])
    east  = int(shape.exterior.coords.xy[0][1])

    ns_dist = north - south # 6250 m
    ew_dist = east  - west  # 5000 m

    ns_step = ns_dist / 2
    ew_step = ew_dist / 4

    # Print error if tile is too large
    if ns_dist * ew_dist > 6250 * 5000:
        print 'The tile is larger than usuall. Maybe more squares are needed!'

    # Split the area in 8 tiles, so the number of pixels remains below 16MP
    bbox = [(west,               south, west +     ew_step, south + ns_step),
            (west +     ew_step, south, west + 2 * ew_step, south + ns_step),
            (west + 2 * ew_step, south, west + 3 * ew_step, south + ns_step),
            (west + 3 * ew_step, south, east              , south + ns_step),
            (west,               south + ns_step, west +     ew_step, north),
            (west +     ew_step, south + ns_step, west + 2 * ew_step, north),
            (west + 2 * ew_step, south + ns_step, west + 3 * ew_step, north),
            (west + 3 * ew_step, south + ns_step, east              , north)]

    # Get WMS layer

    wmsurl   = 'http://geodata1.nationaalgeoregister.nl/luchtfoto/wms'
    wms      = owslib.wms.WebMapService(wmsurl)
    wmslayer = ['luchtfoto_png']

    img = []

    for i in range(8):
        f = wms.getmap(layers = wmslayer, srs='EPSG:28992', bbox = bbox[i],
                       size=(ew_step*2, ns_step*2), format = 'image/png')
        img.append(plt.imread(f)*255)

    rgb = np.vstack((np.hstack((img[4], img[5], img[6], img[7])),
                     np.hstack((img[0], img[1], img[2], img[3]))))

    rgb = rgb.astype(np.uint8)

    np.save(r'D:\WS\aerial_photographs\rgb_'+tile, rgb)
