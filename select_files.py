'''
#==============================================================================
This script creates a txt file in which all the links to the files which
should be downloaded are listed. The txt file will be used for downloading
.laz-files and aerial photographs.

The script will also create a tilegrid, which contains the names and
locations of the tiles which will be created later. This tilegrid will be
saved in a csv file. This tilegrid file will be used for selecting the area
to visualize in the animation script.

Input:
------
- Coordinates of the points which should be covered by the laz-files.
- Coordinates of the tiles which form the edges of the grid

Output:
...\filelist.txt : File with all the laz-files to be downloaded
...\tiledict.csv : File with all the name and location of the data-tiles
#==============================================================================
'''

# Import packages
#------------------------------------------------------------------------------

import owslib.wfs
import owslib.wms
import numpy as np
import json
import shapely.geometry
import matplotlib.pyplot as plt
import csv
from matplotlib import collections

# write = True if output should be written to file, else False
WRITE = True

# Give the main folder
mainfolder = r'D:\rongen\Documents\External node data\\'

# Get the tile properties with the WFS
#------------------------------------------------------------------------------

# Enter url for Web Feature Service (boxes)
wfsurl = 'http://geodata.nationaalgeoregister.nl/ahn2/wfs'
wfs = owslib.wfs.WebFeatureService(wfsurl, version="2.0.0")
wfslayer = wfs.contents['ahn2:ahn2_bladindex']

# Get boxes from WFS
f = wfs.getfeature(typename=[wfslayer.id], outputFormat="json")
data = json.load(f)
shapes = []
for feature in data['features']:
    shapes.append(shapely.geometry.asShape(feature['geometry'])[0])

if WRITE:
    filename = open(mainfolder+'filelist.txt','w')

#==============================================================================
# ENTER THE COORDINATES OF THE POINTS WHICH SHOULD BE INCLUDED
#==============================================================================

coordinates = shapely.geometry.MultiPoint(
        [[22000, 385000],
         [27000, 385000],
         [32000, 385000],
         [37000, 385000],
         [22000, 390000],
         [27000, 390000],
         [42000, 385000],
         [52000, 385000],
         [57000, 385000],
         [22000, 380000],
         [27000, 380000],
         [32000, 380000],
         [37000, 380000],
         [42000, 380000],
         [47000, 380000],
         [52000, 380000],
         [57000, 380000],
         [37000, 372000],
         [42000, 372000],
         [47000, 372000],
         [52000, 372000]
         ])

#==============================================================================
# ENTER THE UPPER LEFT AND LOWER RIGHT BOX AS CORNERS FOR THE MINOR GRID
#==============================================================================

corners = shapely.geometry.MultiPoint(
        [[22000, 390000],
         [58000, 370000]])

# Check which shapes intresect with the coordinates
#------------------------------------------------------------------------------
fig, ax = plt.subplots()
fig.set_size_inches(13,8)

included_shapes = []

for i in range(len(shapes)):
    xy = np.array(shapes[i].exterior.coords)
    shape_top = xy[0,1]
    shape_bottom = xy[2,1]
    shape_left = xy[0,0]
    shape_right = xy[1,0]

for i in range(len(shapes)):

    # Check which box the coorcinates are in
    for p in coordinates:
        if p.within(shapes[i]):
            included_shapes.append(shapes[i])
            if WRITE:
                filename.write('http://geodata.nationaalgeoregister.nl/ahn2/extract/ahn2_gefilterd/g'+data['features'][i]['properties']['bladnr']+'.laz.zip\n')
                filename.write('http://geodata.nationaalgeoregister.nl/ahn2/extract/ahn2_uitgefilterd/u'+data['features'][i]['properties']['bladnr']+'.laz.zip\n')

            ax.text(shapes[i].centroid.coords[0][0], shapes[i].centroid.coords[0][1],
                data['features'][i]['properties']['bladnr'], color = 'white',
                horizontalalignment='center',
                verticalalignment='center')

    for c in corners:
        if (shapes[i].bounds[3] > c.coords[0][1] and
            shapes[i].bounds[1] < c.coords[0][1] and
            shapes[i].bounds[0] < c.coords[0][0] and
            shapes[i].bounds[2] > c.coords[0][0]):

            # Check where the corners of the grid are
            if shapes[i].bounds[0] < corners[0].coords[0][0]:
                topleft = [shapes[i].bounds[0], shapes[i].bounds[3]]
            if shapes[i].bounds[2] > corners[0].coords[0][0]:
                downright = [shapes[i].bounds[2], shapes[i].bounds[1]]

col = collections.LineCollection([s.exterior.coords[:] for s in included_shapes], colors = 'k', lw = 2)
ax.add_collection(col, autolim = True)
ax.set_aspect('equal')
#ax.autoscale_view()

# Get the aerial photgraphs from the background
#------------------------------------------------------------------------------

# Enter url for Web Mapping Service (background)
wmsurl='http://geodata1.nationaalgeoregister.nl/luchtfoto/wms'
wms=owslib.wms.WebMapService(wmsurl)
wmslayer=['luchtfoto_png']

# Set the limits for the background
xdist = downright[0] - topleft[0]
ydist = topleft[1] - downright[1]


bbox = (topleft[0] - 0.1*xdist, downright[1] - 0.1*ydist,
        downright[0] + 0.1*xdist, topleft[1] + 0.1*ydist)


# Get the background from WMS
f = wms.getmap(layers = wmslayer, srs='EPSG:28992', bbox = bbox,
               size=(1000, int(ydist/xdist*1000)),
               format ='image/png', transparent=True)

img = plt.imread(f)
imgpl = ax.imshow(img, extent=[bbox[0],bbox[2],bbox[1],bbox[3]], alpha = 0.75)

ax.axis('scaled')
ax.set_xlim(bbox[0], bbox[2])
ax.set_ylim(bbox[1], bbox[3])


# Make minor grid for tiles
#------------------------------------------------------------------------------

# Bladnr tile sizes are 6250 m vertical by 5000 m horizontal, we will make
# tiles of 500 x 500 m

xstep = 500
ystep = 500

xlimits = np.hstack((np.arange(topleft[0], downright[0], xstep), downright[0]))
ylimits = np.hstack((np.arange(downright[1], topleft[1], xstep), topleft[1]))

tiledict = {}
tiles = []

for i in range(len(xlimits)-1):
    for j in range(len(ylimits)-1):
        # Add a little buffer, so 'shearing' tiles will not be included
        coords = ((xlimits[i]+0.1, ylimits[j]+0.1),
                  (xlimits[i+1]-0.1, ylimits[j]+0.1),
                  (xlimits[i+1]-0.1, ylimits[j+1]-0.1),
                  (xlimits[i]+0.1, ylimits[j+1]-0.1),
                  (xlimits[i]+0.1, ylimits[j]+0.1))
        tile = shapely.geometry.polygon.Polygon(coords)

        for shape in included_shapes:
            if tile.intersects(shape):
                tag = 'V'+'%0.2d'%j+'H'+'%0.2d'%i
                tiledict[tag] = [xlimits[i], ylimits[j], xlimits[i+1], ylimits[j+1]]
                tiles.append(tile.exterior.coords)
                break

col = collections.LineCollection(tiles, colors = 'r', lw = 1, alpha = 0.5)
ax.add_collection(col, autolim = True)
if WRITE == True:
    filename.close()

    csvf = open(mainfolder+'tiledict.csv', 'w')
    w = csv.writer(csvf)
    for key, val in tiledict.items():
        w.writerow([key, val])
    csvf.close()

plt.show()
