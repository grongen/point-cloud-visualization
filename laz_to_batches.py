'''
#==============================================================================
This script extracts the downloaded laz files and puts batches into numpy
arrays. These numpy arrays will be stored in a folder. Also a batchdictionairy
will be created which contains information about which are the x and y limits
of the batches. This helps later with rearranging the batches into tiles. The
data is extracted in batches, because a whole laz-file does not fit into the
memory of a normal computer. Also the colors from the aerial photgraphs are
combined with the coordinates in this step.

The script also has the possibility to increase or decrease the saturation of
the colors, since they are sort of gray from themselves. This can be done by
setting the saturation_factor. A factor between 1.5 and 2 is recommended for
more lively colors. A factor of 1.0 does not change anything.
#==============================================================================
'''


#==============================================================================
# INPUT:
mainfolder = r'D:\rongen\Documents\External node data\\'
saturation_factor = 1.5
npts = 10000000  # The number of points per badge
#==============================================================================


# Import packages
#------------------------------------------------------------------------------
import liblas
import numpy as np
import owslib.wms
import owslib.wfs
import shapely.geometry
import json


# get a list with files to be used
#------------------------------------------------------------------------------
filelist = np.loadtxt(mainfolder+'filelist.txt', dtype = 'str')

files = []
for j in filelist:
    files.append(mainfolder+'LAZfiles\\'+j[-14:-4])

count = 0
las = []
for j in files:
    las.append(liblas.file.File(j))
    count += las[-1].get_header().get_count()

print '=====================================================\n',\
      'The number of files is:', len(files)
print 'The number of points in the total set is:',count,'\n', \
      '====================================================='


# Create function to read out the pointdata
#------------------------------------------------------------------------------

def las2rows(lasfile, start, stop):
    '''
    Read out the x, y and z points. The coordinates are given in one long 1D-
    array, so it has to be reshaped after.
    '''
    read = lasfile.read
    for i in xrange(start, stop):

        point = read(i)
        yield point.x
        yield point.y
        yield point.z

# Get WFS data for box limits
#------------------------------------------------------------------------------

# Enter url for Web Feature Service (boxes)
wfsurl = 'http://geodata.nationaalgeoregister.nl/ahn2/wfs'
wfs = owslib.wfs.WebFeatureService(wfsurl, version="2.0.0")
wfslayer = wfs.contents['ahn2:ahn2_bladindex']

# Get boxes from WFS
f = wfs.getfeature(typename=[wfslayer.id], outputFormat="json")
WFSdata = json.load(f)


# Loop through files and badges
#------------------------------------------------------------------------------

# Loop through files
for f in range(len(files)):

    # Create dictionary to save batch properties
    batchdict = {}

    # Get borders of the box
    for feature in WFSdata['features']:
        if feature['properties']['bladnr'] == files[f][-9:-4]:
            shape = shapely.geometry.asShape(feature['geometry'])[0]

    # get the boundaries of the tiles
    north = int(shape.exterior.coords.xy[1][0])
    south = int(shape.exterior.coords.xy[1][2])
    west  = int(shape.exterior.coords.xy[0][0])
    east  = int(shape.exterior.coords.xy[0][1])

    # Load colors to memory
    rgb = np.load(mainfolder+r'aerial_photographs\rgb_'+files[f][-9:-4]+'.npy')

    # Create a range to set the badge limits, add the last point so it can be used
    # as start and end
    badgestarts = range(0, las[f].get_header().get_count(), npts)
    badgestarts.append(las[f].get_header().get_count())

    # Loop through batches
    for i in xrange(len(badgestarts)-1):

        # Get x, y and z data from lasfile
        xyz = np.fromiter(las2rows(las[f], badgestarts[i], badgestarts[i+1]),
                          np.float32, (badgestarts[i+1]-badgestarts[i])*3)
        xyz = xyz.reshape((len(xyz)/3,3))
        print xyz

        # Allocate array for colors
        colors = np.empty((len(xyz),3), dtype = np.uint8)

        # number of pixels in x and y direction
        xpix = int(rgb.shape[1])
        ypix = int(rgb.shape[0])

        # xrel and yrel are the i'th and j'th pixel from TOPLEFT
        xrel = np.floor(((xyz[:, 0]-west)/(east-west)*xpix)).astype(np.uint16)
        yrel = np.floor(((north-xyz[:, 1])/(north-south)*ypix)).astype(np.uint16)

        # Modify xrel and yrel values which 'do not belong in the box'
        xrel[xrel == rgb.shape[1]] = rgb.shape[1] - 1
        yrel[yrel == rgb.shape[0]] = rgb.shape[0] - 1

        # Get colors matching with x and y coordinates
        colors[:, 0] = rgb[yrel, xrel][:, 0]
        colors[:, 1] = rgb[yrel, xrel][:, 1]
        colors[:, 2] = rgb[yrel, xrel][:, 2]

        if saturation_factor != 1.0:
            from skimage import color
            colors_array = colors[:,np.newaxis,:]
            hsv = color.rgb2hsv(colors_array)
            hsv[:,0,1] *= saturation_factor
            colors = (color.hsv2rgb(hsv)*255).astype('uint8').squeeze()

        # Save xyz coordinates
        np.save(mainfolder+'batches\\xyz_'+files[f][-10:-4]+'_'+str(i),
                xyz.astype(np.float32))
        print('Saved '+'xyz_'+files[f][-10:-4]+'_'+str(i))

        # Save colors
        np.save(mainfolder+'batches\\col_'+files[f][-10:-4]+'_'+str(i),
                colors.astype(np.uint8))
        print('Saved '+'col_'+files[f][-10:-4]+'_'+str(i))

        # Append batchproperties
        batchdict['xyz_'+files[f][-10:-4]+'_'+str(i)] = [min(xyz[:,0]), max(xyz[:,0]),
                        min(xyz[:,1]), max(xyz[:,1])]


    # Write the dictionary to csv file
    csvf = open(mainfolder+r'batches\batchdict_'+files[f][-10:-4]+'.csv', 'w')
    w = csv.writer(csvf)
    for key, val in batchdict.items():
        w.writerow([key, val])
    csvf.close()