# -*- coding: utf-8 -*-
"""
#==============================================================================
This script can be used to adjust the colors of your tiles, may be not be as
wished
#==============================================================================
"""

import glob
from skimage import color
import numpy as np

npointsdict = {}
f = open(r'D:\WS\NPYtiles\memory_maps\npoints_per_set.txt')
for line in f:
    npointsdict[line.split(';')[0]] = int(line.split(';')[1])

files = glob.glob(r'D:\WS\NPYtiles\memory_maps\*col.dat')

for f in files[700:]:
    tag = f[-13:-7]
    rgb = np.memmap(f, dtype = 'uint8', shape = tuple((npointsdict[tag],3)))

    rgbarray = np.array(rgb)[:,np.newaxis,:]
    hsv = color.rgb2hsv(rgbarray)
    hsv[:,0,1] *= 1.5
    rgbarray = (color.hsv2rgb(hsv)*255).astype('uint8')

    saveloc = r'D:\WS\NPYtiles\memory_maps\saturated\{}.dat'.format(f[-13:-4])

    mmap = np.memmap(saveloc, dtype = 'uint8', mode = 'w+', shape = rgbarray.squeeze().shape)
    mmap[:] = rgbarray.squeeze()[:]