# -*- coding: utf-8 -*-
"""
#==============================================================================
This script rearranges the from the laz files extracted batches into tiles. It
reads out the created batchdictionaries, and takes the points form the relevant
batches. The scripts saves the tile data to numpy memory maps, of which parts
can be quickly read out.

The script can also shuffle the data, so a 'random' slice can be taken more
quickly from the data. Taking the first N arguments goes faster than taking the
N arguments with a certain step.
#==============================================================================
"""

#==============================================================================
# Input:
mainfolder = r'D:\rongen\Documents\External node data\\'
SHUFFLE = True
#==============================================================================

# Import packages
#------------------------------------------------------------------------------
import csv
import numpy as np
import glob

# Recreate dictionaries from import
#------------------------------------------------------------------------------
tiledict = {}
for key, val in csv.reader(open(mainfolder+'tiledict.csv')):
    # key = 'VyyHxx' ,val = [left, bottom, right, top]
    tiledict[key] = [float(i) for i in val[1:-2].split(',')]

batchdict = {}
dicts = glob.glob(r'D:\rongen\Documents\External node data\batches\batchdict*')
for d in dicts:
    batch = open(d, mode = 'r')
    for line in batch:
        key = line.split('"')[0][:-1]
        val = line.split('"')[1].replace('[','').replace(']','')
        batchdict[key] = [float(i) for i in val[1:-2].split(',')]
    batch.close()

batchlims = np.array(batchdict.values())



# Check which tiles are already ensembled
#------------------------------------------------------------------------------
filesinfolder = glob.glob(mainfolder+'point_cloud_tiles\*')
done = set([f[-13:-7] for f in filesinfolder])

# If it does not exist already, open a file to store the number of points per
# memory map file:
#------------------------------------------------------------------------------
npoints = open(mainfolder+'npoints_per_set.txt', 'a+')

# Create new tiles
#------------------------------------------------------------------------------
for i in tiledict.keys():
    if i in done and i != 'V07H40':
        print 'Tile {} is already ensembled'.format(i)
        continue


    print 'Processing tile:',i
    # left bound of the tile is smaller than the xmax of the batch
    cond1 = tiledict[i][0] < batchlims[:,1]
    # Right bound of the tile is larger than the xmin of the batch
    cond2 = tiledict[i][2] > batchlims[:,0]
    # Lower bound of the tile is smaller than the ymax of the batch
    cond3 = tiledict[i][1] < batchlims[:,3]
    # Upper bound of the tile is larger than the ymin of the batch
    cond4 = tiledict[i][3] > batchlims[:,2]

    selection = np.where(cond1 * cond2 * cond3 * cond4 == True)

    newxyz = []
    newcol = []

    for j in selection[0]:
        fname = batchdict.keys()[j]+'.npy'
        xyz = np.load(mainfolder+'batches\\'+fname, mmap_mode = 'r')
        col = np.load(mainfolder+'batches\\'+'col_'+fname[4:], mmap_mode = 'r')

        index = (xyz[:,0] > tiledict[i][0]) * (xyz[:,0] < tiledict[i][2]) * \
                (xyz[:,1] > tiledict[i][1]) * (xyz[:,1] < tiledict[i][3])

        if len(index[index == True]) > 0:
            newxyz.append(xyz[index,:])
            newcol.append(col[index,:])


    if len(newxyz) == 1:
        newxyz = newxyz[0]
        newcol = newcol[0]
    elif len(newxyz) > 1:
        newxyz = np.vstack(newxyz)
        newcol = np.vstack(newcol)

    # Shuffle data if SHUFFLE is True, so a quicker 'random' slice can be taken
    #--------------------------------------------------------------------------
    if SHUFFLE:
        xyzrgb = np.c_[newxyz, newcol]
        if len(xyzrgb) != 0:
            np.random.shuffle(xyzrgb)
        newxyz = xyzrgb[:,:3].astype(np.float32)
        neqrgb = xyzrgb[:,3:].astype(np.uint8)

    # Create the memory maps:
    #--------------------------------------------------------------------------
    datashape = newxyz.shape
    if datashape[0]:
        xyzmmap = np.memmap(r'D:\WS\NPYtiles\memory_maps\\{}xyz.dat'.format(i),
                            'float32', mode = 'w+', shape = datashape)
        xyzmmap[:] = newxyz[:]
        del xyzmmap

        rgbmmap = np.memmap(r'D:\WS\NPYtiles\memory_maps\\{}rgb.dat'.format(i),
                            'uint8', mode = 'w+', shape = datashape)
        rgbmmap[:] = newcol[:]
        del rgbmmap

        npoints.write('{};{}\n'.format(i, datashape[0]))

#    np.save(mainfolder+'point_cloud_tiles\\'+i+'xyz', newxyz)
#    np.save(mainfolder+'point_cloud_tiles\\'+i+'col', newcol)