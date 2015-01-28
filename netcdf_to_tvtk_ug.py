# -*- coding: utf-8 -*-
"""
This script can be used to extract data from a netcdf file, and put this into an
tvtk.unstructuredGrid, saved as xml file.

Created on Thu Jan 22 11:26:05 2015

@author: rongen
"""


# coding: utf-8

import netCDF4
import numpy as np
from tvtk.api import tvtk
import mayavi.mlab as mlab
from mayavi.modules.surface import Surface
from matplotlib.mlab import griddata



ds = netCDF4.Dataset(r'D:\rongen\Desktop\Model results delfland\subgrid_map.nc')


xc = ds.variables['FlowElemContour_x'][:]
yc = ds.variables['FlowElemContour_y'][:]
xcc = ds.variables['FlowElem_xcc'][:]
ycc = ds.variables['FlowElem_ycc'][:]
zc = ds.variables['bath'][:]
s1 = ds.variables['s1'][:]

def get_points(xcell, ycell, xnode, ynode, zcell):
    xr = np.arange(np.min(xnode), np.max(xnode)+1, 100)
    yr = np.arange(np.min(ynode), np.max(ynode)+1, 100)
    xi, yi = np.meshgrid(xr, yr)

    x, y, z = xcell, ycell, zcell
    zi = griddata(x, y, z, xi, yi, interp = 'linear')

    points = np.c_[xnode.ravel(), ynode.ravel(), np.zeros_like(xnode.ravel())]

    b = np.ascontiguousarray(points).view(np.dtype((np.void, points.dtype.itemsize * points.shape[1])))
    _, idx = np.unique(b, return_index=True)

    unique_points = points[idx]

    xindex = ((unique_points[:,0] - xr[0]) / 100).astype(np.int64)
    yindex = ((unique_points[:,1] - yr[0]) / 100).astype(np.int64)

    unique_points[:,2] = zi[yindex, xindex]

    unique_points = np.vstack((unique_points[0,:]+0.5, unique_points))

    return unique_points

unique_points = get_points(xcc, ycc, xc, yc, (zc[0,:] + s1[200,:]))

xy_nodes_flat = np.c_[xc.ravel(), yc.ravel()]

def checkwhere(elm):
    c1 = np.where(elm[0] == unique_points[:,0])[0]
    c2 = np.where(elm[1] == unique_points[:,1])[0]
    return np.intersect1d(c1, c2)[0]

indices = np.array([checkwhere(i) for i in xy_nodes_flat])
indices = indices.reshape((len(indices)/4,4))


# Number of cells
n_cells = int(indices.shape[0])

# Array with number of nodes per cell
elemtype = np.sum(indices > 0, 1)

# Array with cell types, in this case all Quad
cell_types = np.ones(len(elemtype))*tvtk.Quad().cell_type

# Array with number of nodes per cell, and index of nodes
cells = np.c_[elemtype, indices].ravel()

# Array with offset, needed to read out cells
offsets = np.r_[[0], np.cumsum(elemtype + 1)]

# Convert cells to tvtk CellArray
cell_array = tvtk.CellArray()
cell_array.set_cells(n_cells, cells)

for i in range(176,len(s1)):
    unique_points = get_points(xcc, ycc, xc, yc, (zc[0,:] + s1[i,:]))
    grid = tvtk.UnstructuredGrid(points = unique_points)
    grid.set_cells(cell_types, offsets, cell_array)
    grid.point_data.scalars = unique_points[:,2]
    grid.point_data.scalars.name = 'wlev'

    w = tvtk.XMLUnstructuredGridWriter(input = grid, file_name = r'D:\rongen\Documents\External node data\3Digrid_wlev_{}.xml'.format(i))
    w.write()

fig = mlab.gcf()
engine = mlab.get_engine()

mlab.pipeline.add_dataset(grid)
engine.add_module(Surface())
mlab.show()
