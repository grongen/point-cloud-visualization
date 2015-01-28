# -*- coding: utf-8 -*-
'''
This script can be used to extract data from neifs files and put them into
tvtk.unstructuredGrids, saved as xml files.

Created on Fri Dec 12 11:22:01 2014

@author: rongen
'''

#==============================================================================
# Import packages
#==============================================================================

import nefis
import numpy as np
import struct
from tvtk.api import tvtk
import matplotlib.pyplot as plt


#==============================================================================
# Set file locations
#==============================================================================

#fileloc = r'P:\1209395-schelde\simulaties\run26\wc52\\'
fileloc = r'P:\1209395-schelde\simulaties\run25_hydroguus\\'

dat_file = fileloc+'trim-a112_dom2_2011.dat'
def_file = fileloc+'trim-a112_dom2_2011.def'

#==============================================================================
# Define fucntion to extract nefis data
#==============================================================================
def get_from_nefis(grp_name, elm_name, t_step, dat_file, def_file,
                   coding = ' ', ac_type = 'r'):

    '''
    Get data from a nefis file

    Parameters
    ----------
    grp_name : list of group names for the elements to be returned
    elm_name : list of element names to be returned
    dat_file : name of the .dat file
    def_file : name of the .def file

    Returns
    -------
    data : List with the data structures in it
    '''

    # Create nefis file (fp is the file parameter)
    error, fp = nefis.crenef(dat_file, def_file, coding, ac_type)

    # Function to set the user index
    def get_usr_index(t):
        usr_index = np.arange(15).reshape(5,3)
        usr_index[0,0] = t # begin
        usr_index[0,1] = t # end
        usr_index[0,2] = 1 # step
        usr_index[1,0] = 0
        usr_index[1,1] = 0
        usr_index[1,2] = 0
        usr_index[2,0] = 0
        usr_index[2,1] = 0
        usr_index[2,2] = 0
        usr_index[3,0] = 0
        usr_index[3,1] = 0
        usr_index[3,2] = 0
        usr_index[4,0] = 0
        usr_index[4,1] = 0
        usr_index[4,2] = 0
        np.ascontiguousarray(usr_index, dtype=np.int32)
        return usr_index

    # Function to set the user order
    def get_usr_order(mode = 0):
        if mode == -1:
            usr_order = np.arange(5)
            usr_order[0] = -1
            usr_order[1] = -1
            usr_order[2] = -1
            usr_order[3] = -1
            usr_order[4] = -1
            np.ascontiguousarray(usr_order, dtype = np.int32)
        else:
            usr_order = np.arange(5)
            usr_order[0] = 1
            usr_order[1] = 2
            usr_order[2] = 3
            usr_order[3] = 4
            usr_order[4] = 5
            np.ascontiguousarray(usr_order, dtype = np.int32)
        return usr_order

    # Function to get data from file
    def get_data(grp_name, elm_name, step = 1):
        # Set user order (dimensions)
        usr_order = get_usr_order(-1)

        props = nefis.inqelm(fp, elm_name, usr_order)

        shape = usr_order[usr_order > 0]

        length = shape.prod()
        nbyts = length * props[2]

        usr_order = get_usr_order()
        usr_index = get_usr_index(step)

        raw = nefis.getelt(fp, grp_name, elm_name, usr_index, usr_order, nbyts)

        fmt = '%df' % length
        data = np.array(struct.unpack(fmt, raw[1])).reshape(shape[1], shape[0])

        return data


    data = []
    for grp, elm, t in zip(grp_name, elm_name, t_step):
        data.append(get_data(grp, elm, t))

    return data

#==============================================================================
# Get data from nefis file
#==============================================================================

def nefis2ug(dat_file, def_file, layers, typ):

    if typ == 'wlev':
        grp  = ['map-series']
        elm  = ['S1']
        name = 'water level'
        save = 'wlev'
    elif typ == 'eros':
        grp  = ['map-sed-series']
        elm  = ['DPS']
        name = 'bottom change'
        save = 'erosion'

    grp_list = grp*layers+\
               ['map-const',
                'map-const',
                'map-const',
                'map-const']

    elm_list = elm*layers+\
               ['XCOR',
                'YCOR',
                'XZ',
                'YZ']

    t_step   = list(np.arange(1,layers+1))+[1, 1, 1, 1]

    nefis_data = get_from_nefis(grp_list, elm_list, t_step, dat_file, def_file)

    vars = {}

    vars['XNODE']  = nefis_data.pop(elm_list.index('XCOR'))
    vars['YNODE']  = nefis_data.pop(elm_list.index('YCOR')-1)
    vars['XCC']    = nefis_data.pop(elm_list.index('XZ')-2)
    vars['YCC']    = nefis_data.pop(elm_list.index('YZ')-3)
    vars['ZNODE']  = np.c_[nefis_data]
#    vars['ZNODE'][vars['ZNODE'] > 20] = 20


    global a
    a = vars['ZNODE']



    #==========================================================================
    # Create input for Unstructured Grid
    #==========================================================================

    def create_ug(xnode, ynode, xcc, ycc, depth, scalars):
        # Correct for spikes in bathymetry
        depth[depth < -10] = -10.
        scalars[scalars < -10] = -10
        scalars[scalars >  10] =  10

        # Grid nodes
        points = np.c_[xnode.ravel(), ynode.ravel(), depth.ravel()*-1]
        index = np.zeros_like(depth, dtype = 'bool')

        # Check which nodes belong to which cell center
        nm_nodes = np.arange(xnode.size).reshape(ynode.shape)

        # Array with node index per cells
        flowelemnode = np.dstack((nm_nodes[:-1,:-1], nm_nodes[1:, :-1],
                                  nm_nodes[ 1:, 1:], nm_nodes[:-1, 1:]))

        # Filter out the values for which the cell center lies on x = 0.
        # These are boundary condition cells et cetera
        flowelemnode = flowelemnode[xcc[1:, 1:] != 0, :]
        index[xcc[1:, 1:] != 0] = 1

        # Number of cells
        n_cells = flowelemnode.shape[0]

        # Array with number of nodes per cell
        elemtype = np.sum(flowelemnode > 0, 1)

        # Array with cell types, in this case all Quad
        cell_types = np.ones(len(elemtype))*tvtk.Quad().cell_type

        # Array with number of nodes per cell, and index of nodes
        cells = np.c_[elemtype, flowelemnode].ravel()

        # Array with offset, needed to read out cells
        offsets = np.r_[[0], np.cumsum(elemtype + 1)]

        # Convert cells to tvtk CellArray
        cell_array = tvtk.CellArray()
        cell_array.set_cells(n_cells, cells)

        #======================================================================
        # Create unstructured grid
        #======================================================================
        grid = tvtk.UnstructuredGrid(points = points)
        grid.set_cells(cell_types, offsets, cell_array)
        grid.point_data.scalars = scalars.ravel()
        grid.point_data.scalars.name = name
#        grid.cell_data.scalars = (depth)[index].ravel()
#        grid.cell_data.scalars.name = 'bathymetry'

        return grid

    for i in range(layers):
        if typ == 'wlev':
            t = i % 48 + 48
            grid = create_ug(vars['XNODE'], vars['YNODE'], vars['XCC'], vars['YCC'], vars['ZNODE'][t], vars['ZNODE'][t])
        elif typ == 'eros':
            grid = create_ug(vars['XNODE'], vars['YNODE'], vars['XCC'], vars['YCC'], vars['ZNODE'][i], vars['ZNODE'][0] - vars['ZNODE'][i])
#        plt.close('all')
#        fig, ax = plt.subplots()
#        print 'Max: {}, Min: {}'.format(np.max(vars['ZNODE'][i]), np.min(vars['ZNODE'][i]))
#        ax.imshow(vars['ZNODE'][i], vmin = 3, vmax = 20)
#        fig.savefig(r'D:\rongen\Documents\External node data\Grids\testfigs\{}.png'.format(i))

        w = tvtk.XMLUnstructuredGridWriter(input = grid,
                file_name = r'D:\WS\Grids\{}_t{:03d}_{}.xml'.format(save, int(i), 1))
        w.write()
        print 'Processed grid domain 1 for t = {}'.format(i)

nefis2ug(dat_file, def_file, 285, 'wlev')