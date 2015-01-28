# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 16:53:16 2015

@author: rongen
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 16:51:34 2015

@author: rongen
"""
import mayavi.mlab as mlab
from tvtk.api import tvtk
from mayavi.modules.glyph import Glyph
from mayavi.tools.sources import MGlyphSource
import timeit
import numpy as np

npointsdict = {}
f = open(r'D:\WS\NPYtiles\memory_maps\npoints_per_set.txt')
for line in f:
    npointsdict[line.split(';')[0]] = int(line.split(';')[1])


tag = 'V09H52'

xyz    = np.memmap(r'D:\WS\NPYtiles\memory_maps\{}xyz.dat'.format(tag), dtype = 'float32', shape = (npointsdict[tag], 3))
colors = np.memmap(r'D:\WS\NPYtiles\memory_maps\{}col.dat'.format(tag), dtype = 'uint8',   shape = (npointsdict[tag], 3))




index1 = xyz[:,2] < -1.4
#index2 = (colors[:,0] < 100) * (colors[:,0] > 60)
#slc = slice(npointsdict[tag])
#slc = ~(index1 * index2)
slc = ~index1

fig = mlab.gcf()
engine = mlab.get_engine()

data_source  = MGlyphSource()
data_source.reset(x = xyz[slc, 0], y = xyz[slc, 1], z = xyz[slc, 2])
vtk_data_source = mlab.pipeline.add_dataset(data_source.dataset, name = 'data')
data_source.m_data = vtk_data_source

col = tvtk.UnsignedCharArray()
col.from_array(colors[slc])
vtk_data_source.mlab_source.dataset.point_data.scalars=col
vtk_data_source.mlab_source.dataset.modified()

glyph = Glyph()
engine.add_filter(glyph, vtk_data_source)
glyph.glyph.glyph_source.glyph_source.glyph_type = 'vertex'
mlab.show()

print 'Original length: {}\nNew length: {}'.format(len(colors), colors[slc].shape[0])

colmmap = np.memmap(r'D:\WS\NPYtiles\memory_maps\updated\{}col.dat'.format(tag), dtype = 'uint8', mode = 'w+', shape = colors[slc].shape)
colmmap[:] = colors[slc]
xyzmmap = np.memmap(r'D:\WS\NPYtiles\memory_maps\updated\{}xyz.dat'.format(tag), dtype = 'float32', mode = 'w+', shape = colors[slc].shape)
xyzmmap[:] = xyz[slc]

del colmmap
del xyzmmap