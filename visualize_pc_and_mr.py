'''
#==============================================================================
This script visualizes a point cloud with model results in it.

It forms the link between Mayavi and a point cloud / model results database.
#==============================================================================
'''

#==============================================================================
# INPUT:

# The start position of the focal point
start_position = [32000, 385000, 0]

# Show lidar data
show_lidar = True

# Show model results. The exact appearance of the model results can be adjusted
# in the model results class itself, or in Mayavi when running the visualizer.
show_model_results = True

# Mask points, gives a better distribution of points over the screen, but adds
# something to the computation time
mask_points = False

# The total points to be plotted, can be changes in the visualization by
# pressing 1, 2, 7, 9, resp: /10, /2, *2, *10.
total_points = 100000

# The mainfolder from which the all the data is loaded
mainfolder = r'D:\WS\\'
#==============================================================================


#==============================================================================
# import packages
#==============================================================================

import numpy as np
import mayavi.mlab as mlab
from tvtk.api import tvtk
import csv
import math
from mayavi.filters.mask_points import MaskPoints
from mayavi.modules.glyph import Glyph
from mayavi.tools.sources import MGlyphSource
import glob
from mayavi.modules.surface import Surface
import os.path
import time



#==============================================================================
# load tile dictionary
#==============================================================================
tiledict = {}
for key, val in csv.reader(open(mainfolder+'tiledict.csv')):
    # key = 'VyyHxx' ,val = val = [left, bottom, right, top]
    tiledict[key] = [float(i) for i in val[1:-2].split(',')]

#==============================================================================
# Load npoints per tile dictionary
#==============================================================================
npointsdict = {}
f = open(mainfolder+r'point_cloud_tiles\\npoints_per_set.txt')
for line in f:
    npointsdict[line.split(';')[0]] = int(line.split(';')[1])

#==============================================================================
# define class for tile
#==============================================================================

class Tile:

    def __init__(self, tag, bounds, use_mask):
        self.tag    = tag
        self.left   = bounds[0]
        self.right  = bounds[2]
        self.top    = bounds[3]
        self.bottom = bounds[1]
        self.center = [0.5*(bounds[0]+bounds[2]), 0.5*(bounds[1]+bounds[3])]

        self.cornors = np.array([[self.left,  self.bottom],
                                 [self.right, self.bottom],
                                 [self.right, self.top   ],
                                 [self.left,  self.top   ]])

        self.use_mask = use_mask

    def __repr__(self):
        return 'Tile '+self.tag+' from (x0,y0) = ('+'%.0f'%self.left + \
               ',' + '%.0f'%self.bottom + ')' + ' to (x1,y1) = (' + \
               '%.0f'%self.right + ',' + '%.0f'%self.top + ')'

    def mmap_data(self):
        self.xyz    = np.memmap(mainfolder+r'point_cloud_tiles\\'+self.tag+'xyz.dat', dtype = 'float32', shape = tuple((npointsdict[self.tag],3)))
        self.colors = np.memmap(mainfolder+r'point_cloud_tiles\\'+self.tag+'col.dat', dtype = 'uint8',   shape = tuple((npointsdict[self.tag],3)))
        self.points_in_set = np.empty(5, dtype = np.uint32)
        self.slices = []
        for i in range(5):
            self.points_in_set[i] = len(self.xyz) / 10**i
            self.slices.append(slice(self.points_in_set[i]))

    def get_level(self, npoints):
        if npoints < self.points_in_set[0]:
            lvl = np.nonzero(npoints < self.points_in_set)[0][-1]
        else:
            lvl = 0
        return lvl

    def load_data_to_screen(self, engine, fig, npoints):
        '''
        Load data to the screen.

        Parameters
        ----------
        step : The step size with which datapoints are skipped.
        '''
        self.lvl = self.get_level(npoints)
        slc = self.slices[self.lvl]
        # Create a glpyh data source, set data to is, add it to the
        # pipeline and create the mlab_source with which properties can be
        # changed.
        if len(self.xyz[slc]) > 0:
            data_source  = MGlyphSource()
            data_source.reset(x = self.xyz[slc, 0], y = self.xyz[slc, 1], z = self.xyz[slc, 2])
            vtk_data_source = mlab.pipeline.add_dataset(data_source.dataset, name = self.tag)
            data_source.m_data = vtk_data_source

            col = tvtk.UnsignedCharArray()
            col.from_array(self.colors[slc])
            vtk_data_source.mlab_source.dataset.point_data.scalars=col
            vtk_data_source.mlab_source.dataset.modified()

            if self.use_mask:
                # Create mask, add it to the engine and set its properties
                self.mask                    = MaskPoints()
                engine.add_filter(self.mask, vtk_data_source)
                self.mask.filter.random_mode = False
                self.mask.filter.maximum_number_of_points = npoints
            # Create a glyph to visualize the points, add it to the engine and
            # set its properties
            glyph = Glyph()
            if self.use_mask:
                engine.add_filter(glyph, self.mask)
            else:
                engine.add_filter(glyph, vtk_data_source)
            glyph.glyph.glyph_source.glyph_source.glyph_type = 'vertex'
            glyph.name = self.tag

    #        print 'Npoints =', self.mask.filter.maximum_number_of_points

#==============================================================================
# class for tilegrid
#==============================================================================

class Tilegrid:

    def __init__(self, use_mask):
        self.dictionary = tiledict
        self.tiles      = []
        self.tags       = []
        self.corners    = []
        self.centers    = []
        self.npoints    = []
        self.vis_points = np.zeros(len(tiledict))
        self.vis_points_old = np.zeros(len(tiledict))

        # Get x range and y range for tiles
        xr = []
        yr = []

        # Append all x and y coordinates from the dictionary
        for i in self.dictionary.values():
            xr.append(i[0])
            xr.append(i[2])
            yr.append(i[1])
            yr.append(i[3])

        # Sort out double values
        self.xr = np.unique(xr)
        self.yr = np.unique(np.array(yr).round(3))

        # Add tile objects to self.tiles
        remove_entries = []
        for i in range(len(tiledict)):
            if os.path.isfile(mainfolder+r'point_cloud_tiles\{}xyz.dat'.format(self.dictionary.keys()[i])):
                print 'Processing tile:',self.dictionary.keys()[i]
                self.tiles.append(Tile(self.dictionary.keys()[i],
                                       self.dictionary.values()[i], use_mask))
                self.tiles[-1].mmap_data()
                self.tags.append(self.dictionary.keys()[i])
                self.corners.append(self.tiles[-1].cornors)
                self.centers.append(self.tiles[-1].center)

                self.npoints.append(int(len(self.tiles[-1].xyz)))
            else:
                remove_entries.append(i)

        newdict = dict(self.dictionary)
        for i in remove_entries:
            del newdict[self.dictionary.keys()[i]]
        self.dictionary = newdict
        # Make one big [x,y] array from the corners
        self.corners = np.vstack(self.corners)

        self.centers = np.array(self.centers)

        self.npoints = np.array(self.npoints)
    def update_proportions(self, camera_position, in_focal_range, total):
        '''
        Tilegrid method which calculates the distance between the camera and
        each center of a tile.

        Assumed is that the center of each tile has a z-coordinate of 0 m.
        '''

        index = np.in1d(self.tags, list(in_focal_range))

        dx = self.centers[index, 0] - camera_position[0]
        dy = self.centers[index, 1] - camera_position[1]
        dz = camera_position[2]

        dist = (dx**2 + dy**2 + dz**2)**0.5

        alpha = np.arctan(dz/(dx**2+dy**2)**0.5)

        proportion = dist**-2 * self.npoints[index] * np.sin(alpha)

        proportion = proportion / sum(proportion)

        self.number_of_visible_points = np.zeros(len(self.tiles), dtype = np.uint32)
        self.number_of_visible_points[index] = np.minimum((proportion * total), self.npoints[index]).astype(np.uint32)



#==============================================================================
# Class for model results
#==============================================================================

class ModelResults:

    def __init__(self, file_locs, engine, scene):

        self.engine = engine
        self.scene = scene
        self.file_locs = file_locs
        self.count = 0
        self.file_locs2 = glob.glob(mainfolder+'grids\wlev\*.xml')
        self.change_data()



    def change_data(self, frame = None):

        cam, foc = mlab.move()
        mlab.view(focalpoint = np.array([foc[0], 211457, 0]))
        for _ in range(5):
            for i in self.scene.children:
                if i.name == 'gridsurface' or i.name == 'gridsurface2' or i.name == 'waterlevel' or i.name == 'gridlines':
                    i.remove()


        if frame == 'forward':
            self.count = (self.count + 1) % len(self.file_locs)
            frame = np.copy(self.count)

        elif frame == 'back':
            self.count = (self.count - 1)
            frame = np.copy(self.count)

        elif frame != None:
            self.count = (self.count + 1) % len(self.file_locs)

        else:
            frame = np.copy(self.count)

        read = tvtk.XMLUnstructuredGridReader(file_name = self.file_locs[frame])
        self.grid = read.get_output()
        mlab.pipeline.add_dataset(self.grid, 'gridsurface')
        self.engine.add_module(Surface())
        mlab.title('{:02d}:{:02d}'.format(int(np.floor(frame/2)%24), int(frame%2 * 30)), size = 0.5)
        for i in self.scene.children:
            if i.name == 'gridsurface':
                surface = i.children[0].children[0]
                module_manager = i.children[0]
                module_manager.scalar_lut_manager.label_text_property.font_family = 'times'
                module_manager.scalar_lut_manager.label_text_property.italic = False
                module_manager.scalar_lut_manager.label_text_property.color = (0.0, 0.0, 0.0)
                module_manager.scalar_lut_manager.label_text_property.font_size = 8
                module_manager.scalar_lut_manager.title_text_property.font_family = 'times'
                module_manager.scalar_lut_manager.title_text_property.italic = False
                module_manager.scalar_lut_manager.title_text_property.color = (0.0, 0.0, 0.0)
                module_manager.scalar_lut_manager.scalar_bar_representation.position2 = np.array([ 0.09600313,  0.57926078])
                module_manager.scalar_lut_manager.scalar_bar_representation.position = np.array([ 0.89399687,  0.32073922])
                module_manager.scalar_lut_manager.use_default_name = False
                module_manager.scalar_lut_manager.scalar_bar.title = 'bottom change [m]'
                module_manager.scalar_lut_manager.number_of_labels = 9
                text = i.children[0].children[1]
                text.property.font_family = 'times'
                text.property.color = (0.0, 0.0, 0.0)
        surface.enable_contours = True
        surface.contour.auto_contours = False
        surface.contour.contours = list(np.arange(-20,20,0.25))
        surface.contour.filled_contours = True

        cmap = [ (212, 53, 53), (215, 215, 215), (101, 194, 113)]
        self.change_repr('gridsurface', self.engine, colormap = cmap, range = [-10,10], show_bar = True)

        read2 = tvtk.XMLUnstructuredGridReader(file_name = self.file_locs2[frame])
        self.wlev = read2.get_output()
        mlab.pipeline.add_dataset(self.wlev, 'waterlevel')
        self.engine.add_module(Surface())
        cmap = [ (25, 125, 180), (25, 125, 180)]
        self.change_repr('waterlevel', self.engine, colormap = cmap, alpha = 0.25)

        for i in self.scene.children:
            if i.name == 'waterlevel':
                surface = i.children[0].children[0]
        surface.enable_contours = True
        surface.contour.auto_contours = False
        surface.contour.contours = list(np.arange(-20,20,0.5))
        surface.contour.filled_contours = True

#        # Grid line on the bathymetry
#        lines = read.get_output()
#        mlab.pipeline.add_dataset(lines, 'gridlines')
#        self.engine.add_module(Surface())
#        cmap = [ (255, 255, 255), (255, 255, 255)]
#        self.change_repr('gridlines', self.engine, colormap = cmap, alpha = 1.0, representation = 'wireframe')

        return self.count




    #==========================================================================
    # Function to create colormap
    #==========================================================================
    def create_colormap(self, col, nsteps):
        '''
        Creates a colormap based on an (uneven) number of input colors. This
        function basically makes a linspace between all the rgb values that
        are given, creating a colormap. Note that for now the number of colors
        should be uneven, or else the output will not have the right number of
        values.

        Parameters
        ----------
        colors: list with rgb tuples
        nsteps: length of the rgb array to be outputted

        Returns
        -------
        (nsteps, 3) shaped array with rgb values.
        '''
        colorlist = []
        for i in range(len(col)-1):
            colorlist.append(np.c_[
                np.linspace(col[i][0], col[i+1][0], nsteps/(len(col)-1)),
                np.linspace(col[i][1], col[i+1][1], nsteps/(len(col)-1)),
                np.linspace(col[i][2], col[i+1][2], nsteps/(len(col)-1))])

        colors = np.vstack(colorlist)
        return colors

    #==========================================================================
    # Function to change layout properties
    #==========================================================================
    def change_repr(self, name, engine, representation = 'surface',
                    show_bar = False, alpha = 1.0, colormap = 'RdBu',
                    range = None):
        '''
        Change the layout of the unstructured grids on screen
        '''

        for child in engine.scenes[0].children:
            if child.name == name:

                if representation != 'surface':
                    child.children[0].children[0].actor.property.\
                        representation = representation

                scl = child.children[0].scalar_lut_manager

                if show_bar:
                    scl.show_scalar_bar = True
                    scl.show_legend = True

                if isinstance(colormap, str):
                    if colormap != scl.lut_mode:
                        scl.lut_mode = colormap

                else:
                    lut = self.create_colormap(colormap, 256)
                    scl.lut.table = np.hstack(
                        (lut, (255 * np.ones(len(lut)))[:,np.newaxis]))

                if range:
                    scl.data_range = np.array(range)
                    scl.use_default_range = False

                if alpha < 1.0:
                    lut = scl.lut.table.to_array()
                    lut[:, -1] = np.ones(256) * alpha * 255.
                    scl.lut.table = lut

#==============================================================================
# Camera class
#==============================================================================

class Camera:

    def __init__(self, camera, start_pos, fig):
        print 'Camera created'
        self.position = camera.position
        self.fp       = np.array(start_pos) # array([x, y, z])
        self.va       = camera.view_angle

        # The on screen crosshead
        self.crosshead = mlab.points3d(self.fp[0], self.fp[1], self.fp[2],
                                       mode = '2dcross', scale_factor = 1,
                                       color = (0, 0, 0), name = 'crosshead',
                                       figure = fig, transparent = True)

        mlab.view(focalpoint = self.fp, distance = 10000)

        self.update_geometry()

#        self.range_box = mlab.plot3d(self.fr[:,0], self.fr[:,1], self.fr[:,2],
#                                     color = (1, 0, 0), tube_radius = 50,
#                                     name = 'rangebox', figure = fig)

        self.update_screen()

    def update_geometry(self):
        '''
        Camera method to calculates the geometry of the view.
        This means the angles between the viewing line and the axes and the
        edges of the screen.
        '''
        # Get the position of the focal point and camera
        azm, elev, self.distance, self.fp = mlab.view()
        self.position, self.fp = mlab.move()

        # Convert angles to radians
        self.azm = azm * 2 * np.pi / 360.
        elev = 90 - elev
        self.elev = elev * 2 * np.pi / 360.

        # Calculate vertical distance between focal point and camera
        dz = self.position[2] - self.fp[2]

        # Calculate horizontal distances to screen bottom and top
        h_to_bottom    = dz / math.tan(self.elev) - \
                         dz / math.tan(0.5*self.va/(180/math.pi)+self.elev)
        # Correct for top of screen not intersecting terrain in front of focal
        # point
        upper_angle = (self.elev - 0.5 * self.va/(180/math.pi))
        if upper_angle <= 0:
            h_to_top   = dz * 100 - dz / math.tan(self.elev)
        else:
            h_to_top   = dz / math.tan(upper_angle) - dz / math.tan(self.elev)

        # Calculate absolute distances to screen bottom and top
        a_to_bottom    = dz / math.sin(self.elev + 0.5 * self.va/(180/math.pi))
        # Correct for top of screen not intersecting terrain in front of focal
        # point
        if upper_angle <= 0:
            a_to_top = (10001*dz**2)**0.5
        else:
            a_to_top       = dz / math.sin(self.elev - 0.5 * self.va/(180/math.pi))

        # Calculate horizontal distances to left and right edges of the screen
        bottom_to_edge = math.tan(0.9*self.va/(180/math.pi))*a_to_bottom
        top_to_edge    = math.tan(0.9*self.va/(180/math.pi))*a_to_top

        # Calculate the edges of the focal range, and put them in an
        # x and y-array like: [leftdown, rightdown, rightup, leftup, leftdown]

        fr_x = np.array([self.fp[0]+math.cos(self.azm)*h_to_bottom +
                         math.sin(self.azm)*bottom_to_edge,
                         self.fp[0]+math.cos(self.azm)*h_to_bottom -
                         math.sin(self.azm)*bottom_to_edge,
                         self.fp[0]-math.cos(self.azm)*h_to_top -
                         math.sin(self.azm)*top_to_edge,
                         self.fp[0]-math.cos(self.azm)*h_to_top +
                         math.sin(self.azm)*top_to_edge,
                         self.fp[0]+math.cos(self.azm)*h_to_bottom +
                         math.sin(self.azm)*bottom_to_edge])

        fr_y = np.array([self.fp[1]+math.sin(self.azm)*h_to_bottom -
                         math.cos(self.azm)*bottom_to_edge,
                         self.fp[1]+math.sin(self.azm)*h_to_bottom +
                         math.cos(self.azm)*bottom_to_edge,
                         self.fp[1]-math.sin(self.azm)*h_to_top +
                         math.cos(self.azm)*top_to_edge,
                         self.fp[1]-math.sin(self.azm)*h_to_top -
                         math.cos(self.azm)*top_to_edge,
                         self.fp[1]+math.sin(self.azm)*h_to_bottom -
                         math.cos(self.azm)*bottom_to_edge])

        # Create a 3 by 5 array to with the 5 (2x point 1) focal range points
        self.fr = np.transpose(np.vstack((fr_x, fr_y, 30*np.ones(5))))

    def update_screen(self):
        '''
        Camera method which updates the focal point and focal range on screen.
        '''
        self.crosshead.mlab_source.dataset.points = [tuple(self.fp)]
#        self.range_box.mlab_source.dataset.points = self.fr

    def move(self, direction):
        '''
        Camera method to move the camera. The direction is determined by key-
        presses. The stepsize is determined with the mpf (meters per frame)
        keyword argument.

        Parameters
        ----------
        direction : direction in which the camera should move
        '''

        # Set the step size per frame (meters per frame)
        mpf = self.position[2] / 10.

        # Move the camera depending on the received direction
        if direction == 'up':
            mlab.view(focalpoint = self.fp+[-mpf*math.cos(self.azm),
                      -mpf*math.sin(self.azm), 0], distance =self.distance)
        elif direction == 'left':
            mlab.view(focalpoint = self.fp+[ mpf*math.sin(self.azm),
                      -mpf*math.cos(self.azm), 0], distance =self.distance)
        elif direction == 'right':
            mlab.view(focalpoint = self.fp+[-mpf*math.sin(self.azm),
                       mpf*math.cos(self.azm), 0], distance =self.distance)
        elif direction == 'down':
            mlab.view(focalpoint = self.fp+[ mpf*math.cos(self.azm),
                       mpf*math.sin(self.azm), 0], distance =self.distance)

#==============================================================================
# Make visualizer class
#==============================================================================
class Visualizer:
    '''
    The visualizer is a class which processes the data viewed on the window.
    It has a camera instance and tilegrid instance, to manage the data and
    checks which data must be displayed, and which data must not. It can be
    seen as the manager of the program.

    Parameters
    ----------
    tiledict : A python dictionary which contains the metadata of the tiles.
    start_position : The start position of the focal range. Makes sure that the
        camera will start at the right position, or a specifically chosen
        position.
    '''

    def __init__(self, start_position, total_points, show_lidar, mask_points,
                 show_model_results):
        print 'Visualizer created'
        self.fig = mlab.gcf()
        istyle = tvtk.InteractorStyleTerrain()
        self.iactor = self.fig.scene.interactor
        self.iactor.interactor_style = istyle

        # Add interaction observers
        istyle.add_observer('KeyPressEvent',           self.Callback)
        istyle.add_observer('EndInteractionEvent',     self.Callback)
        istyle.add_observer('InteractionEvent',        self.Callback)
        istyle.add_observer('MouseWheelBackwardEvent', self.Callback)
        istyle.add_observer('MouseWheelForwardEvent',  self.Callback)

        # Total number of points
        self.total_points = total_points


        # The first tile from the dictionary is taken as the active tile
        self.activetiles = set()
        # The scene is where all the data (children) are added to and the
        # camera is in
        self.engine      = mlab.get_engine()
        self.scene       = self.engine.scenes[0]
        # Set background color
        self.scene.scene.background = (208/255., 238/255., 232/255.)
        # The camera is gives the view on the data
        self.camera      = Camera(self.scene.scene.camera, start_position,
                                  self.fig)
        self.show_lidar = show_lidar
        if self.show_lidar:
            self.tilegrid    = Tilegrid(mask_points)

            print 'Tilegrid created'
        if show_model_results:
            self.model_results = ModelResults(glob.glob(mainfolder+'Grids\erosion\*.xml'),
                                          self.engine, self.scene)

        self.use_mask = mask_points

        self.update_screen_data()



    # Define function for updating dataset
    def update_screen_data(self):
        '''
        Visualizer method which updates the topographic data which is viewe d.
        It adds or removes tiles with data dependend on the focal range of the
        camera.
        '''

        new = self.get_big_box()

        if self.show_lidar:

            self.tilegrid.update_proportions(self.camera.position, new, self.total_points)

            # Add set
            add = new.difference(self.activetiles)

            for i in list(add):
                j = self.tilegrid.tags.index(i)
                self.tilegrid.tiles[j].load_data_to_screen(self.engine, self.fig,
                                     self.tilegrid.number_of_visible_points[j])

            # Remove sets
            remove = self.activetiles.difference(new)

            for i in list(remove):
                for j in self.scene.children:
                    if j.name == i:
                        j.remove()

            self.activetiles = new

            # Update sets
            for i in list(new):
                j = self.tilegrid.tags.index(i)
                tile = self.tilegrid.tiles[j]

                new_lvl = tile.get_level(self.tilegrid.number_of_visible_points[j])
                if new_lvl != tile.lvl:

                    for children in self.scene.children:
                        if children.name == i:
                            children.remove()

                    tile.load_data_to_screen(self.engine, self.fig,
                                     self.tilegrid.number_of_visible_points[j])


    #                tile.data_source.reset(x = tile.xyz[new_lvl][:, 0], y = tile.xyz[new_lvl][:, 1], z = tile.xyz[new_lvl][:, 2])

                if self.use_mask:
                    # If the differences are small, only update the mask
                    if (self.tilegrid.number_of_visible_points[j] > tile.mask.filter.maximum_number_of_points * 1.11 or
                        self.tilegrid.number_of_visible_points[j] < tile.mask.filter.maximum_number_of_points * 0.90):
                        tile.mask.filter.maximum_number_of_points = int(self.tilegrid.number_of_visible_points[j])

    def point_in_triangles(self, tr1, tr2, p):
        '''
        Function which checks if a certain coordinate lies within one of
        two triangles. The two triangles form the focal range, so this
        fucntion checks which tiles are in the focal range.

        The four point per tile which are checked, are the corner point.
        This means that an edge of the focal range can be within a tile,
        without crossing the corner point. Therefor the corner_in_tile
        function is defined below.

        This function is bases on a the script of Perro Azul available on:
        http://jsfiddle.net/PerroAZUL/zdaY8/1/

        Parameters
        ----------
        tr1 : triangle 1. 2 columns (x, y) and 3 rows (3 coordinates)
        tr2 : triangle 2. 2 columns (x, y) and 3 rows (3 coordinates)
        p : array of points to be checked

        Returns
        -------
        out : Array with bool for every 4 points. 4 points since it is assumed
        that 4 subsequent point belong to 1 square.
        '''

        # Calculate the areas of the two triangles
        A1 = (0.5 * (-tr1[1,1] * tr1[2,0] + tr1[0,1] * (-tr1[1,0] +
              tr1[2,0]) + tr1[0,0] * (tr1[1,1] - tr1[2,1]) + tr1[1,0] *
              tr1[2,1]))
        A2 = (0.5 * (-tr2[1,1] * tr2[2,0] + tr2[0,1] * (-tr2[1,0] +
              tr2[2,0]) + tr2[0,0] * (tr2[1,1] - tr2[2,1]) + tr2[1,0] *
              tr2[2,1]))

        # Check whether the area is negative or positive. The area can be
        # negative when the points are defined (anti-)clockwise
        sign1 = -1 if A1 < 0 else 1
        sign2 = -1 if A2 < 0 else 1

        s1 = (tr1[0,1] * tr1[2,0] - tr1[0,0] * tr1[2,1] + (tr1[2,1] -
              tr1[0,1]) * p[:,0] + (tr1[0,0] - tr1[2,0]) * p[:,1]) * sign1
        s2 = (tr2[0,1] * tr2[2,0] - tr2[0,0] * tr2[2,1] + (tr2[2,1] -
              tr2[0,1]) * p[:,0] + (tr2[0,0] - tr2[2,0]) * p[:,1]) * sign2

        t1 = (tr1[0,0] * tr1[1,1] - tr1[0,1] * tr1[1,0] + (tr1[0,1] -
              tr1[1,1]) * p[:,0] + (tr1[1,0] - tr1[0,0]) * p[:,1]) * sign1
        t2 = (tr2[0,0] * tr2[1,1] - tr2[0,1] * tr2[1,0] + (tr2[0,1] -
              tr2[1,1]) * p[:,0] + (tr2[1,0] - tr2[0,0]) * p[:,1]) * sign2

        # Check whether s>0 and t>0 and (s+t)<2*A
        check1 = (np.where(s1>0, True, False) * np.where(t1>0, True, False)
                  * np.where((s1+t1) < 2*A1*sign1, True, False))
        check2 = (np.where(s2>0, True, False) * np.where(t2>0, True, False)
                  * np.where((s2+t2) < 2*A2*sign2, True, False))

        # Add checks for the two triangles
        check = check1 + check2

        # Reshape and take the horizontal sum, since only one in four
        # (edges) point need to be within a triangle to display the tile
        check = check.reshape(len(check)/4, 4)
        check = np.sum(check, axis = 1) > 0

        return check

    def corner_in_tile(self, edges, keys):
        '''
        Function which checks which tile is covered by the corner point.
        This function is made to check for the tiles which are in the
        focal range, but are not found with the point_in_triangle function.
        '''
        corners = self.camera.fr[:4, :2]
        cornertiles = set()

        # Check which tile is around a corner point
        for i in range(4): # Always four corners
            index = ((corners[i, 0] > edges[:, 0]) *
                     (corners[i, 0] < edges[:, 2]) *
                     (corners[i, 1] > edges[:, 1]) *
                     (corners[i, 1] < edges[:, 3]))
            # Check if the corner is in a tile
            if list(np.nonzero(index)[0]):
                # Add tile to cornertiles set
                for j in np.nonzero(index)[0]:
                    tile = keys[int(j)]
                    cornertiles.add(tile)

        return cornertiles


    def get_big_box(self):
        '''
        This function calculates which tiles should fit in the view
        '''


        # Get triangle coordinates from focal range
        triangle1 = self.camera.fr[:3,  :2] # [ left_bot, right_bot, right_top]
        triangle2 = self.camera.fr[-3:, :2] # [ right_top, left_top, left_bot]

        # Call the point_in_triangles function to get the covered tiles
        if self.show_lidar:
            covered_pc1 = self.point_in_triangles(triangle1, triangle2, self.tilegrid.corners)
            covered_pc2 = self.corner_in_tile(np.array(self.tilegrid.dictionary.values()), self.tilegrid.dictionary.keys())

        # Make a set with the tiles which should be displayed
        if self.show_lidar:
            new_pc_tiles = set(map(self.tilegrid.tags.__getitem__,
                               list(np.where(covered_pc1)[0])))
            new_pc_tiles = new_pc_tiles.union(covered_pc2)


        if self.show_lidar:
            return new_pc_tiles


    def Callback(self, obj, event):
        '''
        Visualizer method which processes the interaction events.
        '''
#        self.scene.scene.disable_render = True
        if event == 'InteractionEvent':
            self.camera.update_geometry()
#            self.camera.update_screen()

        else:
            if event == 'KeyPressEvent':
                # First move the focal point and camera
                key = obj.GetInteractor().GetKeyCode()
                if key == '8':
                    self.camera.move('up')
                if key == '4':
                    self.camera.move('left')
                if key == '6':
                    self.camera.move('right')
                if key == '5':
                    self.camera.move('down')

                if key == '1':
                    self.total_points /= 10
                if key == '2':
                    self.total_points /= 2
                if key == '7':
                    self.total_points *= 2
                if key == '9':
                    self.total_points *= 10
                print self.total_points
                 # For looping trough data
                if key == '.':
                    count = self.model_results.change_data('forward')
                    mlab.view(focalpoint=[47874, 372152, 0.0], azimuth = 88.76, elevation = 81.82, distance=1852)
#                    mlab.view(focalpoint=[47210, 374628, 0.0], azimuth = 160, elevation = 78.76, distance=10000)
                if key == ',':
                    count = self.model_results.change_data('back')
                    mlab.view(focalpoint=[46634, 372930, 0.0], azimuth = 94.01, elevation = 81.45, distance=1852)
                    #                    mlab.view(focalpoint=[47210, 374628, 0.0], azimuth = 160, elevation = 78.76, distance=10000)
                # Get position properties
                if key == 'g':
                    az, el, dst, fp = mlab.view()
                    print '--------\nAzimuth: {}\nElevation: {}\nDistance: {}\nFocal point: x={}, y={}, z={}'.format(az, el, dst, fp[0], fp[1], fp[2])
                # Make a movie
                if key == 'm':
                    self.scene.scene.anti_aliasing_frames = 0
                    self.engine.current_scene.scene.off_screen_rendering = True
                    for i in range(285):
                        self.model_results.change_data(i)
#                        mlab.view(focalpoint=[47210, 374628, 0.0], azimuth = 160, elevation = 78.76, distance=10000)
                        mlab.view(focalpoint=[47874, 372152, 0.0], azimuth = 88.76, elevation = 81.82, distance=1852)
                        self.scene.scene.save(r'C:\Users\rongen\Desktop\Video\flats\snapshot{:03d}.png'.format(i))
                        time.sleep(1)

            if event == 'MouseWheelForwardEvent':
                self.camera.distance = self.camera.distance * 0.9
                mlab.view(focalpoint=self.camera.fp, distance=self.camera.distance)

            if event == 'MouseWheelBackwardEvent':
                self.camera.distance = self.camera.distance * 1.111
                mlab.view(focalpoint=self.camera.fp, distance=self.camera.distance)

            self.camera.update_geometry()
            self.camera.update_screen()
            self.update_screen_data()
#        self.scene.scene.disable_render = False

#==============================================================================
# Launch visualizer
#==============================================================================

vis = Visualizer(start_position, total_points, show_lidar, mask_points,
                 show_model_results)

mlab.show()