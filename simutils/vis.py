from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as coloropts
import matplotlib
import imageio
from glob import glob
import os
import re

def check_points(themesh, markers, idx):
    """Function for checking if a set of points is on the desired boundary.
    Useful for debugging mostly.
    
    Arguments:
        themesh (DOLFIN mesh) - mesh to take coordinates from.
        markers (DOLFIN boundary markers) - set of boundary markers to read from.
        idx (int) - the boundary marker we are interested in seeing what points exist inside.
        
    """
    for x in themesh.coordinates():
            if markers[idx].inside(x, True): print('%s is on the specified boundary' % x)

    return 

def arrange_contour_plot(coords_x, coords_y, u_vals, ax1, ax2, title, vmin=0, 
                         vmax=2, iso=0.5, annulus=True, plot_border=False, expensive=True,
                         paper=False):
    """Function for setting up contour plots. NB: lists of coordinates and values must be
    related correctly (ie. lengths the same and come in sets (x, y, u))
    
    Arguments:
        coords_x (list/array) - 1D array of all x coordinates
        coords_y (list/array) - 1D array of all y coordinates
        u_vals (list/array) - 1D array of all function values
        ax1 (matplotlib axis) - Axis object in which to plot the solution
        ax2 (matplotlib axis) - Axis object to insert the colorbar at
        title (string) - Title for the plot
        vmin (float) - Minimum value of u for the colourbar
        vmax (float) - Maximum value of u for the colourbar
        iso (float) - value(s) of the isolines to mark
        paper (boolean) - whether to trim the borders and make the plots look nicer.
    """
    triang = tri.Triangulation(coords_x, coords_y)
    # Cut out points below a minimum radius so they aren't plotted
    # This will need modified for a non-annular/radially symmetric mesh
    if annulus:
        min_r = np.sqrt(np.min(np.power(coords_x, 2) + np.power(coords_y, 2)))
        max_r = np.sqrt(np.max(np.power(coords_x[coords_x > 0], 2) + np.power(coords_y[coords_x > 0], 2)))
        triang.set_mask(np.logical_or((np.hypot(coords_x[triang.triangles].mean(axis=1),
                        coords_y[triang.triangles].mean(axis=1)) < min_r),
                        np.logical_and(coords_x[triang.triangles].mean(axis=1) < 0, 
                    coords_y[triang.triangles].mean(axis=1) > -1*min_r)))
        thetas = np.linspace(-np.pi/2, np.pi/2, num=200)
        min_r_points_x, min_r_points_y = min_r * np.cos(thetas), min_r * np.sin(thetas)
        max_r_points_x, max_r_points_y = max_r * np.cos(thetas), max_r * np.sin(thetas)
    
    else:
        max_y = np.max(coords_y)
        x_points = np.linspace(0, np.max(coords_x), num=200)
        max_y_points_y = np.ones_like(x_points) * max_y
        min_y_points_y = np.ones_like(x_points) * 0

    levels = np.linspace(0, vmax, 500)
    if not expensive:
        refiner = tri.UniformTriRefiner(triang)
        tri_refi, u_test_refi = refiner.refine_field(u_vals, subdiv=1)

        cntr = ax1.tricontourf(tri_refi, u_test_refi, levels=levels, cmap='plasma', extend='neither')

        clbr = plt.colorbar(cntr, cax=ax2, extend='neither')
    else:

        u_vals = np.round(u_vals, 6)
        cntr = ax1.tricontourf(triang, u_vals, levels=levels, cmap='plasma', extend='neither')

        clbr = plt.colorbar(cntr, cax=ax2, norm=(0, vmax), extend='neither')
    

    
    if plot_border:
        if annulus:
            ax1.plot(min_r_points_x, min_r_points_y, color='black')
            ax1.plot(max_r_points_x, max_r_points_y, color='black')
        else:
            ax1.plot(x_points, min_y_points_y, color='black')
            ax1.plot(x_points, max_y_points_y, color='black')
        
    
    #clbr.boundaries = [0, 2]
    ax1.set_title(title)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    if paper:
        ax1.set_title('')
        if annulus:
            # Close up
            #ax1.set_xticks([0, 0.25, 0.5])
            #ax1.set_xlim([0, 0.5])
            #ax1.set_ylim([0.75, 1.25])
            #ax1.set_yticks([0.75, 1, 1.25])
            # Normal
            ax1.set_xticks([0, (min_r+max_r)/2])
            ax1.set_xlim([0, max_r*1.1])
            ax1.set_ylim([0, max_r*1.1])
            ax1.set_yticks([0, (min_r+max_r)/2])
        
        else:
            ax1.set_xticks([0, 1.5])
            ax1.set_xlim([0, 1.570796])
            ax1.set_ylim([0, 0.4])
            ax1.set_yticks([0])
            

        ax2.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax2.set_ylim([0, 1])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_xlabel('')
        ax1.set_ylabel('')
    

    return cntr

def save_img(filepath, id):
    """Simple function for saving a solution as an image.

    Arguments:
        filepath (string) - Location to save file in and prefix for file
        id (string) - Unique file ID for this plot

    Returns:
        filepath (string) - As above
    """
    plt.savefig(os.path.abspath(os.path.join(filepath, \
     'soln' + id + '.png')), format='png', bbox_inches='tight')
    return filepath

def atof(text):
    """Function for checking if a string can be turned into a float.
    Arguments:
        text (string) - text to be checked.

    Returns:
        testfloat (float or None) - text converted to float if possible, otherwise none.
    """
    try:
        testfloat = float(text)
        return testfloat
    except ValueError:
        return None

def natural_keys(text):
    return [atof(c) for c in re.split('(\d+\.\d+)', text)]


def clear_imgs(path):
    """Remove a set of images from a directory.

    Arguments:
        path (string) - path from which to remove the images.

    """
    filelist = glob(path + '*.png')
    for filename in filelist:
        os.remove(filename)

    return


def color_assign_criteria(values, max_val, min_val):
    """Assigns a color to each value in array based on where they sit on a linear scale
    determined by a maximum and minimum point. 
    Arguments:
        :values: (np array) - Values at which to evaluate the modulator
        :max_val: (float) - Maximum value to use for the scale
        :min_val: (float) - Minimum value to use for the scale
    Returns:
        :modulator:
        """
    values = (values - min_val) / (max_val-min_val)
    
    cmap = matplotlib.cm.get_cmap('viridis')
    modulator = cmap(values)


    return modulator