#from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as coloropts
import matplotlib
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

def arrange_contour_plot(coords_x, coords_y, u_vals, ax1, title, ax2=None, vmin=0, 
                         vmax=1, iso=0.5, annulus=True, plot_border=False, expensive=True,
                         paper=False, early_times=False):
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
    low_u_mask = u_vals < 0.0001
    u_vals[low_u_mask] = 0.0001
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


    levels = np.linspace(0, vmax, 101)
    if not expensive:
        refiner = tri.UniformTriRefiner(triang)
        tri_refi, u_test_refi = refiner.refine_field(u_vals, subdiv=1)

        cntr = ax1.tricontour(tri_refi, u_test_refi, levels=levels, cmap='plasma', extend='neither')

        clbr = plt.colorbar(cntr, cax=ax2, extend='neither')
    else:

        u_vals = np.round(u_vals, 4)
        cntr = ax1.tricontourf(triang, u_vals, levels=levels, cmap='plasma', extend='neither', antialiased=False)
        if ax2 is not None:
            clbr = plt.colorbar(cntr, cax=ax2, norm=(0, vmax), extend='neither')
    
    if iso is not None:
        iso_mask = np.abs(u_vals - 0.5) < 0.005
        iso_x = coords_x[iso_mask]
        iso_y = coords_y[iso_mask]
        sort_key = np.argsort(np.power(iso_x, 2) + np.power(iso_y, 2))
        ax1.plot(iso_x[sort_key], iso_y[sort_key], color='black', linewidth=0.6) # marker='.', s=1

    if plot_border:
        if annulus:
            ax1.plot(min_r_points_x, min_r_points_y, color='grey', linewidth=0.6)
            ax1.plot(max_r_points_x, max_r_points_y, color='grey', linewidth=0.6)
        else:
            ax1.plot(x_points, min_y_points_y, color='grey', linewidth=0.6)
            ax1.plot(x_points, max_y_points_y, color='grey', linewidth=0.6)
    
        
    
    #clbr.boundaries = [0, 2]
    ax1.set_title(title)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    if paper:
        ax1.set_title('')
        if annulus:
            if early_times:
                # Close up
                #ax1.set_xticks([0, 0.25, 0.5])
                ax1.set_xticks([])
                ax1.set_xlim([0, 0.5])
                ax1.set_ylim([0.725, 1.275])
                #ax1.set_yticks([0.75, 1, 1.25])
                ax1.set_yticks([])
                plt.subplots_adjust(wspace=0.1, hspace=0.03)
            else:
                # Normal
                ax1.set_xticks([0, (min_r+max_r)/2])
                ax1.set_xticklabels([r'$0$', r'$1$'])
                #ax1.set_xticklabels([])
                ax1.set_xlim([0, 1.2*1.1*3.2*3.8/(3*0.728*4.4)]) # 1.896
                ax1.set_ylim([0, 1.2*1.1])
                ax1.set_yticks([0, (min_r+max_r)/2])
                ax1.set_yticklabels([r'$0$', r'$1$'])
                #ax1.set_yticklabels([])
                #plt.subplots_adjust(wspace=0.15)
        
        else:
            ax1.set_xticks([0, 1])
            ax1.set_xlim([0, 1.2*1.1])
            ax1.set_ylim([0, 1.2*1.1])
            ax1.set_yticks([])
            ax1.set_xticklabels([r'$0$', r'$1$'])
            ax1.yaxis.set_tick_params(width=5)
            #ax1.set_xticklabels([])
            #ax1.set_yticklabels([r'$0$', r'$0.32$'])
            #plt.subplots_adjust(wspace=0.15)
            
        if ax2 is not None:
            ax2.set_yticks([0, 0.5, 1])
            ax2.set_ylim([0, 1])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
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

def set_size(width, fraction=1, subplots=(1, 1), padding_adjustment = 1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = padding_adjustment * fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)