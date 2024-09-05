#Postprocessing utilities - collection of functions useful for running FENICs simulations.
import adios4dolfinx
from dolfinx import fem
from basix.ufl import element
import configparser
from mpi4py import MPI
import numpy as np
import os
from csv import writer
import math

def get_derivative(u):
    """Function for grabbing derivatives in the x and y direction of a FENICs
    function (ie. it is grad). Relies on the existence of an appropriate FENICs
    FunctionSpace for the derivatives.

    Arguments:
        u (FENICs function) - function to take derivatives of.

    Returns:
        udx (FENICs function) - derivative of u in x direction.
        udy (FENICs function) - derivative of u in y direction.
    """
    udx = u.dx(0)
    udy = u.dx(1)

    return udx, udy

def get_curvature(u, space):
    """Shorthand for finding the curvature field of a FENICs function.

    Arguments:
        u (FENICs function) - scalar function to find curvature of.
        Q (FENICs function space) - space to evaluate the derivatives in.

    Returns:
        kappa (FENICs function) - scalar field containing the curvature of u.

    """
    udx, udy = get_derivative(u, space)
    udxx, udxy = get_derivative(udx, space)
    udyx, udyy = get_derivative(udy, space)

    kappa = - (udxx * (udy * udy) - 2 * udx * udy * udxy
             + udyy * (udx * udx)) / ((udx * udx + udy * udy) ** 1.5)

    return kappa


# def extract_coordinates(u, mesh, space, annulus=True):
#     """Extracts the coordinates of the entire mesh, and the values of a function on it.

#     The order of all the outputs is consistent, which is the point of making this function.
    
#     Arguments:
#         u (FENICs function) - fenics function defined on space.
#         mesh (DOLFIN mesh) - mesh upon which u can be evaluated.
#         space (FENICs Sobolev space) - space upon which to evaluate u.
#         annulus (bool, default True) - hack to remove a point from the annulus mesh that is unphysical.
    
#     Returns:
#         dof_x (numpy array) - list of x values of all coordinates.
#         dof_y (numpy array) - list of y values of all coordinates.
#         nodal_values (numpy array) - list of values of u evaluated at all coordinates.
#     """
    
#     space.tabulate
#     # Get dimension of the space.
#     n = space.dim()
#     d = mesh.geometry().dim()

#     # Change mesh coordinates into an ordered list by coordinate
#     dof_coordinates = mesh.coordinates()
#     dof_coordinates.resize((n, d))
#     dof_x = dof_coordinates[:, 0]
#     dof_y = dof_coordinates[:, 1]
    
#     if not annulus:
#         # Get nodal values of u.
#         u_vals = u.compute_vertex_values()
    
#      # Hack to get rid of the point at zero - specific to my annulus mesh.
#     if annulus:
#         # Get nodal values of u.
#         u_proj = project(u, space)
#         nodal_values = np.round(u_proj.vector().get_local(), decimals=5)
#         u_vals = nodal_values[vertex_to_dof_map(space)]
#         dof_x = np.delete(dof_x, 4)
#         dof_y = np.delete(dof_y, 4)
#         u_vals = np.delete(u_vals, 4)
    
#     # Remove nans if there are any.
#     nans = np.isnan(u_vals)
    
#     dof_x = dof_x[np.logical_not(nans)]
#     dof_y = dof_y[np.logical_not(nans)]
#     u_vals = u_vals[np.logical_not(nans)]

#     return dof_x, dof_y, u_vals


def get_indices_isoline(u, mesh, space, iso=0.5, thresh=0.005, annulus=True):
    """Function for grabbing the coordinates of all points on an isoline.
    
    Arguments:
        u (FENICs function) - function to find the isoline of.
        mesh (Dolfin mesh) - mesh upon which to evaluate the function
        iso (float) - number between 0 and 1.
        thresh (float) - comparison threshold to consider numbers in the isoline.
        annulus (bool) - whether using the annulus (1) or rectangular mesh (0)

    Returns:
        isocoords (np array) - 2xN array of isoline coordinates where there
        are N coordinates.
    """
    dof_x, dof_y, nodal_values = extract_coordinates(u, mesh, space, annulus=annulus)
    iso_idxs = (abs(nodal_values - iso) < thresh)
    if iso_idxs.size == 0:
        return 0, 0, 0
    x_vals = dof_x[iso_idxs]
    y_vals = dof_y[iso_idxs]
    
    return x_vals, y_vals, iso_idxs

def calculate_isoline_properties_annulus(u, mesh, space, iso_idxs):
    """Function for getting the properties of a given set of coordinates.
    
    
    """
    udx, udy = get_derivative(u, space)

    udx_vals = extract_coordinates(udx, mesh, space)[2]
    udy_vals = extract_coordinates(udy, mesh, space)[2]
    udx_vals = udx_vals[iso_idxs]
    udy_vals = udy_vals[iso_idxs]
    kappa = get_curvature(u, space)
    kappa_vals = extract_coordinates(kappa, mesh, space)[2]
    try:
        kappa_vals = kappa_vals[iso_idxs]
    except:
        kappa_vals =[]
        print('Failed to get curvature.  ')

    grad_vector = np.array([udx_vals, udy_vals])
    norm = np.power((grad_vector[0, :] * grad_vector[0, :] 
           + grad_vector[1, :] * grad_vector[1, :]), 0.5)
    
    angles = np.arccos(grad_vector[0, :] / norm) - np.pi / 2

    return angles, kappa_vals

def calculate_speed_fisher(u, mesh, space, iso_idxs, model_params, annulus=True):
    """Function for calculating the speed a la Buenzli & Simpson 2021
    """

    u_vals = extract_coordinates(u, mesh, space, annulus=annulus)[2]
    u_vals = u_vals[iso_idxs]

    udx, udy = get_derivative(u, space)
    udxx = get_derivative(udx, space)[0]
    udyy = get_derivative(udy, space)[1]
    
    udx_vals = extract_coordinates(udx, mesh, space, annulus=annulus)[2]
    udy_vals = extract_coordinates(udy, mesh, space, annulus=annulus)[2]
    udx_vals = udx_vals[iso_idxs]
    udy_vals = udy_vals[iso_idxs]

    udxx_vals = extract_coordinates(udxx, mesh, space, annulus=annulus)[2]
    udyy_vals = extract_coordinates(udyy, mesh, space, annulus=annulus)[2]
    udxx_vals = udxx_vals[iso_idxs]
    udyy_vals = udyy_vals[iso_idxs]

    mod_grad_u = np.sqrt(np.power(udx_vals, 2) + np.power(udy_vals, 2))
    speed = (float(model_params[0]['dbar'])*(udxx_vals + udyy_vals) + float(model_params[0]['sbar'])*u_vals*(1-u_vals))/mod_grad_u

    return speed

def calculate_isoline_properties_rect(u, mesh, space, iso_idxs):
    """Function for getting the properties of a given set of coordinates.
    
    
    """
    kappa = get_curvature(u, space)
    kappa_vals = extract_coordinates(kappa, mesh, space, annulus=False)[2]
    kappa_vals = kappa_vals[iso_idxs]

    return kappa_vals


def prepare_objects(config, filepath):
    
    soln_file = filepath + 'soln.bp'

    times = np.linspace(
        config.getfloat('sim.times', 't0'),
        config.getfloat('sim.times', 'tf'),
        int(config.getint('sim.times', 'n_t')/100)+1
    )

    msh = adios4dolfinx.read_mesh(soln_file, MPI.COMM_WORLD)
    V = fem.functionspace(msh, element('DG', msh.basix_cell(), config.getint('sim.times', 'degree')))
    coords = V.tabulate_dof_coordinates()
    u = fem.Function(V)

    return coords, u, V, times, soln_file, msh




