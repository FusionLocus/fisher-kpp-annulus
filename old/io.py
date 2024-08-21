from fenics import *
import numpy as np
import json
import os
import argparse
import csv

def load_rect_fisher(simfile):
    """Load in the simulation parameters for a given problem on the rectangular mesh.
    
    Argument:
        simfile (string) - location of the file with the simulation parameters
    
    Returns:
        T (float) - Final time
        num_steps (int) - Number of time steps
        dt (float) - Time increment
        Lx (float) - Length of the rectanglular mesh
        Ly (float) - Width of the rectangular mesh
        tol (float) - Tolerance for nodes to be counted on the boundary
        sbar (float) - Reaction term scaling parameter
        dbar (float) - Diffusion term scaling parameter
        mesh_path (string) - Path to the mesh file to load in
        dirichlet (string) - type of boundary condition (dirichlet, neumann or robin)
    """
    
    with open(simfile, 'r') as f:
        parameters = json.load(f)

    T = float(parameters['T']) # final time
    num_steps = int(parameters['num_steps']) # number of time steps
    dt = T / num_steps # time step size
    Lx = float(parameters['Lx']) # Length of the rectanglular mesh
    Ly = float(parameters['Ly']) # Width of the rectangular mesh
    tol = float(parameters['tol'])
    model_params = parameters['model_params']
    mesh_path = parameters['meshpath']
    raw_bcs = parameters['BC']
    u_0 = parameters['u_0']
    simid = parameters['simid']

    return T, num_steps, dt, Lx, Ly, tol, model_params, mesh_path, raw_bcs, u_0, simid


def load_curved_fisher(simfile):
    """Load in the simulation parameters for a given problem on the annular mesh.
    
    Argument:
        simfile (string) - location of the file with the simulation parameters
    
    Returns:
        T (float) - Final time
        num_steps (int) - Number of time steps
        dt (float) - Time increment
        Re (float) - Outer radius of the annulus
        Ri (float) - Inner radius of the annulus
        tol (float) - Tolerance for nodes to be counted on the boundary
        sbar (float) - Reaction term scaling parameter
        dbar (float) - Diffusion term scaling parameter
        mesh_path (string) - Path to the mesh file to load in
        raw_bcs (dictionary) - Dictionary containing boundary condition information
        u_0 (string) - expression describing the initial cell density
    
    """
    
    with open(simfile, 'r') as f:
        parameters = json.load(f)

    T = float(parameters['T']) # final time
    num_steps = int(parameters['num_steps']) # number of time steps
    dt = T / num_steps # time step size
    Re = float(parameters['Ro']) # Outer radius
    Ri = float(parameters['Ri']) # Inner radius
    tol = float(parameters['tol'])
    model_params = parameters['model_params']
    mesh_path = parameters['meshpath']
    raw_bcs = parameters['BC']
    u_0 = parameters['u_0']
    simid = parameters['simid']

    return T, num_steps, dt, Re, Ri, tol, model_params, mesh_path, raw_bcs, u_0, simid

def save_soln(vals_list, path, id):
    """Function for exporting and saving solution properties for visualisation later
    
    Arguments:
        vals (list) - list of arrays of different properties to save.
        path (string) - filepath to save solution properties in.
        id (string) - ID to give this individual file.
        header (list) - List of header information for te csv file.  
    
    Returns:
        location (string) - filepath to location of the solution.
    """
    filepath = os.path.join(path, 'soln_t=' + str(id) + '.csv')
    if os.path.exists(filepath):
        with open(filepath, 'a', newline='') as fd:
            csv_writer = csv.writer(fd)
            for vals in vals_list:
                csv_writer.writerow(vals)
    else:
        with open(filepath, 'w', newline='') as fd:
            csv_writer = csv.writer(fd)
            for vals in vals_list:
                csv_writer.writerow(vals)

    return filepath

def load_soln(filename, periodicity=3):
    """Load data from a file, formatted in rows of datafield per timestep
    
    Arguments:
        filename (string) - Name of file to load in.
        periodicity (int) - Number of rows per timestep.
    
    Returns:
        things

    """

    with open(filename, newline='') as csvfile:
        rawdata_reader = csv.reader(csvfile, delimiter=',')
        finaldata = []
        firstTime = True
        
        c = 0
        for row in rawdata_reader:
            criteria = c % periodicity
            if criteria == 0:
                if not firstTime:
                    finaldata.append(temp)
                firstTime = False
                temp = np.zeros((4, len(row)))

            temp[criteria, :] = row 
            c += 1

    return finaldata