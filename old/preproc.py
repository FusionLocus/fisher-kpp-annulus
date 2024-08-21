#Simulation preprocessing utilities - useful for loading in problems.
from fenics import *
import numpy as np
import json
import os
from glob import glob
import argparse

def parse_expression(expressionarray):
    """Turn an array containing a string expression with modifiable parameters into a formatted
    string.

    Argument:
        expressionarray (array of strings) - array containing the expression to parse and its parameters.
    """
    expression = expressionarray[0]
    if len(expressionarray) > 1 and not isinstance(expressionarray, str):
        expression = expression.format(*expressionarray[1:])

        return expression, expressionarray[1:]

    elif len(expressionarray) == 1:
        return expression
    
    elif isinstance(expressionarray, str):
        return expressionarray


def make_mesh(meshpath):
    """Read and produce mesh from a converted .msh file.

    Argument:
        meshpath (string) - Relative file path to the location of the .msh file.

    Return:
        mesh (DOLFIN mesh object) - mesh for use in a FENICs problem.
        mvc (DOLFIN mesh value collection)
        mf (DOLFIN mesh function)
    """
    mesh = Mesh()
    with XDMFFile(meshpath + '/mesh.xdmf') as infile:
        infile.read(mesh)

    mvc = MeshValueCollection('size_t', mesh, 1)

    with XDMFFile(meshpath + '/mf.xdmf') as infile:
        infile.read(mvc, 'name_to_read' )

    mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    return mesh, mvc, mf


def prepare_folders(path, simid):

    mode = 0o777
    simpath = os.path.join(path, simid)
    if not os.path.exists(simpath):
        os.mkdir(simpath)
        os.chmod(simpath, mode)
    
    imgpath = os.path.join(simpath, 'imgs/')
    if not os.path.exists(imgpath):
        os.mkdir(imgpath, mode)
        os.chmod(imgpath, mode)
    else:
        filelist = glob(os.path.join(imgpath, '*'))
        for filename in filelist:
            os.remove(filename)
    
    solnpath = os.path.join(simpath, 'soln/')
    if not os.path.exists(solnpath):
        os.mkdir(solnpath, mode)
        os.chmod(solnpath, mode)
    else:
        filelist = glob(os.path.join(solnpath, '*'))
        for filename in filelist:
            os.remove(filename)


    return simpath
