# Functions for constructing a problem
from fenics import *
from dolfin import *
import simutils.preproc as preproc
import json
import sympy
import math

class BoundaryDef(SubDomain):
    def __init__(self, boundary_D):
        cpp.mesh.SubDomain.__init__(self)
        expression = preproc.parse_expression(boundary_D)
        if len(expression) > 1 and (isinstance(expression, list) or isinstance(expression, tuple)):
            expression = expression[0]
        
        self.boundary_descriptor = eval(expression)

    tol = 1e-14

    def inside(self, x, on_boundary):
        return self.boundary_descriptor(x, on_boundary)

def setup_BCs(themesh, raw_bcs, V, u, v, bypass=[]):
    """Function for reading and parsing boundary conditions.
    Arguments:
        themesh (DOLFIN mesh) - mesh upon which to construct the bcs
        raw_bcs (dictionary) - dictionary from a .json containing information about the bcs.
        V (FENICs function space) - function space containing measure information.
        u (FENICs function) - Trial function
        v (FENICs function) - Test function
        bypass (list) - If empty, generate boundary markers. If full, should contain
        boundary markers object and also a measure for those markers ds

    Returns:
        bcs (List) - List of dirichlet boundary condition objects
        integrals_VN (List) - List of von-neumman bc integrals
        integrals_R_L (List) - list of linear robin bc integrals 
        integrals_R_a (List) - list of bilinear robin bc integrals
        ds (FENICs measure) - Redefined measure to take into account the boundary markers
        boundary_markers (DOLFIN boundary markers) - Boundary markers

    """
    
    markers = []
    bcs = []
    keys = raw_bcs[0].keys()
    if not bypass:
        boundary_markers = MeshFunction('size_t', themesh, themesh.topology().dim()-1)
        c = 0
        # Go through each boundary and mark it
        for key in keys:
            current_marker = BoundaryDef(raw_bcs[0][key]['boundary_D'])
            markers.append(current_marker)
            markers[c].mark(boundary_markers, c)
            c += 1

        # Redefine the measure
        ds =  Measure('ds', domain=themesh, subdomain_data=boundary_markers)
    
    elif bypass:
        boundary_markers = bypass[0]
        ds = bypass[1]

    # Put each type of boundary condition in a list
    bcs = []
    integrals_VN = []
    integrals_R_L = []
    integrals_R_a = []

    c = 0 
    for key in keys:
        # Dirichlet boundary conditions
        if raw_bcs[0][key]['type'] == 'd':
            expression = preproc.parse_expression(raw_bcs[0][key]['val1'])
            current_bc = DirichletBC(V, expression,
                          boundary_markers, c)
            bcs.append(current_bc)
        
        # Robin boundary conditions
        elif raw_bcs[0][key]['type'] == 'r':
            if float(raw_bcs[0][key]['val1']) != 0:
                r = float(preproc.parse_expression(raw_bcs[0][key]['val1']))
                print('r={0}'.format(r))
                integrals_R_a.append(r*u*v*ds(c))
            if float(raw_bcs[0][key]['val2']) != 0:
                s = float(preproc.parse_expression(raw_bcs[0][key]['val2']))
                integrals_R_L.append(r*s*v*ds(c))

        # Von-Neumann boundary conditions
        elif raw_bcs[0][key]['type'] == 'vn':
            if float(raw_bcs[0][key]['val1']) != 0:
                g = float(preproc.parse_expression(raw_bcs[0][key]['val1']))
                integrals_VN.append(g*v*ds(c))

        c += 1
    if bypass:
        return bcs, integrals_VN, integrals_R_L, integrals_R_a
    else:
        return bcs, integrals_VN, integrals_R_L, integrals_R_a, ds, boundary_markers

def setup_fisher(mesh_path, raw_bcs, u0, dt, sbar, dbar):
    """Function to set up a fisher problem with appropriate boundary conditions"""
    
    themesh, mvc, mf = preproc.make_mesh(mesh_path)

    V = FunctionSpace(themesh, 'P', 1)
    Q = FunctionSpace(themesh, 'P', 1)

    # Define initial condition
    u_0 = Expression(preproc.parse_expression(u0)[0], degree=2)

    # Define initial values
    u_n1 = interpolate(u_0, V)
    u_n2 = interpolate(u_0, V)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    
    # Define boundary conditions
    bcs, integrals_VN, integrals_R_L, integrals_R_a, ds, boundary_markers = setup_BCs(themesh, raw_bcs, V, u, v)
    
    # Definition of the fisher problem
    a = u*v*dx + dt*(dbar*dot(grad(u), grad(v))*dx 
    + sbar*u*u_n1*v*dx - sbar*u*v*dx - sum(integrals_R_a))
    
    L = u_n1*v*dx + dt*(-sum(integrals_R_L) + sum(integrals_VN))

    u = Function(V)

    return themesh, V, Q, u_0, a, L, u, u_n1, u_n2, bcs, ds, boundary_markers