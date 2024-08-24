# Functions for constructing a problem
import math
from mpi4py import MPI
import ufl
import numpy as np
from dolfinx import fem, mesh, io, plot
from dolfinx.io.gmshio import model_to_mesh
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import adios4dolfinx
import gmsh
import os
import sys
from basix.ufl import element
from petsc4py.PETSc import ScalarType
import logging

#from dolfinx.fem.petsc import

##### Code for running a fisher-KPP simulation #####

class FisherProblem():
    """All utilities to run a Fisher solver and save the results to file.
    """
    def __init__(self, parameters):

        self.params = parameters
        self.dt = (parameters['tf'] - parameters['t0'])/parameters['n_t']

        return

    def get_mesh(self, bypass=False):
        
        if self.params['case'] == 'annulus':
            model = make_annulus(self.params)
        
        elif self.params['case'] == 'rectangle':
            model = make_rectangle(self.params)

        msh, cell_tags, facet_tags = model_to_mesh(model, 
                                MPI.COMM_WORLD,
                                0)
        
        self.element = element(self.params['element_type'], 
                               msh.basix_cell(),
                               self.params['degree'])

        self.msh = msh
        gmsh.finalize()
        # Define space, u at current and previous timestep
        self.V = fem.functionspace(self.msh, self.element)

        return

    def setup_fkpp_problem(self):
        """Requires that self.msh and self.element have been defined (e.g. by running self.get_mesh())
        Implements the problem using a Crank-Nicolson time discretisation (at this point, the problem
        is semi-discretised as is only discrete in time, not in space. FENICSX will handle the spatial
        discretisation).
        """


        self.u, self.u_n = fem.Function(self.V), fem.Function(self.V)
        self.v = ufl.TestFunction(self.V)

        dt, D, k = self.dt, self.params['d'], self.params['k']

        # For ease of notation, write u and v as references to their class definitions.
        u, u_n, v = self.u, self.u_n, self.v

        # Give u and u_n their initial values (which we take to be the same).
        u.interpolate(self.params['ic'])
        u_n.interpolate(self.params['ic'])



        # Assumes either Dirichlet or no-flux boundary conditions.
        self.F = (u*v - u_n*v)*ufl.dx + 0.5*dt*(
            D*ufl.inner(ufl.nabla_grad(u), ufl.nabla_grad(v))
            + D*ufl.inner(ufl.nabla_grad(u_n), ufl.nabla_grad(v))
            - k*u*(1-u)*v - k*u_n*(1-u_n)*v)*ufl.dx


        return

    def setup_dirichlet_bc(self, marker_func):

        facets = mesh.locate_entities_boundary(
            self.msh,
            dim=1,
            marker= marker_func
        )
        dofs = fem.locate_dofs_topological(V=self.V, 
                                           entity_dim=1,
                                           entities=facets)

        self.bc = fem.dirichletbc(value=ScalarType(self.params['u_d']), 
                                  dofs=dofs, 
                                  V=self.V)

        return

    def setup_solution_file(self):
        """Creates a folder for the solution and to store any relevant plots."""
        self.soln_file = self.params['output_dir']+'soln.bp'
        
        adios4dolfinx.write_mesh(self.soln_file, self.msh)


        return

    def setup_solver(self):
        """Produce a non-linear solver using the F defined by setup_fkpp_problem().
        """
        problem = NonlinearProblem(self.F, self.u, bcs=[self.bc])
        self.solver = NewtonSolver(MPI.COMM_WORLD, problem)
        self.solver.convergence_criterion = "incremental"
        self.solver.rtol = 1e-6
        self.solver.report = True


        return

    def run_simulation(self, verbose=True, logger=None, save_at_end=False):
        t = self.params['t0']
        if not verbose and logger is not None:
            if MPI.COMM_WORLD.rank == 0:
                logger.info('Initialising simulation run...')
        adios4dolfinx.write_function(self.soln_file, self.u, time=t, name='u')

        for n in range(self.params['n_t']):
            t += self.dt
            
            self.solver.solve(self.u)
            self.u.x.scatter_forward()
            umin = np.min(self.u.x.array)
            umax = np.max(self.u.x.array)
            if MPI.COMM_WORLD.rank == 0:
                if verbose:
                    print(f'Iterating, t={t:.2f}, umax={umax:.4f}, umin={umin:.4f}')
                elif not verbose and logger is not None:
                    logger.info(f'Iterating, t={t:.2f}, umax={umax:.4f}, umin={umin:.4f}')

            if not save_at_end:
                adios4dolfinx.write_function(self.soln_file, self.u, time=np.round(t, 4), name='u')

            self.u_n.x.array[:] = self.u.x.array

        if save_at_end:
            adios4dolfinx.write_function(self.soln_file, self.u, time=np.round(t, 4), name='u')
            if MPI.COMM_WORLD.rank == 0:
                logger.info(f'Saving at end of simulation only. t={t:.2f}, umax={umax:.4f}, umin={umin:.4f}')
        
        return
    
def make_annulus(params):

    lc = params['lc']
    Ri = params['r0'] - params['delta']/2
    Re = params['r0'] + params['delta']/2
    
    gmsh.initialize()
    # Outer annulus arc points
    factory = gmsh.model.geo

    factory.addPoint(
        0, Re, 0, lc, 1 # (x, y, z, lc, tag)
    )

    factory.addPoint(Re, 0, 0, lc, 2)
    factory.addPoint(0, -Re, 0, lc, 3)

    # Inner annulus arc points
    factory.addPoint(0, -Ri, 0, lc, 4)
    factory.addPoint(Ri, 0, 0, lc, 5)
    factory.addPoint(0, Ri, 0, lc, 6)

    # Centre of the annulus
    factory.addPoint(0, 0, 0, lc, 7)

    # Implement the annulus arcs and line boundaries
    factory.addCircleArc(1, 7, 2, 1)
    factory.addCircleArc(2, 7, 3, 2)
    factory.addLine(3, 4, 3)
    factory.addCircleArc(4, 7, 5, 4)
    factory.addCircleArc(5, 7, 6, 5)
    factory.addLine(6, 1, 6)

    # Produce curve loop
    factory.addCurveLoop([1, 2, 3, 4, 5, 6], 7)
    factory.addPlaneSurface([7], 8)


    factory.synchronize()

    gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4, 5, 6], 9)
    gmsh.model.addPhysicalGroup(2, [8], 10)

    gmsh.model.mesh.generate(2)
    #if MPI.COMM_WORLD.rank == 0:
    #    gmsh.write(os.path.join(params['output_dir'], 'mesh.msh'))
    
    return gmsh.model

def make_rectangle(params):

    lc = params['lc']
    Lx = params['lx']
    Ly = params['ly']
    
    gmsh.initialize()
    # Corners of the rectangle
    factory = gmsh.model.geo

    factory.addPoint(
        0, 0, 0, lc, 1 # (x, y, z, lc, tag)
    )
    factory.addPoint(Lx, 0, 0, lc, 2)
    factory.addPoint(Lx, Ly, 0, lc, 3)
    factory.addPoint(0, Ly, 0, lc, 4)

    factory.addLine(1, 2, 1)
    factory.addLine(2, 3, 2)
    factory.addLine(3, 4, 3)
    factory.addLine(4, 1, 4)

    # Produce curve loop
    factory.addCurveLoop([1, 2, 3, 4], 5)
    factory.addPlaneSurface([5], 6)

    factory.synchronize()

    gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4], 7)
    gmsh.model.addPhysicalGroup(2, [6], 8)

    gmsh.model.mesh.generate(2)
    print('test-1.0')
    #if MPI.COMM_WORLD.rank == 0:
    #    gmsh.write(os.path.join(params['output_dir'], 'mesh.msh'))
    
    return gmsh.model