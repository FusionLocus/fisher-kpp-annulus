###### Implementation of the FKPP problem.
from mpi4py import MPI
import configparser
import ufl
import numpy as np
from simutils import probconstruct
import os
from dolfinx import fem
from basix.ufl import element
def run_simulation(config, verbose=True, logger=None, save_at_end=False, mesh_bypass=None, dry_run=False):
    ###### Handle parameter inputs #####
    parameters = {}

    for param in config['sim']:
        parameters[param] = config['sim'][param]

    parameters['n_t'] = config.getint('sim.times', 'n_t')
    parameters['degree'] = config.getint('sim.times', 'degree')
    parameters['t0'] = config.getfloat('sim.times', 't0')
    parameters['tf'] = config.getfloat('sim.times', 'tf')
    parameters['u_d'] = config.getfloat('ic', 'u_d')
    parameters['ic_func'] = config['ic']['func']
    print(parameters['ic_func'])


    for param in config['geometry']:
        parameters[param] = config.getfloat('geometry', param)

    for param in config['param']:
        parameters[param] = config.getfloat('param', param)

    fisher_sim = probconstruct.FisherProblem(parameters)
    fisher_sim.get_mesh(bypass=mesh_bypass)
    fisher_sim.setup_dirichlet_bc(lambda x: np.logical_and(np.isclose(x[0], 0), x[1] >= 0))
    fisher_sim.setup_fkpp_problem()
    if dry_run:
        W = fem.functionspace(fisher_sim.msh, element('DG', fisher_sim.msh.basix_cell(), 0))
        print(W.tabulate_dof_coordinates().shape)
        return
    fisher_sim.setup_solution_file()
    fisher_sim.setup_solver()
    fisher_sim.run_simulation(verbose=verbose, logger=logger,
                              save_sparse=save_at_end)

    return fisher_sim.msh

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('./realisations/test-r0-0.4-delta-0.4/config.ini')
    run_simulation(config, save_at_end=False,  dry_run=True)
