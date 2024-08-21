###### Implementation of the FKPP problem.
from mpi4py import MPI
import configparser
import ufl
import numpy as np
from simutils import probconstruct

def run_simulation(config, verbose=True, logger=None, save_at_end=False):
    ###### Handle parameter inputs #####
    parameters = {}

    for param in config['sim']:
        parameters[param] = config['sim'][param]

    parameters['n_t'] = config.getint('sim.times', 'n_t')
    parameters['degree'] = config.getint('sim.times', 'degree')
    parameters['t0'] = config.getfloat('sim.times', 't0')
    parameters['tf'] = config.getfloat('sim.times', 'tf')
    parameters['u_d'] = config.getfloat('ic', 'u_d')
    parameters['ic'] = lambda x : np.logical_and(np.isclose(x[0], 0), x[1] >= 0)*config.getfloat('ic', 'value')


    for param in config['geometry']:
        parameters[param] = config.getfloat('geometry', param)

    for param in config['param']:
        parameters[param] = config.getfloat('param', param)

    fisher_sim = probconstruct.FisherProblem(parameters)
    fisher_sim.get_mesh()
    fisher_sim.setup_dirichlet_bc(lambda x: np.logical_and(np.isclose(x[0], 0), x[1] >= 0))
    fisher_sim.setup_fkpp_problem()
    fisher_sim.setup_solution_file()
    fisher_sim.setup_solver()
    fisher_sim.run_simulation(verbose=verbose, logger=logger,
                              save_at_end=save_at_end)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('./realisations/base-halfannulus/config.ini')
    run_simulation(config)
