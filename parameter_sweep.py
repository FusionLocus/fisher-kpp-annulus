###### Implementation of the FKPP problem.
from mpi4py import MPI
import configparser
import ufl
import numpy as np
from simutils import probconstruct
from fkpp import run_simulation
import os
import logging


config = configparser.ConfigParser()
config.read('./realisations/parameter-sweep/config.ini')


logger = logging.getLogger(__name__)
logging.basicConfig(filename=config['sim']['output_dir_main'] + 'log.log',
                        encoding='utf-8',
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s')


print(MPI.COMM_WORLD.rank)
r0_min = config.getfloat('param.r0', 'minimum')
r0_max = config.getfloat('param.r0', 'maximum')
n_r0 = config.getint('param.r0', 'n')

r0_vals = np.linspace(r0_min, r0_max, num=n_r0)

delta_min = config.getfloat('param.delta', 'minimum')
delta_max = config.getfloat('param.delta', 'maximum')
n_delta = config.getint('param.delta', 'n')

comm = MPI.COMM_WORLD

delta_vals = np.linspace(delta_min, delta_max, num=n_delta)

for r0_val in r0_vals:
    for delta_val in delta_vals:
        r0_delta_str = f'r0={r0_val:.2f}, delta={delta_val:.2f}'
        if r0_val - delta_val/2 <= 0:
            if MPI.COMM_WORLD.rank == 0:
                logging.info(f'Skipping: No longer annulus geometry with ' + r0_delta_str)
                print(f'Skipping: No longer annulus geometry with ' + r0_delta_str)
        
        else:
            config['geometry']['delta'] = str(delta_val)
            config['geometry']['r0'] = str(r0_val)

            folder_path = config['sim']['output_dir_main'] + r0_delta_str
            if not os.path.exists(os.path.join(folder_path, 'soln.bp')):
                
                config['sim']['output_dir'] = folder_path +'/'
                if MPI.COMM_WORLD.rank == 0:
                    if not os.path.exists(folder_path):
                        os.mkdir(folder_path)
                        for i in range(MPI.COMM_WORLD.Get_size()):
                            if i > 0:
                                req = comm.irecv(source=i)
                                req.wait()
                    logging.info(f'Initialising simulation with ' + r0_delta_str)
                else:
                    req = comm.isend('hi', dest=0)
                    req.wait()
                    
                run_simulation(config, verbose=False, logger=logger)

            else:
                if MPI.COMM_WORLD.rank == 0:
                    logging.info('Skipping: Simulation already exists with ' + r0_delta_str)
                    print('Skipping: Simulation already exists with ' + r0_delta_str)

            

