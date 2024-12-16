import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import time
import subprocess
import numpy as np

def check_processes(procs, max_procs, n_solvers, t_delta=1):
    while True:
        running_procs = np.sum([procs[i].poll() is None for i in range(len(procs))])
        if running_procs <= max_procs - n_solvers:
            break
        time.sleep(t_delta)


if __name__ == "__main__":
    
    # Number of CPUs (threads) used
    max_procs = 1

    # Setting for paper reproduction
    # path_dir = 'results'
    # problems = ['HPA101-0', 'HPA102-0', 'HPA103-0', 'HPA101-1', 'HPA102-1', 'HPA103-1', 'HPA101-2', 'RoverTrajectory', 'MOPTA08']
    # itrials = np.arange(1,12)
    # ns_init = 30
    # ns_max = 2000
    # ns_max_gp = 1000
    # ns_max_saasbo = 500
    # verbose = False

    # Smoke test
    path_dir = 'results'    # path to the output directory
    problems = ['HPA101-0'] # problem name, see the above list or src/define_problems.py
    itrials = [1]           # run number also used as random seed 
    ns_init = 30            # number of initial samples for each trust region
    ns_max = 70             # number of maximum function evaluation
    ns_max_gp = 40          # number of maximum function evaluation for GP-EI
    ns_max_saasbo = 31      # number of maximum function evaluation for SAASBO
    verbose = True

    # for TuRBO-m-REI
    acqf='EI'               # acquisition function in TuRBO: 'EI' or 'TS'
    racqf='qREI'            # region-averaged acquisition function: 'REI', 'qREI', 'EI', 'qEI, 'RUCB', 'UCB', or None (None results in the usual TuRBO)
    n_trust_regions = 1     # 1:TuRBO-1, m>1:TuRBO-m
    batch_size = 1          # batch size for local seach in TuRBO
    n_init_region = ns_init # number of initial sample points when TuRBO starts with REI

    # Other setting
    dim = 0                 # for Botorch problems
    popsize = 0             # for CMA-ES (0 means the use of default setting)


    procs = []
    for problem in problems:
        print('Problem: ' + problem)
        for itrial in itrials:
            print('Trial: ' + str(itrial))

            # TuRBO-m-REI
            check_processes(procs, max_procs, 1)
            procs.append(subprocess.Popen(['python', 'src/turbo_rei.py', path_dir, problem, str(ns_init), str(ns_max), str(dim), str(0), str(itrial), str(verbose), str(n_init_region), str(n_trust_regions), str(batch_size), acqf, racqf, 'cpu']))

            # TuRBO-m-REI(restart)
            check_processes(procs, max_procs, 1)
            procs.append(subprocess.Popen(['python', 'src/turbo_rei.py', path_dir, problem, str(ns_init), str(ns_max), str(dim), str(0), str(itrial), str(verbose), str(0), str(n_trust_regions), str(batch_size), acqf, racqf, 'cpu']))

            # TuRBO-m
            check_processes(procs, max_procs, 1)
            procs.append(subprocess.Popen(['python', 'src/turbo_rei.py', path_dir, problem, str(ns_init), str(ns_max), str(dim), str(0), str(itrial), str(verbose), str(0), str(n_trust_regions), str(batch_size), acqf, str(None), 'cpu']))


            # SAASBO
            check_processes(procs, max_procs, 1)
            procs.append(subprocess.Popen(['python', 'src/saasbo.py', path_dir, problem, str(ns_init), str(ns_max_saasbo), str(dim), str(0), str(itrial), str(verbose), str(batch_size), 'cpu']))

            # GP-LogEI
            check_processes(procs, max_procs, 1)
            procs.append(subprocess.Popen(['python', 'src/gpei.py', path_dir, problem, str(ns_init), str(ns_max_gp), str(dim), str(0), str(itrial), str(verbose), str(1), 'cpu']))
            
            # CMAES
            check_processes(procs, max_procs, 1)
            procs.append(subprocess.Popen(['python', 'src/cmaes.py', path_dir, problem, str(ns_init), str(ns_max), str(dim), str(0), str(itrial), str(verbose), 'CMAES', str(popsize)]))

            time.sleep(1)
    check_processes(procs, 0, 0)
    print('finished')
