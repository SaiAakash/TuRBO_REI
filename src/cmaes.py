import os
import sys
import time
import numpy as np
import pandas as pd
import torch

from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem

from define_problems import DefineProblems

class BotorchToPymoo(ElementwiseProblem):
    def __init__(self, problem, dim=0, dim_emb=0, noise_std=None, negate=True):
        self.func = DefineProblems(problem, dim, dim_emb, noise_std, negate)
        self.nx = self.func.dim
        self.nf = 1
        self.ng = 0
        self.lb = np.zeros(self.nx)
        self.ub = np.ones(self.nx)
        super().__init__(n_var=self.nx, n_obj=self.nf, n_constr=self.ng, xl=self.lb, xu=self.ub)
    def _evaluate(self, x, out, *args, **kwargs):
        f = -self.func(torch.tensor(x)).cpu().numpy()
        out["F"] = f


def pymoo_ec(path_dir, problem, ns_init, ns_max, dim, dim_emb, itrial=1, verbose=True, solver='CMAES', popsize=20):
    if popsize == 0:
        popsize = None
    print(problem, solver, itrial)
    fun = BotorchToPymoo(problem, dim, dim_emb)

    if solver == 'CMAES':
        algorithm = eval(solver+'(popsize=popsize)')
    else:
        algorithm = eval(solver+'(pop_size=popsize)')

    path = os.path.join(path_dir, problem, solver)
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            pass

    path_time = os.path.join(path, 'times'+str(itrial)+'.csv')
    path_file = os.path.join(path, 'solutions'+str(itrial)+'.csv')
    if not os.path.exists(path_time):
        t0 = time.perf_counter()
        res = minimize(fun,
                    algorithm,
                    termination=('n_evals', ns_max), 
                    seed=itrial,
                    save_history=True, 
                    verbose=verbose)
        t1 = time.perf_counter()
        for i, hist in enumerate(res.history):
            if i == 0:
                pop_all = np.hstack([np.full([hist.pop.shape[0],1], hist.n_gen), hist.pop.get('X'), hist.pop.get('F'), hist.pop.get('G')])
            else:
                pop_all = np.vstack([pop_all, np.hstack([np.full([hist.pop.shape[0],1], hist.n_gen), hist.pop.get('X'), hist.pop.get('F'), hist.pop.get('G')])])
        columns = np.hstack(['n', 'x'+np.arange(1,1+fun.n_var).astype(str).astype(object), 'f'+np.arange(1,1+fun.n_obj).astype(str).astype(object), 'g'+np.arange(1,1+fun.n_constr).astype(str).astype(object), 'time'])
        df = pd.DataFrame(np.hstack([pop_all, np.full([pop_all.shape[0],1], (t1-t0)/pop_all.shape[0])]), columns=columns)
        df.to_csv(path_file, index=False)
        with open(path_time, 'a') as file:
            np.savetxt(file, np.array([[itrial, t1-t0]]), delimiter=',')
        del res, pop_all, df
    # print(problem, solver, itrial)


if __name__ == "__main__": 
    (path_dir, problem, ns_init, ns_max, dim, dim_emb, itrial, verbose, solver, popsize) = sys.argv[1:]
    pymoo_ec(path_dir, problem, int(ns_init), int(ns_max), int(dim), int(dim_emb), int(itrial), eval(verbose), solver, int(popsize))
