"""
implementation of saasbo based on BoTorch documentation
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.quasirandom import SobolEngine

from botorch import fit_fully_bayesian_model_nuts
from botorch.acquisition.analytic import ExpectedImprovement, LogExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf

from define_problems import DefineProblems


WARMUP_STEPS = 256
NUM_SAMPLES = 128
THINNING = 16


def saasbo(path_dir, problem, ns_init, ns_max, dim=0, dim_emb=0, itrial=1, verbose=True, batch_size=5, device='cpu', restart=True):
    solver = 'SAASBO-'+str(batch_size)
    print(problem, solver, itrial)
    if device=='auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    dtype = torch.double
    fun = DefineProblems(problem, dim, dim_emb, noise_std=None, negate=True).to(device=device, dtype=dtype)
    dim = fun.dim
    if dim_emb > dim:
        nx = dim_emb
    else:
        nx = dim
    path = os.path.join(path_dir, problem, solver)
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            pass

    path_time = os.path.join(path, 'times'+str(itrial)+'.csv')
    path_file = os.path.join(path, 'solutions'+str(itrial)+'.csv')
    if not os.path.exists(path_file):
        t0 = time.perf_counter()
        times = []

        # main part
        rng = np.random.default_rng(itrial)
        seed = int(rng.integers(low=0, high=2**16, dtype=np.int64))
        xs = SobolEngine(dimension=nx, scramble=True, seed=seed).draw(ns_init).to(dtype=dtype, device=device)
        ys = torch.tensor([fun(x) for x in xs]).unsqueeze(-1).to(dtype=dtype, device=device)
        if verbose:
            print(f"{len(ys)}) Objective: {ys.max():.2e}")
        t = time.perf_counter()
        times.extend([(t - t0)/ns_init]*ns_init)

        columns = np.hstack(['x'+np.arange(1,xs.shape[1]+1).astype(str).astype(object), 'f'+np.arange(1,ys.shape[1]+1).astype(str).astype(object), 'time'])
        df = pd.DataFrame(np.hstack([xs.cpu().numpy(), (-ys).cpu().numpy(), np.array(times).reshape(-1,1)]), columns=columns)
        df.to_csv(path_file, index=False)

        while ys.shape[0] < ns_max:
            t = time.perf_counter()
            gp = SaasFullyBayesianSingleTaskGP(
                train_X=xs,
                train_Y=ys,
                train_Yvar=torch.full_like(ys, 1e-6),
                outcome_transform=Standardize(m=1),
            )
            fit_fully_bayesian_model_nuts(
                gp,
                warmup_steps=WARMUP_STEPS,
                num_samples=NUM_SAMPLES,
                thinning=THINNING,
                disable_progbar=True,
            )

            if batch_size<=1:
                EI = LogExpectedImprovement(gp, ys.max())
                candidates, acq_values = optimize_acqf(
                    EI,
                    bounds=torch.cat((torch.zeros(1, nx), torch.ones(1, nx))).to(dtype=dtype, device=device),
                    q=1,
                    num_restarts=10,
                    raw_samples=512,
                )
            else:
                EI = qLogExpectedImprovement(model=gp, best_f=ys.max())
                candidates, acq_values = optimize_acqf(
                    EI,
                    bounds=torch.cat((torch.zeros(1, nx), torch.ones(1, nx))).to(dtype=dtype, device=device),
                    q=batch_size,
                    num_restarts=10,
                    raw_samples=512,
                )

            ys_next = torch.tensor([fun(x) for x in candidates]).unsqueeze(-1).to(dtype=dtype, device=device)
            xs = torch.cat((xs, candidates))
            ys = torch.cat((ys, ys_next))

            if verbose:
                print(f"{len(ys)}) Objective: {ys.max():.2e}")

            tt = time.perf_counter()
            times.extend([(tt - t)/batch_size]*batch_size)

            df = pd.DataFrame(np.hstack([xs.cpu().numpy(), (-ys).cpu().numpy(), np.array(times).reshape(-1,1)]), columns=columns)
            df.to_csv(path_file, index=False)

        t1 = time.perf_counter()
        df = pd.DataFrame(np.hstack([xs.cpu().numpy(), (-ys).cpu().numpy(), np.array(times).reshape(-1,1)]), columns=columns)
        df.to_csv(path_file, index=False)
        with open(path_time, 'a') as file:
            np.savetxt(file, np.array([[itrial, t1-t0]]), delimiter=',')
        del xs, ys, df
    

    elif restart:
        df = pd.read_csv(path_file)
        if len(df) < ns_max:
            xs = torch.tensor(df.loc[:,df.columns.str.contains('x')].values).to(dtype=dtype, device=device)
            ys = -1*torch.tensor(df.loc[:,df.columns.str.contains('f')].values.reshape(-1,1)).to(dtype=dtype, device=device)
            times = df['time'].values.tolist()

            t0 = time.perf_counter()
            # main part
            rng = np.random.default_rng(itrial)
            seed = int(rng.integers(low=0, high=2**16, dtype=np.int64))

            if verbose:
                print(f"{len(ys)}) Objective: {ys.max():.2e}")

            columns = df.columns

            while ys.shape[0] < ns_max:
                t = time.perf_counter()
                gp = SaasFullyBayesianSingleTaskGP(
                    train_X=xs,
                    train_Y=ys,
                    train_Yvar=torch.full_like(ys, 1e-6),
                    outcome_transform=Standardize(m=1),
                )
                fit_fully_bayesian_model_nuts(
                    gp,
                    warmup_steps=WARMUP_STEPS,
                    num_samples=NUM_SAMPLES,
                    thinning=THINNING,
                    disable_progbar=True,
                )

                if batch_size<=1:
                    EI = LogExpectedImprovement(gp, ys.max())
                    candidates, acq_values = optimize_acqf(
                        EI,
                        bounds=torch.cat((torch.zeros(1, nx), torch.ones(1, nx))).to(dtype=dtype, device=device),
                        q=1,
                        num_restarts=10,
                        raw_samples=512,
                    )
                else:
                    EI = qLogExpectedImprovement(model=gp, best_f=ys.max())
                    candidates, acq_values = optimize_acqf(
                        EI,
                        bounds=torch.cat((torch.zeros(1, nx), torch.ones(1, nx))).to(dtype=dtype, device=device),
                        q=batch_size,
                        num_restarts=10,
                        raw_samples=512,
                    )

                ys_next = torch.tensor([fun(x) for x in candidates]).unsqueeze(-1).to(dtype=dtype, device=device)
                xs = torch.cat((xs, candidates))
                ys = torch.cat((ys, ys_next))

                if verbose:
                    print(f"{len(ys)}) Objective: {ys.max():.2e}")

                tt = time.perf_counter()
                times.extend([(tt - t)/batch_size]*batch_size)

                df = pd.DataFrame(np.hstack([xs.cpu().numpy(), (-ys).cpu().numpy(), np.array(times).reshape(-1,1)]), columns=columns)
                df.to_csv(path_file, index=False)

            t1 = time.perf_counter()
            df = pd.DataFrame(np.hstack([xs.cpu().numpy(), (-ys).cpu().numpy(), np.array(times).reshape(-1,1)]), columns=columns)
            df.to_csv(path_file, index=False)
            with open(path_time, 'a') as file:
                np.savetxt(file, np.array([[itrial, t1-t0]]), delimiter=',')
            del xs, ys, df


if __name__ == "__main__": 
    if len(sys.argv) == 11:
        (path_dir, problem, ns_init, ns_max, dim, dim_emb, itrial, verbose, batch_size, device) = sys.argv[1:]
        saasbo(path_dir, problem, int(ns_init), int(ns_max), int(dim), int(dim_emb), int(itrial), eval(verbose), int(batch_size), device)
    elif len(sys.argv) == 12:
        (path_dir, problem, ns_init, ns_max, dim, dim_emb, itrial, verbose, batch_size, device, restart) = sys.argv[1:]
        saasbo(path_dir, problem, int(ns_init), int(ns_max), int(dim), int(dim_emb), int(itrial), eval(verbose), int(batch_size), device, eval(restart))
