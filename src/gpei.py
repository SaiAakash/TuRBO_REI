import os
import sys
import time
import warnings
import time

import numpy as np
import pandas as pd
import torch
from torch.quasirandom import SobolEngine
from botorch import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import normalize, unnormalize
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.analytic import ExpectedImprovement, LogExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from gpytorch.kernels import MaternKernel, RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints.constraints import GreaterThan

from define_problems import DefineProblems


warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
SMOKE_TEST = os.environ.get("SMOKE_TEST")

class GPLogEI():
    def __init__(self, device, dtype, path):
        self.device = device
        self.dtype = dtype
        self.path_file = path


    def generate_initial_data(self, dim, fun, n=6):
        # generate training data
        seed = int(self.rng.integers(low=0, high=2**16, dtype=np.int64))
        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        train_xn = sobol.draw(n=n).to(dtype=self.dtype, device=self.device)
        train_obj = torch.tensor([fun(x) for x in unnormalize(train_xn.detach(), bounds=fun.bounds)], dtype=self.dtype, device=self.device).unsqueeze(-1)
        return train_xn, train_obj


    def initialize_model(self, train_xn, train_obj, NOISE, MIN_INFERRED_NOISE_LEVEL=1e-4, KERNEL='Matern5', OUTPUT_SCALE=True, state_dict=None):
        # define models for objective and constraint
        train_yvar = torch.full_like(train_obj, 0.0)
        if KERNEL == 'Gaussian':
            covar_module = RBFKernel(ard_num_dims = train_xn.shape[-1])
        else:
            covar_module = MaternKernel(nu=2.5, ard_num_dims=train_xn.shape[-1], lengthscale_prior=GammaPrior(3.0, 6.0))
        if OUTPUT_SCALE:
            covar_module = ScaleKernel(covar_module, outputscale_prior=GammaPrior(2.0, 0.15))
        if NOISE:
            noise_prior = GammaPrior(1.1, 0.05)
            noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
            likelihood = GaussianLikelihood(noise_prior=noise_prior, noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL, transform=None, initial_value=noise_prior_mode))
            model = SingleTaskGP(train_xn, train_obj, likelihood=likelihood, covar_module=covar_module, outcome_transform=Standardize(m=1))
        else:
            sys.exit()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return mll, model


    def optimize_acqf_and_get_observation(self, acq_func, fun, batch_size=1):
        """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.nbounds,
            q=batch_size,
            num_restarts=10,
            raw_samples=512,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        # observe new values
        new_xn = candidates.detach().to(dtype=self.dtype, device=self.device)
        new_obj = torch.tensor([fun(x) for x in unnormalize(new_xn.detach(), bounds=fun.bounds)], dtype=self.dtype, device=self.device).unsqueeze(-1)
        return new_xn, new_obj
    

    def optimize(self, dim, fun, max_iter, seed=0, batch_size=1, n_init=10, NOISE=True, MIN_INFERRED_NOISE_LEVEL=1e-4, KERNEL='Matern5', OUTPUT_SCALE=True, USE_STATE_DICT=False, verbose=False):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.nbounds = torch.stack([torch.zeros(dim), torch.ones(dim)]).to(device=self.device, dtype=self.dtype)

        if not os.path.exists(self.path_file):
            t0 = time.perf_counter()
            train_xn, train_obj = self.generate_initial_data(dim, fun, n_init)
            t1 = time.perf_counter()
            times = np.full(n_init, (t1-t0)/n_init)

            columns = np.hstack(['x'+np.arange(1,train_xn.shape[1]+1).astype(str).astype(object), 'f'+np.arange(1,train_obj.shape[1]+1).astype(str).astype(object), 'time'])
            df = pd.DataFrame(np.hstack([unnormalize(train_xn.detach(), bounds=fun.bounds).cpu().numpy(), (-train_obj).cpu().numpy(), times.reshape(-1,1)]), columns=columns)
            df.to_csv(self.path_file, index=False)
        else:
            df = pd.read_csv(self.path_file)
            train_xn = torch.tensor(df.loc[:,df.columns.str.contains('x')].values).to(dtype=self.dtype, device=self.device)
            train_obj = -1*torch.tensor(df.loc[:,df.columns.str.contains('f')].values.reshape(-1,1)).to(dtype=self.dtype, device=self.device)
            times = df['time'].values
            columns = np.hstack(['x'+np.arange(1,train_xn.shape[1]+1).astype(str).astype(object), 'f'+np.arange(1,train_obj.shape[1]+1).astype(str).astype(object), 'time'])

        t0 = time.perf_counter()
        mll, model = self.initialize_model(train_xn, train_obj, NOISE, MIN_INFERRED_NOISE_LEVEL, KERNEL, OUTPUT_SCALE)

        # run N_BATCH rounds of BayesOpt after the initial random batch
        counter = 0
        while train_xn.shape[0] < max_iter:
            counter += 1

            # fit the models
            fit_gpytorch_mll(mll)

            # for best_f, we use the best observed noisy values as an approximation
            if batch_size<=1:
                EI = LogExpectedImprovement(
                    model=model,
                    best_f=train_obj.max(),
                )
            else:
                EI = qLogExpectedImprovement(
                    model=model,
                    best_f=train_obj.max(),
                )

            # optimize and get new observation
            new_xn, new_obj = self.optimize_acqf_and_get_observation(EI, fun, batch_size)

            # update training points
            train_xn = torch.cat([train_xn, new_xn])
            train_obj = torch.cat([train_obj, new_obj])

            t1 = time.perf_counter()
            times = np.hstack([times, np.full(new_xn.shape[0]-1, np.nan), t1-t0])
            t0 = t1

            # reinitialize the models so they are ready for fitting on next iteration
            # use the current state dict to speed up fitting
            if USE_STATE_DICT:
                mll, model = self.initialize_model(train_xn, train_obj, NOISE, MIN_INFERRED_NOISE_LEVEL, KERNEL, OUTPUT_SCALE, state_dict=model.state_dict())
            else:
                mll, model = self.initialize_model(train_xn, train_obj, NOISE, MIN_INFERRED_NOISE_LEVEL, KERNEL, OUTPUT_SCALE)

            if verbose:
                print(train_obj.shape[0], train_obj.max().unsqueeze(-1).cpu().numpy()[0])
            
            if counter%100==0:
                df = pd.DataFrame(np.hstack([unnormalize(train_xn.detach(), bounds=fun.bounds).cpu().numpy(), (-train_obj).cpu().numpy(), times.reshape(-1,1)]), columns=columns)
                df.to_csv(self.path_file, index=False)
            
        return unnormalize(train_xn.detach(), bounds=fun.bounds), train_obj, times.reshape(-1,1)



def botorch_ei(path_dir, problem, ns_init, ns_max, dim=0, dim_emb=0, itrial=1, verbose=True, batch_size=5, device='cpu'):
    if batch_size > 1:
        solver = 'GP-qLogEI-'+str(batch_size)
    else:
        solver = 'GP-LogEI-'+str(batch_size)
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
    gpei = GPLogEI(device, dtype, path_file)
    t0 = time.perf_counter()
    xs, ys, times = gpei.optimize(nx, fun, ns_max, seed=itrial, batch_size=batch_size, n_init=ns_init, verbose=verbose)
    t1 = time.perf_counter()
    columns = np.hstack(['x'+np.arange(1,xs.shape[1]+1).astype(str).astype(object), 'f'+np.arange(1,ys.shape[1]+1).astype(str).astype(object), 'time'])
    df = pd.DataFrame(np.hstack([xs.cpu().numpy(), (-ys).cpu().numpy(), times]), columns=columns)
    df.to_csv(path_file, index=False)
    with open(path_time, 'a') as file:
        np.savetxt(file, np.array([[itrial, t1-t0]]), delimiter=',')
    del xs, ys, df
    # print(problem, solver, itrial)


if __name__ == "__main__": 
    (path_dir, problem, ns_init, ns_max, dim, dim_emb, itrial, verbose, batch_size, device) = sys.argv[1:]
    botorch_ei(path_dir, problem, int(ns_init), int(ns_max), int(dim), int(dim_emb), int(itrial), eval(verbose), int(batch_size), device)
