"""
implementation of turbo based on BoTorch documentation
"""
import os
import sys
import time
import math
import warnings
from typing import Callable, Union, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.quasirandom import SobolEngine
from botorch.acquisition.analytic import ExpectedImprovement, LogExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
import gpytorch
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from rei import qRegionalExpectedImprovement, LogRegionalExpectedImprovement
from rucb import RegionalUpperConfidenceBound
from define_problems import DefineProblems


SMOKE_TEST = os.environ.get("SMOKE_TEST")
warnings.simplefilter('ignore')

@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


class TuRBO():
    def __init__(self, device, dtype, path='results.csv'):
        self.device = device
        self.dtype = dtype
        self.path_file = path

    def update_state(self, state, Y_next):
        if max(Y_next) > state.best_value:
            state.success_counter += 1
            state.failure_counter = 0
        else:
            state.success_counter = 0
            state.failure_counter += 1

        if state.success_counter == state.success_tolerance:  # Expand trust region
            state.length = min(2.0 * state.length, state.length_max)
            state.success_counter = 0
        elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
            state.length /= 2.0
            state.failure_counter = 0

        state.best_value = max(state.best_value, max(Y_next).item())
        if state.length < state.length_min:
            state.restart_triggered = True
        return state


    def get_initial_points(self, dim, n_pts, seed=0):
        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        X_init = sobol.draw(n=n_pts).to(dtype=self.dtype, device=self.device)
        return X_init


    def generate_batch(
        self, 
        state,
        model,  # GP model
        X,  # Evaluated points on the domain [0, 1]^d
        Y,  # Function values
        batch_size,
        n_candidates=None,  # Number of candidates for Thompson sampling
        num_restarts=10,
        raw_samples=512,
        acqf="TS",  # "EI" or "TS"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert acqf in ("TS", "EI")
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

        # Scale the TR to be proportional to the lengthscales
        x_center = X[Y.argmax(), :].clone()
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

        if acqf == "TS":
            dim = X.shape[-1]
            sobol = SobolEngine(dim, scramble=True)
            pert = sobol.draw(n_candidates).to(dtype=self.dtype, device=self.device)
            pert = tr_lb + (tr_ub - tr_lb) * pert

            # Create a perturbation mask
            prob_perturb = min(20.0 / dim, 1.0)
            mask = torch.rand(n_candidates, dim, dtype=self.dtype, device=self.device) <= prob_perturb
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=self.device)] = 1

            # Create candidate points from the perturbations and the mask
            X_cand = x_center.expand(n_candidates, dim).clone()
            X_cand[mask] = pert[mask]

            # Sample on the candidate points

            # replacement is True for simplicity
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=True)
            with torch.no_grad():
                posterior = thompson_sampling.model.posterior(
                    X_cand,
                    observation_noise=None,
                    posterior_transform=None,
                )
                samples = posterior.rsample(sample_shape=torch.Size([batch_size]))
                X_next = thompson_sampling.maximize_samples(X_cand, samples, batch_size)
                acq_value = torch.max(samples, dim=1)[0].reshape((-1, 1))

        elif acqf == "EI":
            if batch_size<=1:
                ei = LogExpectedImprovement(model, Y.max())
                X_next, acq_value = optimize_acqf(
                    ei,
                    bounds=torch.stack([tr_lb, tr_ub]),
                    q=1,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                )
            else:
                ei = qLogExpectedImprovement(model, Y.max())
                X_next, acq_value = optimize_acqf(
                    ei,
                    bounds=torch.stack([tr_lb, tr_ub]),
                    q=batch_size,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                )

        return X_next, acq_value


    def evaluate_on_init_points(
        self, 
        eval_fn: Callable[[np.ndarray], float],
        dim: int,
        n_init: int,
        rng: np.random.Generator,
    ):
        seed = int(rng.integers(low=0, high=2**16, dtype=np.int64))
        X_turbo = self.get_initial_points(dim, n_init, seed=seed)
        y_turbo = torch.tensor(
            [eval_fn(x) for x in X_turbo], dtype=self.dtype, device=self.device
        ).unsqueeze(-1)

        return (X_turbo, y_turbo)


    def x_next_and_acqui_vals_on_tr(
        self, 
        dim,
        X_turbo,
        y_turbo,
        state,
        batch_size,
        n_candidates,
        num_restarts,
        raw_samples,
        max_cholesky_size,
        noise_constraint,
        n_gp_max=2000,
        acqf="TS",
    ):
        mask = self.select_sample(X_turbo, y_turbo, n_gp_max)
        train_Y = (y_turbo - y_turbo.mean()) / y_turbo.std()
        likelihood = GaussianLikelihood(
            noise_constraint=Interval(noise_constraint[0], noise_constraint[1])
        )
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
            MaternKernel(
                nu=2.5,
                ard_num_dims=dim,
                lengthscale_constraint=Interval(0.005, 4.0),
            )
        )
        model = SingleTaskGP(
            X_turbo[mask], train_Y[mask], covar_module=covar_module, likelihood=likelihood
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Do the fitting and acquisition function optimization inside the Cholesky context
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            # Fit the model
            fit_gpytorch_mll(mll)

            # Create a batch
            X_next, aqui_vals = self.generate_batch(
                state=state,
                model=model,
                X=X_turbo[mask],
                Y=train_Y[mask],
                batch_size=batch_size,
                n_candidates=n_candidates,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                acqf=acqf,
            )
        return (X_next, aqui_vals)


    def from_hyper_cube(self, x: torch.tensor, lb: np.ndarray, ub: np.ndarray) -> torch.Tensor:
        if lb is None:
            return x
        else:
            return x * (ub - lb) + lb


    def sampling_from_global_model(
        self, 
        X_hist, 
        y_hist,
        bounds, 
        eval_fn: Callable[[np.ndarray], float],
        dim: int,
        n_init: int,
        rng: np.random.Generator,
        max_cholesky_size,
        length_init = 0.8,
        q_batch = 1,
        racqf = 'qREI',
        MIN_INFERRED_NOISE_LEVEL=1e-4,
    ):
        train_y = (y_hist - y_hist.mean()) / y_hist.std()
        
        covar_module = MaternKernel(nu=2.5, ard_num_dims=X_hist.shape[-1], lengthscale_prior=GammaPrior(3.0, 6.0))
        covar_module = ScaleKernel(covar_module, outputscale_prior=GammaPrior(2.0, 0.15))
        noise_prior = GammaPrior(1.1, 0.05)
        noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
        likelihood = GaussianLikelihood(noise_prior=noise_prior, noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL, transform=None, initial_value=noise_prior_mode))
        model = SingleTaskGP(X_hist, train_y, likelihood=likelihood, covar_module=covar_module, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Do the fitting and acquisition function optimization inside the Cholesky context
        seed = int(rng.integers(low=0, high=2**16, dtype=np.int64))
        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        X_dev = sobol.draw(n=128).to(dtype=self.dtype, device=self.device)
        # move to the center
        X_dev[torch.argmin(torch.sum((X_dev - 0.5)**2, axis=1)),:] = 0.5

        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            fit_gpytorch_mll(mll)

            if racqf == 'qREI':
                seed = int(rng.integers(low=0, high=2**16, dtype=np.int64))
                racq_function = qRegionalExpectedImprovement(
                    X_dev=X_dev,
                    model=model,
                    best_f=train_y.max(),
                    sampler=SobolQMCNormalSampler(sample_shape=torch.Size([256]), seed=seed),
                    length = length_init,
                    bounds = bounds,
                )
            elif racqf == 'REI':
                racq_function = LogRegionalExpectedImprovement(
                    X_dev=X_dev,
                    model=model,
                    best_f=train_y.max(),
                    length = length_init,
                    bounds = bounds,
                )
            elif racqf == 'EI':
                racq_function = LogExpectedImprovement(model, train_y.max())
            elif racqf == 'qEI':
                seed = int(rng.integers(low=0, high=2**16, dtype=np.int64))
                racq_function = qLogExpectedImprovement(model, train_y.max(), sampler=SobolQMCNormalSampler(sample_shape=torch.Size([256]), seed=seed))
            elif racqf == 'RUCB':
                racq_function = RegionalUpperConfidenceBound(
                    model=model,
                    beta=0.1,
                    X_dev=X_dev,
                    length = length_init,
                    bounds = bounds,
                )
            elif racqf == 'UCB':
                racq_function = UpperConfidenceBound(model, beta=0.1)

            candidates, _ = optimize_acqf(
                            acq_function=racq_function,
                            bounds=bounds,
                            q=q_batch,
                            num_restarts=10,
                            raw_samples=512,  # used for intialization heuristic
                            options={"batch_limit": 5, "maxiter": 200},
                            sequential=True,
                            )
            # observe new values
            X_center = candidates.detach()

        if n_init > 1:
            seed = int(rng.integers(low=0, high=2**16, dtype=np.int64))
            sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
            X_min = (X_center - 0.5*length_init).clamp_min(bounds[0])
            X_max = (X_center + 0.5*length_init).clamp_max(bounds[1])
            X_turbo = torch.empty([0, dim]).to(dtype=self.dtype, device=self.device)
            for i in range(q_batch):
                X_init = sobol.draw(n_init-1).to(dtype=self.dtype, device=self.device)
                X_turbo = torch.cat((X_turbo, X_center[i:i+1], X_init*(X_max[i:i+1] - X_min[i:i+1]) + X_min[i:i+1]), dim=0)
        else:
            X_turbo = X_center.clone()

        y_turbo = torch.tensor(
            [eval_fn(x) for x in X_turbo], dtype=self.dtype, device=self.device
        ).unsqueeze(-1)

        return (X_turbo, y_turbo)


    def select_sample(self, X, y, n_gp_max=2000):
        # Subsampling for GP: Normalize the objective function and design variables, calculate the distance on a log scale based on regret from the best value in the objective direction, and select using MaxMinGreedy.
        if X.shape[0] > n_gp_max:
            # consider distance in regret dimention 
            y_max = y.max()
            y_min = y.min()
            regret = (y_max - y)/(y_max - y_min)
            r_min2 = regret[regret>0].min()
            regret = (torch.log(regret + r_min2) - torch.log(r_min2))/(torch.log(1 + r_min2) - torch.log(r_min2))
            XY = torch.hstack([X, regret])
            # greedy maxmin selection
            mask = torch.full([XY.shape[0]], False)
            mask[regret[:,0].argmin()] = True
            while mask.sum() < n_gp_max:
                i_maxmin = torch.cdist(XY, XY[mask], p=2).min(axis=1)[0].argmax()
                mask[i_maxmin] = True
        else:
            mask = torch.full([X.shape[0]], True)
        return mask


    def turbo1(
        self, 
        dim: int,
        fun: Callable[[torch.Tensor], float],
        max_iter: int,
        batch_size: int = 10,
        n_init: int = 10,
        n_init_region: int = 0,
        n_gp_max: int = 2000,
        max_cholesky_size: int = 2000,
        seed: Union[int, None] = None,
        noise_constraint: Tuple[float, float] = (1e-4, 1e0),
        acqf: str = 'TS',
        racqf = 'qREI',
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        def eval_objective(x):
            """This is a helper function we use to unnormalize and evalaute a point"""
            return fun(unnormalize(x, fun.bounds))

        NUM_RESTARTS = 10 if not SMOKE_TEST else 2
        RAW_SAMPLES = 512 if not SMOKE_TEST else 4
        N_CANDIDATES = min(5000, max(2000, 200 * dim)) if not SMOKE_TEST else 4

        assert racqf in ('REI', 'qREI', 'EI', 'qEI', 'RUCB', 'UCB') or racqf is None, "racqf must be 'REI', 'qREI', 'EI', 'qEI', 'RUCB', 'UCB', or None"

        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        normal_bounds = torch.stack([torch.zeros(dim), torch.ones(dim)]).to(device=self.device, dtype=self.dtype)

        if os.path.exists(self.path_file):
            # restart
            df = pd.read_csv(self.path_file)
            columns = df.columns
            times = df['time'].values
            itr = df['TR'].astype(int).values[-1]
            X_hist = torch.tensor(df.loc[:,df.columns.str.contains('x')].values).to(dtype=self.dtype, device=self.device)
            y_hist = -1*torch.tensor(df.loc[:,df.columns.str.contains('f')].values.reshape(-1,1)).to(dtype=self.dtype, device=self.device)
            tr_hist = torch.tensor(df.loc[:,df.columns.str.contains('TR')].astype(int).values)
            X_turbo = X_hist[tr_hist[:,0]==itr,:]
            y_turbo = y_hist[tr_hist[:,0]==itr,:]
            state = TurboState(dim, batch_size=batch_size)
            if (racqf is not None) and n_init_region>0 and itr==0:
                n = n_init + n_init_region
            else:
                n = n_init
            for i in range((y_turbo.shape[0] - n)//batch_size):
                state = self.update_state(state=state, Y_next=y_turbo[n+i*batch_size:n+(i+1)*batch_size,:])
        else:
            # Initial sample
            t0 = time.perf_counter()
            if (racqf is not None) and n_init_region>0:
                itr = 0
                state = TurboState(dim, batch_size=batch_size)
                X_init, y_init = self.evaluate_on_init_points(eval_objective, dim, n_init_region, rng)
                tr_init = torch.full([n_init_region, 1], itr, dtype=torch.int)
                
                # q Regional Expected Improvement
                X_turbo, y_turbo = self.sampling_from_global_model(X_init, y_init, normal_bounds, eval_objective, dim, n_init, rng, max_cholesky_size, state.length, q_batch=1, racqf=racqf)

                # Mix initial and selected samples
                X_turbo = torch.vstack([X_init.detach(), X_turbo.detach()])
                y_turbo = torch.vstack([y_init.detach(), y_turbo.detach()])
                X_hist = X_turbo.detach()
                y_hist = y_turbo.detach()
                tr_hist = torch.vstack([tr_init, torch.full([n_init, 1], itr, dtype=torch.int)])
            else:
                itr = 0
                state = TurboState(dim, batch_size=batch_size)
                X_turbo, y_turbo = self.evaluate_on_init_points(eval_objective, dim, n_init, rng)
                X_hist = X_turbo.detach()
                y_hist = y_turbo.detach()
                tr_hist = torch.full([n_init, 1], itr, dtype=torch.int)

            t1 = time.perf_counter()
            times = np.full(tr_hist.shape[0], (t1-t0)/tr_hist.shape[0])
            columns = np.hstack(['x'+np.arange(1,X_hist.shape[1]+1).astype(str).astype(object), 'f'+np.arange(1,y_hist.shape[1]+1).astype(str).astype(object), 'TR', 'time'])
            df = pd.DataFrame(np.hstack([X_hist.cpu().numpy(), -y_hist.cpu().numpy(), tr_hist.cpu().numpy(), times.reshape(-1,1)]), columns=columns)
            df.to_csv(self.path_file, index=False)

        if verbose:
            print(
                f"{len(X_hist)}) Objective: {y_hist[tr_hist[:,0]==itr].max():.2e}, TR: {itr}, TR length: {state.length:.2e}"
            )
        
        counter = 0
        while X_hist.shape[0] < max_iter:
            t0 = time.perf_counter()
            if not state.restart_triggered or len(X_hist) + n_init > max_iter:
                counter += batch_size
                X_next, _ = self.x_next_and_acqui_vals_on_tr(
                    dim,
                    X_turbo,
                    y_turbo,
                    state,
                    batch_size,
                    N_CANDIDATES,
                    NUM_RESTARTS,
                    RAW_SAMPLES,
                    max_cholesky_size,
                    noise_constraint,
                    n_gp_max,
                    acqf,
                )

                Y_next = torch.tensor(
                    [eval_objective(x) for x in X_next], dtype=self.dtype, device=self.device
                ).unsqueeze(-1)

                # Update state
                state = self.update_state(state=state, Y_next=Y_next)

                # Append data
                X_turbo = torch.cat((X_turbo, X_next), dim=0)
                y_turbo = torch.cat((y_turbo, Y_next), dim=0)
                X_hist = torch.cat((X_hist, X_next), dim=0)
                y_hist = torch.cat((y_hist, Y_next), dim=0)
                tr_hist = torch.cat((tr_hist, torch.full([batch_size, 1], itr, dtype=torch.int)), dim=0)

                t1 = time.perf_counter()
                times = np.hstack([times, np.full(X_next.shape[0]-1, np.nan), t1-t0])

            if state.restart_triggered and len(X_hist) + n_init <= max_iter:
                # restart TR
                counter += n_init
                if verbose:
                    print("Restarting TR")

                state = TurboState(dim, batch_size=batch_size)
                if racqf is not None:
                    # Subsampling for GP
                    mask = self.select_sample(X_hist, y_hist, n_gp_max)
                    # Region-Averaged Acquisition Function
                    X_turbo, y_turbo = self.sampling_from_global_model(
                        X_hist[mask], y_hist[mask], normal_bounds, eval_objective, dim, n_init, rng, max_cholesky_size, state.length, q_batch=1, racqf=racqf
                    )
                else:
                    # random restart
                    X_turbo, y_turbo = self.evaluate_on_init_points(eval_objective, dim, n_init, rng)
                X_hist = torch.cat((X_hist, X_turbo), dim=0)
                y_hist = torch.cat((y_hist, y_turbo), dim=0)
                itr += 1
                tr_hist = torch.cat((tr_hist, torch.full([n_init, 1], itr, dtype=torch.int)), dim=0)

                t1 = time.perf_counter()
                times = np.hstack([times, np.full(X_turbo.shape[0]-1, np.nan), t1-t0])

            if counter>=100:
                counter = 0
                df = pd.DataFrame(np.hstack([X_hist.cpu().numpy(), -y_hist.cpu().numpy(), tr_hist.cpu().numpy(), times.reshape(-1,1)]), columns=columns)
                df.to_csv(self.path_file, index=False)

            # Print current status
            if verbose:
                print(
                    f"{len(X_hist)}) Objective: {y_hist[tr_hist[:,0]==itr].max():.2e}, TR: {itr}, TR length: {state.length:.2e}"
                )
                
        return (
            torch.tensor(
                np.array([self.from_hyper_cube(x, fun.bounds[0], fun.bounds[1]).cpu().numpy() for x in X_hist]),
                dtype=self.dtype,
                device=self.device,
            ),
            y_hist,
            tr_hist,
            times.reshape(-1,1),
        )


    def turbom(
        self, 
        dim: int,
        fun: Callable[[torch.Tensor], float],
        max_iter: int,
        batch_size: int = 10,
        n_trust_regions: int = 5,
        n_init: int = 10,
        n_init_region: int = 0,
        n_gp_max: int = 2000,
        max_cholesky_size: int = 2000,
        seed: Union[int, None] = None,
        noise_constraint: Tuple[float, float] = (1e-4, 1e0),
        acqf: str = 'TS',
        racqf = 'qREI',
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        def eval_objective(x):
            """This is a helper function we use to unnormalize and evalaute a point"""
            return fun(unnormalize(x, fun.bounds))
        
        if n_init_region>0 and n_trust_regions>1:
            assert racqf in ('qREI', 'qEI') or racqf is None, "racqf must be 'qREI', 'qEI', or None for TuRBO-m"
        else:
            assert racqf in ('REI', 'qREI', 'EI', 'qEI', 'RUCB', 'UCB') or racqf is None, "racqf must be 'REI', 'qREI', 'EI', 'qEI', 'RUCB', 'UCB', or None"

        t0 = time.perf_counter()

        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        normal_bounds = torch.stack([torch.zeros(dim), torch.ones(dim)]).to(device=self.device, dtype=self.dtype)

        X_hist = torch.empty([0, dim]).to(dtype=self.dtype, device=self.device)
        y_hist = torch.empty([0, 1]).to(dtype=self.dtype, device=self.device)
        tr_hist = torch.empty([0, 1], dtype=torch.int)
        tr_active = np.arange(n_trust_regions)
        states = []

        # Initial sample
        if (racqf is not None) and n_init_region>0:
            for i in tr_active:
                states.append(TurboState(dim, batch_size=batch_size))
            X_init, y_init = self.evaluate_on_init_points(eval_objective, dim, n_init_region, rng)
            # q Regional Expected Improvement
            X_turbo, y_turbo = self.sampling_from_global_model(
                X_init, y_init, normal_bounds, eval_objective, dim, n_init, rng, max_cholesky_size, states[0].length, q_batch=n_trust_regions, racqf=racqf
            )
            tr_init = torch.full([n_init_region, 1], -1, dtype=torch.int)
            X_hist = torch.vstack([X_init, X_turbo.detach()])
            y_hist = torch.vstack([y_init, y_turbo.detach()])
            tr_hist = torch.vstack([tr_init, torch.tensor(np.tile(np.arange(n_trust_regions, dtype=int), [n_init, 1]).T.reshape(-1,1))])
        else:
            for i in tr_active:
                X_turbo, y_turbo = self.evaluate_on_init_points(eval_objective, dim, n_init, rng)
                X_hist = torch.vstack([X_hist, X_turbo.detach()])
                y_hist = torch.vstack([y_hist, y_turbo.detach()])
                tr_hist = torch.vstack([tr_hist, torch.full([n_init, 1], i, dtype=torch.int)])
                states.append(TurboState(dim, batch_size=batch_size))
        # Print current status
        if verbose:
            print(
                f"{len(X_hist)}) Best value: {y_hist.max():.2e}, TR: {tr_hist[y_hist.argmax(),0]}, TR length: {states[tr_hist[y_hist.argmax(),0]].length:.2e}"
            )

        NUM_RESTARTS = 10 if not SMOKE_TEST else 2
        RAW_SAMPLES = 512 if not SMOKE_TEST else 4
        N_CANDIDATES = min(5000, max(2000, 200 * dim)) if not SMOKE_TEST else 4

        t1 = time.perf_counter()
        if (racqf is not None) and n_init_region>0:
            times = np.full(n_init*(n_trust_regions+1), (t1-t0)/n_init)
        else:
            times = np.full(n_init*n_trust_regions, (t1-t0)/n_init)
        t0 = time.perf_counter()

        columns = np.hstack(['x'+np.arange(1,X_hist.shape[1]+1).astype(str).astype(object), 'f'+np.arange(1,y_hist.shape[1]+1).astype(str).astype(object), 'TR', 'time'])

        counter = 0
        while X_hist.shape[0] < max_iter:
            counter += 1
            X_cand = torch.empty([0, dim]).to(dtype=self.dtype, device=self.device)
            y_cand = torch.empty([0, 1]).to(dtype=self.dtype, device=self.device)
            tr_cand = torch.empty([0, 1], dtype=torch.int)

            # Generate candidates from each TR
            for i in tr_active:
                mask = tr_hist[:,0]==i
                X_turbo = X_hist[mask]
                y_turbo = y_hist[mask]

                _X_cand, _y_cand = self.x_next_and_acqui_vals_on_tr(
                    dim,
                    X_turbo,
                    y_turbo,
                    states[i],
                    batch_size,
                    N_CANDIDATES,
                    NUM_RESTARTS,
                    RAW_SAMPLES,
                    max_cholesky_size,
                    noise_constraint,
                    n_gp_max,
                    acqf,
                )

                X_cand = torch.vstack([X_cand, _X_cand])
                y_cand = torch.vstack([y_cand, _y_cand])
                tr_cand = torch.vstack([tr_cand, torch.full([batch_size, 1], i, dtype=torch.int)])
            
            # Select next candidates
            idx = torch.argsort(y_cand[:,0], descending=True)
            X_next = X_cand[idx,:][:batch_size,:]
            y_next = torch.tensor([eval_objective(x) for x in X_next], dtype=self.dtype, device=self.device).unsqueeze(-1)
            tr_next = tr_cand[idx.cpu(),:][:batch_size,:]

            # Update state
            for i in tr_next.unique().cpu().numpy():
                states[i] = self.update_state(state=states[i], Y_next=y_next[tr_next[:,0]==i])

            # Append data
            X_hist = torch.vstack([X_hist, X_next])
            y_hist = torch.vstack([y_hist, y_next])
            tr_hist = torch.vstack([tr_hist, tr_next])

            t1 = time.perf_counter()
            times = np.hstack([times, np.full(X_next.shape[0]-1, np.nan), t1-t0])
            t0 = time.perf_counter()

            if counter%100==0:
                df = pd.DataFrame(np.hstack([X_hist.cpu().numpy(), -y_hist.cpu().numpy(), tr_hist.cpu().numpy(), times.reshape(-1,1)]), columns=columns)
                df.to_csv(self.path_file, index=False)

            # Print current status
            if verbose:
                print(
                    f"{len(X_hist)}) Best value: {y_hist.max():.2e}, TR: {tr_next[y_next.argmax(),0]}, TR length: {states[tr_next[y_next.argmax(),0]].length:.2e}"
                )
            
            # Restart
            for i in tr_next.unique().cpu().numpy():
                if states[i].restart_triggered and len(X_hist) + n_init <= max_iter:
                    if verbose:
                        print(f"Restarting TR: {i}")

                    i_tr = len(states)
                    states.append(TurboState(dim, batch_size=batch_size))
                    if racqf is not None:
                        # Subsampling for GP
                        mask = self.select_sample(X_hist, y_hist, n_gp_max)
                        # q Regional Expected Improvement
                        X_turbo, y_turbo = self.sampling_from_global_model(
                            X_hist[mask], y_hist[mask], normal_bounds, eval_objective, dim, n_init, rng, max_cholesky_size, states[i_tr].length, q_batch=1, racqf=racqf
                        )
                    else:
                        X_turbo, y_turbo = self.evaluate_on_init_points(eval_objective, dim, n_init, rng)
                    X_hist = torch.vstack([X_hist, X_turbo.detach()])
                    y_hist = torch.vstack([y_hist, y_turbo.detach()])
                    tr_hist = torch.vstack([tr_hist, torch.full([n_init, 1], i_tr)])
                    tr_active = np.append(tr_active[tr_active!=i], i_tr)

                    t1 = time.perf_counter()
                    times = np.hstack([times, np.full(X_turbo.shape[0]-1, np.nan), t1-t0])
                    t0 = time.perf_counter()

        return (
            torch.tensor(
                np.array([self.from_hyper_cube(x, fun.bounds[0], fun.bounds[1]).cpu().numpy() for x in X_hist]),
                dtype=self.dtype,
                device=self.device,
            ),
            y_hist,
            tr_hist,
            times.reshape(-1,1),
        )


    def optimize(self, dim, fun, n_max, batch_size=1, n_trust_regions=1, n_init=2, n_init_region=0, n_gp_max=2000, seed=0, noise_constraint=(1e-4,1e0), acqf='EI', racqf='qREI', verbose=False):
        if n_trust_regions<=1:
            return self.turbo1(dim, fun, n_max, batch_size, n_init, n_init_region, n_gp_max, seed=seed, noise_constraint=noise_constraint, acqf=acqf, racqf=racqf, verbose=verbose)
        else:
            return self.turbom(dim, fun, n_max, batch_size, n_trust_regions, n_init, n_init_region, n_gp_max, seed=seed, noise_constraint=noise_constraint, acqf=acqf, racqf=racqf, verbose=verbose)


def turbo(path_dir, problem, ns_init, ns_max, dim=0, dim_emb=0, itrial=1, verbose=True, n_init_region=0, n_trust_regions=1, batch_size=5, acqf='EI', racqf='qREI', device='cpu'):
    n_gp_max = 2000
    seed = itrial
    noise_constraint = (1e-4, 1e0)

    solver = 'TuRBO-'+str(n_trust_regions) + '-' +acqf
    if racqf is not None:
        solver = solver + '-' + racqf
        if n_init_region==0:
            solver = solver + '(restart)'
    solver = solver + '-' + str(batch_size)
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
    path = os.path.join(path_dir, problem+'_'+ str(nx) +'D', solver)
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            pass

    path_time = os.path.join(path, 'times'+str(itrial)+'.csv')
    path_file = os.path.join(path, 'solutions'+str(itrial)+'.csv')
    t0 = time.perf_counter()
    turbo = TuRBO(device, dtype, path_file)
    xs, ys, tr, times = turbo.optimize(nx, fun, ns_max, batch_size, n_trust_regions, ns_init, n_init_region, n_gp_max, seed, noise_constraint, acqf, racqf, verbose)
    t1 = time.perf_counter()
    xs = xs.cpu().numpy()
    ys = -ys.cpu().numpy()
    tr = tr.cpu().numpy()

    columns = np.hstack(['x'+np.arange(1,xs.shape[1]+1).astype(str).astype(object), 'f'+np.arange(1,ys.shape[1]+1).astype(str).astype(object), 'TR', 'time'])
    df = pd.DataFrame(np.hstack([xs, ys, tr, times]), columns=columns)
    df.to_csv(path_file, index=False)
    with open(path_time, 'a') as file:
        np.savetxt(file, np.array([[itrial, t1-t0]]), delimiter=',')
    del turbo, xs, ys, tr, times, df


if __name__ == "__main__": 
    (path_dir, problem, ns_init, ns_max, dim, dim_emb, itrial, verbose, n_init_region, n_trust_regions, batch_size, acqf, racqf, device) = sys.argv[1:]
    if racqf in ('None', 'none'):
        racqf = None
    turbo(path_dir, problem, int(ns_init), int(ns_max), int(dim), int(dim_emb), int(itrial), eval(verbose), int(n_init_region), int(n_trust_regions), int(batch_size), acqf, racqf, device)
