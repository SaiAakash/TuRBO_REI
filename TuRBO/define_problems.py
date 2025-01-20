import numpy as np
import torch
from torch import Tensor
from botorch.test_functions.synthetic import *
from botorch.test_functions.base import BaseTestProblem

from benchmark.mopta08.mopta08 import Mopta08
from benchmark.ebo.rover_function import RoverTrajectory
from benchmark.rom import SiROM


class BotorchHPA(BaseTestProblem):
    def __init__(
        self,
        problem_name,
        n_div=4,
        level=1,
        NORMALIZED=True,
        noise_std=None,
        negate=True,
    ):
        self.hpa = eval(
            problem_name
            + "(n_div="
            + str(n_div)
            + ", level="
            + str(level)
            + ", NORMALIZED="
            + str(NORMALIZED)
            + ")"
        )
        self.nx = self.hpa.nx
        self.nf = self.hpa.nf
        self.ng = self.hpa.ng
        if NORMALIZED:
            self.lb = np.zeros(self.nx)
            self.ub = np.ones(self.nx)
        else:
            self.lb = self.hpa.lbound
            self.ub = self.hpa.ubound
        self.dim = self.nx
        self._bounds = [(l, u) for (l, u) in zip(self.lb, self.ub)]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        if self.ng > 0:
            f, g = self.hpa(X.cpu().numpy().reshape(-1))
        else:
            f = self.hpa(X.cpu().numpy().reshape(-1))
        return torch.tensor(f)


def DefineProblems(problem, dim=0, dim_emb=0, noise_std=None, negate=True, n_div=4):
    if "HPA" in problem:
        name = problem[:6]
        level = int(problem[-1])
        return BotorchHPA(name, n_div, level, noise_std=noise_std, negate=True)
    elif "MOPTA08" in problem:
        return Mopta08(noise_std=noise_std, negate=True)
    elif "RoverTrajectory" in problem:
        if len(problem) > 15:
            n_points = int(problem[15:])
        else:
            n_points = 30
        return RoverTrajectory(n=n_points, noise_std=noise_std, negate=False)
    elif "SiROM" in problem:
        problem_name = problem.split("-")[1]
        return SiROM(problem_name, dim=6, noise_std=noise_std, negate=False)
    else:
        # Botorch functions
        if dim > 0:
            try:
                fun = eval(
                    problem
                    + "(dim="
                    + str(dim)
                    + ", noise_std="
                    + str(noise_std)
                    + ", negate="
                    + str(negate)
                    + ")"
                )
            except:
                fun = eval(
                    problem
                    + "(noise_std="
                    + str(noise_std)
                    + ", negate="
                    + str(negate)
                    + ")"
                )
            return fun
        else:
            return eval(
                problem
                + "(noise_std="
                + str(noise_std)
                + ", negate="
                + str(negate)
                + ")"
            )
