import os
import subprocess
import sys
import tempfile
from pathlib import Path
from platform import machine

import numpy as np
import torch
from torch import Tensor
from botorch.test_functions.base import BaseTestProblem

# The original class is modified by Fujitsu Limited to fit BaseTestProblem in botorch
class Mopta08(BaseTestProblem):
    def __init__(self, noise_std=None, negate=True):
        self.dim = 124
        self._bounds = np.vstack((np.zeros(self.dim), np.ones(self.dim))).T
        super().__init__(noise_std=noise_std, negate=negate)

        self.sysarch = 64 if sys.maxsize > 2 ** 32 else 32
        self.machine = machine().lower()

        if self.machine == "armv7l":
            assert self.sysarch == 32, "Not supported"
            self._mopta_exectutable = "mopta08_armhf.bin"
        elif self.machine == "x86_64":
            assert self.sysarch == 64, "Not supported"
            self._mopta_exectutable = "mopta08_elf64.bin"
        elif self.machine == "i386":
            assert self.sysarch == 32, "Not supported"
            self._mopta_exectutable = "mopta08_elf32.bin"
        elif self.machine == "amd64":
            assert self.sysarch == 64, "Not supported"
            self._mopta_exectutable = "mopta08_amd64.exe"
        else:
            raise RuntimeError("Machine with this architecture is not supported")
        
        self._mopta_exectutable = os.path.join(
            Path(__file__).parent, self._mopta_exectutable
        )
        self.directory_file_descriptor = tempfile.TemporaryDirectory()
        self.directory_name = self.directory_file_descriptor.name

    def evaluate_true(self, X: Tensor) -> Tensor:
        x = X.squeeze()
        with open(os.path.join(self.directory_name, "input.txt"), "w+") as tmp_file:
            for _x in x:
                tmp_file.write(f"{_x.detach().cpu().numpy()}\n")
        popen = subprocess.Popen(
            self._mopta_exectutable,
            stdout=subprocess.PIPE,
            cwd=self.directory_name,
        )
        popen.wait()
        output = (
            open(os.path.join(self.directory_name, "output.txt"), "r")
            .read()
            .split("\n")
        )
        output = [x.strip() for x in output]
        output = torch.tensor([float(x) for x in output if len(x) > 0])
        value = output[0]
        constraints = output[1:]
        return (value + 10*torch.sum(torch.clip(constraints, 0))).unsqueeze(-1)
