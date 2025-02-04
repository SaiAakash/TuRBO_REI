from abc import ABC
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize as opMin
from scipy.integrate import solve_ivp
from scipy.stats.qmc import LatinHypercube as lhc
from scipy.stats import qmc

import torch
from TuRBO import turbo
from botorch.test_functions.base import BaseTestProblem


class ROM(ABC):
    def __init__(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        output_column: str,
        initial_conditions: np.ndarray,
        params_range: dict,
        pso_kwargs: dict,
    ):
        self.training_data = training_data
        self.validation_data = validation_data
        self.output_column = output_column
        self.target_data = (
            self.training_data[["Time", self.output_column]].to_numpy().transpose()
        )
        self.training_bc = (
            training_data[["Time", "Tfluid1", "Tfluid2"]].to_numpy().transpose()
        )
        self.validation_bc = (
            validation_data[["Time", "Tfluid1", "Tfluid2"]].to_numpy().transpose()
        )
        self.initial_conditions = initial_conditions
        self.mrange = params_range["m"]
        self.krange = params_range["k"]
        self.hrange1 = params_range["h1"]
        self.hrange2 = params_range["h2"]
        self.expanded_params_range = {
            "m1": self.mrange,
            "m3": self.mrange,
            "k1": self.krange,
            "k2": self.krange,
            "h1": self.hrange1,
            "h2": self.hrange2,
        }
        # if output_column in ["S11", "S12", "S22", "S33"]:
        #     self.expanded_params_range["c1"] = [-1.0, 1.0]
        #     self.expanded_params_range["c2"] = [-1.0, 1.0]
        #     self.expanded_params_range["c3"] = [-1.0, 1.0]
        self.pso_kwargs = pso_kwargs

    def dTdtLinear(self, t, y, mArray, kArray, hArray, BCArray):
        """
        Transient response, linear n mass system.

        y is current temperature (nx1)
        t is itme
        mArray = mass array, nx1
        kArray = (n-1) x 1
        hArray = 2x1

        BCArray = Array of time vs fluid temperature (external boundary conditions)
            BCArray[0] = array of times (i x 1)
            BCArray[1] = array of first temperature variation (i x 1)
            BCArray[2] = array of second temperature variation(i x 1)

        """

        # catch on non-physical values (negatives)
        mArray[mArray < 1e-9] = 1e-9
        kArray[kArray < 0] = 0
        hArray[hArray < 0] = 0

        nTF = BCArray.shape[0] - 1  # number of temperature entries
        n = mArray.size  # number of masses

        TF = np.array(
            [np.interp(t, BCArray[0], BCArray[1]), np.interp(t, BCArray[0], BCArray[2])]
        )

        TF = TF.reshape(nTF, 1)
        T = y.reshape(n, 1)

        # Initialise arrays
        mMat = np.zeros([n, n])
        kMat = np.zeros([n, n])
        hMatExt = np.zeros([n, nTF])
        hMatInt = np.zeros([n, n])

        for i in range(0, n):
            mMat[i, i] = mArray[i]

        # loop over k values
        for j in range(0, n - 1):
            kMat[j, j] = kMat[j, j] - kArray[j]
            kMat[j + 1, j + 1] = kMat[j + 1, j + 1] - kArray[j]
            kMat[j, j + 1] = kMat[j, j + 1] + kArray[j]
            kMat[j + 1, j] = kMat[j + 1, j] + kArray[j]

        # h arrays
        hMatExt[0, 0] = hArray[0]
        hMatExt[n - 1, nTF - 1] = hArray[1]

        hMatInt[0, 0] = -hArray[0]
        hMatInt[n - 1, n - 1] = -hArray[1]

        # Rate of change
        dTdt = np.matmul(
            np.linalg.inv(mMat),
            (np.matmul((kMat + hMatInt), T) + np.matmul(hMatExt, TF)),
        )

        return dTdt.flatten()

    def WeightedErrorFunction(self, C, testMassTemps, target):
        """

        Parameters
        ----------
        Function to return error

        testMassTemps : data to calcualte error for. Values against time [mxn array]
            test[0] = times
            test[1 to m-1] = temperatures at each of the m masses
        target : data to be reproduced. Values against time [2xk array]
            target[0] = times
            target[1] = temperatures

        C : weighting array for test temperatures, m-1 x 1 (1 weight for each mass)

        Returns: Scalar error value
        -------
        """

        # interpolate test value to target times, ensure points over dwells where solver may have only a few points

        # weighted sum of temps
        tempTestWeighted = np.matmul(C, testMassTemps[1:, :])

        # add target points every 1s
        interpTimes = np.linspace(target[0][0], target[0][-1], np.int32(target[0][-1]))
        interpTimes = np.append(interpTimes, target[0])  # add all existing points.
        interpTimes = np.unique(interpTimes)  # unique, sorted entries
        # interpolate test and target data to the new time points
        targetValsInterp = np.interp(interpTimes, target[0], target[1])
        testValsInterp = np.interp(interpTimes, testMassTemps[0], tempTestWeighted)
        # calculate errors
        errorVals = np.abs((targetValsInterp - testValsInterp))

        return errorVals.mean()

    def TargetFunction(
        self,
        modelParameters,
        boundaryConditionArray,
        initialConditions,
        targetData,
        weightOpt=False,
    ):
        """
        Parameters
        ----------
        Function to calculate transient response for a set of model parameters and boundary conditions
        Return error against an input set of 'target' response data

        modelParameters: 1D array with model parameters for a n mass linear network:
            length 2n+1
            modelParameters[0:n] = masses (n x 1)
            modelParameters[n:2n-1] = k values (n-1 x 1)
            modelParameters[2n-1:] = h values (2x1)

        boundaryConditionArray: Array of time vs fluid temperature (external boundary conditions)
            boundaryConditionArray[0] = array of times (i x 1)
            boundaryConditionArray[1] = array of first temperature variation (i x 1)
            boundaryConditionArray[2] = array of second temperature variation(i x 1)

        initialConditions: intital mass temperatures (at t=0) (nx1)

        targetData: data to calcualte error for. Values against time [2xj array]
            targetData[0] = times
            targetData[1] = temperatures

        Returns: Scalar error value
        -------
        """

        # separate model parameters
        n = int((modelParameters.size - 1) / 2)
        mArray = modelParameters[0:n]
        kArray = modelParameters[n : 2 * n - 1]
        hArray = modelParameters[2 * n - 1 :]

        # start and end time
        tSpan = [boundaryConditionArray[0][0], boundaryConditionArray[0][-1]]

        # solve differential eqn.
        sol = solve_ivp(
            lambda t, y: self.dTdtLinear(
                t, y, mArray, kArray, hArray, boundaryConditionArray
            ),
            tSpan,
            initialConditions,
            method="Radau",
        )
        # atol=1e-6,
        # rtol=1e-3)
        # setup output
        C = np.array([0, 1, 0])
        testMassTemps = np.vstack([sol.t, sol.y])

        # Optimise minimise for weighting of mass temperatures
        if weightOpt == True:
            CGuess = np.ones(n)
            optArgs = (testMassTemps, targetData)
            # optimise for C array
            optSoln = opMin(self.WeightedErrorFunction, CGuess, args=optArgs)
            # output
            C = optSoln.x

        # tempTestWeighted = np.matmul(C,testMassTemps[1:,:])
        # testData = np.array([sol.t,tempTestWeighted])

        err = self.WeightedErrorFunction(C, testMassTemps, targetData)

        return err, C

    def ParticleSwarmOptimise(self):
        """
        Particle swarm optimiser for 3 mass system

        trainingBoundayConditions:
            Array of time vs fluid temperature (external boundary conditions)
            boundaryConditionArray[0] = array of times (i x 1)
            boundaryConditionArray[1] = array of first temperature variation (i x 1)
            boundaryConditionArray[2] = array of second temperature variation(i x 1)

        initialConditions: intital mass temperatures (at t=0) (3x1)

        trainingData:
            data to calcualte error for. Values against time [2xj array]
            targetData[0] = times
            targetData[1] = temperatures

        inputsPSO:
            Dictionary containing data to define particle swarm:
            inputsPSO={}
            inputsPSO["nSwarm"] = 20 # number of particles in the swarm
            inputsPSO["npars"] = 7 # number of parameters
            inputsPSO["nIter"] = 3 # number of iterations
            inputsPSO["weightOpt"] =False# Optimise C factors or not?

            # range of inputs to sample
            inputsPSO["mrange"] = [0.1,10]  - mass parameters
            inputsPSO["krange"] = [0.0001,0.1] - parameters
            inputsPSO["hrange1"] = [1e-3, 10] - first h value, with 1st BC
            inputsPSO["hrange2"] = [1e-9, 1e-8] - second h value, with 2nd BC

        """

        # Unpack inputs parameters
        nSwarm = self.pso_kwargs["nSwarm"]
        nIter = self.pso_kwargs["nIter"]
        weightOpt = self.pso_kwargs["weightOpt"]
        mrange = self.mrange
        krange = self.krange
        hrange1 = self.hrange1
        hrange2 = self.hrange2

        # number of optimisation parameters
        npars = 7

        # Optimiser parameters
        vStdDev = 0.1
        # velocity update factors
        wBestAll = 0.5  # pull towards global best
        wBestPoints = 0.01  # pull towards point best
        w = 0.5  # inertia

        rangeArr = np.array(
            [mrange, [1, 1], mrange, krange, krange, hrange1, hrange2]
        ).transpose()

        # Initialise arrays
        # initial positions, current and best for each point, distribute in log space

        # latin hypercube sample:
        sampleFunc = lhc(d=npars)
        samplePts = sampleFunc.random(n=nSwarm)

        # particle, parameter, iteration
        particlePaths = np.zeros([nSwarm, npars + 1, nIter])

        xPointsCurrent = rangeArr[0] * np.exp(
            samplePts * np.log(rangeArr[1] / rangeArr[0])
        )
        xPointsBest = xPointsCurrent

        # intital velocity -log
        vPoints = vStdDev * np.multiply(
            np.random.randn(nSwarm, npars), np.log(xPointsCurrent)
        )

        # errors, current and best for each point
        errPointsCurrent = np.ones(nSwarm) * 1e6
        errPointsBest = np.ones(nSwarm) * 1e6

        minErrArr = np.zeros(nIter)
        minErrAll = 1e6

        xBestAll = []

        print("-------------- Start PSO --------------")

        for i in range(0, nIter):

            # calculate error at each point
            for n in range(0, nSwarm):
                errPointsCurrent[n], C = self.TargetFunction(
                    xPointsCurrent[n],
                    self.training_bc,
                    self.initial_conditions,
                    self.target_data,
                    weightOpt,
                )
                # print(errPointsCurrent[n])

                # update best point
                if errPointsCurrent[n] < errPointsBest[n]:
                    errPointsBest[n] = errPointsCurrent[n]
                    xPointsBest[n] = xPointsCurrent[n]

                # update best global if needed
                if errPointsCurrent[n] < minErrAll:
                    minErrAll = errPointsCurrent[n]
                    xBestAll = xPointsCurrent[n]
                    cBestAll = C

            # Vector towards global best and point best - log space
            vBestAll = np.log(xBestAll) - np.log(xPointsCurrent)
            vBestPoints = np.log(xPointsBest) - np.log(xPointsCurrent)

            # update velocity
            r = np.random.rand(2)
            vPoints = (
                w * vPoints
                + r[0] * wBestAll * vBestAll
                + r[1] * wBestPoints * vBestPoints
            )

            # store stuff
            particlePaths[:, :, i] = np.hstack(
                [xPointsCurrent, errPointsCurrent.reshape(nSwarm, 1)]
            )

            # update locations, add +/- 10% scatter
            xPointsCurrent = np.exp(np.log(xPointsCurrent) + vPoints) * (
                0.9 + np.random.rand(nSwarm, npars) * 0.2
            )

            # catch any updates outside ranges
            for k in range(0, npars):
                xPointsCurrent[:, k][xPointsCurrent[:, k] < rangeArr[0][k]] = rangeArr[
                    0
                ][k]
                xPointsCurrent[:, k][xPointsCurrent[:, k] > rangeArr[1][k]] = rangeArr[
                    1
                ][k]

            minErrArr[i] = minErrAll
            print("--------   iter", i + 1, minErrAll)

            outputsPSO = {}
            outputsPSO["cBest"] = cBestAll
            outputsPSO["mkhBest"] = xBestAll
            outputsPSO["minError"] = minErrAll
            outputsPSO["errorArray"] = minErrArr

        return outputsPSO

    def ErrorCalculation(
        self,
        model_parameters,
        t_span_factor=1.0,
        validation_mode=False,
        optimize_c=False,
    ):
        if validation_mode:
            data = self.validation_bc
            target_data = (
                self.validation_data[["Time", self.output_column]]
                .to_numpy()
                .transpose()
            )
        else:
            data = self.training_bc
            target_data = self.target_data

        # To reduce the time span for the integrator
        tSpan = np.array([data[0][0], math.ceil(t_span_factor * data[0][-1])])
        new_target_data = target_data[
            :, np.where(target_data[0, :] <= tSpan[-1])
        ].squeeze(1)

        m = model_parameters["m"]
        k = model_parameters["k"]
        h = model_parameters["h"]
        c = model_parameters["c"]

        sol = solve_ivp(
            lambda t, y: self.dTdtLinear(t, y, m, k, h, data),
            tSpan,
            y0=self.initial_conditions,
            method="Radau",
        )

        testMassTemps = np.vstack([sol.t, sol.y])
        # weighted output
        Y = np.matmul(c, sol.y)

        if self.output_column == "Temp":
            # Calculate error
            error = self.WeightedErrorFunction(
                c,
                testMassTemps=testMassTemps,
                target=new_target_data,
            )

        elif self.output_column in ["S11", "S12", "S22", "S33"]:
            if optimize_c:
                CGuess = np.ones(c.shape[0])
                optArgs = (testMassTemps, self.target_data)
                # optimise for C array
                optSoln = opMin(self.WeightedErrorFunction, CGuess, args=optArgs)
                # output
                c = optSoln.x

                Y = np.matmul(c, sol.y)
            # Calculate the error
            error = self.WeightedErrorFunction(
                c, testMassTemps=testMassTemps, target=new_target_data
            )

        return error, sol, c, Y


class SiROM(BaseTestProblem):
    def __init__(self, problem, dim=6, noise_std=None, negate=True):
        self.problem = problem
        self.dim = dim
        # if problem == "Temp":
        #     self.dim = 6
        # elif problem in ["S11", "S12", "S22", "S33"]:
        #     self.dim = 9
        # else:
        #     raise ValueError("Invalid problem name")
        self._bounds = np.vstack((np.zeros(self.dim), np.ones(self.dim))).T
        super().__init__(noise_std=noise_std, negate=negate)

        # Read the CSVs
        self.training_data = pd.read_csv(
            "../data/Abaqus_n28_Interface_Mid_training.csv", skipinitialspace=True
        )
        self.validation_data = pd.read_csv(
            "../data/Abaqus_n28_Interface_Mid_validation.csv", skipinitialspace=True
        )

        # initial conditions
        nGuess = 3
        # if self.problem == "Temp":
        #     self.initialTempsGuess = np.ones(nGuess) * 20
        # elif self.problem in ["S11", "S12", "S22", "S33"]:
        #     self.initialTempsGuess = np.zeros(nGuess)
        self.initialTempsGuess = np.ones(nGuess) * 20

        # Set the parameter ranges
        self.params_range = {
            "m": [0.1, 10],
            "k": [0.00001, 0.1],
            "h1": [1e-3, 10],
            "h2": [1e-9, 1e-8],
        }

        # Define a ROM object
        self.rom = ROM(
            self.training_data,
            self.validation_data,
            output_column=self.problem,
            initial_conditions=self.initialTempsGuess,
            params_range=self.params_range,
            pso_kwargs={},
        )

    def scale_samples(self, samples, lower_bounds, upper_bounds):
        return qmc.scale(samples, lower_bounds, upper_bounds)

    def evaluate_true(
        self,
        X,
        kwargs={
            "scale_parameters": True,
            "return_candidates": False,
            "optimize_c": True,
        },
    ):
        candidate_dict = {}
        error_candidates = []

        # Retrieve parameter ranges
        lower_bounds = [
            self.rom.expanded_params_range[key][0]
            for key in self.rom.expanded_params_range.keys()
        ]
        upper_bounds = [
            self.rom.expanded_params_range[key][1]
            for key in self.rom.expanded_params_range.keys()
        ]

        # Expand dims when te data array is 1D
        if not isinstance(X, np.ndarray):
            X = X.numpy()
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if kwargs["scale_parameters"]:
            X = self.scale_samples(X, lower_bounds, upper_bounds)

        # Calculate the error for each candidate
        for j in range(X.shape[0]):
            candidate_dict = {}
            candidate_dict["m"] = np.array([X[j, 0], 1.0, X[j, 1]])
            candidate_dict["k"] = np.array([X[j, 2], X[j, 3]])
            candidate_dict["h"] = np.array([X[j, 4], X[j, 5]])
            # if self.problem in ["S11", "S12", "S22", "S33"]:
            #     candidate_dict["c"] = np.array([X[j, 6], X[j, 7], X[j, 8]])
            # else:
            #     candidate_dict["c"] = np.array([0.0, 1.0, 0.0])
            candidate_dict["c"] = np.array([0.0, 1.0, 0.0])
            (
                error,
                _,
                c,
                y,
            ) = self.rom.ErrorCalculation(
                candidate_dict, t_span_factor=1.0, optimize_c=kwargs["optimize_c"]
            )
            error_candidates.append(-1 * error)
            error_tensor = torch.tensor(error_candidates).unsqueeze(-1)

        if self.dim == 6:
            c = np.expand_dims(c, axis=0)
            X = np.hstack((X, c))

        if kwargs["return_candidates"]:
            return error_tensor, X
        else:
            return error_tensor
