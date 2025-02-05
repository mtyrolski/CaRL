import copy

from carl.environment.gym_rubik.rubik_solver.Move import Move

from .. import Solver
from . import (SecondLayerSolver, WhiteCrossSolver, WhiteFaceSolver, YellowCrossSolver, YellowFaceSolver)


class BeginnerSolver(Solver):
    def solution(self):
        cube = copy.deepcopy(self.cube)
        solution = WhiteCrossSolver.WhiteCrossSolver(cube).solution()
        solution += WhiteFaceSolver.WhiteFaceSolver(cube).solution()
        solution += SecondLayerSolver.SecondLayerSolver(cube).solution()
        solution += YellowCrossSolver.YellowCrossSolver(cube).solution()
        solution += YellowFaceSolver.YellowFaceSolver(cube).solution()
        return [Move(m) for m in solution]
