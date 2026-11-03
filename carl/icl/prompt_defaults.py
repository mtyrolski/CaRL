"""Stable defaults for the configurable Luna prompt contract."""
from typing import Final

DEFAULT_RULES: Final[
    str
] = """Sokoban: # wall, - floor, . goal, $ box, * box on goal, @ player,
+ player on goal. Move UP, DOWN, LEFT or RIGHT. You may push one box into a free
cell, never pull or push two boxes. Solve by putting every box on a goal.
Preserve walls, goals, box count and one player. Return only the requested JSON."""

DEFAULT_POLICY: Final[str] = (
    DEFAULT_RULES
    + """\nReturn a probability vector in action order [UP, DOWN, LEFT, RIGHT]
for the NEXT SINGLE primitive action towards the supplied subgoal. Use nonnegative
probabilities with positive sum. This same policy is sampled during calibration
and search. The subgoal is an explicit board; do not invent a multi-action move."""
)

DEFAULT_VALUE: Final[str] = (
    DEFAULT_RULES
    + """\nEstimate value in [0,1]: the discounted continuation return for
solving this board. The examples use discount ** remaining_solution_actions.
Higher is closer to solving; a solved board has value 1."""
)

DEFAULT_VALUE_REMAINING: Final[str] = (
    "Sokoban: # wall, - floor, . goal, $ box, * box on goal, @ player, + player on goal. "
    "Estimate the number of remaining single player moves (including pushes) until all boxes are on goals. "
    "Examples are verified training trajectories, not necessarily optimal. Return 0 for a solved state. "
    "Use the board and the demonstrations to estimate remaining moves; output only the requested JSON."
)

DEFAULT_GENERATOR: Final[str] = (
    DEFAULT_RULES
    + "\nPropose distinct intermediate subgoal boards, different from the input."
)

DEFAULT_FIXED: Final[str] = (
    "k is a FIXED number of primitive moves along a successful trajectory. Propose a useful state k moves ahead, "
    "reachable within k actions when cap=k. The examples are verified k-action transitions. k never denotes a percentile."
)

DEFAULT_RELATIVE: Final[str] = (
    "u is a policy-relative percentile, NOT a number of moves or a success probability. For each training context, "
    "targets were weighted by the frozen policy's exact-hit rates and ranked by mean successful first-hit time. "
    "Ties share the midpoint of their combined percentile interval. Infer this scale from examples without "
    "reconstructing a reference distribution for the query. The controller has a separate fixed action cap L."
)

DEFAULT_ABSOLUTE: Final[str] = (
    "d is the frozen policy's mean successful first-hit execution time in primitive actions, measured by repeated "
    "rollouts. It is not necessarily the trajectory offset. The controller has a separate fixed action cap L."
)
