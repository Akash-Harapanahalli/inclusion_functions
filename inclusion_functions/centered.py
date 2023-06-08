import numpy as np
import interval
import sympy as sp
from .inclusion import InclusionFunction

class CenteredInclusionFunction (InclusionFunction) :
    def __init__(self, f_eqn, x_vars) -> None:
        super().__init__(f_eqn, x_vars)

