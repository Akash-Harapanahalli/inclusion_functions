import numpy as np
import interval
# from interval import NotIntervalException
import sympy as sp

class InclusionFunction :
    def __init__(self, f_eqn, x_vars) -> None:
        self.f_eqn = sp.Matrix(f_eqn)
        self.x_vars = tuple(x_vars)

    def __call__ (self, x) :
        return NotImplementedError

class NaturalInclusionFunction (InclusionFunction) :
    def __init__(self, f_eqn, x_vars) -> None:
        super().__init__(f_eqn, x_vars)
        self.f = sp.lambdify((x_vars,), f_eqn, 'numpy')
    
    def __call__(self, x) :
        # if x.dtype != np.interval :
        #     raise NotIntervalException(x)
        return self.f(x)
