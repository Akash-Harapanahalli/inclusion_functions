import numpy as np
import interval
from interval import get_cent_pert
import sympy as sp
from inclusion import InclusionFunction

# R^n --> R^m
class CenteredInclusionFunction (InclusionFunction) :
    def __init__(self, f_eqn, x_vars) -> None:
        super().__init__(f_eqn, x_vars)
        self.f = None
        self.Df_x = None
        # self.f = sp.lambdify((x_vars,), f_eqn, 'numpy')
        # self.Df_x_sym = self.f_eqn.jacobian((x_vars,))
        # self.Df_x = sp.lambdify((x_vars,), self.Df_x_sym)
    
    def __call__(self, x) :
        # if x.dtype != np.interval :
        #     raise NotIntervalException(x)

        cent, _ = get_cent_pert(x)
        print(self.f(cent).shape)
        return self.f(cent).reshape(-1) + self.Df_x(x)@(x - cent)
