import numpy as np
import interval
from interval import get_cent_pert, as_lu, get_iarray
import sympy as sp
from inclusion import InclusionFunction, standard_ordering

# R^n --> R^m
class CenteredInclusionFunction (InclusionFunction) :
    def __init__(self, f_eqn, x_vars) -> None:
        super().__init__(f_eqn, x_vars)
        # self.f = None
        # self.Df_x = None
        self.f_lam = sp.lambdify((x_vars,), f_eqn, 'numpy')
        self.f = lambda x : self.f_lam(x)[0]
        # self.f = sp.lambdify((x_vars,), f_eqn, 'numpy')
        self.Df_x_sym = self.f_eqn.jacobian((x_vars,))
        self.Df_x = sp.lambdify((x_vars,), self.Df_x_sym)
    
    def __call__(self, x) :
        # if x.dtype != np.interval :
        #     raise NotIntervalException(x)

        cent, _ = get_cent_pert(x)
        return self.f(cent).reshape(-1) + self.Df_x(x)@(x - cent)

class MixedCenteredInclusionFunction (InclusionFunction) :
    def __init__(self, f_eqn, x_vars, orderings=None) -> None:
        super().__init__(f_eqn, x_vars)
        self.n = len(x_vars)
        self.m = len(f_eqn)

        self.f_lam = sp.lambdify((x_vars,), f_eqn, 'numpy')
        self.f = lambda x : np.array(self.f_lam(x))
        self.f_i = [sp.lambdify((x_vars,), f_eqn_i, 'numpy') for f_eqn_i in self.f_eqn]
        self.Df_x_sym = self.f_eqn.jacobian((x_vars,))
        self.Df_x = sp.lambdify((x_vars,), self.Df_x_sym)
        # # print([self.Df_x_sym[:,i] for i in range(self.n)])
        self.Df_x_i = [sp.lambdify((x_vars,), self.Df_x_sym[:,i]) for i in range(self.n)]
        # self.f = None
        # self.Df_x = None

        self.orderings = orderings if orderings is not None else standard_ordering(self.n)
    
    def __call__(self, x) :
        # if x.dtype != np.interval :
        #     raise NotIntervalException(x)

        # J_i = [lambda x : get_lu(Df_x_col(x)) for Df_x_col in self.Df_x_i]

        # return _mixed_cornered_algorithm(self.orderings, self.corners, self.Df_x_i, self.f, x, self.m, self.n)

        ret = []
        cent, _ = get_cent_pert(x)

        for ordering in self.orderings :
            ret_i = 0
            xmcent = (x - cent)
            x_jac = np.copy(cent).astype(np.interval)
            for i in range (self.n) :
                x_jac[ordering[i]] = x[ordering[i]]
                J_i = self.Df_x_i[i](x_jac)
                ret_i = ret_i + J_i@np.atleast_1d(xmcent[i])
            ret.append(as_lu(ret_i + self.f(cent).reshape(-1)).T.reshape(-1))

        ret = np.array(ret)
        l, u = np.max(ret[:,:self.m], axis=0), np.min(ret[:,self.m:], axis=0)
        return get_iarray(l,u)

