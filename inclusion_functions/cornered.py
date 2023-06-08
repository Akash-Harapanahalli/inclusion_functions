import numpy as np
import interval
from interval import NotIntervalException, get_lu
import sympy as sp
from .inclusion import InclusionFunction
from .utils import d_positive

class Corner :
    def __init__(self, edge) -> None:
        self.edge = tuple(edge)

    def __getitem__ (self, key) :
        return self.edge[key]
    
    def __setitem__ (self, key, item) :
        self.edge[key] = item
        
def two_corners (n) :
    return [Corner((0,)*n), Corner((1,)*n)]

# R^n --> R^m
class CorneredInclusionFunction (InclusionFunction) :
    def __init__(self, f_eqn, x_vars) -> None:
        super().__init__(f_eqn, x_vars)
        self.f = sp.lambdify((x_vars,), f_eqn)
        self.Df_x_sym = self.f_eqn.jacobian((x_vars,))
        self.Df_x = sp.lambdify((x_vars,), self.Df_x_sym)
        self.n = len(x_vars)
        self.m = len(f_eqn)

        self.corners = two_corners(self.n)
    
    def __call__(self, x) :
        if x.dtype != np.interval :
            raise NotIntervalException(x)

        _x, x_ = get_lu(x)

        # f_x = self.f_eqn(_x)
        # fx_ = self.f_eqn(x_)
        f_eval = np.vstack((self.f_eqn(_x), self.f_eqn(x_))).T

        J = self.Df_x(x)
        _J, J_ = get_lu(J)
        _Jp, _Jn = d_positive(_J)
        J_p, J_n = d_positive(J_)

        B = np.empty((2*self.m, 2*self.n))
        b = np.empty(2*self.m)

        for corner in self.corners :
            for i in range(self.n) :
                edge = corner[i]

                if edge == 0 :
                    B[:self.m,i] = -_Jn[:,i]
                    B[self.m:,i] =  _Jn[:,i]
                    B[:self.m,i+self.n] = -J_p[:,i]
                    B[self.m:,i+self.n] =  J_p[:,i]
                    b[i] = self.f(replaced)
                else :
                    B[:self.m,i] =  J_p[:,i]
                    B[self.m:,i] = -J_p[:,i]
                    B[:self.m,i+self.n] =  _Jn[:,i]
                    B[self.m:,i+self.n] = -_Jn[:,i]


                b[j] = f_eval[corner[:self.m,j]]
                b[j + self.n] = 
