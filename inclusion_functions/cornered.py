import numpy as np
import interval
from interval import get_lu, as_lu, get_iarray
import sympy as sp
from inclusion import InclusionFunction
from utils import d_positive

class Corner :
    def __init__(self, edge) -> None:
        self.edge = tuple(edge)

    def __getitem__ (self, key) :
        return self.edge[key]
    
    def __setitem__ (self, key, item) :
        self.edge[key] = item
    
    def __repr__ (self) :
        return repr(self.edge)
        
    def __str__ (self) :
        return str(self.edge)
        
def two_corners (n) :
    return [Corner((0,)*n), Corner((1,)*n)]

def all_corners (n) :
    return [Corner(tuple(reversed([(i//(2**j)) % 2 for j in range(n)]))) for i in range(2**n)]

# f_i should be a list
def _cornered_algorithm (corners, _J, J_, f, x, as_iarray=True) :
    _xx_ = as_lu(x).T.reshape(-1,1)
    _Jp, _Jn = d_positive(_J)
    J_p, J_n = d_positive(J_)
    m, n = _J.shape

    B = np.empty((2*m, 2*n))
    b = np.empty((2*m, 1))

    ret = []

    for corner in corners :
        replaced = np.array([(x[i].l if corner[i] == 0 else x[i].u) for i in range(n)])
        # print(replaced)
        for i in range(n) :
            if corner[i] == 0 :
                B[:m,i]   = -_Jn[:,i]
                B[:m,i+n] =  _Jn[:,i]
                B[m:,i]   = -J_p[:,i]
                B[m:,i+n] =  J_p[:,i]
            else :
                B[:m,i]   =  J_p[:,i]
                B[:m,i+n] = -J_p[:,i]
                B[m:,i]   =  _Jn[:,i]
                B[m:,i+n] = -_Jn[:,i]
        b[:m,0] = f(replaced).reshape(-1)
        b[m:,0] = b[:m,0]
        # print(B)
        ret.append((B@_xx_ + b).reshape(-1))
    
    ret = np.array(ret)
    # print(ret)
    l, u = np.max(ret[:,:m], axis=0), np.min(ret[:,m:], axis=0)
    if as_iarray :
        return get_iarray(l,u)
    else :
        return (l, u)

# R^n --> R^m
class CorneredInclusionFunction (InclusionFunction) :
    def __init__(self, f_eqn, x_vars, corners=None) -> None:
        super().__init__(f_eqn, x_vars)
        self.f = sp.lambdify((x_vars,), f_eqn, 'numpy')
        self.f_i = [sp.lambdify((x_vars,), f_eqn_i, 'numpy') for f_eqn_i in self.f_eqn]
        self.Df_x_sym = self.f_eqn.jacobian((x_vars,))
        self.Df_x = sp.lambdify((x_vars,), self.Df_x_sym)
        self.n = len(x_vars)
        self.m = len(f_eqn)

        self.corners = corners if corners is not None else two_corners(self.n)
    
    def __call__(self, x) :
        # if x.dtype != np.interval :
        #     raise NotIntervalException(x)

        J = self.Df_x(x)
        _J, J_ = get_lu(J)

        return _cornered_algorithm(self.corners, _J, J_, self.f, x)
        

        # for corner in self.corners :
        # []

