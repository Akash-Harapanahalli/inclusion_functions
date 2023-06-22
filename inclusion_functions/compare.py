import sympy as sp
import numpy as np
import interval
from cornered import CorneredInclusionFunction, all_corners
from centered import CenteredInclusionFunction
from inclusion import NaturalInclusionFunction

x1, x2 = (x_vars := sp.symbols('x1 x2'))

f_eqn = [
    x1**2, # x1 + x2 + 2*x1*x2,
    x1**3 # (x1 + x2)**2
]

f_nif = NaturalInclusionFunction (f_eqn, x_vars)
f_coif = CorneredInclusionFunction(f_eqn, x_vars, all_corners(2))
f_ceif = CenteredInclusionFunction(f_eqn, x_vars)
x = np.array([
    np.interval(-1,1),
    np.interval(-0.1,0.1)
])

print(f_nif(x))
print(f_coif(x))
print(f_ceif(x))