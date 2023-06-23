import sympy as sp
import numpy as np
import interval
# from cornered import MixedCorneredInclusionFunction, CorneredInclusionFunction, all_corners, two_corners
# from cornered import *
# from centered import *
from inclusion import *
from ReachMM.utils import run_times

N = 10000

x1, x2 = (x_vars := sp.symbols('x1 x2'))
n = len(x_vars)

f_eqn = [
    (x1 + x2)**2,
    x1 + x2 + 2*x1*x2
]

# f_eqn = [
#     (x1 + x2)**2,
#     4*sp.sin((x1 - x2)/4)
# ]

f_nif = NaturalInclusionFunction (f_eqn, x_vars)
f_coif = FastCorneredInclusionFunction(f_eqn, x_vars, all_corners(n))
f_mcoif = FastMixedCorneredInclusionFunction(f_eqn, x_vars, two_orderings(n), all_corners(n))
f_ceif = CenteredInclusionFunction(f_eqn, x_vars)
f_mceif = MixedCenteredInclusionFunction(f_eqn, x_vars, two_orderings(n))
x = np.array([
    np.interval(-0.1,0.1),
    np.interval(-0.1,0.1)
])

print('Natural:')
ret, times = run_times (N, f_nif, x, rt_disable_bar=True); print(ret); print(f'{np.mean(times):.4e} \pm {np.std(times):.4e}\n')
print('Cornered:')
ret, times = run_times (N, f_coif, x, rt_disable_bar=True); print(ret); print(f'{np.mean(times):.4e} \pm {np.std(times):.4e}\n')
print('Mixed Cornered:')
ret, times = run_times (N, f_mcoif, x, rt_disable_bar=True); print(ret); print(f'{np.mean(times):.4e} \pm {np.std(times):.4e}\n')
print('Centered')
ret, times = run_times (N, f_ceif, x, rt_disable_bar=True); print(ret); print(f'{np.mean(times):.4e} \pm {np.std(times):.4e}\n')
print('Mixed Centered')
ret, times = run_times (N, f_mceif, x, rt_disable_bar=True); print(ret); print(f'{np.mean(times):.4e} \pm {np.std(times):.4e}\n')

