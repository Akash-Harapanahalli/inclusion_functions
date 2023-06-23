import numpy as np
import interval
from interval import from_cent_pert, get_lu, get_iarray
import sympy as sp
from ReachMM import NeuralNetwork
from centered import CenteredInclusionFunction
from cornered import CorneredInclusionFunction, two_corners, all_corners
import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

net = NeuralNetwork('../../ReachMM/examples/vehicle/models/100r100r2')

print(net)
W1 = net.seq[0].weight.cpu().detach().numpy()
b1 = net.seq[0].bias.cpu().detach().numpy().reshape(-1,1)
W2 = net.seq[2].weight.cpu().detach().numpy()
b2 = net.seq[2].bias.cpu().detach().numpy().reshape(-1,1)
W3 = net.seq[4].weight.cpu().detach().numpy()
b3 = net.seq[4].bias.cpu().detach().numpy().reshape(-1,1)

cent = np.array([8,7,-2*np.pi/3,2])
pert = np.array([0.001,0.001,0.001,0.001])
x0 = from_cent_pert(cent, pert)
_x, x_ = get_lu(x0)

x1, x2, x3, x4 = (xvars := sp.symbols('x1 x2 x3 x4'))
print(xvars)

# x = sp.symbols('x')
# sig = lambda x : sp.tanh(x)
def sig (x) :
    return sp.Matrix([sp.tanh(xi) for xi in x])

net.seq[1] = torch.nn.Tanh()
net.seq[3] = torch.nn.Tanh()
# del(net.seq[2])
# del(net.seq[2])
print(net)

global_input = torch.zeros([1,4], dtype=torch.float32)
bnn = BoundedModule(net, global_input)
x_L = torch.tensor(_x.reshape(1,-1), dtype=torch.float32)
x_U = torch.tensor(x_.reshape(1,-1), dtype=torch.float32)
ptb = PerturbationLpNorm(norm=np.inf, x_L=x_L, x_U=x_U)
input = BoundedTensor(global_input, ptb)
print('CROWN')
u_lb, u_ub = bnn.compute_bounds(x=(input,), method='CROWN')
print(get_iarray(u_lb.detach().cpu().numpy(), u_ub.cpu().detach().numpy()).reshape(-1))
print('IBP')
u_lb, u_ub = bnn.compute_bounds(x=(input,), method='IBP')
print(get_iarray(u_lb.detach().cpu().numpy(), u_ub.cpu().detach().numpy()).reshape(-1))

# print(f_eqn)

f_nat = lambda x : (W3@np.tanh(W2@np.tanh(W1@x + b1) + b2) + b3)
# f_nat = lambda x : (W3@np.tanh(W1@x + b1) + b3)
print('Natural: ')
print(f_nat(x0.reshape(-1,1)).reshape(-1))

# f_eqn = W3@sig(W2@sig(W1@sp.Matrix(xvars) + b1) + b2) + b3
# f_eqn = W3@sig(W1@sp.Matrix(xvars) + b1) + b3
f_eqn = sig(W1@sp.Matrix(xvars) + b1)

# # f_cent = CenteredInclusionFunction(f_eqn, xvars)
# f_cent = CorneredInclusionFunction(f_eqn, xvars, all_corners(4))
# f_cent.f = lambda x : W3@np.tanh((W1@x).reshape(-1,1) + b1) + b3
# # f2 = lambda x : W3@np.tanh(x) + b3
# print(W3.shape)
# print(W1.shape)
# f_cent.Df_x = lambda x : W3@(np.diag((1 - np.tanh(W1@x.reshape(-1,1) + b1)**2).reshape(-1)))@W1

# print('Cornered: ')
# print(f_cent(x0).reshape(-1))

# f_cent0 = lambda x: (W1@x.reshape(-1,1)) + b1

f_cent = CorneredInclusionFunction(f_eqn, xvars, two_corners(4))
# f_cent = CorneredInclusionFunction(f_eqn, xvars, all_corners(4))
f_cent.f = lambda y : W2@np.tanh((W1@y.reshape(-1,1) + b1).reshape(-1,1)) + b2
f_cent.Df_x = lambda y : W2@(np.diag((1 - np.tanh(W1@y.reshape(-1,1) + b1)**2).reshape(-1)))@W1

f_cent2 = CorneredInclusionFunction(f_eqn, xvars, two_corners(100))
f_cent2.f = lambda z : W3@np.tanh(z.reshape(-1,1)) + b3
f_cent2.Df_x = lambda z : W3@(np.diag((1 - np.tanh(z)**2).reshape(-1)))

print('Centered: ')
print(f_cent2(f_cent(x0.reshape(-1)).reshape(-1)).reshape(-1))
# print(f_cent2(f_cent(f_cent0(x0).reshape(-1)).reshape(-1)).reshape(-1))

# f_corn = CorneredInclusionFunction(f_eqn, xvars, all_corners(4))
# print('Cornered: ')
# print(f2(f_corn(x0).reshape(-1,1)))

# print(W1, W2, b1, b2)
