import numpy as np
import interval
from interval import from_cent_pert, get_lu, get_iarray
import sympy as sp
from ReachMM import NeuralNetwork
from centered import CenteredInclusionFunction
from cornered import CorneredInclusionFunction, all_corners
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
pert = np.array([0.1,0.1,0.1,0.1])
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

f_cent = CenteredInclusionFunction(f_eqn, xvars)
# f_cent = CorneredInclusionFunction(f_eqn, xvars, all_corners(4))
f_cent.f = lambda x : W2@np.tanh((W1@x).reshape(-1,1) + b1) + b2
f2 = lambda x : W3@np.tanh(x) + b3
f_cent.Df_x = lambda x : W2@((1 - np.tanh(W1@x + b1)) @ (1 - np.tanh(W1@x + b1)).T)@W1
print((1 - np.tanh(W1@x0 + b1)).shape)

print('Centered: ')
print(f2(f_cent(x0).reshape(-1,1)))

# f_corn = CorneredInclusionFunction(f_eqn, xvars, all_corners(4))
# print('Cornered: ')
# print(f2(f_corn(x0).reshape(-1,1)))

# print(W1, W2, b1, b2)
