# Fenics solver for 2D linear elasticity, generating both low and high fid solution and also perform low->high projection
# geo: 2D annular
# status: plain strain
# ref: https://fenicsproject.org/pub/tutorial/html/._ftut1008.html
# ref: https://fenics-solid-tutorial.readthedocs.io/en/latest/2DPlaneStrain/2D_Elasticity.html
# ref: https://github.com/CMGLab/IGA-Notes
# ref: https://comet-fenics.readthedocs.io/en/latest/demo/elasticity/2D_elasticity.py.html
# ref: https://jorgensd.github.io/dolfinx-tutorial/

# boundary condition: inner circle radial traction, outer circle traction free
#                     vertical side no horizon displacement, vertical traction free
#                     horizontal side no vertical displacement, horiontal traction free

# Solve low fid problem
import numpy as np
from fenics import *

parameters['allow_extrapolation'] = True

# sym-ed gradient tensor (strain)
def epsilon(u):
    return sym(grad(u)) 

# isotropic stress
def sigma(u, lmd, mu):
    return 2*mu*epsilon(u) + lmd * div(u) * Identity(2) 

# geo info
r_0 = 0.09  # outer circle radius
r_i = 0.075 # inner circle radius


# material properties
E  = 200e9   # young's modulus 
nu = 0.3     # possion's ratio

# lame parameters for plain strain
lmd = E*nu/ ( (1+nu)*(1-2*nu) )
mu  = E/ ( 2*(1+nu) )

p = 1 # polynomial degree

# radial traction
P_r = 30e6
mesh       = Mesh('low.xml')
bdies      = MeshFunction("size_t", mesh, 'low_facet_region.xml')

# mesh info
x = SpatialCoordinate(mesh)
ds = ds(subdomain_data=bdies) 

# define function space 
U   = VectorFunctionSpace(mesh, "CG", p) 

# assign bc to only one component, the other one is shear traction-free
bc_bot = DirichletBC(U.sub(1), 0.0 , bdies, 8) # 8 means bot, check .geo file
bc_top = DirichletBC(U.sub(0), 0.0 , bdies, 7) # 7 means top, check .geo file

bcs = [bc_bot, bc_top] # group bc

# traction decomposition back to carti
t_r = as_vector((P_r*x[0]/r_i,P_r*x[1]/r_i))

# print(assemble( dot(t_r, FacetNormal(mesh))*ds(5)) ) # 5 means inner circ bdy, check .geo file

# trial and test function are form the same functional space
u = TrialFunction(U)
v = TestFunction(U)

# set bilinear form
A = inner(sigma(u, lmd, mu) , epsilon(v))*dx

# set linear form
L = dot(v, t_r) * ds(5)  # 5 means inner circ bdy

# solve the low-fidelity problem
u_h_low = Function(U)
solve(A==L, u_h_low, bcs)

# sol_save = File("Sol/low-fid.pvd")
# sol_save << u_h_low

# intepolate low-fid solution to high fid mesh
#--------------------------------------------#
mesh_H       = Mesh('high.xml')
# define function space 
U_H   = VectorFunctionSpace(mesh_H, "CG", p) 
u_H_interp = Function(U_H)
u_H_interp.interpolate(u_h_low)


sol_H_save = File("Sol/low-to-high.pvd")
sol_H_save << u_H_interp
#--------------------------------------------#