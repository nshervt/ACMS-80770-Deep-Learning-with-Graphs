# fenics solver for 2D linear elasticity, generating both low and high fid solution and also perform low->high projection
# geo: 2D cantilever
# status: plain stress
# ref: https://fenicsproject.org/pub/tutorial/html/._ftut1008.html
# ref: https://fenics-solid-tutorial.readthedocs.io/en/latest/2DPlaneStrain/2D_Elasticity.html
# ref: https://github.com/CMGLab/IGA-Notes
# ref: https://comet-fenics.readthedocs.io/en/latest/demo/elasticity/2D_elasticity.py.html
# ref: https://jorgensd.github.io/dolfinx-tutorial/

# Boundary condition:
#                   Left (x=0): d = 0


# we project the low fid solution to high fid mesh via interpolation

from dolfin import *
from fenics import *

parameters['allow_extrapolation'] = True # allow non-matching meshes

# sym-ed gradient tensor (strain)
def epsilon(u):
    return sym(grad(u)) 

# isotropic stress
def sigma(u, lmd, mu):
    return 2*mu*epsilon(u) + lmd * div(u) * Identity(2) 

f = Constant((0,-1)) # external loading

#----------------------Meshing--------------------------#
mesh_low = Mesh('low.xml')

k   = 1 # polynomial order

# Define function space 
U   = VectorFunctionSpace(mesh_low, "CG", k) 
#------------------------------------------------------#

#-------------------material properties-----------------#
E  = 1e6     # young's modulus 
nu = 0.3     # possion's ratio

# lame parameters for plain strain
lmd = E*nu/ ( (1+nu)*(1-2*nu) )
mu  = E/ ( 2*(1+nu) )

lmd = 2*lmd*mu /(lmd + 2*mu) # if plan-stress
#--------------------------------------------------------#

#-------------------boundary conditions------------------#
def uD(x, on_boundary):   # x = 0, u = 0
    return near(x[0], 0)

bc_x0 = DirichletBC(U, Constant((0., 0.)), uD)
#--------------------------------------------------------#

#-------------------Variational forms--------------------#
# trial and test function are form the same functional space
u = TrialFunction(U)
v = TestFunction(U)

# set bilinear form
A = inner(sigma(u, lmd, mu) , epsilon(v))*dx

# set linear form
l = dot(f, v) * dx
#--------------------------------------------------------#

#------------------solve and save------------------------#
# solve the problem
u_h_low = Function(U)
solve(A==l, u_h_low, bc_x0)

# sol_save = File("Sol/low-fid.pvd")
# sol_save << u_h_low
#--------------------------------------------------------#


#---------------projection from low to high--------------#
# import high-fid mesh
mesh_high  = Mesh('high.xml')
U_High     = VectorFunctionSpace(mesh_high, "CG", k) 
u_H_interp = Function(U_High)
u_H_interp.interpolate(u_h_low)

sol_H_save = File("Sol/low-to-high.pvd")
sol_H_save << u_H_interp
#--------------------------------------------------------#