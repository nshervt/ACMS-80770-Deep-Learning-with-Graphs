"""
Adiabatic premixed flame

Requires: cantera >= 2.5.0, matplotlib >= 2.0
"""

import cantera as ct
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

pc = ['Multi']
lt = ['-','--','.']
phi_t = [1]
fuel_t = ['CH4:1']
# inp_file = ['gri30.yaml']

inp_file = ['gri30.yaml','reduced_drgep_22.cti','reduced_drg_40.cti']

from matplotlib import cm,rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)

# Simulation parameters
p = ct.one_atm     # pressure [Pa]
Tin = 300.0        # unburned gas temperature [K]
width = 0.03       # m
loglevel = 0       # amount of diagnostic output (0 to 8)

sL = [
               ['CO2', 'CH4', 'H2O', 'CO'],
               ['C2H6', 'O2', 'H2O', 'CO2']]

# customLines = [Line2D([0], [0], color='k', linestyle='solid'),
#                Line2D([0], [0], color='k', linestyle='dashed'),
#                Line2D([0], [0], color='k', marker='+')]

mF = [
               [1, 1, 1, 1, 10  ],
               [1, 1, 10, 10]]

# Solution object used to compute mixture properties, set to the state of the
# upstream fuel-air mixture

# Define the oxidizer composition, here air with 21 mol-% O2 and 79 mol-% N2
air = "O2:0.21, N2:0.79"

# Set the mixture composition according to the equivalence ratio

# Set up flame object
f = {}
B={}
C={}
thickness = {}


for i in range (len(fuel_t)):
      for j in range(len(inp_file)):
                  
            p = ct.one_atm     # pressure [Pa]
            Tin = 300.0        # unburned gas temperature [K]
            width = 0.03       # m
            loglevel = 1       # amount of diagnostic output (0 to 8)
            gas = ct.Solution(inp_file[j])
            gas.TP = Tin, p
            
            air = "O2:0.21, N2:0.79"
            gas.set_equivalence_ratio(phi=phi_t[i], fuel=fuel_t[i], oxidizer=air)

            f[j] = ct.FreeFlame(gas, width=0.03)
            f[j].set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
            f[j].show_solution()

            # Solve with mixture-averaged transport model
            f[j].transport_model = pc[i]
            f[j].solve(loglevel=loglevel, auto=True)

      # write the velocity, temperature, and mole fractions to a CSV file
            #f[j].write_csv('premixed_flame.csv', quiet=False)

            f[j].show_solution()
            
            fuel = [0,13,24]
            fuel_clr = ["r","b","g","y","m"]
            
            z= f[j].flame.grid
            T = f[j].T
            size = len(z)-1
            grad = np.zeros(size)
            for r in range(size):
                  grad[r] = (T[r+1]-T[r])/(z[r+1]-z[r])
            thickness[j] = (max(T) -min(T)) / max(grad)
            #print ('laminar flame thickness = ', thickness)
            #print('lame speed = {0:7f} m/s'.format(f[i].velocity[0]))

            # C = f[j].Y[gas.species_index((sL[i]))]

            plt.plot(f[j].flame.grid*100,f[j].T, str(fuel_clr[j])+str(lt[j]), label=str(pc[i]))
            plt.xlabel('$x$ (cm)')
            plt.ylabel('$T$')
            fig = matplotlib.pyplot.gcf()
            
plt.close()

j = 0           
plt.plot(f[j].flame.grid*100,f[j].T, str(fuel_clr[j])+str(lt[j]), label=str(inp_file[j]))
j=1
plt.plot(f[j].flame.grid*100,f[j].T, str(fuel_clr[j])+str(lt[j]), label=str(inp_file[j]))
j=2
plt.plot(f[j].flame.grid*100,f[j].T, str(fuel_clr[j])+str(lt[j]), label=str(inp_file[j]))

plt.xlabel('$x$ (cm)')
plt.ylabel('$T$')
plt.legend(["True mechanism","DRGEP","DRG"])
plt.show()
print("flame velocity-->", "| detailed = ",f[0].velocity[0],"| drgep = ", f[1].velocity[0],"| drg = ", f[2].velocity[0])
print("flame thickness-->","| detailed = ", thickness[0],"| drgep = ",thickness[1],"| drg = ",thickness[2])

print("done")
