# 2D mesh partitioning, the parallel way
# the script is for generating the low-fid meshes from a high-fid mesh

from Partitioning_Tools import *
import numpy as np
import meshio
from math import floor
from mpi4py import MPI
import os
from mgmetis.parmetis import part_mesh_kway

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# create pathes
path0  = 'Partitioned_Mesh/Rankwised_Nodes/'
os.makedirs(path0,exist_ok=True)

path1  = 'Partitioned_Mesh/Shared_Nodes/'
os.makedirs(path1,exist_ok=True)

path2  = 'Partitioned_Mesh/Rankwised_Elements/'
os.makedirs(path2,exist_ok=True)

path3  = 'Partitioned_Mesh/'
os.makedirs(path3,exist_ok=True)


# High fid mesh target
mesh_name =  "high.msh" 

#------------------------Step1--------------------------------#
#Get geometry and discretization: gmsh and read with meshio
#Credit: https://pypi.org/project/meshio/

if rank == 0: 
	Mesh      =  meshio.read(mesh_name)  		# import mesh
	Cells     =  (Mesh.cells_dict)['triangle']  # tri3
	Points    =  Mesh.points                    # nodal points
	nELE      =  len(Cells[:,0])    			# number of element

	# find the elmdist (vtxdist) array: credit: meshPartition.py from the CVFES
	nEach = floor(nELE/size)      # average number of element to each P
	nLeft = nELE - nEach * size   # leftovers evenly distributed to the last a few Ps
	elmdist = np.append( (nEach + 1) * np.arange(nLeft +1), \
									( (nEach+1)*nLeft) + nEach * np.arange(1, size -nLeft +1))

	elmdist = elmdist.astype(np.int64) # convert to integers
else:
	Cells,Facets,Points,elmdist = None, None, None, None # placeholders

# broadcast the data to each processor
Cells   = comm.bcast(Cells,   root=0)
Points  = comm.bcast(Points,  root=0)
elmdist = comm.bcast(elmdist, root=0)

# allocate element nodes to each processors, i.e.: find the array eptr, eind
P_start   = elmdist[rank]
P_end     = elmdist[rank+1]

eptr = np.zeros(P_end-P_start+1, dtype=np.int64)
eind = np.empty(0, dtype=np.int64)

for idx, ele in enumerate(Cells[P_start:P_end]):
	eptr[idx+1] = eptr[idx] + len(ele)
	eind = np.append(eind, ele[:])	

# using mgmetis.parmetis:
_, epart = part_mesh_kway(size, eptr, eind)

# gather the partitioned data, use Gatherv function to concatenate partitioned array of different size
recvbuf = None
if rank == 0:
    recvbuf = np.empty(len(Cells), dtype='int')
comm.Gatherv(epart,recvbuf,root=0)
recvbuf = comm.bcast(recvbuf, root=0)

Local_ele_list, Local_nodal_list = rankwise_dist(rank, recvbuf, Points, Cells)

# Collect the shared nodes information, this is arguably the most important part of this solver
rank_nodal_num  = comm.gather(len(Local_nodal_list),root=0) # no idea why comm.Gather is not working here, tbd
rank_nodal_list = comm.gather(Local_nodal_list,root=0)
rank_nodal_num,rank_nodal_list = comm.bcast(rank_nodal_num,root=0), comm.bcast(rank_nodal_list, root=0)
shared_nodes = find_shared_nodes(rank,size,rank_nodal_num,rank_nodal_list) # find the shared nodes for each processor

# save shared nodes information
np.savetxt(path0+'Rank='+str(rank)+'_local_nodes.csv',rank_nodal_list[rank],delimiter=',',fmt='%d')
np.savetxt(path1+'Rank='+str(rank)+'_shared.csv',shared_nodes,delimiter=',',fmt='%d')
np.savetxt(path2+'Rank='+str(rank)+'_elements.csv',Local_ele_list,delimiter=',',fmt='%d')

G_shared_nodes = comm.gather(shared_nodes,root=0)
if rank == 0:
	Global_shared = sort_shared(G_shared_nodes)
	np.savetxt(path1+'Global_shared.csv',Global_shared,delimiter=',',fmt='%d')

# write rankwise mesh
save_path = path3 + 'Rank-' + str(rank) + '.vtk'
Write_mesh_to_vtk(3, Points, Cells[Local_ele_list], Local_nodal_list , rank, save_path)