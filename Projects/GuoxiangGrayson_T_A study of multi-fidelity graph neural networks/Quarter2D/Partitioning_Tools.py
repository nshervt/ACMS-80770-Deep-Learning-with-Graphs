# this code contains tools for the distributed solver
import numpy as np
import meshio
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# this function find the global dof number of a global node
# inputs:
#      d: dimension
#      ls: list of constrained direction 
#      P: list of global node id
# output:
#      list of constrained global dof
def node_to_dof(d, ls, P):
	global_dof = []
	for g in P:
		for i in ls:
			global_dof.append( int( d*g ) + i )
	return global_dof


# localize global nodal and element number of a particular processor
# inputs:
#       rank: processor number
#       recvbuf: mesh part
#       Points: global coordinates
#       Cells: global cell number
# outputs:
#       ele_list: local element list
#       nodel_list: local nodal list
def rankwise_dist(rank,recvbuf, Points, Cells):
	ele_list   = []
	nodal_list = []

	for idx, node in enumerate(recvbuf):
		if node == rank:
			ele_list.append(idx)
			for n in Cells[idx]:
				if n not in nodal_list:
					nodal_list.append(n)
	return ele_list, nodal_list	


# This function finds the shared nodes for each processor, note that one processor can share
# data from each of the rest processors
def find_shared_nodes(rank,size,rank_nodal_num,rank_nodal_list):
	shared_nodes = []
	my_nodes     = rank_nodal_list[rank]
	my_num       = rank_nodal_num[rank]
	# print('My rank is:'+str(rank)+', number of nodes is:'+str(my_num)+', node list is:'+str(my_nodes))

	for r in range(size):
		if r != rank:
			for idx in rank_nodal_list[r]:
				if idx in my_nodes and idx not in shared_nodes:
					shared_nodes.append(idx)
	return shared_nodes 


# find the global sorted shared nodes
def sort_shared(G_shared_nodes):
	sorted = []
	for i in G_shared_nodes:
		for j in i:
			if j not in sorted:
				sorted.append(j)

	return np.sort(sorted)


# function to map global nodal number to rankwised nodal number
def local_mat_node(G_ID, L_N):
	rank_mat_ID = []
	for g in G_ID:
		for idx, node in enumerate(L_N):
			if node==g:
				rank_mat_ID.append(idx)
				continue
	return rank_mat_ID


# function to write results to rankwise file
# inputs:
#        n_basis: number of basis fucntion per ele
#        Points : global point array
#        Local_Cells : local (rankwise cell array)
# 		 Local_N_list: local (rankwise node list)
# 		 rank        : processor label
#        save_path   : save path
def Write_mesh_to_vtk(n_basis, Points, Local_Cells,Local_N_list,rank,save_path):
	
	rank_wise_cell   = np.zeros((len(Local_Cells),n_basis))
	rank_wise_point  = np.zeros((len(Local_N_list),3))

	# localize points and cells
	for idx,ele in enumerate(Local_Cells):
		local_nodes = local_mat_node(ele,Local_N_list)
		rank_wise_cell[idx,:]=local_nodes
		for i in ele:
			rank_wise_point[local_mat_node([i],Local_N_list),:] = Points[i,:]

	if n_basis == 3:
		rank_wise_cell = [("triangle",rank_wise_cell)]
	else: 
		print('element type not supported yet!')

	meshio.write_points_cells(save_path, rank_wise_point , rank_wise_cell,{})

	return 0