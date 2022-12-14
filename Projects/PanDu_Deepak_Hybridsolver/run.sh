BATCH_SIZE=4

# Prediction experiments

mpirun -np $((BATCH_SIZE+1)) --oversubscribe python train.py >log

# mpirun -np $((BATCH_SIZE+1)) --oversubscribe python main.py --batch-size $BATCH_SIZE --gpus 1 -dw 2 --model gcn --hidden-size 512 --num-layers 6 --optim adam -lr 5e-5 --data-dir data/NACA0012_interpolate/ -e gcn_interp


 
# # Generalization experiments

# mpirun -np $((BATCH_SIZE+1)) --oversubscribe python main.py --batch-size $BATCH_SIZE --gpus 1 -dw 2 --su2-config coarse.cfg --model cfd_gcn --hidden-size 512 --num-layers 6 --num-end-convs 3 --optim adam -lr 5e-5 --data-dir data/NACA0012_machsplit_noshock --coarse-mesh meshes/mesh_NACA0012_xcoarse.su2 -e cfd_gcn_gen > /dev/null

# mpirun -np $((BATCH_SIZE+1)) --oversubscribe python main.py --batch-size $BATCH_SIZE --gpus 1 -dw 2 --model gcn --hidden-size 512 --num-layers 6 --optim adam -lr 5e-5 --data-dir data/NACA0012_machsplit_noshock/ -e gcn_gen



# Cleanup temporary files (keeps logs)

sh cleanup.sh


#mpirun -np 17 --oversubscribe python main.py --batch-size 16 --gpus 1 -dw 2 --su2-config coarse.cfg --model cfd_gcn --hidden-size 512 --num-layers 6 --num-end-convs 3 --optim adam -lr 5e-5 --data-dir data/NACA0012_interpolate --coarse-mesh meshes/mesh_NACA0012_xcoarse.su2 -e cfd_gcn_interp > /dev/null