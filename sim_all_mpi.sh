#!/bin/bash

nohup mpiexec -np 4 python -m mpi4py test_simulation_mpi.py ./simulated/sim_c3_m3 dphprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
nohup mpiexec -np 4 python -m mpi4py test_simulation_mpi.py ./simulated/sim_c3_m6 dphprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
nohup mpiexec -np 4 python -m mpi4py test_simulation_mpi.py ./simulated/sim_c3_m9 dphprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
nohup mpiexec -np 4 python -m mpi4py test_simulation_mpi.py ./simulated/sim_c3_m12 dphprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &

nohup mpiexec -np 4 python -m mpi4py test_simulation_mpi.py ./simulated/sim_c6_m3 dphprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
nohup mpiexec -np 4 python -m mpi4py test_simulation_mpi.py ./simulated/sim_c6_m6 dphprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
nohup mpiexec -np 4 python -m mpi4py test_simulation_mpi.py ./simulated/sim_c6_m9 dphprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
nohup mpiexec -np 4 python -m mpi4py test_simulation_mpi.py ./simulated/sim_c6_m12 dphprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &

nohup mpiexec -np 4 python -m mpi4py test_simulation_mpi.py ./simulated/sim_c12_m3 dphprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
nohup mpiexec -np 4 python -m mpi4py test_simulation_mpi.py ./simulated/sim_c12_m6 dphprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
nohup mpiexec -np 4 python -m mpi4py test_simulation_mpi.py ./simulated/sim_c12_m9 dphprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
nohup mpiexec -np 4 python -m mpi4py test_simulation_mpi.py ./simulated/sim_c12_m12 dphprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &

nohup mpiexec -np 4 python -m mpi4py test_simulation_mpi.py ./simulated/sim_c20_m3 dphprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
nohup mpiexec -np 4 python -m mpi4py test_simulation_mpi.py ./simulated/sim_c20_m6 dphprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
nohup mpiexec -np 4 python -m mpi4py test_simulation_mpi.py ./simulated/sim_c20_m9 dphprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
nohup mpiexec -np 4 python -m mpi4py test_simulation_mpi.py ./simulated/sim_c20_m12 dphprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &

# EOF
