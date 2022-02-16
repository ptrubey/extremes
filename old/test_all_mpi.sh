#!/bin/bash

# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_nov_mar.csv ./output dpdln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_nov_mar.csv ./output dpgdln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_nov_mar.csv ./output dppgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_nov_mar.csv ./output dpprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
#
# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_nov_mar.csv ./output dphpgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_nov_mar.csv ./output dphprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
#
# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_nov_mar.csv ./output mdln 50000 20000 30 --nMix 30 > /dev/null 2>&1 &
# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_nov_mar.csv ./output mgdln 50000 20000 30 --nMix 30 > /dev/null 2>&1 &
# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_nov_mar.csv ./output mpgln 50000 20000 30 --nMix 30 > /dev/null 2>&1 &
# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_nov_mar.csv ./output mprgln 50000 20000 30 --nMix 30 > /dev/null 2>&1 &
#
# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_nov_mar.csv ./output mhpgln 50000 20000 30 --nMix 30 > /dev/null 2>&1 &
# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_nov_mar.csv ./output mhprgln 50000 20000 30 --nMix 30 > /dev/null 2>&1 &

# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_nov_mar.csv ./output dppprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1
nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_nov_mar.csv ./output dpppgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1
# EOF
