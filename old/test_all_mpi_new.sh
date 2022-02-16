#!/bin/bash

# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_updated_nov_mar.csv ./output2 dpdln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_updated_nov_mar.csv ./output2 dpgdln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_updated_nov_mar.csv ./output2 dppgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_updated_nov_mar.csv ./output2 dpprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
#
# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_updated_nov_mar.csv ./output2 dphpgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_updated_nov_mar.csv ./output2 dphprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
#
# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_updated_nov_mar.csv ./output2 mdln 50000 20000 30 --nMix 30 > /dev/null 2>&1 &
# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_updated_nov_mar.csv ./output2 mgdln 50000 20000 30 --nMix 30 > /dev/null 2>&1 &
# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_updated_nov_mar.csv ./output2 mpgln 50000 20000 30 --nMix 30 > /dev/null 2>&1 &
# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_updated_nov_mar.csv ./output2 mprgln 50000 20000 30 --nMix 30 > /dev/null 2>&1 &
#
# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_updated_nov_mar.csv ./output2 mhpgln 50000 20000 30 --nMix 30 > /dev/null 2>&1 &
# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_updated_nov_mar.csv ./output2 mhprgln 50000 20000 30 --nMix 30 > /dev/null 2>&1 &

# nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_updated_nov_mar.csv ./output2 dppprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
nohup mpiexec -np 8 python -m mpi4py test_generic_mpi.py ./datasets/ivt_updated_nov_mar.csv ./output2 dpppgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
# EOF
