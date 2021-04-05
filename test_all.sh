#!/bin/bash
ECHO "doing everything"

nohup python test_dpd.py 20000 10000 2 2 1e1 > /dev/null 2>&1 &
nohup python test_dpd.py 20000 10000 2 2 1e0 > /dev/null 2>&1 &
nohup python test_dpd.py 20000 10000 2 2 1e-1 > /dev/null 2>&1 &

nohup python test_dpgd.py 20000 10000 2 2 1e1 > /dev/null 2>&1 &
nohup python test_dpgd.py 20000 10000 2 2 1e0 > /dev/null 2>&1 &
nohup python test_dpgd.py 20000 10000 2 2 1e-1 > /dev/null 2>&1 &

nohup python test_dppg.py 20000 10000 2 2 1e1 > /dev/null 2>&1 &
nohup python test_dppg.py 20000 10000 2 2 1e0 > /dev/null 2>&1 &
nohup python test_dppg.py 20000 10000 2 2 1e-1 > /dev/null 2>&1 &

nohup python test_dpprg.py 20000 10000 2 2 1e1 > /dev/null 2>&1 &
nohup python test_dpprg.py 20000 10000 2 2 1e0 > /dev/null 2>&1 &
nohup python test_dpprg.py 20000 10000 2 2 1e-1 > /dev/null 2>&1 &

nohup python test_md.py 20000 10000 2 10 > /dev/null 2>&1 &
nohup python test_md.py 20000 10000 2 20 > /dev/null 2>&1 &
nohup python test_md.py 20000 10000 2 30 > /dev/null 2>&1 &

nohup python test_mgd.py 20000 10000 2 10 > /dev/null 2>&1 &
nohup python test_mgd.py 20000 10000 2 20 > /dev/null 2>&1 &
nohup python test_mgd.py 20000 10000 2 30 > /dev/null 2>&1 &

nohup python test_mpg.py 20000 10000 2 10 > /dev/null 2>&1 &
nohup python test_mpg.py 20000 10000 2 20 > /dev/null 2>&1 &
nohup python test_mpg.py 20000 10000 2 30 > /dev/null 2>&1 &

nohup python test_mprg.py 20000 10000 2 10 > /dev/null 2>&1 &
nohup python test_mprg.py 20000 10000 2 20 > /dev/null 2>&1 &
nohup python test_mprg.py 20000 10000 2 30 > /dev/null 2>&1 &

nohup python test_dppgln.py 20000 10000 2 2 1e1 > /dev/null 2>&1 &
nohup python test_dppgln.py 20000 10000 2 2 1e0 > /dev/null 2>&1 &
nohup python test_dppgln.py 20000 10000 2 2 1e-1 > /dev/null 2>&1 &

nohup python test_dpprgln.py 20000 10000 2 2 1e1 > /dev/null 2>&1 &
nohup python test_dpprgln.py 20000 10000 2 2 1e0 > /dev/null 2>&1 &
nohup python test_dpprgln.py 20000 10000 2 2 1e-1 > /dev/null 2>&1 &

nohup python test_dpgdln.py 20000 10000 2 2 1e1 > /dev/null 2>&1 &
nohup python test_dpgdln.py 20000 10000 2 2 1e0 > /dev/null 2>&1 &
nohup python test_dpgdln.py 20000 10000 2 2 1e-1 > /dev/null 2>&1 &

nohup python test_dpdln.py 20000 10000 2 2 1e1 > /dev/null 2>&1 &
nohup python test_dpdln.py 20000 10000 2 2 1e0 > /dev/null 2>&1 &
nohup python test_dpdln.py 20000 10000 2 2 1e-1 > /dev/null 2>&1 &

# nohup mpiexec -np 5 python -m mpi4py test_dppgln.py 20000 10000 2 2 1e1 > /dev/null 2>&1 &
# nohup mpiexec -np 5 python -m mpi4py test_dppgln.py 20000 10000 2 2 1e0 > /dev/null 2>&1 &
# nohup mpiexec -np 5 python -m mpi4py test_dppgln.py 20000 10000 2 2 1e-1 > /dev/null 2>&1 &
#
# nohup mpiexec -np 5 python -m mpi4py test_dpprgln.py 20000 10000 2 2 1e1 > /dev/null 2>&1 &
# nohup mpiexec -np 5 python -m mpi4py test_dpprgln.py 20000 10000 2 2 1e0 > /dev/null 2>&1 &
# nohup mpiexec -np 5 python -m mpi4py test_dpprgln.py 20000 10000 2 2 1e-1 > /dev/null 2>&1 &

# nohup mpiexec -np 5 python -m mpi4py test_dpgdln.py 20000 10000 2 2 1e1 > /dev/null 2>&1 &
# nohup mpiexec -np 5 python -m mpi4py test_dpgdln.py 20000 10000 2 2 1e0 > /dev/null 2>&1 &
# nohup mpiexec -np 5 python -m mpi4py test_dpgdln.py 20000 10000 2 2 1e-1 > /dev/null 2>&1 &
#
# nohup mpiexec -np 5 python -m mpi4py test_dpdln.py 20000 10000 2 2 1e1 > /dev/null 2>&1 &
# nohup mpiexec -np 5 python -m mpi4py test_dpdln.py 20000 10000 2 2 1e0 > /dev/null 2>&1 &
# nohup mpiexec -np 5 python -m mpi4py test_dpdln.py 20000 10000 2 2 1e-1 > /dev/null 2>&1 &




# EOF
