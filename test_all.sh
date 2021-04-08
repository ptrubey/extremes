#!/bin/bash
ECHO "doing everything"

# nohup python test_dpd.py 50000 20000 30 2 1e1 > /dev/null 2>&1 &
# nohup python test_dpd.py 50000 20000 30 2 1e0 > /dev/null 2>&1 &
# nohup python test_dpd.py 50000 20000 30 2 1e-1 > /dev/null 2>&1 &

# nohup python test_dpgd.py 50000 20000 30 2 1e1 > /dev/null 2>&1 &
# nohup python test_dpgd.py 50000 20000 30 2 1e0 > /dev/null 2>&1 &
# nohup python test_dpgd.py 50000 20000 30 2 1e-1 > /dev/null 2>&1 &

# nohup python test_dppg.py 50000 20000 30 2 1e1 > /dev/null 2>&1 &
# nohup python test_dppg.py 50000 20000 30 2 1e0 > /dev/null 2>&1 &
# nohup python test_dppg.py 50000 20000 30 2 1e-1 > /dev/null 2>&1 &

# nohup python test_dpprg.py 50000 20000 30 2 1e1 > /dev/null 2>&1 &
# nohup python test_dpprg.py 50000 20000 30 2 1e0 > /dev/null 2>&1 &
# nohup python test_dpprg.py 50000 20000 30 2 1e-1 > /dev/null 2>&1 &

# nohup python test_md.py 50000 20000 30 10 > /dev/null 2>&1 &
# nohup python test_md.py 50000 20000 30 20 > /dev/null 2>&1 &
# nohup python test_md.py 50000 20000 30 30 > /dev/null 2>&1 &

# nohup python test_mgd.py 50000 20000 30 10 > /dev/null 2>&1 &
# nohup python test_mgd.py 50000 20000 30 20 > /dev/null 2>&1 &
# nohup python test_mgd.py 50000 20000 30 30 > /dev/null 2>&1 &

# nohup python test_mpg.py 50000 20000 30 10 > /dev/null 2>&1 &
# nohup python test_mpg.py 50000 20000 30 20 > /dev/null 2>&1 &
# nohup python test_mpg.py 50000 20000 30 30 > /dev/null 2>&1 &

# nohup python test_mprg.py 50000 20000 30 10 > /dev/null 2>&1 &
# nohup python test_mprg.py 50000 20000 30 20 > /dev/null 2>&1 &
# nohup python test_mprg.py 50000 20000 30 30 > /dev/null 2>&1 &

# nohup python test_dppgln.py 50000 20000 30 2 1e1 > /dev/null 2>&1 &
# nohup python test_dppgln.py 50000 20000 30 2 1e0 > /dev/null 2>&1 &
# nohup python test_dppgln.py 50000 20000 30 2 1e-1 > /dev/null 2>&1 &

# nohup python test_dpprgln.py 50000 20000 30 2 1e1 > /dev/null 2>&1 &
# nohup python test_dpprgln.py 50000 20000 30 2 1e0 > /dev/null 2>&1 &
# nohup python test_dpprgln.py 50000 20000 30 2 1e-1 > /dev/null 2>&1 &

# UP TO HERE DONE

nohup python test_dpgdln.py 50000 20000 30 2 1e1 > /dev/null 2>&1 &
nohup python test_dpgdln.py 50000 20000 30 2 1e0 > /dev/null 2>&1 &
nohup python test_dpgdln.py 50000 20000 30 2 1e-1 > /dev/null 2>&1 &

nohup python test_dpdln.py 50000 20000 30 2 1e1 > /dev/null 2>&1 &
nohup python test_dpdln.py 50000 20000 30 2 1e0 > /dev/null 2>&1 &
nohup python test_dpdln.py 50000 20000 30 2 1e-1 > /dev/null 2>&1 &

nohup python test_mpgln.py 50000 20000 30 10 > /dev/null 2>&1 &
nohup python test_mpgln.py 50000 20000 30 20 > /dev/null 2>&1 &
nohup python test_mpgln.py 50000 20000 30 30 > /dev/null 2>&1 &

nohup python test_mprgln.py 50000 20000 30 10 > /dev/null 2>&1 &
nohup python test_mprgln.py 50000 20000 30 20 > /dev/null 2>&1 &
nohup python test_mprgln.py 50000 20000 30 30 > /dev/null 2>&1 &

nohup python test_mgdln.py 50000 20000 30 10 > /dev/null 2>&1 &
nohup python test_mgdln.py 50000 20000 30 20 > /dev/null 2>&1 &
nohup python test_mgdln.py 50000 20000 30 30 > /dev/null 2>&1 &

nohup python test_mdln.py 50000 20000 30 10 > /dev/null 2>&1 &
nohup python test_mdln.py 50000 20000 30 20 > /dev/null 2>&1 &
nohup python test_mdln.py 50000 20000 30 30 > /dev/null 2>&1 &

# nohup mpiexec -np 5 python -m mpi4py test_dppgln.py 50000 20000 30 2 1e1 > /dev/null 2>&1 &
# nohup mpiexec -np 5 python -m mpi4py test_dppgln.py 50000 20000 30 2 1e0 > /dev/null 2>&1 &
# nohup mpiexec -np 5 python -m mpi4py test_dppgln.py 50000 20000 30 2 1e-1 > /dev/null 2>&1 &
#
# nohup mpiexec -np 5 python -m mpi4py test_dpprgln.py 50000 20000 30 2 1e1 > /dev/null 2>&1 &
# nohup mpiexec -np 5 python -m mpi4py test_dpprgln.py 50000 20000 30 2 1e0 > /dev/null 2>&1 &
# nohup mpiexec -np 5 python -m mpi4py test_dpprgln.py 50000 20000 30 2 1e-1 > /dev/null 2>&1 &

# nohup mpiexec -np 5 python -m mpi4py test_dpgdln.py 50000 20000 30 2 1e1 > /dev/null 2>&1 &
# nohup mpiexec -np 5 python -m mpi4py test_dpgdln.py 50000 20000 30 2 1e0 > /dev/null 2>&1 &
# nohup mpiexec -np 5 python -m mpi4py test_dpgdln.py 50000 20000 30 2 1e-1 > /dev/null 2>&1 &
#
# nohup mpiexec -np 5 python -m mpi4py test_dpdln.py 50000 20000 30 2 1e1 > /dev/null 2>&1 &
# nohup mpiexec -np 5 python -m mpi4py test_dpdln.py 50000 20000 30 2 1e0 > /dev/null 2>&1 &
# nohup mpiexec -np 5 python -m mpi4py test_dpdln.py 50000 20000 30 2 1e-1 > /dev/null 2>&1 &




# EOF
