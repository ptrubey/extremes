ECHO "doing everything"

REM start /low cmd /k python test_dpd.py 20000 10000 2 2 1e1
REM start /low cmd /k python test_dpd.py 20000 10000 2 2 1e0
REM start /low cmd /k python test_dpd.py 20000 10000 2 2 1e-1

REM start /low cmd /k python test_dpgd.py 20000 10000 2 2 1e1
REM start /low cmd /k python test_dpgd.py 20000 10000 2 2 1e0
REM start /low cmd /k python test_dpgd.py 20000 10000 2 2 1e-1

REM start /low cmd /k python test_dppg.py 20000 10000 2 2 1e1
REM start /low cmd /k python test_dppg.py 20000 10000 2 2 1e0
REM start /low cmd /k python test_dppg.py 20000 10000 2 2 1e-1

REM start /low cmd /k python test_dpprg.py 20000 10000 2 2 1e1
REM start /low cmd /k python test_dpprg.py 20000 10000 2 2 1e0
REM start /low cmd /k python test_dpprg.py 20000 10000 2 2 1e-1

REM start /low cmd /k python test_md.py 20000 10000 2 10
REM start /low cmd /k python test_md.py 20000 10000 2 20
REM start /low cmd /k python test_md.py 20000 10000 2 30

REM start /low cmd /k python test_mgd.py 20000 10000 2 10
REM start /low cmd /k python test_mgd.py 20000 10000 2 20
REM start /low cmd /k python test_mgd.py 20000 10000 2 30

REM start /low cmd /k python test_mpg.py 20000 10000 2 10
REM start /low cmd /k python test_mpg.py 20000 10000 2 20
REM start /low cmd /k python test_mpg.py 20000 10000 2 30

REM start /low cmd /k python test_mprg.py 20000 10000 2 10
REM start /low cmd /k python test_mprg.py 20000 10000 2 20
REM start /low cmd /k python test_mprg.py 20000 10000 2 30

REM start /low cmd /k python test_dppgln.py 20000 10000 2 2 1e1
REM start /low cmd /k python test_dppgln.py 20000 10000 2 2 1e0
REM start /low cmd /k python test_dppgln.py 20000 10000 2 2 1e-1

REM start /low cmd /k python test_dpprgln.py 20000 10000 2 2 1e1
REM start /low cmd /k python test_dpprgln.py 20000 10000 2 2 1e0
REM start /low cmd /k python test_dpprgln.py 20000 10000 2 2 1e-1

REM start /low cmd /k python test_dpgdln.py 20000 10000 2 2 1e1
REM start /low cmd /k python test_dpgdln.py 20000 10000 2 2 1e0
REM start /low cmd /k python test_dpgdln.py 20000 10000 2 2 1e-1

REM start /low cmd /k python test_dpdln.py 20000 10000 2 2 1e1
REM start /low cmd /k python test_dpdln.py 20000 10000 2 2 1e0
REM start /low cmd /k python test_dpdln.py 20000 10000 2 2 1e-1

REM start /low cmd /k python test_mpgln.py 20000 10000 2 10
REM start /low cmd /k python test_mpgln.py 20000 10000 2 20
REM start /low cmd /k python test_mpgln.py 20000 10000 2 30

REM start /low cmd /k python test_mprgln.py 20000 10000 2 10
REM start /low cmd /k python test_mprgln.py 20000 10000 2 20
REM start /low cmd /k python test_mprgln.py 20000 10000 2 30

REM start /low cmd /k python test_mgdln.py 20000 10000 2 10
REM start /low cmd /k python test_mgdln.py 20000 10000 2 20
REM start /low cmd /k python test_mgdln.py 20000 10000 2 30

REM start /low cmd /k python test_mdln.py 20000 10000 2 10
REM start /low cmd /k python test_mdln.py 20000 10000 2 20
REM start /low cmd /k python test_mdln.py 20000 10000 2 30

REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_dppgln.py 20000 10000 2 2 1e1
REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_dppgln.py 20000 10000 2 2 1e0
REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_dppgln.py 20000 10000 2 2 1e-1

REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_dpprgln.py 20000 10000 2 2 1e1
REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_dpprgln.py 20000 10000 2 2 1e0
REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_dpprgln.py 20000 10000 2 2 1e-1

REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_dpgdln.py 20000 10000 2 2 1e1
REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_dpgdln.py 20000 10000 2 2 1e0
REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_dpgdln.py 20000 10000 2 2 1e-1

REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_dpdln.py 20000 10000 2 2 1e1
REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_dpdln.py 20000 10000 2 2 1e0
REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_dpdln.py 20000 10000 2 2 1e-1

REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_mpgln.py 20000 10000 2 10
REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_mpgln.py 20000 10000 2 20
REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_mpgln.py 20000 10000 2 30

REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_mprgln.py 20000 10000 2 10
REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_mprgln.py 20000 10000 2 20
REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_mprgln.py 20000 10000 2 30

REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_mgdln.py 20000 10000 2 10
REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_mgdln.py 20000 10000 2 20
REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_mgdln.py 20000 10000 2 30

REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_mdln.py 20000 10000 2 10
REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_mdln.py 20000 10000 2 20
REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_mdln.py 20000 10000 2 30

start /low cmd /k python test_dppn.py 20000 10000 2 2 1e1
start /low cmd /k python test_dppn.py 20000 10000 2 2 1e0
start /low cmd /k python test_dppn.py 20000 10000 2 2 1e-1



REM EOF