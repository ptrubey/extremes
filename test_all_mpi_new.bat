REM DOING EVERYTHING

start /low cmd /k mpiexec -np 5 python -m mpi4py test_generic_mpi.py ./datasets/ivt_updated_nov_mar.csv ./output2 dpdln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1
start /low cmd /k mpiexec -np 5 python -m mpi4py test_generic_mpi.py ./datasets/ivt_updated_nov_mar.csv ./output2 dpgdln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1
start /low cmd /k mpiexec -np 5 python -m mpi4py test_generic_mpi.py ./datasets/ivt_updated_nov_mar.csv ./output2 dppgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1
start /low cmd /k mpiexec -np 5 python -m mpi4py test_generic_mpi.py ./datasets/ivt_updated_nov_mar.csv ./output2 dpprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1
start /low cmd /k mpiexec -np 5 python -m mpi4py test_generic_mpi.py ./datasets/ivt_updated_nov_mar.csv ./output2 mdln 50000 20000 30 --nMix 30
start /low cmd /k mpiexec -np 5 python -m mpi4py test_generic_mpi.py ./datasets/ivt_updated_nov_mar.csv ./output2 mgdln 50000 20000 30 --nMix 30
start /low cmd /k mpiexec -np 5 python -m mpi4py test_generic_mpi.py ./datasets/ivt_updated_nov_mar.csv ./output2 mpgln 50000 20000 30 --nMix 30
start /low cmd /k mpiexec -np 5 python -m mpi4py test_generic_mpi.py ./datasets/ivt_updated_nov_mar.csv ./output2 mprgln 50000 20000 30 --nMix 30

REM EOF
