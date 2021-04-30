REM DOING EVERYTHING

REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 dpdln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1
REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 dpgdln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1
REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 dppgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1
REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 dpprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1
REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 mdln 50000 20000 30 --nMix 30
REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 mgdln 50000 20000 30 --nMix 30
REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 mpgln 50000 20000 30 --nMix 30
REM start /low cmd /k mpiexec -np 5 python -m mpi4py test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 mprgln 50000 20000 30 --nMix 30

REM EOF
