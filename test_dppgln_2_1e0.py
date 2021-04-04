from model_pgln import DPPGLN_Chain, DPPGLN_Result
from projgamma import GammaPrior
# import pt_mpi as pt
import pt
import numpy as np
from data import Data_From_Raw
from pandas import read_csv
from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
rank = 0
size = 5

pt.MPI_MESSAGE_SIZE = 2**20

# if rank > 0:
#     chain = pt.PTSlave(comm = comm, statmodel = DPPGLN_Chain)
#     chain.watch()

if rank == 0:
    raw  = read_csv('./datasets/ivt_nov_mar.csv')
    data = Data_From_Raw(raw, True)
    data.write_empirical('./output/dppgln/empirical.csv')

    model = pt.PTMaster(
        # comm,
        statmodel = DPPGLN_Chain,
        temperature_ladder = 1.05 ** np.array(range(size - 1)),
        data = data,
        prior_eta = GammaPrior(2.,1e0)
        )
    model.sample(20000)
    model.write_to_disk('./output/dppgln/results_2_1e0.db', 10000, 2)
    model.complete()

    res = DPPGLN_Result('./output/dppgln/results_2_1e0.db')
    res.write_posterior_predictive('./output/dppgln/postpred_2_1e0.csv')

# EOF
