from dp_pgln2 import DPMPG_Chain, DPMPG_Result
from projgamma import GammaPrior
import pt_mpi as pt
# import pt
import numpy as np
from data import Data_From_Raw
from pandas import read_csv
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
pt.MPI_MESSAGE_SIZE = 2**20

if rank > 0:
    chain = pt.PTSlave(comm = comm, statmodel = DPMPG_Chain)
    chain.watch()

if rank == 0:
    raw  = read_csv('./datasets/ivt_nov_mar.csv')
    data = Data_From_Raw(raw, True)
    data.write_empirical('./output/dppgln2/empirical.csv')

    model = pt.PTMaster(
        comm,
        temperature_ladder = 1.05 ** np.array(range(size - 1)),
        data = data,
        prior_eta = GammaPrior(2.,.1)
        )
    model.sample(10000)
    model.write_to_disk('./output/dppgln2/results_2_1e-1.db', 5000, 1)
    model.complete()

    # res = DPMPG_Result('./output/dppgln/results_2_1e-1.db') # not working yet.
    # res.write_posterior_predictive('./output/dppgln/postpred_2_1e-1.csv')

## relevant to non-MPI version
# if __name__ == '__main__':
#     raw  = read_csv('./datasets/ivt_nov_mar.csv')
#     data = Data_From_Raw(raw, True)
#     data.write_empirical('./output/dppgln2/empirical.csv')
#
#     model = pt.PTMaster(
#         statmodel = DPMPG_Chain,
#         temperature_ladder = 1.3 ** np.array(range(4)),
#         data = data,
#         prior_eta = GammaPrior(2.,.1),
#         )
#     model.sample(10)
#     model.write_to_disk('./output/dppgln2/results_2_1e-1.db', 5, 1)
#
#     res = DPMPG_Result('./output/dppgln2/results_2_1e-1.db')
#     res.write_posterior_predictive('./output/dppgln2/postpred.csv')

# EOF
