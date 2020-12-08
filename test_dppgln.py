from dp_pgln import DPMPG_Chain, ResultDPMPG
import pt_mpi as pt
# import pt
import numpy as np
from data import Data_From_Raw
from pandas import read_csv
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# rank = 0
# size = 5

pt.MPI_MESSAGE_SIZE = 2**20

if rank > 0:
    chain = pt.PTSlave(comm = comm, statmodel = DPMPG_Chain)
    chain.watch()

if rank == 0:
    raw  = read_csv('./datasets/ivt_nov_mar.csv')
    data = Data_From_Raw(raw, True)
    data.write_empirical('./output/dpmpg_empirical_decluster.csv')

    model = pt.PTMaster(
        comm,
        # statmodel = DPMPG_Chain,
        temperature_ladder = 1.05 ** np.array(range(size - 1)),
        data = data,
        )
    model.sample(20000)
    model.write_to_disk('./output/dppgln_results_decluster2.db', 10000, 1)
    model.complete()

#
# if __name__ == '__main__':
#     raw  = read_csv('./datasets/ivt_nov_mar.csv')
#     data = Data_From_Raw(raw, True)
#     data.write_empirical('./output/dpmpg_empirical_decluster.csv')
#
#     model = pt.PTMaster(
#         statmodel = DPMPG_Chain,
#         temperature_ladder = 1.3 ** np.array(range(4)),
#         data = data,
#         )
#     model.sample(10000)
#     model.write_to_disk('./output/dppgln_results_decluster2.db', 5000, 1)

    # res = ResultDPMPG('./output/dpmpg_results_decluster.db')
    # res.write_posterior_predictive('./output/dpmpg_postpred_decluster.csv')

# EOF
