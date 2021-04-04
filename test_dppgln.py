from argparser import argparser_dp as argparser
from model_pgln import DPPGLN_Chain as Chain, DPPGLN_Result as Result
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
# pt.MPI_MESSAGE_SIZE = 2**20

rank = 0
size = 5

model_type  = 'dppgln'
default_in  = './datasets/ivt_nov_mar.csv'
default_emp = os.path.join('./results', model_type, 'empirical.csv')
out_base    = os.path.join('./results', model_type, 'results_{}_{}.db')
pp_base     = os.path.join('./results', model_type, 'postpred_{}_{}.csv')

if rank > 0:
    chain = pt.PTSlave(comm = comm, statmodel = Chain)
    chain.watch()

if rank == 0:
    args = argparser()

    raw  = read_csv(default_in)
    data = Data_From_Raw(raw, True)
    data.write_empirical(default_emp)

    model = pt.PTMaster(
        # comm,
        statmodel = Chain,
        temperature_ladder = 1.05 ** np.array(range(size - 1)),
        data = data,
        prior_eta = GammaPrior(double(args.eta_shape), double(args.eta_rate))
        )
    model.sample(int(args.nSamp))
    out_path = out_base.format(args.eta_shape, args.eta_rate)
    model.write_to_disk(out_path, int(args.nKeep), int(args.nThin))
    model.complete()

    res = DPPGLN_Result(out_path)
    pp_path = pp_base.format(args.eta_shape, args.eta_rate)
    res.write_posterior_predictive(pp_path)

# EOF
