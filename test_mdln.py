from argparser import argparser_fm as argparser
from model_dln import MDLN_Chain as Chain, MDLN_Result as Result
# import pt_mpi as pt
import pt
import numpy as np
from data import Data_From_Raw
from pandas import read_csv
from mpi4py import MPI
import os
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
# pt.MPI_MESSAGE_SIZE = 2**20

rank = 0
size = 5

model_type  = 'mdln'
default_in  = './datasets/ivt_nov_mar.csv'
default_emp = os.path.join('./output', model_type, 'empirical.csv')
out_base    = os.path.join('./output', model_type, 'results_{}.db')
pp_base     = os.path.join('./output', model_type, 'postpred_{}.csv')

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
        nMix = int(args.nMix),
        )
    model.sample(int(args.nSamp))
    out_path = out_base.format(args.nMix)
    model.write_to_disk(out_path, int(args.nKeep), int(args.nThin))
    model.complete()

    res = Result(out_path)
    pp_path = pp_base.format(args.nMix)
    res.write_posterior_predictive(pp_path)

# EOF
