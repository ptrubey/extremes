from argparser import argparser_generic as argparser
from projgamma import GammaPrior
import models_mpi as models
from data import Data_From_Raw
import pt_mpi as pt
import numpy as np
import os
from pandas import read_csv
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

pt.MPI_MESSAGE_SIZE = 2**24
args = argparser()

Chain = models.Chains[args.model]
Result = models.Results[args.model]

if rank > 0:
    chain = pt.PTSlave(comm = comm, statmodel = Chain)
    chain.watch()

if rank == 0:
    raw  = read_csv(args.in_path)
    data = Data_From_Raw(raw, decluster = args.decluster, quantile = args.quantile)

    if args.model.startswith('dp'):
        emp_path = os.path.join(
            args.out_folder, args.model, 'empirical.csv',
            )
        out_path = os.path.join(
            args.out_folder, args.model, 'results_{}_{}.db'.format(args.eta_shape, args.eta_rate),
            )
        pp_path = os.path.join(
            args.out_folder, args.model, 'postpred_{}_{}.csv'.format(args.eta_shape, args.eta_rate),
            )
        model = pt.PTMaster(
            comm,
            temperature_ladder = 1.05 ** np.array(range(size - 1)),
            data = data,
            prior_eta = GammaPrior(float(args.eta_shape), float(args.eta_rate))
            )
    elif args.model.startswith('m'):
        emp_path = os.path.join(
            args.out_folder, args.model, 'empirical.csv',
            )
        out_path = os.path.join(
            args.out_folder, args.model, 'results_{}.db'.format(args.nMix),
            )
        pp_path = os.path.join(
            args.out_folder, args.model, 'postpred_{}.csv'.format(args.nMix),
            )
        model = pt.PTMaster(
            comm,
            temperature_ladder = 1.05 ** np.array(range(size - 1)),
            data = data,
            nMix = int(args.nMix),
            )
    else:
        raise ValueError

    data.write_empirical(emp_path)
    model.sample(int(args.nSamp))
    model.write_to_disk(out_path, int(args.nKeep), int(args.nThin))
    model.complete()

    res = Result(out_path)
    res.write_posterior_predictive(pp_path)

# EOF
