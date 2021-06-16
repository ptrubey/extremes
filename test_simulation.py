from argparser import argparser_simulation as argparser
from simulate_data import Data
from projgamma import GammaPrior
from pandas import read_csv
import models
import os

if __name__ == '__main__':
    p = argparser()

    Chain  = models.Chains[p.model]
    Result = models.Results[p.model]

    data = Data(os.path.join(p.in_path, 'data.db'))

    if p.model.startswith('dp'):
        out_path = os.path.join(
            p.in_path, p.model, 'results_{}_{}.db'.format(p.eta_shape, p.eta_rate),
            )
        pp_path = os.path.join(
            p.in_path, p.model, 'postpred_{}_{}.csv'.format(p.eta_shape, p.eta_rate),
            )
        model = Chain(data, prior_eta = GammaPrior(float(p.eta_shape), float(p.eta_rate)))

    elif p.model.startswith('m'):
        out_path = os.path.join(
            p.in_path, p.model, 'results_{}.db'.format(p.nMix),
            )
        pp_path = os.path.join(
            p.in_path, p.model, 'postpred_{}.csv'.format(p.nMix),
            )
        model = Chain(data, nMix = int(p.nMix))
    elif p.model.startswith('v'):
        out_path = os.path.join(
            p.in_path, p.model, 'results.db',
            )
        pp_path = os.path.join(
            p.in_path, p.model, 'postpred.csv',
            )
        model = Chain(data)
    else:
        raise ValueError

    model.sample(int(p.nSamp))

    if not os.path.exists(os.path.split(out_path)[0]):
        os.mkdir(os.path.split(out_path)[0])
    model.write_to_disk(out_path, int(p.nKeep), int(p.nThin))

    res = Result(out_path)

    res.write_posterior_predictive(pp_path)

# EOF
