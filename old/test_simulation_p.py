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
        emp_path = os.path.join(p.in_path, p.model, 'empirical.csv')
        out_path = os.path.join(
            p.in_path, p.model, 'results_{}.db'.format(p.p),
            )
        pp_path = os.path.join(
            p.in_path, p.model, 'postpred_{}.csv'.format(p.p),
            )
        model = Chain(
            data, p = float(p.p), prior_eta = GammaPrior(float(p.eta_shape), float(p.eta_rate)),
            )
    else:
        raise ValueError

    data.write_empirical(emp_path)

    model.sample(int(p.nSamp))

    model.write_to_disk(out_path, int(p.nKeep), int(p.nThin))

    res = Result(out_path)

    res.write_posterior_predictive(pp_path)

# EOF
