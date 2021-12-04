from argparser import argparser_varying_p as argparser
from data import Data_From_Raw as Data
from projgamma import GammaPrior
from pandas import read_csv
import models
import os

if __name__ == '__main__':
    p = argparser()

    Chain  = models.Chains[p.model]
    Result = models.Results[p.model]
    raw  = read_csv(p.in_path).values
    data = Data(raw, decluster = True, quantile = 0.95)
    if p.model.startswith('dp'):
        emp_path = os.path.join(p.out_path, p.model, 'empirical.csv')
        out_path = os.path.join(
            p.out_path, p.model, 'results_{}.db'.format(p.p),
            )
        model = Chain(
            data,
            p = float(p.p),
            prior_eta = GammaPrior(float(p.eta_shape), float(p.eta_rate)),
            )
    else:
        raise ValueError

    data.write_empirical(emp_path)
    model.sample(int(p.nSamp))
    model.write_to_disk(out_path, int(p.nKeep), int(p.nThin))

# EOF
