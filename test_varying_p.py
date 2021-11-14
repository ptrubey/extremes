from argparser import argparser_varying_p as argparser
from data import Data_From_Sphere as Data
from projgamma import GammaPrior
from pandas import read_csv
import models
import os

if __name__ == '__main__':
    p = argparser()

    Chain  = models.Chains[p.model]
    Result = models.Results[p.model]
    data   = Data(os.path.join(p.in_path, 'data.csv'))
    if p.model.startswith('dp'):
        out_path = os.path.join(
            p.in_path, p.model, 'results_{}.db'.format(p.p),
            )
        model = Chain(
            data,
            p = float(p.p),
            prior_eta = GammaPrior(float(p.eta_shape), float(p.eta_rate)),
            )
    else:
        raise ValueError

    model.sample(int(p.nSamp))
    model.write_to_disk(out_path, int(p.nKeep), int(p.nThin))

# EOF
