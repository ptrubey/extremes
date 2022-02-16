from argparser import argparser_generic as argparser
from data import Data_From_Raw
from projgamma import GammaPrior
from pandas import read_csv
import models
import os

class Object(object):
    pass

if __name__ == '__main__':
    # p = argparser()
    p = Object()

    p.model = "dpppgln"
    p.out_folder = "./output2"
    p.in_path = './datasets/ivt_updated_nov_mar.csv'
    p.decluster = 'True'
    p.quantile = '0.95'
    p.eta_shape = '2'
    p.eta_rate = '1e0'
    p.nSamp = '5000'
    p.nKeep = '2000'
    p.nThin = '3'

    Chain  = models.Chains[p.model]
    Result = models.Results[p.model]

    raw  = read_csv(p.in_path)
    # so nice, I tried it twice.
    try:
        data = Data_From_Raw(raw, decluster = eval(p.decluster), quantile = float(p.quantile))
    except:
        data = Data_From_Raw(raw, decluster = eval(p.decluster), quantile = float(p.quantile))

    if p.model.startswith('dp'):
        emp_path = os.path.join(
            p.out_folder, p.model, 'empirical.csv',
            )
        out_path = os.path.join(
            p.out_folder, p.model, 'results_{}_{}.db'.format(p.eta_shape, p.eta_rate),
            )
        pp_path = os.path.join(
            p.out_folder, p.model, 'postpred_{}_{}.csv'.format(p.eta_shape, p.eta_rate),
            )
        model = Chain(data, prior_eta = GammaPrior(float(p.eta_shape), float(p.eta_rate)))

    elif p.model.startswith('m'):
        emp_path = os.path.join(
            p.out_folder, p.model, 'empirical.csv',
            )
        out_path = os.path.join(
            p.out_folder, p.model, 'results_{}.db'.format(p.nMix),
            )
        pp_path = os.path.join(
            p.out_folder, p.model, 'postpred_{}.csv'.format(p.nMix),
            )
        model = Chain(data, nMix = int(p.nMix))
    elif p.model.startswith('v'):
        emp_path = os.path.join(
            p.out_folder, p.model, 'empirical.csv',
            )
        out_path = os.path.join(
            p.out_folder, p.model, 'results.db',
            )
        pp_path = os.path.join(
            p.out_folder, p.model, 'postpred.csv',
            )
        model = Chain(data)
    else:
        raise ValueError

    data.write_empirical(emp_path)

    model.sample(int(p.nSamp))

    model.write_to_disk(out_path, int(p.nKeep), int(p.nThin))

    res = Result(out_path)

    res.write_posterior_predictive(pp_path)

# EOF
