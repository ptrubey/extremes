from argparser import argparser_dp as argparser
from model_projresgamma import DPPRG_Chain as Chain, DPPRG_Result as Result
from data import Data_From_Raw
from pandas import read_csv

model_type  = 'dpprg'
default_in  = './datasets/ivt_nov_mar.csv'
default_emp = os.path.join('./results', model_type, 'empirical.csv')
out_base    = os.path.join('./results', model_type, 'results_{}_{}.db')
pp_base     = os.path.join('./results', model_type, 'postpred_{}_{}.csv')

if __name__ == '__main__':
    args = argparser()

    raw  = read_csv(default_in)
    data = Data_From_Raw(raw, True)
    data.write_empirical(default_emp)

    model = Chain(
            data,
            prior_eta = GammaPrior(double(args.eta_shape), double(args.eta_rate)),
            )
    model.sample(int(args.nSamp))

    out_path = out_base.format(args.eta_shape, args.eta_rate)
    model.write_to_disk(out_path, int(args.nKeep), int(args.nThin))
    res = Result(out_path)

    pp_path = pp_base.format(args.eta_shape, args.eta_rate)
    res.write_posterior_predictive(pp_path)

# EOF
