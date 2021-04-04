from argparser import argparser_fm as argparser
from model_projgamma import MPG_Chain as Chain, MPG_Result as Result
from data import Data_From_Raw
from pandas import read_csv

model_type  = 'mpg'
default_in  = './datasets/ivt_nov_mar.csv'
default_emp = os.path.join('./results', model_type, 'empirical.csv')
out_base    = os.path.join('./results', model_type, 'results_{}.db')
pp_base     = os.path.join('./results', model_type, 'postpred_{}.csv')

if __name__ == '__main__':
    args = argparser()

    raw  = read_csv(default_in)
    data = Data_From_Raw(raw, True)
    data.write_empirical(default_emp)

    model = Chain(data, nMix = int(args.nMix))
    model.sample(int(args.nSamp))

    out_path = out_base.format(args.nMix)
    model.write_to_disk(out_path, int(args.nKeep), int(args.nThin))
    res = Result(out_path)

    pp_path = pp_base.format(args.nMix)
    res.write_posterior_predictive(pp_path)

# EOF
