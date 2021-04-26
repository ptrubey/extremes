from argparser import argparser_v as argparser
from model_dirichlet import VD_Chain as Chain, VD_Result as Result
from data import Data_From_Raw
from pandas import read_csv
import os

model_type  = 'vd'
default_in  = './datasets/ivt_nov_mar.csv'
default_emp = os.path.join('./output', model_type, 'empirical.csv')
out_base    = os.path.join('./output', model_type, 'results.db')
pp_base     = os.path.join('./output', model_type, 'postpred.csv')

if __name__ == '__main__':
    args = argparser()

    raw  = read_csv(default_in)
    data = Data_From_Raw(raw, True)
    data.write_empirical(default_emp)

    model = Chain(data)
    model.sample(int(args.nSamp))

    out_path = out_base
    model.write_to_disk(out_path, int(args.nKeep), int(args.nThin))
    res = Result(out_path)

    pp_path = pp_base
    res.write_posterior_predictive(pp_path)

# EOF
