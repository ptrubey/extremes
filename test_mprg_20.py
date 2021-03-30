from model_projresgamma import *
from data import Data_From_Raw
from pandas import read_csv

path = './datasets/ivt_nov_mar.csv'

if __name__ == '__main__':
    raw = read_csv(path)
    data = Data_From_Raw(raw, True)
    data.write_empirical('./output/mprg/empirical.csv')

    model = MPRG_Chain(data, nMix = 20)
    model.sample(20000)
    model.write_to_disk('./output/mprg/results_20.db', 10000, 2)
    res = MPRG_Result('./output/mprg/results_20.db')
    res.write_posterior_predictive('./output/mprg/postpred_20.csv')

# EOF
