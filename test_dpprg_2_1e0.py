from model_projresgamma import *
from data import Data_From_Raw
from pandas import read_csv

path = './datasets/ivt_nov_mar.csv'

if __name__ == '__main__':
    raw = read_csv(path)
    data = Data_From_Raw(raw, True)
    data.write_empirical('./output/dpprg/empirical.csv')

    model = DPPRG_Chain(
            data,
            prior_eta = GammaPrior(2., 1e0),
            )
    model.sample(20000)
    model.write_to_disk('./output/dpprg/results_2_1e0.db', 10000, 2)
    res = DPPRG_Result('./output/dpprg/results_2_1e0.db')
    res.write_posterior_predictive('./output/dpprg/postpred_2_1e0.csv')

# EOF
