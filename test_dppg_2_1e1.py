from model_projgamma import *
from data import Data_From_Raw
from pandas import read_csv

path = './datasets/ivt_nov_mar.csv'

if __name__ == '__main__':
    raw = read_csv(path)
    data = Data_From_Raw(raw, True)
    data.write_empirical('./output/dppg/empirical.csv')

    model = DPPG_Chain(
            data,
            prior_eta = GammaPrior(2., 1e1),
            )
    model.sample(20000)
    model.write_to_disk('./output/dppg/results_2_1e1.db', 10000, 2)
    res = DPPG_Result('./output/dppg/results_2_1e1.db')
    res.write_posterior_predictive('./output/dppg/postpred_2_1e1.csv')

# EOF
