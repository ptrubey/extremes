from model_dirichlet import *
from data import Data_From_Raw
from pandas import read_csv

path = './datasets/ivt_nov_mar.csv'

if __name__ == '__main__':
    raw = read_csv(path)
    data = Data_From_Raw(raw, True)
    data.write_empirical('./output/dpd/empirical.csv')

    model = DPD_Chain(
            data,
            prior_eta = GammaPrior(2., 1e-1),
            )
    model.sample(20000)
    model.write_to_disk('./output/dpd/results_2_1e-1.db', 10000, 2)
    res = DPD_Result('./output/dpd/results_2_1e-1.db')
    res.write_posterior_predictive('./output/dpd/postpred_2_1e-1.csv')

# EOF
