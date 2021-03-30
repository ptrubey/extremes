from model_dirichlet import *
from data import Data_From_Raw
from pandas import read_csv

path = './datasets/ivt_nov_mar.csv'

if __name__ == '__main__':
    raw = read_csv(path)
    data = Data_From_Raw(raw, True)
    data.write_empirical('./output/md/empirical.csv')

    model = MD_Chain(data, nMix = 20)
    model.sample(20000)
    model.write_to_disk('./output/md/results_20.db', 10000, 2)
    res = MD_Result('./output/md/results_20.db')
    res.write_posterior_predictive('./output/md/postpred_20.csv')

# EOF
