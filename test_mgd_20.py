from model_gendirichlet import *
from data import Data_From_Raw
from pandas import read_csv

path = './datasets/ivt_nov_mar.csv'

if __name__ == '__main__':
    raw = read_csv(path)
    data = Data_From_Raw(raw, True)
    data.write_empirical('./output/mgd/empirical.csv')

    model = MGD_Chain(data, nMix = 20)
    model.sample(20000)
    model.write_to_disk('./output/mgd/results_20.db', 10000, 2)
    res = MGD_Result('./output/mgd/results_20.db')
    res.write_posterior_predictive('./output/mgd/postpred_20.csv')

# EOF
