from m_projgamma import MPG, MPGResult
from data import Data_From_Raw
from pandas import read_csv
from random import shuffle

path = './datasets/ivt_nov_mar.csv'

if __name__ == '__main__':
    raw  = read_csv(path)
    # cols = raw.columns.values.tolist()
    # shuffle(cols)
    # raw = raw.reindex(columns = cols)
    data = Data_From_Raw(raw, True)
    data.write_empirical('./output/mpg/empirical.csv')

    mpg = MPG(data, nMix = 30)
    mpg.initialize_sampler(10000)
    mpg.sample(10000)
    mpg.write_to_disk('./output/mpg/results_30.db', 5000, 1)

    res = MPGResult('./output/mpg/results_30.db')
    res.write_posterior_predictive('./output/mpg/postpred_30.csv')

# EOF
