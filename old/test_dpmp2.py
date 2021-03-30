from dp_mprobit2 import DPMP, ResultDPMP, Data_From_Raw
# from data import Data_From_Raw
from pandas import read_csv
from random import shuffle

path = './datasets/ivt_nov_mar.csv'

if __name__ == '__main__':
    raw  = read_csv(path)
    # cols = raw.columns.values.tolist()
    # shuffle(cols)
    # raw  = raw.reindex(columns = cols)
    data = Data_From_Raw(raw, True)
    data.write_empirical('./output/dpmp/empirical.csv')

    dpmp = DPMP(data)
    dpmp.sample(10000)
    dpmp.write_to_disk('./output/dpmp/results.db', 5001,1)

    res = ResultDPMP('./output/dpmp/results.db')
    res.write_posterior_predictive('./output/dpmp/postpred.csv')

# EOF
