from dp_projgamma import DPMPG, ResultDPMPG
from projgamma import GammaPrior
from data import Data_From_Raw
from pandas import read_csv
from random import shuffle


path = './datasets/ivt_nov_mar.csv'


if __name__ == '__main__':
    raw  = read_csv(path)
    data = Data_From_Raw(raw, True)
    data.write_empirical('./output/dpmpg/empirical.csv')

    dpmpg = DPMPG(
        data,
        fixed_eta = 200.
        )
    dpmpg.initialize_sampler(10000)
    dpmpg.sample(10000)
    dpmpg.write_to_disk('./output/dpmpg/results_fixed_high.db', 5000,1)

    res = ResultDPMPG('./output/dpmpg/results_fixed_high.db')
    res.write_posterior_predictive('./output/dpmpg/postpred_fixed_high.csv')

# EOF
