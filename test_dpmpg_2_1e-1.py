from dp_projgamma import DPMPG, ResultDPMPG
from projgamma import GammaPrior
from data import Data_From_Raw
from pandas import read_csv
from random import shuffle


path = './datasets/ivt_nov_mar.csv'


if __name__ == '__main__':
    raw  = read_csv(path)
    # cols = raw.columns.values.tolist()
    # shuffle(cols)
    # raw  = raw.reindex(columns = cols)
    data = Data_From_Raw(raw, True)
    data.write_empirical('./output/dpmpg/empirical.csv')

    dpmpg = DPMPG(
        data,
        prior_eta = GammaPrior(2.,0.1)
        )
    dpmpg.initialize_sampler(10000)
    dpmpg.sample(10000)
    dpmpg.write_to_disk('./output/dpmpg/results_2_1e-1.db', 5000,1)

    res = ResultDPMPG('./output/dpmpg/results_2_1e-1.db')
    res.write_posterior_predictive('./output/dpmpg/postpred_2_1e-1.csv')

# EOF
