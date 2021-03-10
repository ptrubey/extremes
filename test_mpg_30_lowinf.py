from m_projgamma import MPG, MPGResult, GammaPrior, DirichletPrior
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

    mpg = MPG(
        data = data,
        nMix = 30,
        prior_alpha = GammaPrior(0.1,0.1),
        prior_beta = GammaPrior(0.1,0.1),
        prior_eta = DirichletPrior(0.25),
        )
    mpg.sample(500000)
    mpg.write_to_disk('./output/mpg/results_30_lowinf.db', 400000, 20)

    res = MPGResult('./output/mpg/results_30_lowinf.db')
    res.write_posterior_predictive('./output/mpg/postpred_30_lowinf.csv')

# EOF
