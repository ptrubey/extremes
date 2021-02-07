from simplex import *
from data import Data_From_Raw
from pandas import read_csv

path = './datasets/ivt_nov_mar.csv'

if __name__ == '__main__':
    # raw = read_csv(path)
    # data = Data_From_Raw(raw, True)
    # data.write_empirical('./output/fmix/empirical.csv')
    #
    # fmix = FMIX_Chain(
    #         data, 30,
    #         GammaPrior(1.,1.),
    #         DirichletPrior(1.),
    #         )
    # fmix.initialize_sampler(10000)
    # fmix.sample(10000)
    # fmix.write_to_disk('./output/fmix/results.db',5000,1)

    res = FMIX_Result('./output/fmix/results.db')
    res.write_posterior_predictive('./output/postpred.csv')


# EOF
