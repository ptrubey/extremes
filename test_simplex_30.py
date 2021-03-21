from simplex import *
from data import Data_From_Raw
from pandas import read_csv

path = './datasets/ivt_nov_mar.csv'

if __name__ == '__main__':
    raw = read_csv(path)
    data = Data_From_Raw(raw, True)
    data.write_empirical('./output/fmix/empirical.csv')

    fmix = FMIX_Chain(
            data, 30,
            GammaPrior(0.5,0.5),
            GammaPrior(2.,2.),
            DirichletPrior(1.),
            )
    fmix.sample(1000)
    fmix.write_to_disk('./output/fmix/test.db',500,5)

    #res = FMIX_Result('./output/fmix/test.db')
    #res.write_posterior_predictive('./output/fmix/test.csv')


# EOF
