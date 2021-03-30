from simplex import *
from data import Data_From_Raw
from pandas import read_csv

path = './datasets/ivt_nov_mar.csv'

if __name__ == '__main__':
    raw = read_csv(path)
    data = Data_From_Raw(raw, True)
    data.write_empirical('./output/dpmix/empirical.csv')

    model = DPSimplex_Chain(
            data,
            GammaPrior(0.5, 0.5,),
            GammaPrior(2., 2.,),
            GammaPrior(2., 1e-1)
            )
    model.sample(1000)
    model.write_to_disk('./output/dpmix/test.db', 500, 5)
    res = DPSimplex_Result('./output/dpmix/test.db')
    res.write_posterior_predictive('./output/dpmix/test.csv')

# EOF
