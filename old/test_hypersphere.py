from hypersphere import *
from data import Data_From_Raw
from pandas import read_csv

path = './datasets/ivt_nov_mar.csv'

if __name__ == '__main__':
    raw = read_csv(path)
    data = Data_From_Raw(raw, True)
    # model = MPG_Chain(
    #         data,
    #         10,
    #         DirichletPrior(0.5),
    #         GammaPrior(0.5, 0.5),
    #         GammaPrior(2.0, 2.0),
    #         GammaPrior(0.5, 0.5),
    #         GammaPrior(2.0, 2.0),
    #         )
    # model.sample(20000)
    # model.write_to_disk('./output/mpg/results_test.db', 10000, 2)
    # res = MPG_Result('./output/mpg/results_test.db')
    # res.write_posterior_predictive('./output/dpmix/postpred_test.csv')
    # model = DPPG_Chain(
    #         data,
    #         GammaPrior(2., 0.5),
    #         )
    # model.sample(20000)
    # model.write_to_disk('./output/dpmpg/results_test.db', 10000, 2)
    res = DPPG_Result('./output/dpmpg/results_test.db')
    res.write_posterior_predictive('./output/dpmpg/postpred_test.csv')

# EOF
