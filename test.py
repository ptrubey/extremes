from mixture_projgamma import MPG
from projgamma import Data_From_Raw
from pandas import read_csv

path = './datasets/ivt_nov_mar.csv'

if __name__ == '__main__':
    raw  = read_csv(path)
    data = Data_From_Raw(raw)
    mpg = MPG(data, nMix = 20)
    mpg.initialize_sampler(10000)
    mpg.sample(10000)


# EOF
