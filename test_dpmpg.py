from dpm_projgamma import DPMPG
from projgamma import Data_From_Raw
from pandas import read_csv

path = './datasets/ivt_nov_mar.csv'

if __name__ == '__main__':
    raw  = read_csv(path)
    data = Data_From_Raw(raw)
    dpmpg = DPMPG(data)
    dpmpg.initialize_sampler(10000)
    dpmpg.sample(10000)
    # self = nppg
    # k = 1
    # self.samples_delta[k] = self.samples_delta[k-1]


# EOF
