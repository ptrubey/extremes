import sys
from simplex import *
from data import Data_From_Raw
from pandas import read_csv

path = './datasets/ivt_nov_mar.csv'
# cols = [int(x) for x in sys.argv[1:]]

col_idx = [int(x) for x in sys.argv[1:]]

for x in sys.argv[1:]:
    print(x)

if True:
    emp_path = './output/fmix_3d/empirical_{}_{}_{}.csv'.format(*col_idx)
    res_path = './output/fmix_3d/results_{}_{}_{}.db'.format(*col_idx)
    out_path = './output/fmix_3d/postpred_{}_{}_{}.csv'.format(*col_idx)

    raw = read_csv(path).iloc[:,col_idx]
    data = Data_From_Raw(raw, True)
    data.write_empirical(emp_path.format(*col_idx))

    fmix = FMIX_Chain(data, 10, GammaPrior(0.1,0.1), DirichletPrior(1.))
    fmix.sample(50000)
    fmix.write_to_disk(res_path, 25000, 5)
    res = FMIX_Result(res_path)
    res.write_posterior_predictive(out_path)

# EOF
