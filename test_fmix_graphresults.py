"""
Plot the
"""
import numpy as np
import pandas as pd
import itertools as it
from collections import namedtuple
import glob
import os
import plotly.express as px
from random import sample

from data import to_euclidean
from simplex import to_simplex

#---------------------------
IO_Path = namedtuple('IO_Path', 'emp pp out colnames')
#--------------------------
def polar_to_simplex(data):
    return to_simplex(to_euclidean(data))

if __name__ == "__main__":
    base_path = './output/fmix_3d'
    empirical_path = os.path.join(base_path, 'empirical_{}_{}_{}.csv')
    postpred_path  = os.path.join(base_path, 'postpred_{}_{}_{}.csv')
    out_path       = os.path.join(base_path, 'ternary_{}_{}_{}.png')
    cols = list(it.combinations(range(8), 3))

    paths = []

    for col_set in cols:
        if os.path.exists(postpred_path.format(*col_set)):
            paths.append(
                IO_Path(
                    empirical_path.format(*col_set),
                    postpred_path.format(*col_set),
                    out_path.format(*col_set),
                    {'a' : 'C_{}'.format(col_set[0]),
                     'b' : 'C_{}'.format(col_set[1]),
                     'c' : 'C_{}'.format(col_set[2])},
                )
            )

    for path in paths:
        emp = pd.DataFrame(polar_to_simplex(pd.read_csv(path.emp).values),
                            columns = [path.colnames[x] for x in ['a','b','c']])
        emp['source'] = 'Empirical'
        pp  = pd.DataFrame(polar_to_simplex(pd.read_csv(path.pp).values),
                            columns = [path.colnames[x] for x in ['a','b','c']])
        pp['source'] = 'PostPred'
        selected_rows = sample(range(pp.shape[0]), emp.shape[0])

        out_df = pd.concat([emp, pp.iloc[selected_rows,]], axis = 0)

        fig = px.scatter_ternary(out_df, color = 'source', **path.colnames)
        fig.write_image(path.out)

        del(fig)
        del(emp)
        del(pp)

# EOF
