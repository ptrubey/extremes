import numpy as np
import pandas as pd
import sqlite3 as sql

sql_paths = [
    # './simulated/sphere2/result_240227.sql',
    # './simulated/sphere2/result_240306.sql',
    # './simulated/sphere2/result_240305.sql',
    # './simulated/sphere2/result_240313.sql',
    # './simulated/sphere2/result_240314.sql',
    # './simulated/sphere2/result_240315.sql',
    # './simulated/sphere2/result_240317.sql',
    './simulated/sphere2/result_240324.sql',
    # './simulated/sphere2/result_240330.sql',
    # './simulated/sphere2/result_240331.sql',
    './simulated/sphere2/result_240404.sql',   # mean = -2, -2
    # './simulated/sphere2/result_240417.sql', # mean = -3, -2
    # './simulated/sphere2/result_240418.sql', # mean = -4, -3
    './simulated/sphere2/result_240710.sql',   # uniform
    './simulated/sphere2/result_240723.sql',   # pregamed
    './simulated/sphere2/result_240802.sql',   # pregamed2
    # './simulated/sphere2/result_240912.sql',   # MVarPYPG
    # './simulated/sphere2/result_240913.sql',   # MVarPYPG
    # './simulated/sphere2/result_240914.sql',   # MVarPYPG
    './simulated/sphere2/result_240918.sql',   # MVarPYPG
    ]

def get_table(path, table):
    with sql.connect(path) as conn:
        df = pd.read_sql('select * from {}'.format(table), con = conn)
    df['respath'] = path
    return df

dfs = [get_table(path, 'energy') for path in sql_paths]
dfs[0] = dfs[0][dfs[0].model != 'spypg']
dfs[0].model = 'MCMC'
dfs[1].model = 'VB Random'
dfs[2].model = 'VB Uniform'
dfs[3].model = 'VB Pregamed 1'
dfs[4].model = 'VB Pregamed 2'
dfs[5].model = 'VB Gibbs'
df = pd.concat(dfs)
df.to_csv('./simulated/sphere2/performance.csv', index = False)