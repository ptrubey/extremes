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
    # './simulated/sphere2/result_240404.sql', # mean = -2, -2
    # './simulated/sphere2/result_240417.sql', # mean = -3, -2
    # './simulated/sphere2/result_240418.sql', # mean = -4, -3
    './simulated/sphere2/result_240419.sql', # mean = -4, -2
    ]

def get_table(path, table):
    with sql.connect(path) as conn:
        df = pd.read_sql('select * from {}'.format(table), con = conn)
    df['respath'] = path
    return df

dfs = [get_table(path, 'energy') for path in sql_paths]
dfs[0] = dfs[0][dfs[0].model != 'spypg']
df = pd.concat(dfs)
df.to_csv('~/Desktop/performance.csv', index = False)
