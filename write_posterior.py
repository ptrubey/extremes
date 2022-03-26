from models import Results
import pandas as pd
import numpy as np

if __name__ == "__main__":

    in_paths = [
        './test/results_mdpppg.pkl',
        './test/results_mdppprg.pkl',
        ]
    out_paths = [
        './test/postpred_mdpppg.csv',
        './test/postpred_mdppprg.csv',
        ]
    models = [
        'mdpppg',
        'mdppprg',
        ]
    for in_path, out_path, model in zip(in_paths, out_paths, models):
        result = Results[model](in_path)
        postpred = result.generate_posterior_predictive_gammas()
        pd.DataFrame(postpred).to_csv(out_path, index = False)
        pass
    




    pass

