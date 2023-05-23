# Compute Distance / Divergence between Distributions
import numpy as np
import pandas as pd
import os, glob
from collections import namedtuple
from energy import energy_score_full
from models import Results

np.seterr(under = 'ignore')
epsilon = 1e-30

class PostPredLoss(object):
    def energy_score_Linf(self):
        Vnew = self.generate_posterior_predictive_hypercube()
        return energy_score_full(Vnew, self.data.V)

def ResultFactory(modelname, path):
    class Result(Results[modelname], PostPredLoss):
        pass
    return Result(path)

PPLResult = namedtuple('PPLResult', 'model source es')

def ppl_generation(model):
    result = ResultFactory(*model)
    pplr = PPLResult(
        model[0],
        os.path.splitext(os.path.split(model[1])[1])[0],
        result.energy_score_Linf(),
        )
    return pplr

if __name__ == '__main__':
    model_types = ['sdppprg','sdppprgln','sdpppg','sdpppgln']
    base_path = './output/new'

    models = []
    for model_type in model_types:
        results = glob.glob(
            os.path.join(base_path, '*_result_{}.pkl'.format(model_type)),
            )
        for result in results:
            models.append((model_type, result))

    pplrs = []
    for model in models:
        print('Processing model {}   '.format(model[0]), end = ' ')
        pplrs.append(ppl_generation(model))
    
    df = pd.DataFrame(pplrs)
    df.to_csv(os.path.join(base_path, 'energy_score.csv'), index = False)

# EOF
