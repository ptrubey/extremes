"""Varying p Results"""
import glob
import os
import pandas as pd

if __name__ == '__main__':
    results = []
    paths = glob.glob('./varying_p/*/post_pred_loss_results.csv')
    for path in paths:
        # Get the folder ID
        scenario = os.path.split(os.path.split(path)[0])[1]
        temp = pd.read_csv(path)
        temp['scenario'] = scenario
        results.append(temp)
    combined = pd.concat(results)
    combined.to_csv('./varying_p/post_pred_loss_results.csv', index = False)



# EOF
