import glob, os, pickle, numpy, pandas
from collections import namedtuple

Result = namedtuple('Result', 'data app time')
paths = glob.glob("./ad/*/*_results_1e-1_1e-1.pkl")

if __name__ == '__main__':
    results = []
    for path in paths:
        with open(path, 'rb') as file:
            res = pickle.load(file)
        time = res['time']
        dataset = os.path.split(os.path.split(path)[0])[1]
        application = os.path.split(path)[1].split('_', 1)[0]
        results.append(Result(dataset, application, time / 3600))
    
    pandas.DataFrame(results).to_csv('./ad/model_run_times.csv', index = False)


# EOF 