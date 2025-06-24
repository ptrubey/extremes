import numpy as np
import pandas as pd
from data import Data_From_Raw as Data

paths = [
    './datasets/ad_cardio_x.csv',
    './datasets/ad_cover_x.csv',
    './datasets/ad_mammography_x.csv',
    './datasets/ad_pima_x.csv',
    './datasets/ad_satellite_x.csv',
    ]

def generate_data(path, decluster = False):
    raw = pd.read_csv(path).values
    data = Data(raw, decluster)
    return data


# EOF
