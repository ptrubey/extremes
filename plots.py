import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

hist_options = {'density' : True, 'stacked' : True, 'histtype' : 'stepfilled'}

def histogram(figure, series):
    pass

def stacked_histogram(figure, series, bins):
    plt.hist(series, bins = bins, **hist_options)
    return

def stacked_histogram_grid(figure, data_1, data_2):
    pass




# EOF
