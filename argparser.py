import argparse

def argparser_dp():
    p = argparse.ArgumentParser()
    p.add_argument('nSamp')
    p.add_argument('nKeep')
    p.add_argument('nThin')
    p.add_argument('eta_shape')
    p.add_argument('eta_rate')
    return p.parse_args()

def argparser_fm():
    p = argparse.ArgumentParser()
    p.add_argument('nSamp')
    p.add_argument('nKeep')
    p.add_argument('nThin')
    p.add_argument('nMix')
    return p.parse_args()

def argparser_v():
    p = argparse.ArgumentParser()
    p.add_argument('nSamp')
    p.add_argument('nKeep')
    p.add_argument('nThin')
    return p.parse_args()

def argparser_generic():
    p = argparse.ArgumentParser()
    p.add_argument('in_path')
    p.add_argument('out_folder')
    p.add_argument('model')
    p.add_argument('nSamp')
    p.add_argument('nKeep')
    p.add_argument('nThin')
    p.add_argument('--nMix')
    p.add_argument('--eta_shape')
    p.add_argument('--eta_rate')
    p.add_argument('--quantile', default = 0.95)
    return p.parse_args()

def argparser_ppl():
    p = argparse.ArgumentParser()
    p.add_argument('path')
    return p.parse_args()

if __name__ == '__main__':
    p = argparser_dp()
