import subprocess
import glob
import os

if __name__ == '__main__':
    paths = glob.glob('./varying_p/*')
    for path in paths:
        subprocess.run(['python', './postpred_loss.py', path])
    
    pass 

# EOF
