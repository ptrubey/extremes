from re import L
import time
import numpy as np

class DirichletProcessSampler(object):
    print_string_during = '\rSampling {:.1%} Completed in {}, {} Clusters'
    print_string_after  = '\rSampling 100% Completed in {}, {} Clusters Avg.'
    
    def time_elapsed(self):
        elapsed = time.time() - self.start_time
        if elapsed < 60:
            return '{:.0d} Seconds'.format(elapsed)
        elif elapsed < 3600: 
            return '{:.2d} Minutes'.format(elapsed / 60)
        else:
            return '{:.2d} Hours'.format(elapsed / 3600)
        pass

    def sample(self, ns):
        self.initialize_sampler(ns)
        self.start_time = time.time()
        print('Sampling 0% Completed', end = '')
        while (self.curr_iter < ns):
            if (self.curr_iter % 100) == 0:
                ps = self.print_string_during.format(
                    self.curr_iter / ns,
                    self.time_elapsed(),
                    self.curr_delta.max() + 1,
                    )
                print(ps, end = '')
                self.iter_sample()
        ps = self.print_string_after.format(
            self.time_elapsed, 
            self.samples.delta[(ns//2):].max(axis = 1).mean() + 1,
            )
        print(ps)
        return

# EOF
