"""
samplers.py
--------------
Proto-classes for MCMC samplers.

both assume the existence of:
    obj.initialize_sampler(ns)
    obj.iter_sample()
    obj.curr_iter

DirichletProcessSampler assumes the existence of:
    obj.samples.delta
    obj.curr_delta
"""
import time
import numpy as np

class BaseSampler(object):
    print_string_during = '\rSampling {:.1%} Completed in {}'
    print_string_after = '\rSampling 100% Completed in {}'

    @property
    def time_elapsed(self):
        """ returns current time elapsed since sampling start in human readable format """
        elapsed = time.time() - self.start_time
        if elapsed < 60:
            return '{:.0f} Seconds'.format(elapsed)
        elif elapsed < 3600: 
            return '{:.2f} Minutes'.format(elapsed / 60)
        else:
            return '{:.2f} Hours'.format(elapsed / 3600)
        pass

    def sample(self, ns):
        """ Run the Sampler """
        self.initialize_sampler(ns)
        self.start_time = time.time()
        
        print('\rSampling 0% Completed', end = '')

        while (self.curr_iter < ns):
            if (self.curr_iter % 100) == 0:
                ps = self.print_string_during.format(self.curr_iter / ns, self.time_elapsed)
                print(ps.ljust(80), end = '')
            self.iter_sample()
        
        ps = self.print_string_after.format(self.time_elapsed)
        print(ps)
        return

class DirichletProcessSampler(BaseSampler):
    print_string_during = '\rSampling {:.1%} Completed in {}, {} Clusters'
    print_string_after  = '\rSampling 100% Completed in {}, {} Clusters Avg.'
    
    @property
    def curr_cluster_count(self):
        """ Returns current cluster count """
        return self.curr_delta.max() + 1
    
    def average_cluster_count(self, ns):
        acc = self.samples.delta[(ns//2):].max(axis = 1).mean() + 1
        return '{:.2f}'.format(acc)

    def sample(self, ns):
        """ Run the sampler """
        self.initialize_sampler(ns)
        self.start_time = time.time()
        
        print('\rSampling 0% Completed', end = '')
        
        while (self.curr_iter < ns):
            if (self.curr_iter % 100) == 0:
                ps = self.print_string_during.format(
                    self.curr_iter / ns, self.time_elapsed, self.curr_cluster_count,
                    )
                print(ps.ljust(80), end = '')
            self.iter_sample()
        
        ps = self.print_string_after.format(self.time_elapsed, self.average_cluster_count(ns))
        print(ps)
        return

# EOF
