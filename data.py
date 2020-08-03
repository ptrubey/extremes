""" class definition for data object -- read in either angular or real datas
and pre-compute relevant datasets for later use. """

class Data(object):
    """ Generic Data object--Assumes input path contains angular data """
    theta = None
    sins  = None
    coss  = None
    sinp  = None
    Yl    = None

    @staticmethod
    def euclidean(theta):
        """ casts angles in radians onto unit hypersphere in Euclidean space """
        coss = np.vstack((np.cos(theta).T, 1)).T
        sins = np.vstack((1, np.sin(theta).T)).T
        sinp = np.cumprod(sins, axis = 1)
        return coss * sinp

    @staticmethod
    def angular(y):
        """ casts data into angles (specifically radians) """
        n, k = y.shape
        theta = np.empty((n, k - 1)))
        for i in range(k - 1):
            theta[:,i] = arccos(y[:,i] / norm(y[:,i:], axis = 1))
        return theta

    def __fill_out(self):
        self.coss  = np.vstack((np.cos(self.theta).T, 1)).T
        self.sins  = np.vstack((1, np.sin(self.theta).T)).T
        self.sinp  = np.cumprod(self.sins, axis = 1)
        self.Yl    = self.coss * self.sinp
        return

    def __init__(self, path):
        self.theta = read.csv(path)
        self.__fill_out()
        return

class Data_From_Raw(Data):
    """ Data object -- assumes input path contains real data for which the
    marginal Pareto parameters (and thresholds) need to be estimated. """
    # Data Processing Steps
    raw = None   # Raw data
    Z   = None   # Standardized Pareto Marginals
    V   = None   # Z Projected onto Unit Hypercube
    # Marginal Pareto parameters
    a   = None   # Scale
    b   = None   # Threshold
    xi  = None   # Extremal Index

    @staticmethod
    def standardize(Y):
        """ Computes Estimates for Marginal Pareto Parameters, and returns data
        post standardization (values above 1 follow standard Pareto). returns 4
        values:  standardized data, threshold, scale, and extremal index for
        each column. """
        pass

    @staticmethod
    def cast_to_hypercube(Z):
        """ Computes row-max of Z, and divides row by row max. Returns those
        observations for which the row max was > 1 (extreme), along with a list
        of indices indicating what each row's index was in the original data. """
        pass

    def __init__(self, path):
        self.raw   = read.csv(path)
        self.Z, self.a, self.b, self.xi = self.standardize(self.raw)
        self.V, self.keep_idx = self.cast_to_hypercube(self.Z)
        self.theta = self.angular(V)
        self.__fill_out()
        return

# EOF
