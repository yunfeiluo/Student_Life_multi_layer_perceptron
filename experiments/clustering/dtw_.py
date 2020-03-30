import numpy as np

class DTW_clusters:
    def __init__(self, n_clusters, random_state, tol):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.tol = tol
        self.cluster_centers_ = list() # list of numpy array
    
    def fit(self, data):
        # random centers, test purpose
        np.random.seed(self.random_state)
        max_ts_len = max([len(i) for i in data])
        pt_dim = len(data[0][0])
        for i in range(self.n_clusters):
            len_ = np.random.randint(max_ts_len)
            #len_ = max_ts_len
            center = np.random.rand(len_, pt_dim)
            self.cluster_centers_.append(center)

        # TODO
        return self
    
    def predict(self, pts):
        res = list()
        for pt in pts:
            dists = [self.dtw_dist(c, pt) for c in self.cluster_centers_]
            res.append(np.argmin(dists))
        return res

    # helper functions
    def dist(self, p1, p2):
        return np.linalg.norm(p1-p2, ord=2)

    def dtw_dist(self, ts1, ts2):
        DTW = dict()
        DTW[(0, 0)] = 0
        
        for i in range(len(ts1)):
            for j in range(len(ts2)):
                if i == 0 and j == 0:
                    continue
                cost = self.dist(ts1[i], ts2[j])
                min_ = None
                if i - 1 >= 0 and j - 1 >= 0:
                    min_ = min(DTW[(i-1, j)], DTW[(i, j-1)], DTW[(i-1, j-1)])
                elif i - 1 >= 0:
                    min_ = DTW[(i-1, j)]
                elif j - 1 >= 0:
                    min_ = DTW[(i, j-1)]
                DTW[(i, j)] = cost + min_
        
        return DTW[(len(ts1) - 1, len(ts2) - 1)]
