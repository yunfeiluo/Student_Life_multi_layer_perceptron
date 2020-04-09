# ---------------------------------------------------------------- 
# Independent Study 496, Student Stree Prediction
#
# file_name: density_based_clustering.py
# Functionality: Class, density_based_clustering: do clustering based on density
# Author: Yunfei Luo
# Start date: EST Apr.9th.2020
# Last update: EST Apr.9th.2020
# ----------------------------------------------------------------

import numpy as np
from sklearn.cluster import OPTICS

class density_based_clustering:
    def __init__(self, eps, min_samples, cluster_method, metric):
        self.eps = eps
        self.min_samples = min_samples
        self.cluster_method = cluster_method # xi or dbscan
        self.metric = metric
        self.dist_matrix = list()
        self.labels_ = list()
    
    def compute_dist_matrix(self, pts):
        compute_dist = None
        if self.metric == 'l1':
            compute_dist = lambda x1, x2: np.linalg.norm(x1-x2, 1)
        elif self.metric == 'l2':
            compute_dist = lambda x1, x2: np.linalg.norm(x1-x2, 2)
        else:
            print('The distance computing type is not available yet...')
            exit()
        dist_matrix = list()
        for pt1 in pts:
            row = list()
            for pt2 in pts:
                row.append(np.linalg.norm(pt1 - pt2))
            dist_matrix.append(row)
        return np.array(dist_matrix)
    
    def cluster_outliers(self, outliers_ind):
        '''
        a. Put each outlier to the cluster where its closest points belongs to;
        b. Repeat a. until no change made
        c. Treat the rest outliers as in one group
        '''
        # Cluster the outliers that are close to the exist clusters
        hasChange = True
        while hasChange:
            hasChange = False
            for i in outliers_ind:
                if self.labels_[i] != -1:
                    continue
                min_ind = -1
                min_dist = np.inf
                for j in range(len(self.pts)):
                    if i != j:
                        if self.dist_matrix[i][j] < min_dist:
                            min_dist = self.dist_matrix[i][j]
                            min_ind = j
                if self.labels_[min_ind] != -1:
                    self.labels_[i] = self.labels_[min_ind]
                    hasChange = True
        
        # cluster the rest outliers
        outliers_ind = [i for i in outliers_ind if self.labels_[i] == -1]
        while len(outliers_ind) > 0:
            group_id = max(self.labels_) + 1
            i = outliers_ind[0]
            self.labels_[i] = group_id

            while True:
                # find closest node in outliers
                min_ind = -1
                min_dist = np.inf
                for j in outliers_ind:
                    if i != j:
                        if self.dist_matrix[i][j] < min_dist:
                            min_dist = self.dist_matrix[i][j]
                            min_ind = j
                
                # if the closest is grouped, break
                if len(outliers_ind) == 1:
                    self.labels_[i] = self.labels_[min_ind]
                    break
                elif self.labels_[min_ind] != -1:
                    break
                else:
                    self.labels_[min_ind] = group_id
                i = min_ind
            
            # update ouliers list (delete those with group)
            outliers_ind = [i for i in outliers_ind if self.labels_[i] == -1]
            

        '''
        ## pseudo-code
        while left > 0:
            choose the first node i, put it in a new group
            loop: 
                find i's closest node j
                if j haven't group
                    put it in the group
                else
                    break
                i = j
            update outliers_ind
        '''
                

    def fit(self, pts):
        # clustering
        self.pts = np.array(pts)
        if self.metric == 'precomputed':
            self.dist_matrix = self.pts
            clusters = OPTICS(min_samples=self.min_samples, max_eps=self.eps, cluster_method=self.cluster_method, metric='precomputed').fit(self.dist_matrix)
            self.labels_ = clusters.labels_
        else:
            self.dist_matrix = self.compute_dist_matrix(self.pts)
            clusters = OPTICS(min_samples=self.min_samples, max_eps=self.eps, cluster_method=self.cluster_method).fit(self.pts)
            self.labels_ = clusters.labels_

        # clustering outliers
        outliers_ind = [i for i in range(len(clusters.labels_)) if clusters.labels_[i] == -1]
        self.cluster_outliers(outliers_ind)
        
        return self
