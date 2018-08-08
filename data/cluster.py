import random

import numpy as np

"""
k-mediods algorithms
"""


class Clustering:
    """
    Init a new clustering from a given number of clusters and distance matrix
    :param k, integer, the number of clusters
    :param distMatrix, numpy array, the distance matrix
    """

    def __init__(self, k, distMatrix):
        self._distMatrix = distMatrix
        self._numClusters = k
        self._nsamples = distMatrix.shape[0]
        # find maximal distance between elements
        d = 0.0
        for i in range(0, self._nsamples):
            for j in range(i + 1, self._nsamples):
                if self._distMatrix[i, j] > d:
                    d = self._distMatrix[i, j]
        self._maxdist = d
        # start with order indices sequence
        self._indices = np.array(range(0, self._nsamples))
        # randomly select medioids indices
        for i in range(0, k):
            p = random.randint(0, self._nsamples - i - 1)
            # swap selected position and next medioid position
            q = self._indices[self._nsamples - i - 1]
            self._indices[self._nsamples - i - 1] = self._indices[p]
            self._indices[p] = q
        # value is the cluster number for the sample
        self._clusters = np.empty((self._nsamples), dtype=int)
        # initial clustering
        self._clusterize()

    """
    Returns the medioids, i.e. clusters centers
    """

    @property
    def medioids(self):
        return [self._indices[self._nsamples - i] for i in range(1, self._numClusters + 1)]

    """
    Returns the current distance matrix
    """

    @property
    def distanceMatrix(self):
        return self._distMatrix

    """
    Returns a list of couples with first element the sample index and second the cluster number
    """

    @property
    def clusters(self):
        return [(self._indices[i], self._clusters[i]) for i in range(0, self._nsamples)]

    """
    Computes distance between elements in the array holding indices
    """

    def _distance(self, i, j):
        return self._distMatrix[self._indices[i], self._indices[j]]

    """
      Computes distance between an element and a mediod
        :param i, index in the element array
        :param m, medioid number
    """

    def _distanceToMedioid(self, i, m):
        return self._distMatrix[self._indices[i], self._indices[self._nsamples - m]]

    """
    Update the clusters arrays
    """

    def _clusterize(self):
        # assign all non mediods to clusters
        for i in range(0, self._nsamples - self._numClusters):
            min = self._distanceToMedioid(i, 1)
            imin = 1
            for j in range(2, self._numClusters + 1):
                d = self._distanceToMedioid(i, j)
                if d < min:
                    min = d
                    imin = j
            # assign to cluster number j
            self._clusters[i] = imin
        # assign medioids to the cluster they define
        for i in range(1, self._numClusters+1):
            print(i)
            self._clusters[self._nsamples - i] = i

    """
    Returns the cost of replacing medioid m by candidate p
    for the non-medioid j
    """

    def _swapCost(self, m, p, j):
        cost = 0.0
        h = self._clusters[j]
        if h == m:
            # we are in the cluster defined by m
            # compute distances from j to all other mediods except m and get the smallest
            min = self._maxdist
            for k in range(1, m):
                d = self._distanceToMedioid(j, k)
                if d < min:
                    min = d
            for k in range(m + 1, self._numClusters):
                d = self._distanceToMedioid(j, k)
                if d < min:
                    min = d
            if min < self._distance(j, p):
                # j will change cluster
                cost = min - self._distanceToMedioid(j, m)
            else:
                # j will stay in its cluster
                cost = self._distance(j, p) - self._distanceToMedioid(j, m)
        else:
            # j is in another cluster
            d = self._distance(j, p) < self._distanceToMedioid(j, h)
            if d < 0.0:
                cost = d
        return cost

    """
    Returns the total cost for swapping medioid m and candidate p
    """

    def _totalSwapCost(self, m, p):
        cost = 0.0
        for j in range(0, p):
            cost += self._swapCost(m, p, j)
        for j in range(p + 1, self._nsamples - self._numClusters):
            cost += self._swapCost(m, p, j)
        return cost

    """
    Implements one iteration of the clustering algorithm
    returns true if a swapping was made false otherwise
    """

    def updateMedioids(self):
        swapMedioid = 1
        swapCandidate = 0
        swap = False
        # init with swapping cost of candidate 0 and first mediod
        d = self._totalSwapCost(1, 0)
        for j in range(1, self._nsamples - self._numClusters):
            # iterate over candidates
            dd = self._totalSwapCost(1, j)
            if dd < d:
                d = dd
                swapCandidate = j
                swapMedioid = 1
        # check other medioids
        for i in range(2, self._numClusters + 1):
            for j in range(0, self._nsamples - self._numClusters):
                # iterate over candidates
                dd = self._totalSwapCost(i, j)
                if dd < d:
                    d = dd
                    swapCandidate = j
                    swapMedioid = i
        # if a swapping pair has been found, perform it
        if d < 0.0:
            a = self._indices[self._nsamples - swapMedioid]
            self._indices[self._nsamples - swapMedioid] = self._indices[swapCandidate]
            self._indices[swapCandidate] = a
            swap = True
        # recompute clustering
        self._clusterize()
        return swap

    """
    Performs the k-medioids algorihm and returns the number of iterations performed
    """

    def performClustering(self):
        iter = 0
        while self.updateMedioids():
            iter += 1
        return iter


if __name__ == '__main__':
    dm = np.fromfunction(lambda i, j: (i - j) * (i - j), (100, 100))
    print('Here we are!')
    cl = Clustering(4, dm)
    print(cl.performClustering())
    print(cl.medioids)
    print(cl.clusters)
