#!/usr/bin/env python3
# engo699/filters/statsFilters.py
# Jeremy Steward

"""
Defines classes and methods which takes in a point cloud dataset and filters
out outliers which do not conform to the specified statistics or relationships
within the data. (e.g. RadialOutlierFilter removes points which do not have the
property of having K nearest neighbours within a given radius.
"""

import numpy as np
from scipy import spatial

from engo699.fitting import fitPlaneTo

class RadialOutlierFilter(object):
    """
    Removes outliers in point cloud data. A point is determined
    as an outlier iff it does not have N number of nearest neighbours
    within the 3D sphere of a given radius.
    """

    def __init__(self, points, num_neighbours, radius):
        self.K                = num_neighbours
        self.radius           = radius
        self.kdtree           = spatial.KDTree(points)
        self.filtered_indices = None

    @property
    def pts(self):
        """
        Converts the KD-Tree with all the points in it to the
        original point cloud that was input into the class.
        """
        return self.kdtree.data

    @pts.setter
    def pts(self, new_points):
        self.kdtree = spatial.KDTree(new_points)
        return

    def filterPoints(self):
        """
        Performs the actual filtering of points.
        We search the KDTree for all nearest neighbours around a given point,
        and if there isn't at least K points nearby, then we return false,
        else we return true. The list will tell us row by row in self.pts
        which points we keep and which we filter.
        """
        self.filtered_indices = []
        ret_indices = []
        for i, pt in enumerate(self.pts):
            dists, nn_indices = self.kdtree.query(
                    pt, k=self.K, p=2, distance_upper_bound=self.radius)

            # If we ask for e.g. 20 neighbours when we query for points,
            # and only 5 nearest neighbours within the distance_upper_bound
            # are found, then the rest of the distances in dists will be
            # represented by np.Inf. We can test for these via np.isinf
            if (len(dists) - len(dists[np.isinf(dists)])) >= self.K:
                ret_indices.append(i)
            else:
                self.filtered_indices.append(i)

        if self.filtered_indices:
            return self.pts[ret_indices, :].copy()
        else:
            return self.pts.copy()

    def numPointsFiltered(self):
        """
        Just a simple function to return the number of points that were
        filtered from the original point cloud via this method. If the
        filter hasn't been run yet, it should return None.
        """
        if self.filtered_indices is None:
            return None
        else:
            return len(self.filtered_indices)

class NonPlanarOutlierFilter(object):
    """
    Removes outliers in point cloud data. A point is determined
    as an outlier iff the estimated plane fit at the location of the
    point with it's K nearest neighbours is within some given threshold.
    """

    def __init__(self, points, K, threshold):
        self.K = K
        self.kdtree = spatial.KDTree(points)
        self.threshold = threshold

    @property
    def pts(self):
        """
        Converts the KD-Tree with all the points in it to the
        original point cloud that was input into the class.
        """
        return self.kdtree.data

    @pts.setter
    def pts(self, new_points):
        self.kdtree = spatial.KDTree(new_points)
        return

    def filterPoints(self):
        """
        Performs the actual filtering of the points.
        Search each point for the nearest neighbours, fit a plane to
        the point and the K nearest neighbours, and if the variance
        of the plane fit is greater than our threshold we filter
        the point.
        """
        self.filtered_indices = []
        ret_indices = []
        for i, pt in enumerate(self.pts):
            dists, nn_indices = self.kdtree.query(pt, k=self.K)

            normal, variance = fitPlaneTo(self.pts[nn_indices, :])

            if variance < self.threshold:
                ret_indices.append(i)
            else:
                self.filtered_indices.append(i)

        if self.filtered_indices:
            return self.pts[ret_indices, :].copy()
        else:
            return self.pts.copy()
