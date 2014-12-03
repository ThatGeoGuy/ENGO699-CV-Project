#!/usr/bin/env python3
# engo699/filters/radial_outlier_filter.py
# Jeremy Steward

"""
A class which takes in a point cloud dataset and filters out
outliers which do not have a set number of nearest neighbours
within some radius.
"""

import numpy as np
from scipy import spatial

class RadialOutlierFilter(object):
    """
    Removes outliers in point cloud data. A point is determined
    as an outlier iff it does not have N number of nearest neighbours
    within the 3D sphere of a given radius.
    """

    def __init__(self, points, num_neighbours, radius):
        self.N                = num_neighbours
        self.radius           = radius
        self.kdtree           = spatial.KDTree(points)
        self.filtered_indices = None

    @property
    def pts(self):
        """
        Converts the KD-Tree with all the points in it to an Nx3
        matrix of points. Rather, it just returns the original point
        cloud that was input into the class.
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
        and if there isn't at least N points nearby, then we return false,
        else we return true. The list will tell us row by row in self.pts
        which points we keep and which we filter.
        """
        self.filtered_indices = []
        for i, pt in enumerate(self.pts):
            dists, nn_indices = self.kdtree.query(
                    pt, k=self.N, p=2, distance_upper_bound=self.radius)

            # If we ask for e.g. 20 neighbours when we query for points,
            # and only 5 nearest neighbours within the distance_upper_bound
            # are found, then the rest of the distances in dists will be
            # represented by np.Inf. We can test for these via np.isinf
            if (len(dists) - len(dists[np.isinf(dists)])) < self.N:
                self.filtered_indices.append(i)

        return self.pts[
                np.all(self.pts != self.pts[self.filtered_indices, :], axis=1), :].copy()

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
