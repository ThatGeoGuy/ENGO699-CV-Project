#!/usr/bin/env python3
# engo699/detectors/ISSDetector.py
# Jeremy Steward

"""
Defines the point detector used by the Intrinsic Shape Signatures algorithm.
"""

from scipy import spatial

class ISSKeypointDetector(object):
    """
    A class to manage the state and functionality used in the Intrinsic Shape
    Signature algorithm for detecting keypoints.
    """

    def __init__(self, points, density_radius, frame_radius):
        self.kdtree = spatial.KDTree(points)
        self.r_d = density_radius
        self.r_f = frame_radius

    @property
    def pts(self):
        """
        Returns the points stored within our KDTree.
        """
        return self.kdtree.data

    @pts.setter
    def pts(self, new_points):
        """
        Sets the points stored within our KDTree to new_points
        """
        self.kdtree = spatial.KDTree(new_points)
        return

    def computeWeights(self):
        """
        Computes a weight for each point pt_i, which is inversely related to the
        number of points in the spherical neighbourhood of radius `density_radius`
        """
        w = []

        for pt in self.pts:
            dists, nn_indices = self.kdtree.query(pt, k=len(self.pts),
                    p=2, distance_upper_bound=self.r_d)

            dists[dists == 0] = np.nan
            w.append(np.nansum(1 / dists))
        self.w = np.array(w)
        return

    def computeWeightedScatterMatrix(self):
        """
        Computes the weighted scatter matrix cov(pt_i) for pt_i using all points pt_j
        within a distance frame_radius.
        """
        
        for pt in self.pts:
             
        return
