#!/usr/bin/env python3
# engo699/detectors/ISSDetector.py
# Jeremy Steward

"""
Defines the point detector used by the Intrinsic Shape Signatures algorithm.
"""

import numpy as np
from scipy import spatial

class ISSKeypointDetector(object):
    """
    A class to manage the state and functionality used in the Intrinsic Shape
    Signature algorithm for detecting keypoints.
    """

    def __init__(self, points, min_nn, density_radius, frame_radius, gamma21, gamma32):
        self.kdtree         = spatial.KDTree(points)
        self.min_neighbours = min_nn
        self.r_d            = density_radius
        self.r_f            = frame_radius
        self.gamma21        = gamma21
        self.gamma32        = gamma32

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

        for i, pt_i in enumerate(self.pts):
            dists, nn_indices = self.kdtree.query(pt_i, k=len(self.pts),
                    p=2, distance_upper_bound=self.r_d)

            # Remove NN indices where dists is infinite -> keep finites
            nn_indices = nn_indices[np.isfinite(dists)]
            # Remove the point itself from it's nearest neighbours
            nn_indices = nn_indices[nn_indices != i]

            tmp = np.nansum(np.linalg.norm(self.pts[nn_indices, :], 2, axis=1))
            if tmp == 0:
                w.append(1)
            else:
                w.append(1 / tmp)

        return np.array(w)

    def computeWeightedScatterMatrices(self, w):
        """
        Computes the weighted scatter matrix cov(pt_i) for pt_i using all points pt_j
        within a distance frame_radius.
        """
        covars = []

        for i, pt_i in enumerate(self.pts):
            dists, nn_indices = self.kdtree.query(pt_i, k=len(self.pts),
                    p=2, distance_upper_bound=self.r_f)

            # Remove NN indices where dists is infinite -> keep finites
            nn_indices = nn_indices[np.isfinite(dists)]
            # Remove the point itself from it's nearest neighbours
            nn_indices = nn_indices[nn_indices != i]

            if len(nn_indices) < self.min_neighbours:
                covars.append(np.zeros((3, 3)))
                continue

            tmp = np.zeros((3, 3))
            for j, pt_j in enumerate(self.pts[nn_indices]):
                # Matrix is necessary since the structure is needed for 1D array
                diff = np.matrix(pt_j - pt_i)
                tmp += w[nn_indices[j]] * (diff.T * diff)
            covars.append(tmp / np.sum(w[nn_indices]))
        return covars

    def detectKeypoints(self):
        """
        Detects and returns the indices of points determined to be keypoints via the
        ISS saliency measure with non-maxima suppression.
        """
        w = self.computeWeights()
        covars = self.computeWeightedScatterMatrices(w)

        saliencies = np.zeros(len(self.pts))
        for i, CV in enumerate(covars):
            eig_vals, eig_vecs = np.linalg.eigh(CV)

            if not np.all(eig_vals == 0):
                lam1 = np.max(eig_vals)
                lam2 = np.max(eig_vals[np.array([0, 1, 2]) != np.argmax(eig_vals)])
                lam3 = np.min(eig_vals)

                if lam3 < 0:
                    continue

                if (lam2 / lam1) < self.gamma21 and (lam3 / lam2) < self.gamma32:
                    saliencies[i] = lam3

        self.keypoint_indices = []
        for i, pt_i in enumerate(self.pts):
            dists, nn_indices = self.kdtree.query(pt_i, k=len(self.pts),
                    p=2, distance_upper_bound=self.r_f)

            # Remove NN indices where dists is infinite -> keep finites
            nn_indices = nn_indices[np.isfinite(dists)]
            # Remove the point itself from it's nearest neighbours
            nn_indices = nn_indices[nn_indices != i]

            if nn_indices.tolist() and saliencies[i] >= np.max(saliencies[nn_indices]):
                self.keypoint_indices.append(i)
        return self.pts[self.keypoint_indices, :].copy()
