#!/usr/bin/env python3
# engo699/filters/WLOP.py
# Jeremy Steward

"""
This module defines functions / classes in order to perform the weighted
locally optimal projection.
"""

import sys
import numpy as np

from engo699.geometry import centroid
from engo699.shapes import createCube

class WLOP(object):
    """
    This class is used to perform the weighted locally optimal projection,
    taking in a point cloud and outputting a point cloud once run.
    """
    # Static methods needed to compute WLOP
    @staticmethod
    def theta(r, h):
        return np.exp(-1 * ((4 * r) / h) ** 2)

    # Normal methods
    def eta(self, r):
        return -r

    def deta_dr(self, r):
        return -1 * np.ones(r.shape)

    def alpha_ij(self, r_xi_to_pj):
        r_xi_to_pj[r_xi_to_pj == 0] = np.nan
        return WLOP.theta(r_xi_to_pj, self.h) / r_xi_to_pj

    def beta_ij(self, r_xi_to_xii):
        r_xi_to_xii[r_xi_to_xii == 0] = np.nan
        return WLOP.theta(r_xi_to_xii, self.h) * \
                np.abs(self.deta_dr(r_xi_to_xii)) / r_xi_to_xii

    # Weight function
    def weight_ij(self, r_xi_to_xii):
        r_xi_to_xii[r_xi_to_xii == 0] = np.nan
        return (1 / (len(r_xi_to_xii) - 1)) + WLOP.theta(r_xi_to_xii, self.h)

    def computePDistances(self):
        d = np.zeros((len(self.P), len(self.P)))
        # Yes I know I should use enumerate but there's no reason
        # here to allocate for pi and pj every iteration when I can
        # shave off precious runtime
        for i in range(0, len(self.P)):
            for j in range(i, len(self.P)):
                d[i, j] = np.linalg.norm(self.P[i, :] - self.P[j, :], 2)
            d[:, i] = d[i, :]
        return d

    def getInitialPoints(self):
        # Initial set of points
        X = createCube(
            np.ceil(self.num_pts ** (1 / 3)),
            self.h,
            centroid(self.P)
        )
        X = X[:self.num_pts, :]
        return X

    def computeProjection(self, num_pts, h, mu):
        """
        Function to actually compute and perform the WLOP.

        Parameters
        ==========
          num_pts:  Number of points in Q, or rather, the number of points
                    we want to project onto P.
          h:        Compact support radius which weights out comparisons
                    with points that are farther than it.
          mu:       Estimated or set Lagrange multiplier in the range of
                    [0, 0.5). A higher value results in a more evenly
                    distributed projection.

        Returns
        =======
          Q:    The weighted locally optimal projection onto P
        """
        # This initialization is necessary to make some of the interim
        # functions simpler / easier
        if not (0 <= mu < 0.50):
            print("ERROR: WLOP - mu must be in range [0, 0.5)", file=sys.stderr)
            return None
        self.h       = h
        self.mu      = mu
        self.num_pts = num_pts

        # Create initial X at iteration 0
        X = self.getInitialPoints()
        Q = np.zeros(X.shape)

        # This never changes, will only be computed once at the beginning
        dist_pj_pjj = self.computePDistances()
        # Below is an optimization instead of the list-comprehension (since
        # there doesn't exist a numpy array comprehension)
        v = np.zeros((len(dist_pj_pjj), len(dist_pj_pjj)))
        for i, dist in enumerate(dist_pj_pjj):
            v[i, :] = self.weight_ij(dist)
        v = np.nansum(v, axis=1)

        # The remaining iterations until convergence
        # Up until here, nothing is different from standard LOP
        it = 0
        while np.any(np.abs(Q - X) > self.TOL):
            if it >= self.MAX_ITER:
                print("WARNING: WLOP - Reached max number of iterations.", file=sys.stderr)
                print("WARNING: WLOP - Breaking loop and returning current projection.", file=sys.stderr)
                break

            for i, xi in enumerate(X):
                dist_xi_pj  = np.linalg.norm(xi - self.P, 2, axis=1)
                diff_xi_xii = xi - X
                dist_xi_xii = np.linalg.norm(diff_xi_xii, 2, axis=1)

                # (re)set our E1 and E2 terms to zero after each point calculation
                E1 = np.zeros(3)
                E2 = np.zeros(3)

                # a = alpha, b = beta
                a = self.alpha_ij(dist_xi_pj)
                b = self.beta_ij(dist_xi_xii)

                # Weights E1 -> v, E2 -> w
                # v is below since it relies on the point pj
                w = self.weight_ij(dist_xi_xii)
                alpha_over_v = a / v
                beta_times_w = b * w

                for j, pj in enumerate(self.P):
                    if np.isnan(alpha_over_v[j]):
                        continue

                    # Calculate E1
                    E1 += pj * (alpha_over_v[j] / np.nansum(alpha_over_v))

                for ii, diff_ii in enumerate(diff_xi_xii):
                    if np.isnan(beta_times_w[ii]):
                        continue

                    # Calculate E2
                    E2 += diff_ii * (beta_times_w[ii] / np.nansum(beta_times_w))

                Q[i, :] = E1 + self.mu * E2
            # Increment iterations and go to next iteration over point cloud
            it += 1
            X = Q.copy()
        return Q

    # Below are "off limits" methods. These shouldn't be called directly
    def __init__(self, pt_cloud, tolerance, max_iter = 40):
        """
        This initializes the class with the point cloud P to project onto,
        the tolerance with which we want our points to project to, and the
        maximum number of iterations before we break (without considering
        convergence).

        Parameters
        ==========
          pt_cloud:     Point cloud P we want to project on to. This is
                        represented as an N x 3 array where N is the number
                        of points and the three columns are for X, Y, and Z
                        data.
          tolerance:    The maximum change we can allow before we consider
                        the method "converged."
          max_iter:     The maximum number of iterations of WLOP before we
                        break out of the algorithm and assume the answer is
                        not converging.

        Returns
        =======
          self:         An object used to calculate the weighted locally
                        optimal projection.
        """
        self.P        = pt_cloud.copy()
        self.TOL      = tolerance
        self.MAX_ITER = max_iter
