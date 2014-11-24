#!/usr/bin/env python3
# engo699/filters/WLOP.py
# Jeremy Steward
# Code to perform the Weighted Locally Optimal Projection

__doc__ = """
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

    @staticmethod
    def eta(r):
        return -r

    @staticmethod
    def deta_dr(r):
        return -1

    # Normal methods
    def alpha_ij(self, xi, pj):
        r = np.linalg.norm(xi - pj, 2)
        return WLOP.theta(r, self.h) / r

    def beta_ij(self, xi, xii):
        r = np.linalg.norm(xi - xii, 2)
        return WLOP.theta(r, self.h) * abs(WLOP.deta_dr(r)) / r

    # Weight function for E1 terms
    def v_j(self, pj):
        return 1 + np.sum(WLOP.theta(np.linalg.norm(pj - p, 2), self.h) for p in self.P)

    # Weight function for E2 terms
    def w_i(self, i, xi, X):
        return 1 + np.sum(WLOP.theta(np.linalg.norm(xi - xj, 2), self.h) \
                for j, xj in enumerate(X) if i != j)

    def getInitialPoints(self):
        X = createCube(
            int(self.num_pts ** (1 / 3)),
            (1 - self.h) / (1 + self.h),
            centroid(self.P)
        )
        # X = self.P
        return X[:self.num_pts, :]

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
        if mu < 0 or mu >= 0.5:
            print("ERROR: WLOP - mu must be in range [0, 0.5)", file=sys.stderr)
            return None
        self.h       = h
        self.mu      = mu
        self.num_pts = num_pts

        # Create initial X at iteration 0
        X = self.getInitialPoints()
        Q = X.copy()

        # The following calculates X at iteration 1
        for i, xi in enumerate(X):
            thetas = np.array([WLOP.theta(np.linalg.norm(pj - xi), self.h) for pj in self.P])
            numer = np.sum(pj * thetas[j] for j, pj in enumerate(self.P))
            denom = np.sum(thetas)

            Q[i, :] = numer / denom

        # The remaining iterations until convergence
        # Up until here, nothing is different from standard LOP
        it = 0
        while np.any((Q - X) > self.TOL):
            if it >= self.MAX_ITER:
                print("WARNING: WLOP - Reached max number of iterations.", file=sys.stderr)
                print("WARNING: WLOP - Breaking loop and returning current projection.", file=sys.stderr)
                break

            X = Q.copy()
            for i, xi in enumerate(X):
                # Calculate E1 term
                alpha_over_vj = [self.alpha_ij(xi, pj) / self.v_j(pj) for pj in self.P]
                sum_alpha_over_vj = sum(alpha_over_vj)
                E1 = sum(pj * alpha_over_vj[j] / sum_alpha_over_vj for j, pj in enumerate(self.P))

                # Calculate E2 term
                beta_times_wi = [self.beta_ij(xi, xii) * self.w_i(i, xi, X) \
                        for ii, xii in enumerate(X) if i != ii]
                sum_beta_times_wi = sum(beta_times_wi)
                E2 = sum((xi - xii) * beta_times_wi[ii] / sum_beta_times_wi for ii, xii in X)
                Q[i, :] = E1 + self.mu * E2

            it += 1

        # DEBUG
        print("Completed projection with {} iterations".format(it), file=sys.stderr)
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

class LOP(WLOP):
    """
    Locally optimal projection as referenced by Lipman et al. All parameters
    are weighted equally, and the eta function falls off much more slowly.
    """
    @staticmethod
    def eta(r):
        return 1 / (3 * (r ** 3))

    @staticmethod
    def deta_dr(r):
        return -1 / (r ** 4)

    # Both weighting factors are just equal to 1
    def v_j(self, pj):
        return 1

    def w_i(self, i, xi, X):
        return 1
