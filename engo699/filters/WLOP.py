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
        # X = createCube(
        #     int(self.num_pts ** (1 / 3)),
        #     (1 - self.h) / (1 + self.h),
        #     centroid(self.P)
        # )
        X = self.P[:self.num_pts, :].copy()
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
        if mu < 0 or mu >= 0.5:
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
        v = np.array([self.weight_ij(row) for row in dist_pj_pjj])

        # The remaining iterations until convergence
        # Up until here, nothing is different from standard LOP
        it = 0
        while np.any(np.abs(Q - X) > self.TOL):
            if it >= self.MAX_ITER:
                print("WARNING: WLOP - Reached max number of iterations.", file=sys.stderr)
                print("WARNING: WLOP - Breaking loop and returning current projection.", file=sys.stderr)
                break

            # DEBUG
            print("Iteration {}.".format(it), file=sys.stderr)
            X = Q.copy()

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
                beta_times_w = b * w

                for j, pj in enumerate(self.P):
                    if np.isnan(v[j,j]):
                        continue

                    alpha_over_v = a / v[j, :]

                    # Calculate E1
                    E1 += pj * (alpha_over_v[j] / np.nansum(alpha_over_v))

                for ii, diff_ii in enumerate(diff_xi_xii):
                    if np.isnan(w[ii]):
                        continue

                    # Calculate E2
                    E2 += diff_ii * (beta_times_w[ii] / np.nansum(beta_times_w))

                Q[i, :] = E1 + self.mu * E2
            # Increment iterations and go to next iteration over point cloud
            it += 1

        # DEBUG
        print("Program completed in {} iterations.".format(it), file=sys.stderr)
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
    def eta(self, r):
        return 1 / (3 * (r ** 3))

    def deta_dr(self, r):
        return -1 / (r ** 4)

    # Both weighting factors are just equal to 1
    def v_j(self, pj):
        return 1

    def w_i(self, i, xi, X):
        return 1
