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
    @np.vectorize
    def eta(self, r):
        return -r

    @np.vectorize
    def deta_dr(self, r):
        return -1

    def alpha_ij(self, r_xi_to_pj):
        return WLOP.theta(r, self.h) / r

    def beta_ij(self, r_xi_to_xii):
        return WLOP.theta(r, self.h) * np.abs(self.deta_dr(r)) / r

    # Weight function
    def weight_ij(self, r_xi_to_xii):
        return (1 / (len(r_xi_to_xii) - 1)) + WLOP.theta(r_xi_to_xii, self.h)

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
        Q = np.zeros(X.shape)

        # This is done since these will never change
        dist_pj_pjj = np.array([np.linalg.norm(pj - pjj, 2) for pj in self.P for pjj in self.P])
        v = self.weight_ij(dist_pj_pjj)

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
                dist_xi_pj  = np.array([np.linalg.norm(xi - pj, 2) for pj in self.P])
                diff_xi_xii = np.array([xi - xii for xii in self.X])
                dist_xi_xii = np.array([np.linalg.norm(diff, 2) for diff in diff_xi_xii])

                # (re)set our E1 and E2 terms to zero after each point calculation
                E1 = np.zeros(3)
                E2 = np.zeros(3)

                # a = alpha, b = beta
                a = self.alpha_ij(dist_xi_pj)
                b = self.beta_ij(dist_xi_xii)

                # Weights E1 -> v, E2 -> w
                # v is defined above outside of the while loop, since they will
                # always be the same
                w = self.weight_ij(dist_xi_xii)

                # NOTE: As per numpy these operations are element by element
                alpha_over_v = a / v
                beta_times_w = b * w
                for j, pj in enumerate(self.P):
                    # Calculate E1
                    E1 += pj * (alpha_over_v[j] / \
                            np.sum(tmp for jj, tmp in enumerate(alpha_over_v) if j != jj))

                for ii, diff_ii in enumerate(diff_xi_xii):
                    E2 += diff_ii * (beta_times_w[ii] / \
                            np.sum(tmp for iii, tmp in enumerate(beta_times_w) if ii != iii))

                Q[i, :] = E1 + self.mu * E2
            # Increment iterations and go to next iteration over point cloud
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
    def eta(self, r):
        return 1 / (3 * (r ** 3))

    def deta_dr(self, r):
        return -1 / (r ** 4)

    # Both weighting factors are just equal to 1
    def v_j(self, pj):
        return 1

    def w_i(self, i, xi, X):
        return 1
