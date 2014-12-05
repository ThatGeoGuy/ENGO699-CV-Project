#!/usr/bin/env python3
# engo699/filters/fitting
# Jeremy Steward

"""
A module containing functions which fit point clouds to geometries
that can be expressed as linear fits (least squares or otherwise).
Examples of these are plane fits, line fits, etc.
"""

import numpy as np

def principalComponents(points):
    """
    Returns the Eigen values and Eigen vectors of the covariance matrix of an
    N x M point cloud array of points, where N is the number of points and M
    is the number of dimensions of each point.
    """
    return np.linalg.eigh(np.cov(points.T))

def fitPlaneTo(points):
    """
    Least Squares plane fit for the points specified in the given point cloud
    array. This is basically the NIST algorithm for PCA, but finds the axis
    which corresponds to the smallest eigenvalue of the covariance matrix of
    our points.
    """
    eig_vals, eig_vecs = principalComponents(points)
    variance = np.min(eig_vals)
    normal = eig_vecs[:, np.argmin(eig_vals)]
    return normal, variance


def fitLineTo(points):
    """
    Least Squares line fit for the points specified in the given point cloud
    array. This is again from the NIST algorithms for fitting via PCA, and
    finds the principal axis which corresponds to the largest eigenvalue of
    the covariance matrix of our points (which in turn means we have the most
    variance along that direction).
    """
    eig_vals, eig_vecs = principalComponents(points)
    variance = np.max(eig_vals)
    direction = eig_vecs[:, np.argmax(eig_vals)]
    return direction, variance
