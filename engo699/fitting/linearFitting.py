#!/usr/bin/env python3
# engo699/filters/fitting
# Jeremy Steward

"""
A module containing functions which fit point clouds to geometries
that can be expressed as linear fits (least squares or otherwise).
Examples of these are plane fits, line fits, etc.
"""

def fitPlaneTo(points):
    """
    Least Squares plane fit for the points specified in the given point cloud
    array. This is basically the NIST algorithm for PCA, but finds the axis
    which corresponds to the smallest eigenvalue of the covariance matrix of
    our points.
    """
    
def fitLineTo(points):
    """
    Least Squares line fit for the points specified in the given point cloud
    array. This is again from the NIST algorithms for fitting via PCA, and
    finds the principal axis which corresponds to the largest eigenvalue of
    the covariance matrix of our points (which in turn means we have the most
    variance along that direction).
    """
