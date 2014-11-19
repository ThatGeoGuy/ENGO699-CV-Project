#!/usr/bin/env python3
# engo699/geometry.py
# Jeremy Steward
# Functions and classes to compute basic geometric quantities

__doc__ = """
Module containing functions to compute basic geometric quantities.
"""

import numpy as np

def centroid(array_like):
    """
    Takes in an array-like object and computes the centroid across all columns.
    This basically equates to taking the mean of each column and returning the
    row vector of these means.

    Parameters
    ==========
      array_like:   A numpy array with NxM columns, where each of the M columns
                    will be averaged across N rows and the resultant row vector
                    returned.

    Returns
    =======
      centroid:     A single row vector (numpy) that contains the centroid (mean) of
                    each dimension.
    """
    return np.array([np.mean(row) for row in array_like.T])
