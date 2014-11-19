#!/usr/bin/env python3
# engo699/filters/WLOP.py
# Jeremy Steward
# Code to perform the Weighted Locally Optimal Projection

__doc__ = """
This module defines functions / classes in order to perform the weighted
locally optimal projection.
"""

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
    def getInitialPoints(self):
        self.X = createCube(self.num_points, self.h, centroid(self.P))

    # Below are "off limits" methods. These shouldn't be called directly
    def __init__(self, pt_cloud):
        self.P = pt_cloud

