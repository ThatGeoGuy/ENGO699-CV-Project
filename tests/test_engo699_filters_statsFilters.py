#!/usr/bin/env python3
# tests/test_engo699_filters_radial_outlier_filter.py
# Jeremy Steward

"""
Tests the statistical filters module.
"""

import unittest
import numpy as np

from engo699.shapes import createCube
from engo699.filters import RadialOutlierFilter, NonPlanarOutlierFilter

class TestRadialOutlierFilter(unittest.TestCase):
    """
    Tests the RadialOutlierFilter class.
    """

    def test_initializes(self):
        """
        Tests that the class initializes member slots properly.
        """
        K = 4
        radius = 2.0
        filtered_indices = None
        pt_cloud = createCube(3, 100)

        rof = RadialOutlierFilter(pt_cloud, K, radius)

        assert K == rof.K
        assert radius == rof.radius
        assert np.all(pt_cloud == rof.pts)
        assert filtered_indices == rof.filtered_indices

    def test_filterPoints(self):
        """
        Should filter only points that are outliers.

        The filter should not filter points that are obviously within the radius,
        but should filter points outside that radius. In this case, only one
        offending point should exist, and it should be filtered out.
        """
        K        = 4
        radius   = 0.200
        pt_cloud = createCube(10, 0.100)

        # Append a point that's way far away
        pt_cloud = np.append(pt_cloud, np.array([[100,100,100]]), axis=0)

        rof = RadialOutlierFilter(pt_cloud, K, radius)
        filtered_cloud = rof.filterPoints()

        assert len(filtered_cloud) == len(pt_cloud) - 1
        assert np.all(filtered_cloud == pt_cloud[:-1, :])
        assert np.all(pt_cloud[rof.filtered_indices] == np.array([100,100,100]))

class TestNonPlanarOutlierFilter(unittest.TestCase):
    """
    Tests the NonPlanarOutlierFilter class.
    """
    
    def test_initializes(self):
        """
        Tests that the class initializes member slots properly.
        """
        K = 8
        threshold = 0.50
        pt_cloud = createCube(10, 10)
        
        npof = NonPlanarOutlierFilter(pt_cloud, K, threshold)

        assert K == npof.K
        assert threshold == npof.threshold
        assert np.all(pt_cloud == npof.pts)
