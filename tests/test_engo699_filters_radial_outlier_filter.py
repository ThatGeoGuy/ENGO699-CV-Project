#!/usr/bin/env python3
# tests/test_engo699_filters_radial_outlier_filter.py
# Jeremy Steward

"""
Tests the radial outlier filter module.
"""

import unittest
import numpy as np

from engo699.shapes import createCube
from engo699.filters.radial_outlier_filter import RadialOutlierFilter

class TestRadialOutlierFilter(unittest.TestCase):
    """
    Tests the RadialOutlierFilter class.
    """

    def test_initializes(self):
        """
        Tests that the class initializes member slots properly.
        """
        N = 4
        radius = 2.0
        filtered_indices = None
        pt_cloud = createCube(3, 100)

        rof = RadialOutlierFilter(pt_cloud, N, radius)

        assert N == rof.N
        assert radius == rof.radius
        assert np.all(pt_cloud == rof.pts)
        assert filtered_indices == rof.filtered_indices

    def test_filterPoints(self):
        """
        The filter should not filter points that are obviously within the radius,
        but should filter points outside that radius. In this case, only one
        offending point should exist, and it should be filtered out.
        """
        N        = 4
        radius   = 0.200
        pt_cloud = createCube(10, 0.100)

        # Append a point that's way far away
        pt_cloud = np.append(pt_cloud, np.array([[100,100,100]]), axis=0)

        rof = RadialOutlierFilter(pt_cloud, N, radius)
        filtered_cloud = rof.filterPoints()

        assert len(filtered_cloud) == len(pt_cloud) - 1
        assert np.all(filtered_cloud == pt_cloud[:-1, :])
        assert np.all(pt_cloud[rof.filtered_indices] == np.array([100,100,100]))
