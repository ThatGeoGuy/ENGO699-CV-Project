#!/usr/bin/env python3
# tests/test_engo699_filters_statsFilters.py
# Jeremy Steward

"""
Tests the statistical filters module.
"""

import unittest
import numpy as np

from engo699.shapes import createCube, createSquarePlane
from engo699.filters import RadialOutlierFilter, NonPlanarOutlierFilter

class TestRadialOutlierFilter(unittest.TestCase):
    """
    Tests the RadialOutlierFilter class.
    """

    def testInitializes(self):
        """
        Tests that the class initializes member slots properly.
        """
        K = 4
        radius = 2.0
        filtered_indices = None
        pt_cloud = createCube(3, 100)

        rof = RadialOutlierFilter(pt_cloud, K, radius)

        self.assertEqual(K, rof.K)
        self.assertEqual(radius, rof.radius)
        self.assertTrue(np.all(pt_cloud == rof.pts))
        self.assertEqual(filtered_indices, rof.filtered_indices)

    def testFilterPoints(self):
        """
        Tests that the filter algorithm works as intended.

        The filter should not filter points that are obviously within the radius,
        but should filter points outside that radius. In this case, only one
        offending point should exist, and it should be filtered out.
        """
        K        = 4
        radius   = 0.200
        pt_cloud = createCube(10, 0.100)

        # Append a point that's way far away
        noisy_cloud = np.append(pt_cloud, np.array([[100,100,100]]), axis=0)

        rof = RadialOutlierFilter(noisy_cloud, K, radius)
        filtered_cloud = rof.filterPoints()

        self.assertEqual(len(filtered_cloud), len(pt_cloud))
        self.assertTrue(np.all(filtered_cloud == pt_cloud))
        self.assertTrue(np.all(noisy_cloud[rof.filtered_indices] == np.array([100,100,100])))

class TestNonPlanarOutlierFilter(unittest.TestCase):
    """
    Tests the NonPlanarOutlierFilter class.
    """

    def testInitializes(self):
        """
        Tests that the class initializes member slots properly.
        """
        K = 8
        threshold = 0.50
        pt_cloud = createCube(10, 10)

        npof = NonPlanarOutlierFilter(pt_cloud, K, threshold)

        self.assertEqual(K, npof.K)
        self.assertEqual(threshold, npof.threshold)
        self.assertTrue(np.all(pt_cloud == npof.pts))

    def testFilterPoints(self):
        """
        Tests that the filter algorithm works as intended.
        """
        # Create a plane
        plane_pts = createSquarePlane(10, 10)
        noisy_pts = np.append(plane_pts, [[100, 100, 100], [101,100,101]], axis=0)

        # Ideally only our two points that were added should be outliers
        K = 8
        threshold = 0.01
        npof = NonPlanarOutlierFilter(noisy_pts, K, threshold)
        filtered_cloud = npof.filterPoints()

        self.assertEqual(len(npof.filtered_indices), 2)
        self.assertEqual(len(filtered_cloud), len(plane_pts))

if __name__ == '__main__':
    unittest.main()
