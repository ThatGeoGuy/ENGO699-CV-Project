#!/usr/bin/env python3
# tests/test_engo699_detectors_ISSDetector.py
# Jeremy Steward

"""
Tests the functions and classes in `engo699/detectors/ISSDetector.py`
"""

import unittest
import numpy as np

from engo699.detectors import ISSKeypointDetector

class TestISSKeypointDetector(unittest.TestCase):
    """
    Tests the ISSKeypointDetector class.
    """

    def testInitializes(self):
        """
        Tests that the object initializes properly.
        """
        points         = np.ones((5000, 3))
        min_nn         = 8
        density_radius = 0.040
        frame_radius   = 1.900
        gamma21        = 1.00
        gamma32        = 1.00

        iss = ISSKeypointDetector(points, min_nn, density_radius, frame_radius, gamma21, gamma32)

        self.assertTrue(np.all(points == iss.pts))
        self.assertEqual(density_radius, iss.r_d)
        self.assertEqual(frame_radius, iss.r_f)
        self.assertEqual(min_nn, iss.min_neighbours)
        self.assertEqual(gamma21, iss.gamma21)
        self.assertEqual(gamma32, iss.gamma32)

    def testComputeWeights(self):
        """
        Tests that the object computes the weights properly.
        I honestly don't know how to test this.
        """

if __name__ == "__main__":
    unittest.main()
