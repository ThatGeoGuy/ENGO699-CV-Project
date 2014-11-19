#!/usr/bin/env python3
# tests/test_engo699_geometry.py
# Jeremy Steward
# Tests the functionality in engo699/geometry.py

import unittest
import numpy as np

import engo699.geometry

centroid = engo699.geometry.centroid

class test_centroid(unittest.TestCase):
    """
    Tests for the function "centroid."
    Test descriptions are in the docstring of each member function.
    """
    def test_centroid_of_zeros(self):
        """
        The centroid of a set of points that are all at the origin should be zero.
        """
        pts_at_origin     = np.zeros((200, 3))
        expected_centroid = np.array([0, 0, 0])

        self.assertTupleEqual(centroid(pts_at_origin).shape, expected_centroid.shape)
        self.assertTrue(np.all(centroid(pts_at_origin) == expected_centroid))

    def test_centroid_is_mean(self):
        """
        Test if the centroid is the mean of the column.
        This is somewhat of a stupid test but it's more of a sanity check.
        """
        pts = np.random.randn(200, 3)
        expected_centroid = np.array(
            [np.mean(pts[:, 0]), np.mean(pts[:, 1]), np.mean(pts[:, 2])]
        )

        self.assertTupleEqual(centroid(pts).shape, expected_centroid.shape)
        self.assertTrue(np.all(centroid(pts) == expected_centroid))
