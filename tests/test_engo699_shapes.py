#!/usr/bin/env python3
# tests/test_WLOP.py
# Jeremy Steward
# Tests the functionality in engo699/filters/WLOP.py

import unittest
import numpy as np

from engo699.shapes import createCube

class test_createCubeOfPoints(unittest.TestCase):
    """
    Tests for the function createCubeOfPoints.
    Expected behaviour is listed in docstrings of member functions.
    """
    def test_output_should_be_Nx3(self):
        """
        The output from WLOP should be an Nx3 matrix, where N is the
        cube of the side-length. e.g. if side-length = 3, then N should
        be 3 ** 3 => 27
        """
        side_length = 3
        extent      = 100
        pts         = createCube(side_length, extent)

        self.assertEqual(pts.shape[0], side_length ** 3)

    def test_cube_centred_on_centroid(self):
        """
        The centroid of all the points in the cube output by the
        function should be the same as the centroid passed into the
        function.
        """
        side_length = 3
        extent      = 100
        centroid    = (100, 100, 100)
        pts         = createCube(side_length, extent, centroid)

        self.assertEqual(
            (np.mean(pts[:,0]), np.mean(pts[:,1]), np.mean(pts[:,2])),
            centroid
        )
