#!/usr/bin/env python3
# tests/test_engo699_shapes.py
# Jeremy Steward
# Tests the functionality in engo699/shapes.py

import unittest
import numpy as np

import engo699.shapes

# Simplify some function names so we don't have namespace hell in our tests
createCube = engo699.shapes.createCube

class test_createCube(unittest.TestCase):
    """
    Tests for the function "createCube."
    Expected behaviour is listed in docstrings of member functions.
    """
    def test_createCube_output_should_be_Nx3(self):
        """
        The output from WLOP should be an Nx3 matrix, where N is the
        cube of the side-length. e.g. if side-length = 3, then N should
        be 3 ** 3 => 27
        """
        side_length = 3
        extent      = 100
        pts         = createCube(side_length, extent)

        self.assertEqual(pts.shape[0], side_length ** 3)

    def test_createCube_cube_centred_on_centroid(self):
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

if __name__ == '__main__':
    unittest.main()
