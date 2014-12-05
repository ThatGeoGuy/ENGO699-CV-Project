#!/usr/bin/env python3
# tests/test_engo699_shapes.py
# Jeremy Steward
# Tests the functionality in engo699/shapes.py

import unittest
import numpy as np

from engo699.fitting import fitPlaneTo
from engo699.geometry import centroid
from engo699.shapes import createCube, createSquarePlane

class TestCreateCube(unittest.TestCase):
    """
    Tests for the function "createCube."
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

class TestCreateSquarePlane(unittest.TestCase):
    """
    Tests for the function "createSquarePlane."
    """

    def test_createSquarePlane_outputPlaneMatchesInputNormal(self):
        """
        Tests whether the plane normal of the output plane matches the normal requested by input.
        """
        dim = 8
        scale = 100
        normal = np.array([0,1,0])

        plane_pts = createSquarePlane(dim, scale, normal=normal)
        surface_normal, variance = fitPlaneTo(plane_pts)

        self.assertTrue(np.all(normal == surface_normal))

    def test_createSquarePlane_outputPlaneAboutInputCentroid(self):
        """
        Tests whether the output plane is centred about the input centroid.
        """
        dim = 8
        scale = 100
        cent = np.array([100,100,100])

        plane_pts = createSquarePlane(dim, scale, centroid=cent)
        calculated_centroid = centroid(plane_pts)

        self.assertTrue(np.all(cent == calculated_centroid))

    def test_createSquarePlane_testPlaneMatchesFunctionOutput(self):
        """
        Test to make sure that the function outputs all the points in a square grid
        and that the normal is [0,0,1]
        """
        dim = 3
        scale = 2
        normal = np.array([0,0,1])

        grid = np.array([[-1, -1,  0],
                         [-1,  0,  0],
                         [-1,  1,  0],
                         [ 0, -1,  0],
                         [ 0,  0,  0],
                         [ 0,  1,  0],
                         [ 1, -1,  0],
                         [ 1,  0,  0],
                         [ 1,  1,  0]])

        plane_pts = createSquarePlane(dim, scale, normal=normal)

        for pt in grid:
            self.assertTrue(pt in plane_pts)

        for pt in plane_pts:
            self.assertTrue(pt in grid)

        self.assertTrue(np.all(normal == fitPlaneTo(plane_pts)[0]))
        self.assertTrue(np.all(normal == fitPlaneTo(grid)[0]))

if __name__ == '__main__':
    unittest.main()
