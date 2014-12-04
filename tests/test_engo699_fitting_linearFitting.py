#!/usr/bin/env python3
# engo699/fitting/linearFitting.py
# Jeremy Steward

"""
Tests the functionality provided by the linearFitting module.
"""

import unittest
import numpy as np

from engo699.fitting import fitPlaneTo, fitLineTo

class TestFitPlaneTo(unittest.TestCase):
    """
    Tests the function fitPlaneTo from the fitting module.
    """

    def test_fitPlaneTo_cardinal_planes(self):
        """
        Cardinal plane data should have a unit normal in the direction of the orthogonal axis.
        """
        # Unit vectors representing cardinal axes
        x_axis = np.array([1,0,0])
        y_axis = np.array([0,1,0])
        z_axis = np.array([0,0,1])

        # Create a random Nx3 (N is arbitrarily 5000, doesn't matter though)
        # which will be used as the basis for creating the three point cloud
        # arrays which will be transformed into cardinal planes.
        rand_array = np.random.rand(5000, 3)
        cardinal_plane_x = rand_array.copy()    # NOTE: .copy() is required
        cardinal_plane_y = rand_array.copy()    # since otherwise we'll end
        cardinal_plane_z = rand_array.copy()    # up modifying rand_array!

        # Make the point clouds into cardinal planes
        # Basically make all the X, Y, or Z values equal to zero
        cardinal_plane_x[:, 0] = 0
        cardinal_plane_y[:, 1] = 0
        cardinal_plane_z[:, 2] = 0

        planes = (cardinal_plane_x, cardinal_plane_x, cardinal_plane_x)
        axes   = (x_axis, y_axis, z_axis)

        for plane_cloud, ax in zip(planes, axes):
            fitNormal, variance = fitPlaneTo(plane_cloud)

            assert np.all(fitNormal == ax) or \
                    np.all((-1 * fitNormal) == ax)

    def test_fitPlaneTo_insufficient_points(self):
        """
        Trying to fit a plane when there are only two points should be an error.
        """

class TestFitLineTo(unittest.TestCase):
    """
    Tests the function fitLineTo from the fitting module.
    """

    def test_fitLineTo_cardinal_axes(self):
        """
        Line data along the cardinal axes should fit to the axis itself.
        """
        # Unit vectors representing cardinal axes
        x_axis = np.array([1,0,0])
        y_axis = np.array([0,1,0])
        z_axis = np.array([0,0,1])

        # Create a random point cloud which will be used to define the
        # "perfect line" point clouds along the cardinal axes
        rand_array = np.random.rand(5000, 3)
        cardinal_line_x = rand_array.copy()
        cardinal_line_y = rand_array.copy()
        cardinal_line_z = rand_array.copy()

        # Modify the cardinal line point clouds so that only the axes of
        # interest is not zero.
        cardinal_line_x[:, 1] = cardinal_line_x[:, 2] = 0
        cardinal_line_y[:, 0] = cardinal_line_y[:, 2] = 0
        cardinal_line_z[:, 0] = cardinal_line_z[:, 1] = 0

        lines = (cardinal_line_x, cardinal_line_y, cardinal_line_z)
        axes  = (x_axis, y_axis, z_axis)

        for line_cloud, ax in zip(lines, axes):
            fit_line, variance = fitLineTo(line_cloud)

            assert np.all(fit_line == ax) or \
                    np.all((-1 * fit_line) == ax)
