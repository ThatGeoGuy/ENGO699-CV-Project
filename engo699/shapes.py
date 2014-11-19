#!/usr/bin/env python3
# engo699/shapes.py
# Jeremy Steward
# Code to create points sets of basic geometric shapes

import numpy as np
from itertools import product

def createCube(dimension, scale, centroid = None):
    """
    Creates a default 'cube' of points, centered about the three-vector
    specified by centroid, with size

    Parameters
    ==========
      dimension:    specifies the number of points along a single edge of the
                    cube. e.g. if you specify 3 then you will receive a cube
                    of 27 points, or a 3x3x3 cube of points.

      scale:        specifies the physical distance between two corners of
                    the cube. e.g. the length of a side in physical space so
                    a cube with scale 1m should have a volume of 1 [m ** 3]

      centroid:     optional parameter. Specifies the point you want to centre
                    the cube about.

    Returns
    =======
      pts:          a point cloud representing the cube. This is implemented
                    as an N x 3 array of coordinates, where N is dimension
                    cubed.
    """
    if centroid is None:
        centroid = np.array([0, 0, 0])
    else:
        centroid = np.array(centroid)

    space = list(np.linspace(-scale / 2, scale / 2, dimension))
    pts   = np.array(list(product(space, repeat=3)))
    return centroid - pts
