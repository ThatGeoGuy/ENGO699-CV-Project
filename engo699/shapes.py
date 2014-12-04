#!/usr/bin/env python3
# engo699/shapes.py
# Jeremy Steward
# Code to create points sets of basic geometric shapes

import numpy as np
from itertools import product

from engo699.utility import RotationMatrix

def createCube(dimension, scale, centroid = None):
    """
    Creates a 'cube' of points, centered about the three-vector specified
    by centroid, with the side length equal to `scale`

    Parameters
    ==========
      dimension:    specifies the number of points along a single edge of the
                    cube. e.g. if you specify 3 then you will receive a cube
                    of 27 points, or a 3x3x3 cube of points.

      scale:        specifies the distance between two corners of the cube.
                    e.g. the length of a side in physical space so a cube with
                    scale 1 should have a volume of 1 [unit ** 3]

      centroid:     optional parameter. Specifies the point you want to centre
                    the cube about. Should be some iterable object with length
                    equal to 3.

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

def createPlane(dimension, scale, normal=None, centroid = None, tolerance = None):
    """
    Creates a plane of points, centred about the three-vector specified
    by centroid, with the length of a side of the plane equal to `scale`

    Parameters
    ==========
      dimension:    specifies the number of points along the length of one edge
                    of the plane.

      scale:        specifies the distance along one side of the plane.

      normal:       optional parameter. A three-vector specifying the plane
                    normal. This defines the plane orientation.

      centroid:     optional parameter. Specifies the point with which
                    the plane is centred about.

      tolerance:    Any absolute value less than this after the resulting
                    plane rotation will be set to zero.

    Returns
    =======
      pts:          a point cloud representing the plane. This is implemented
                    as an N x 3 array of coordinates, where N is the dimension
                    squared.
    """
    if normal is None:
        normal = np.array([0,0,1])
    else:
        normal = np.array(normal)
        normal /= np.linalg.norm(normal, 2)

    if centroid is None:
        centroid = np.array([0,0,0])
    else:
        centroid = np.array(centroid)

    # This is so small floating point errors are removed.
    if tolerance is None:
        tolerance = 1e-12

    space = list(np.linspace(-scale / 2, scale / 2, dimension))
    pts   = np.array(list(product(space, repeat=2)))
    pts   = centroid - np.append(pts, np.zeros((len(pts), 1)), axis=1)

    # Compute the Exponential Map
    # The second axis is [0,0,1] because that's the default rotation for the
    # plane that is created.
    axis = np.cross([0,0,1], normal)
    angle = np.arccos(np.dot([0,0,1], normal)) * 180 / np.pi # for degrees

    M = RotationMatrix.fromExponentialMap(angle, axis)
    pts = np.transpose(np.dot(M, pts.T))
    pts[np.abs(pts) < tolerance] = 0
    return pts
