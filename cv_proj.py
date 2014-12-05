#!/usr/bin/env python3
# cv_proj.py
# Jeremy Steward

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from engo699.filters.WLOP import WLOP
from engo699.shapes import createCube

def main(filepath):
    pts = np.loadtxt(filepath)[:,:-1]

    ## WLOP Stuff
    # template = WLOP(pts, 0.0001, 20)
    # projected_pts = template.computeProjection(6000, 1.100, 0.45)

    # Plot both point cloud and projected point cloud
    # fig1 = plt.figure(1)
    # ax = fig1.add_subplot(111, projection='3d')
    # plt.plot(pts[:,0], pts[:,1], pts[:,2], 'bo')
    # plt.plot(projected_pts[:,0], projected_pts[:,1], projected_pts[:,2], 'ro')
    # plt.show()

    return 0

if __name__ == "__main__":
    # Initialize parser for program
    parser = argparse.ArgumentParser(
        description="Performs key point detection on 3D body points clouds"
    )
    # Positional arguments
    parser.add_argument("filepath", type=str, help="path to the point cloud file you wish to process.")

    args = parser.parse_args()
    main(args.filepath)
