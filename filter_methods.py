#!/usr/bin/env python3
# cv_proj.py
# Jeremy Steward

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from engo699.filters import WLOP, RadialOutlierFilter, NonPlanarOutlierFilter
from engo699.shapes import createCube

def main(filepath, output_dir):
    pts = np.loadtxt(filepath)[:,:3]

    if not output_dir:
        output_dir = "./"
    elif not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    ###################################################################
    ## WLOP Stuff
    ###################################################################
    # template = WLOP(pts, 0.0001, 20)
    # projected_pts = template.computeProjection(6000, 1.0, 0.45)

    # Plot both point cloud and projected point cloud
    # fig1 = plt.figure(1)
    # ax = fig1.add_subplot(111, projection='3d')
    # plt.plot(pts[:,0], pts[:,1], pts[:,2], 'bo')
    # plt.plot(projected_pts[:,0], projected_pts[:,1], projected_pts[:,2], 'ro')
    # plt.show()
    ###################################################################

    ###################################################################
    ## Radial Outlier Filter Stuff
    ###################################################################
    num_nn_rof = 8
    radius = 0.042
    rof = RadialOutlierFilter(pts, num_nn_rof, radius)
    rof_pts = rof.filterPoints()
    filtered_rof_i = rof.filtered_indices

    out1 = np.append(pts, np.zeros((pts.shape[0], 1)), axis=1)
    out1[filtered_rof_i, 3] = 128
    np.savetxt(os.path.join(output_dir, "ROF.txt"), out1)
    np.savetxt(os.path.join(output_dir, "radial_only.txt"), rof_pts)
    ###################################################################

    ###################################################################
    ## Non Planar Outlier Filter Stuff
    ###################################################################
    num_nn_npof = 9
    threshold = 1e-4
    npof = NonPlanarOutlierFilter(pts, num_nn_npof, threshold)
    npof_pts = npof.filterPoints()
    filtered_npof_i = npof.filtered_indices

    out2 = np.append(pts, np.zeros((pts.shape[0], 1)), axis=1)
    out2[filtered_npof_i, 3] = 128
    np.savetxt(os.path.join(output_dir, "NPOF.txt"), out2)
    ###################################################################

    ###################################################################
    ## Both NPOF and ROF sequentially
    ###################################################################
    npof_2 = NonPlanarOutlierFilter(pts, num_nn_npof, threshold)
    npof_2_pts = npof_2.filterPoints()
    filtered_npof_2 = npof_2.filtered_indices

    rof_2 = RadialOutlierFilter(npof_2_pts, num_nn_rof, radius)
    final_pts = rof_2.filterPoints()
    filtered_rof_2 = rof_2.filtered_indices

    final_pts = np.append(final_pts, np.zeros((final_pts.shape[0], 1)), axis=1)
    filter1 = np.append(pts[filtered_npof_2, :],
            128 * np.ones((pts[filtered_npof_2, :].shape[0], 1)), axis=1)
    filter2 = np.append(npof_2_pts[filtered_rof_2, :],
            255 * np.ones((pts[filtered_rof_2, :].shape[0], 1)), axis=1)

    final_pts = np.append(final_pts, filter1, axis=0)
    final_pts = np.append(final_pts, filter2, axis=0)
    np.savetxt(os.path.join("final.txt"), rof_2.filterPoints())
    np.savetxt(os.path.join("final_with_colors.txt"), final_pts)
    ###################################################################

    return 0

if __name__ == "__main__":
    # Initialize parser for program
    parser = argparse.ArgumentParser(
        description="Performs key point detection on 3D body points clouds"
    )
    # Positional arguments
    parser.add_argument("filepath", type=str,
            help="path to the point cloud file you wish to process.")
    parser.add_argument("-o", type=str, dest="output_dir",
            help="specify a folder for the OUTPUT of the program.")

    args = parser.parse_args()
    main(args.filepath, args.output_dir)
