#!/usr/bin/env python3
# iss_backproject.py
# Jeremy Steward

"""
Performs the filtering, Implicit Shape Signature keypoint detection, then
displays the projection of the detected keypoints back onto the original
images.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

from engo699.detectors import ISSKeypointDetector
from engo699.filters import RadialOutlierFilter

fileno = 3

def main(original_dir, filename):
    """
    Main procedure.
    """
    # TODO: Fix below
    if not os.path.isdir(original_dir) or not os.path.isfile(filename):
        raise ValueError("What are you inputting, it's a directory and a file.")

    pts = np.loadtxt(filename)
    xyz = pts[:,:3]

    camera_files = [f for f in os.listdir(original_dir)
                      if re.search("Camera_[1-3]_{0:d}.txt$".format(fileno), f)]

    camera_amp_data = [np.loadtxt(os.path.join(original_dir, f))[:, 3] for f in camera_files]
    camera_amp_data = [(c / np.max(c)) ** 0.25 for c in camera_amp_data]

    rof = RadialOutlierFilter(xyz, 8, 0.040)
    xyz_filtered = rof.filterPoints()

    iss = ISSKeypointDetector(xyz_filtered, 8, 0.040, 0.040, 0.975, 0.975)
    keypoints = iss.detectKeypoints()

    camera_keypoints = [[] for _ in range(len(camera_amp_data))]
    for point in keypoints:
        i = np.argmax(np.sum(xyz == point, axis=1))
        cam = int(pts[i, 3])
        camera_keypoints[cam-1].append(i)

    for j, kypts in enumerate(camera_keypoints):
        plt.figure(j+1)
        plt.hold(True)
        plt.imshow(camera_amp_data[j].reshape(176, 144), cmap=plt.cm.gray)
        rows = pts[kypts, 4]
        cols = pts[kypts, 5]
        plt.scatter(rows, cols, c='r')
    plt.show()

if __name__ == "__main__":
    # >implying I need an argument parser.
    # TODO: Clean up spaghetti below
    range_image_directory = "D:/Users/jeremy/documents/Projects/2014-10-Herve_Motion_Data/Individual_Walking"
    file_of_interest = "D:/Users/jeremy/documents/Projects/2014-12-Herve_Motion_Data_With_Image_Coors/Walking/Walking ({0:d}).txt".format(fileno)
    main(range_image_directory, file_of_interest)
