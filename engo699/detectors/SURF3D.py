#!/usr/bin/env python3
# engo699/detectors/SURF3D.py
# Jeremy Steward

"""
Defines the point detector used by the 3D SURF algorithm.
"""

import numpy as np
from scipy import spatial
from scipy.signal import fftconvolve

from engo699.geometry import centroid
from engo699.shapes import createCube

class SURF3DKeypointDetector(object):
    """
    A class to manage the state and functionality used in the 3D SURF
    keypoint detector.
    """

    def __init__(self, points, num_scales, initial_voxel_dim=256):
        """
        Initializes point cloud data, number of scales, voxel dimensions, and
        then computes the max distance (extent) for our voxel grid.
        """
        self.pts        = points.copy()
        self.num_scales = num_scales
        self.vdim       = initial_voxel_dim

        kdt = spatial.KDTree(pts)
        max_dist = 0
        for pt in points:
            dists, nn_indices = kdt.query(pt, k=len(points), p=2)

            if np.max(np.abs(dists[np.isfinite(dists)])) > max_dist:
                max_dist = np.max(np.abs(dists[np.isfinite(dists)]))

        # the extent is 2 times the max dist because it stretches on both sides
        # of the centroid i.e.: Xmin <---------- C ----------> Xmax
        # Xmax - Xmin would be the extent.
        self.extent   = 2 * max_dist
        self.centroid = centroid(points)
        return

    @staticmethod
    def haar1(axis="x"):
        """
        Returns the 1st Haar Wavelet approximation (in 3D voxels) along the
        desired axis.
        """
        if axis not in ("x", "y", "z"):
            raise ValueError("SURF3DKeypointDetector.haar1 : Invalid axis selection.")

        haar = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if axis == "x":
                        haar[i,j,k] = 1 if i != 1 else -2
                    elif axis == "y":
                        haar[i,j,k] = 1 if j != 1 else -2
                    elif axis == "z":
                        haar[i,j,k] = 1 if k != 1 else -2
        return haar

    @staticmethod
    def haar2(axis1="x", axis2="y"):
        """
        Returns the 2nd Haar Wavelet approximation (in 3D voxels) along the
        desired two axes.
        """
        if axis1 not in ("x", "y", "z"):
            raise ValueError("SURF3DKeypointDetector.haar2 : Invalid axis1 selection.")
        if axis2 not in ("x", "y", "z"):
            raise ValueError("SURF3DKeypointDetector.haar2 : Invalid axis2 selection.")
        if axis1 == axis2:
            raise ValueError("SURF3DKeypointDetector.haar2 : Axes cannot be equal.")

        # THERE BE DRAGONS BELOW
        # This was the easiest way to do this. Deadlines etc.
        haar = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if axis1 == "x":
                        if axis2 == "y":
                            haar[i,j,k] = 0 if (i == 1 or j == 1) \
                                    else 1 if (i < 1 and j > 1) or (i > 1 and j < 1) \
                                    else -1
                        elif axis2 == "z":
                            haar[i,j,k] = 0 if (i == 1 or k == 1) \
                                    else 1 if (i < 1 and k > 1) or (i > 1 and k < 1) \
                                    else -1
                    elif axis1 == "y":
                        if axis2 == "x":
                            haar[i,j,k] = 0 if (j == 1 or i == 1) \
                                    else -1 if (j < 1 and i > 1) or (j > 1 and i < 1) \
                                    else 1
                        elif axis2 == "z":
                            haar[i,j,k] = 0 if (j == 1 or k == 1) \
                                    else 1 if (j < 1 and k > 1) or (j > 1 and k < 1) \
                                    else -1
                    elif axis1 == "z":
                        if axis2 == "x":
                            haar[i,j,k] = 0 if (k == 1 or i == 1) \
                                    else -1 if (k < 1 and i > 1) or (k > 1 and i < 1) \
                                    else 1
                        elif axis2 == "y":
                            haar[i,j,k] = 0 if (k == 1 or j == 1) \
                                    else -1 if (k < 1 and j > 1) or (k > 1 and j < 1) \
                                    else 1
        return haar

    def voxelizePoints(self, vdim):
        """
        Returns a voxel grid for the point cloud pts.
        Extent should be the full extent across the entire voxel cube.
        """
        voxels = np.zeroes((vdim, vdim, vdim))
        vox_width = self.extent / vdim

        xmin = self.centroid[0] - (self.extent / 2)
        ymin = self.centroid[1] - (self.extent / 2)
        zmin = self.centroid[2] - (self.extent / 2)

        for pt in self.pts:
            i = int((pt[0] - xmin) / vox_width)
            j = int((pt[1] - ymin) / vox_width)
            k = int((pt[2] - zmin) / vox_width)

            voxels[i, j, k] += 1
        return voxels

    def detectKeypoints(self):
        """
        Performs the keypoint detection, and returns both the detected
        keypoints as well as the scales they were detected at.
        """
        # Haar wavelets, precomputed for efficiency
        haar1x = SURF3DKeypointDetector.haar1("x")
        haar1y = SURF3DKeypointDetector.haar1("y")
        haar1z = SURF3DKeypointDetector.haar1("z")

        haar2xy = SURF3DKeypointDetector.haar2("x", "y")
        haar2xz = SURF3DKeypointDetector.haar2("x", "z")
        haar2yx = SURF3DKeypointDetector.haar2("y", "x")
        haar2yz = SURF3DKeypointDetector.haar2("y", "z")
        haar2zx = SURF3DKeypointDetector.haar2("z", "x")
        haar2zy = SURF3DKeypointDetector.haar2("z", "y")

        for n in self.num_scales:
            scale = 2 ** n
            current_vdim = self.vdim / scale
            voxels = self.voxelizePoints(current_vdim)

            # Haar wavelet 1
            Lxx = fftconvolve(voxels, haar1x, mode="same")
            Lyy = fftconvolve(voxels, haar1y, mode="same")
            Lzz = fftconvolve(voxels, haar1z, mode="same")
            # Haar wavelet 2
            Lxy = fftconvolve(voxels, haar2xy, mode="same")
            Lxz = fftconvolve(voxels, haar2xz, mode="same")
            Lyx = fftconvolve(voxels, haar2yx, mode="same")
            Lyz = fftconvolve(voxels, haar2yz, mode="same")
            Lzx = fftconvolve(voxels, haar2zx, mode="same")
            Lzy = fftconvolve(voxels, haar2zy, mode="same")

            saliencies = np.zeros((current_vdim, current_vdim, current_vdim))

            # Sometimes array indices are the clearest looping construct
            for i in range(current_vdim):
                for j in range(current_vdim):
                    for k in range(current_vdim):
                        H = np.array([[Lxx[i,j,k], Lxy[i,j,k], Lxz[i,j,k]],
                                      [Lyx[i,j,k], Lyy[i,j,k], Lyz[i,j,k]],
                                      [Lzx[i,j,k], Lzy[i,j,k], Lzz[i,j,k]]])

                        saliencies[i,j,k] = np.linalg.det(H)

            # TODO: Clean this up, possibly don't recompute after calling voxelizePoints
            vox_width = self.extent / current_vdim
            xmin = self.centroid[0] - (self.extent / 2)
            ymin = self.centroid[1] - (self.extent / 2)
            zmin = self.centroid[2] - (self.extent / 2)

            keypoints = []
            for i in range(1, current_vdim - 1):
                for j in range(1, current_vdim - 1):
                    for k in range(1, current_vdim - 1):
                        if np.all(saliencies[i,j,k] >= saliencies[i-1:i+2, j-1:j+2, k-1:k+2]):
                            # Plus 0.5 because we want the middle of the voxel
                            # location, which is 50% of the voxel width.
                            xnew = ((i + 0.5) * vox_width) + xmin
                            ynew = ((j + 0.5) * vox_width) + ymin
                            znew = ((k + 0.5) * vox_width) + zmin

                            keypoints.append([xnew, ynew, znew, scale])
        return np.array(keypoints)[:,:3], np.array(keypoints)[:,4]
