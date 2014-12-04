#!/usr/bin/env python3
# engo699/utility.py
# Jeremy Steward

"""
This module sports some functions which may be useful in dealing with point
clouds in general.
"""

import numpy as np
from numpy import sin, cos

class RotationMatrix(object):
    """
    An object (with no instance methods) which returns a 3D rotation matrix from
    either Euler angles, Quaternions, etc.
    """


    @staticmethod
    def R1(t):
        return np.array([[1,       0,      0],
                         [0,  cos(t), sin(t)],
                         [0, -sin(t), cos(t)]])

    @staticmethod
    def R2(t):
        return np.array([[cos(t), 0, -sin(t)],
                         [     0, 1,       0],
                         [sin(t), 0,  cos(t)]])

    @staticmethod
    def R3(t):
        return np.array([[ cos(t), sin(t),  0],
                         [-sin(t), cos(t),  0],
                         [      0,      0,  1]])

    @staticmethod
    def fromEulerAngles(omega, phi, kappa):
        """
        Generates a 3D rotation matrix from Euler angles in degrees.
        """
        o = omega * np.pi / 180
        p = phi   * np.pi / 180
        k = kappa * np.pi / 180

        r1 = RotationMatrix.R1(o)
        r2 = RotationMatrix.R2(p)
        r3 = RotationMatrix.R3(k)

        M = r3 * r2 * r1
        return M

    @staticmethod
    def fromExponentialMap(angle, axis):
        """
        Generates a 3D rotation matrix from Angle-Axis parameters (angle in
        degrees). Angle-Axis is also known as the Exponential Map, which is
        the proper mathematical name for the concept. Suck it nerds.
        """
        n1 = axis[0]
        n2 = axis[1]
        n3 = axis[2]
        t  = angle * np.pi / 180

        M = np.zeros((3,3))
        M[0, 0] = cos(t) + (n1 ** 2) * (1 - cos(t))
        M[0, 1] = (n1 * n2) * (1 - cos(t)) - (n3 * sin(t))
        M[0, 2] = (n1 * n3) * (1 - cos(t)) + (n2 * sin(t))
        M[1, 0] = (n1 * n2) * (1 - cos(t)) + (n3 * sin(t))
        M[1, 1] = cos(t) + (n2 ** 2) * (1 - cos(t))
        M[1, 2] = (n2 * n3) * (1 - cos(t)) - (n1 * sin(t))
        M[2, 0] = (n1 * n3) * (1 - cos(t)) - (n2 * sin(t))
        M[2, 1] = (n2 * n3) * (1 - cos(t)) + (n1 * sin(t))
        M[2, 2] = cos(t) + (n3 ** 2) * (1 - cos(t))

        return M

    @staticmethod
    def fromQuaternion(q0, q1, q2, q3):
        """
        Generates a 3D rotation matrix from a Quaternion.
        """
        M = np.zeros((3,3))
        M[0, 0] = (q0 ** 2) + (q1 ** 2) - (q2 ** 2) - (q3 ** 2)
        M[0, 1] = 2 * (q1 * q2 - q0 * q3)
        M[0, 2] = 2 * (q1 * q3 - q0 * q2)
        M[1, 0] = 2 * (q1 * q2 + q0 * q3)
        M[1, 1] = (q0 ** 2) - (q1 ** 2) + (q2 ** 2) - (q3 ** 2)
        M[1, 2] = 2 * (q2 * q3 - q0 * q1)
        M[2, 0] = 2 * (q1 * q3 - q0 * q2)
        M[2, 1] = 2 * (q2 * q3 + q0 * q1)
        M[2, 2] = (q0 ** 2) - (q1 ** 2) - (q2 ** 2) + (q3 ** 2)
