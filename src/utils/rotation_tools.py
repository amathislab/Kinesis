# This file contains code adapted from:
#
# 1. SMPLSim (https://github.com/ZhengyiLuo/SMPLSim)
#    Copyright (c) 2024 Zhengyi Luo
#    Licensed under the BSD 3-Clause License.

import numpy as np
from scipy.spatial.transform import Rotation as sRot

def fit_plane(points):
    centroid = points.mean(axis=0)
    centered_points = points - centroid
    _, _, vh = np.linalg.svd(centered_points)
    normal_vector = vh[-1]
    if normal_vector[2] < 0:
        normal_vector = -normal_vector

    # Normalize
    normal_vector /= np.linalg.norm(normal_vector)

    return normal_vector, centroid

def compute_alignment_rotation(normal_vector):
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(normal_vector, z_axis)
    if np.linalg.norm(rotation_axis) < 1e-6:
        if np.dot(normal_vector, z_axis) > 0:
            rotation = sRot.identity()
        else:
            rotation = sRot.from_euler('x', np.pi, degrees=False)
    else:
        # Rotation angle
        rotation_angle = np.arccos(np.clip(np.dot(normal_vector, z_axis), -1.0, 1.0))
        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation = sRot.from_rotvec(rotation_angle * rotation_axis)

    return rotation

def apply_rotation(points, rotations, rotation):
    rotated_translations = rotation.apply(points)

    rotation_matrices = sRot.from_rotvec(rotations).as_matrix()

    adjusted_rotation_matrices = rotation.as_matrix() @ rotation_matrices

    adjusted_rotations = sRot.from_matrix(adjusted_rotation_matrices).as_rotvec()

    return rotated_translations, adjusted_rotations

def correct_humanoid_motion(root_translations, root_rotations):
    normal_vector, centroid = fit_plane(root_translations)
    alignment_rotation = compute_alignment_rotation(normal_vector)
    aligned_translations, aligned_rotations = apply_rotation(root_translations, root_rotations, alignment_rotation)


    return aligned_translations, aligned_rotations

def get_local_facting_axis(root_rotation):
    rotation_matrix = sRot.from_rotvec(root_rotation).as_matrix()
    local_facing_axis = rotation_matrix[:, 0]
    local_facing_axis /= np.linalg.norm(local_facing_axis)
    return local_facing_axis

def create_rotation_around_axis(axis, angle):
    axis_normalized = axis / np.linalg.norm(axis)
    rotation = sRot.from_rotvec(angle * axis_normalized)
    return rotation

def rotate_root_around_facing_axis(root_rotations, angle):
    adjusted_rotations = []
    for rotation in root_rotations:
        local_facing_axis = get_local_facting_axis(rotation)
        rotation_around_axis = create_rotation_around_axis(local_facing_axis, angle)
        root_rotation_obj = sRot.from_rotvec(rotation)
        new_rotation = root_rotation_obj * rotation_around_axis
        adjusted_rotations.append(new_rotation.as_rotvec())

    return adjusted_rotations