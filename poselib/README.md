# poselib

`poselib` is a library for loading, manipulating, and retargeting skeleton poses and motions.

## poselib.core
- `poselib.core.rotation3d`: A set of Torch JIT functions for computing quaternions, transforms, and rotation/transformation matrices.
    - `quat_*` manipulate and create quaternions in [x, y, z, w] format (where w is the real component).
    - `transform_*` handle 7D transforms in [quat, pos] format.
    - `rot_matrix_*` handle 3x3 rotation matrices.
    - `euclidean_*` handle 4x4 Euclidean transformation matrices.
- `poselib.core.tensor_utils`: Provides loading and saving functions for PyTorch tensors.

## poselib.skeleton
- `poselib.skeleton.skeleton3d`: Utilities for loading and manipulating skeleton poses, and retargeting poses to different skeletons.
    - `SkeletonTree` is a class that stores a skeleton as a tree structure. This describes the skeleton topology and joints.
    - `SkeletonState` describes the static state of a skeleton, and provides both global and local joint angles.
    - `SkeletonMotion` describes a time-series of skeleton states and provides utilities for computing joint velocities.