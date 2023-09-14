# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import util
import numpy as np
import torch
import nvdiffrast.torch as dr

###############################################################################
# Functions adapted from https://github.com/NVlabs/nvdiffrec
###############################################################################

def get_random_camera_batch(batch_size, fovy = np.deg2rad(45), iter_res=[512,512], cam_near_far=[0.1, 1000.0], cam_radius=3.0, device="cuda"):
    def get_random_camera():
        proj_mtx = util.perspective(fovy, iter_res[1] / iter_res[0], cam_near_far[0], cam_near_far[1])
        mv     = util.translate(0, 0, -cam_radius) @ util.random_rotation_translation(0.25)
        mvp    = proj_mtx @ mv
        return mv, mvp
    mv_batch = []
    mvp_batch = []
    for i in range(batch_size):
        mv, mvp = get_random_camera()
        mv_batch.append(mv)
        mvp_batch.append(mvp)
    return torch.stack(mv_batch).to(device), torch.stack(mvp_batch).to(device)

def get_rotate_camera(itr, fovy = np.deg2rad(45), iter_res=[512,512], cam_near_far=[0.1, 1000.0], cam_radius=3.0, device="cuda"):
    proj_mtx = util.perspective(fovy, iter_res[1] / iter_res[0], cam_near_far[0], cam_near_far[1])

    # Smooth rotation for display.
    ang    = (itr / 10) * np.pi * 2
    mv     = util.translate(0, 0, -cam_radius) @ (util.rotate_x(-0.4) @ util.rotate_y(ang))
    mvp    = proj_mtx @ mv
    return mv.to(device), mvp.to(device)

def xfm_points(points, matrix):
    '''Transform points.
    Args:
        points: Tensor containing 3D points with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)
    Returns:
        Transformed points in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''
    out = torch.matmul(
        torch.nn.functional.pad(points, pad=(0, 1), mode='constant', value=1.0), torch.transpose(matrix, 1, 2))
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_points contains inf or NaN"
    return out

def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(
        attr, rast, attr_idx, rast_db=rast_db,
        diff_attrs=None if rast_db is None else 'all')

def render_mesh(mesh, mv, mvp, iter_res, return_types = ["mask", "depth"], white_bg=False):
    v_pos_clip = xfm_points(mesh.vertices.unsqueeze(0), mvp)  # Rotate it to camera coordinates
    rast, db = dr.rasterize(
        dr.RasterizeGLContext(), v_pos_clip, mesh.faces.int(), iter_res)

    out_dict = {}
    for type in return_types:
        if type == "mask" :
            img = dr.antialias((rast[..., -1:] > 0).float(), rast, v_pos_clip, mesh.faces.int()) 
        elif type == "depth":
            v_pos_cam = xfm_points(mesh.vertices.unsqueeze(0), mv)
            img, _ = interpolate(v_pos_cam, rast, mesh.faces.int())
        elif type == "normal" :
            normal_indices = (torch.arange(0, mesh.nrm.shape[0], dtype=torch.int64, device='cuda')[:, None]).repeat(1, 3)
            img, _ = interpolate(mesh.nrm.unsqueeze(0).contiguous(), rast, normal_indices.int())
        if white_bg:
            bg = torch.ones_like(img)
            alpha = (rast[..., -1:] > 0).float() 
            img = torch.lerp(bg, img, alpha)
        out_dict[type] = img

        
    return out_dict