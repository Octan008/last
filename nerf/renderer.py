from enum import unique
from multiprocessing.managers import ValueProxy
from xml.etree.ElementPath import prepare_star
from cv2 import transform
import numpy as np
from tqdm import trange
import trimesh

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching
from .utils import custom_meshgrid
from .render_util import *
import copy
import time
dry_black = False

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


@torch.cuda.amp.autocast(enabled=False)
def near_far_from_bound(rays_o, rays_d, bound, type='cube', bound_rate = None, bound_box = None):
    # rays: [B, N, 3], [B, N, 3]
    # bound: int, radius for ball or half-edge-length for cube
    # return near [B, N, 1], far [B, N, 1]

    radius = rays_o.norm(dim=-1, keepdim=True)
    # if bound_rate is not None and bound_box is None:
    #     bound = bound_rate * bound
    if type == 'sphere':
        near = radius - bound # [B, N, 1]
        far = radius + bound

    elif type == 'cube':
        if bound_box is not None:
            bound_box = bound*bound_box
            tmin = (bound_box[0] - rays_o) / (rays_d + 1e-15) # [B, N, 3]
            # print("tmi", tmin.shape)
            tmax = (bound_box[1] - rays_o) / (rays_d + 1e-15)
        else:
            tmin = (-bound - rays_o) / (rays_d + 1e-15) # [B, N, 3]
            tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        # if far < near, means no intersection, set both near and far to inf (1e9 here)
        mask = far < near
        near[mask] = 1e9
        far[mask] = 1e9
        # restrict near to a minimal value
        near = torch.clamp(near, min=0.05)

    return near, far

def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()

def clamp_xyz(xyz, bound, bound_rate, box=None):
        if box is not None:
            return torch.stack([
                xyz[...,0].clamp(bound*box[0, 0], bound*box[1, 0]),
                xyz[...,1].clamp(bound*box[0, 1], bound*box[1, 1]),
                xyz[...,2].clamp(bound*box[0, 2], bound*box[1, 2])
            ],dim=-1)
        return torch.stack([
            xyz[...,0].clamp(-bound*bound_rate[0], bound*bound_rate[0]),
            xyz[...,1].clamp(-bound*bound_rate[1], bound*bound_rate[1]),
            xyz[...,2].clamp(-bound*bound_rate[2], bound*bound_rate[2])
        ],dim=-1)



class NeRFRenderer(nn.Module):
    def __init__(self,
            bound=1,
            cuda_ray=False,
            density_scale=1, # scale up deltas (or sigmas), to make the density grid more sharp. larger value than 1 usually improves performance.
            skeleton_mode=False,
            initiation=True,
            mix_render=False,
            skeleton=None,
            heatmap=False,
            pose_only=False,
            weight_only=False,
            ):
        super().__init__()

        self.skeleton = skeleton
        self.joints = listify_skeleton(self.skeleton)
        self.bound = bound
        self.density_scale = density_scale
        self.heatmap = heatmap
        self.pose_only = pose_only
        self.weight_only=weight_only
        self.nerfonly_mode = False
        self.bound_box_rate = torch.transpose(torch.tensor([
                [-0.5, 0.8],[-0.1, 0.5],[-0.55, 0.55]
                # [-1.0, 1.0],[-1.0, 1.0],[-1.0, 1.0]
            ], device = torch.device("cuda:0")
        ), 0, 1)

        # extra state for cuda raymarching
        self.cuda_ray = cuda_ray
        self.skeleton_mode = skeleton_mode
        self.initiation = initiation
        self.mix_render = mix_render
        # self.bound_rate = torch.tensor([0.7, 0.6, 0.6], device=torch.device("cuda:0"))
        self.bound_rate = torch.tensor([1.0, 1.0, 1.0], device=torch.device("cuda:0"))
        if cuda_ray:
            # density grid
            density_grid = torch.zeros([128] * 3)
            self.register_buffer('density_grid', density_grid)
            self.mean_density = 0
            self.iter_density = 0
            # step counter
            step_counter = torch.zeros(16, 2, dtype=torch.int32) # 16 is hardcoded for averaging...
            self.register_buffer('step_counter', step_counter)
            self.mean_count = 0
            self.local_step = 0

    def bound_box_mask(self, rays_o, rays_d):
        bb = self.bound_box_rate * self.bound
        ta = (-rays_o + bb[0]) / rays_d #[N, 3]
        tb = (-rays_o + bb[1]) / rays_d #[N, 3]
        ts = []
        for i in range(3):
            if i == 0:
                ts.append(torch.min(ta[..., 0], tb[..., 0]))
                ts.append(torch.max(ta[..., 0], tb[..., 0]))
            else:
                ts[0] = torch.max(torch.min(ta[...,i], tb[..., i]), ts[0])
                ts[1] = torch.min(torch.max(ta[...,i], tb[..., i]), ts[1])
        return ts[0] < ts[1]


    def forward(self, x, d):
        raise NotImplementedError()

    def density(self, x):
        raise NotImplementedError()

    def cast(self, x,  skeleton_pose):
        raise NotImplementedError()
    def set_castModel(self, castModel):
        self.cast_model = castModel
    def set_skeleton_mode(self, mode):
        self.skeleton_mode=mode
    def set_nerfonly_mode(self, mode):
        self.nerfonly_mode = mode

    def reset_extra_state(self):
        if not self.cuda_ray:
            return
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0

    def run(self, rays_o, rays_d, num_steps, upsample_steps, bg_color, perturb, joints = None, skeleton_pose=None):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device
        # bound_rate = [0.7, 0.6, 0.4]

        # sample steps
        near, far = near_far_from_bound(rays_o, rays_d, self.bound, type='cube', bound_rate = self.bound_rate, bound_box=self.bound_box_rate)
        # near, far = near_far_from_bound(rays_o, rays_d, self.bound, type='cube', bound_rate = self.bound_rate)

        #print(f'near = {near.min().item()} ~ {near.max().item()}, far = {far.min().item()} ~ {far.max().item()}')
        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0) # [1, T]
        z_vals = z_vals.expand((N, num_steps)) # [N, T]
        z_vals = near + (far - near) * z_vals # [N, T], in [near, far]

        # perturb z_vals
        sample_dist = (far - near) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            # z_vals = z_vals.clamp(near, far) # avoid out of bounds xyzs.

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        tmp = xyzs.clone().detach()

        #ToDo: clamp
        xyzs = clamp_xyz(xyzs, self.bound, self.bound_rate, box= self.bound_box_rate)

        if not self.nerfonly_mode:
            dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)


            if self.skeleton_mode and not self.pose_only:
                if self.initiation or (self.weight_only and self.training):
                    gt_xyzs = xyzs.clone().detach()
                    gt_dirs = dirs.clone().detach()
                xyzs, dirs, weight = self.cast(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), skeleton_pose)
                if self.weight_only and self.training:
                    return weight, compute_weights(gt_xyzs.reshape(-1, 3), self.joints)

            if joints is not None or self.pose_only:
                # xyzs = cast_positions(xyzs.reshape(-1, 3), joints).reshape(-1, num_steps, 3)
                # gt_xyzs = xyzs.clone().detach()
                # gt_dirs = dirs.clone().detach()
                if skeleton_pose is None:
                    #skeletonpose がないとき、現状のskeletonから創る
                    poses = self.skeleton.get_listed_rotations()
                    poses2 = []
                    for ii in range(poses.shape[0]):
                        poses2.append(euler_to_quaternion(poses[ii]))
                    skeleton_pose = torch.stack(poses2, dim=0)
                #Todo: Switch
                transforms = self.skeleton.rotations_to_invs_fast(skeleton_pose, type="quaternion")
                # transforms = self.skeleton.rotations_to_invs(skeleton_pose, type="quaternion")

                xyz_slice = xyzs.reshape(-1, 3).shape[0]
                tmp = torch.cat([xyzs.reshape(-1, 3),(xyzs-dirs).reshape(-1, 3)], dim=0)
                weights = compute_weights(xyzs.reshape(-1, 3), self.joints)
                weights = torch.cat([weights, weights], dim=0)

                tmp =  weighted_transformation(tmp, weights, transforms)
                #debug
                xyzs, dirs = tmp[:xyz_slice], tmp[:xyz_slice] - tmp[xyz_slice:]
                # xyzs += torch.tensor([0,0,1]).to(xyzs.device)
                del tmp, weights, transforms



            xyzs = xyzs.reshape(-1, num_steps, 3)
            dirs = dirs.reshape(-1, num_steps, 3)
            #ToDo: clamp
            xyzs = clamp_xyz(xyzs, self.bound, self.bound_rate, self.bound_box_rate)

        # print('[xyzs]', xyzs.shape, xyzs.dtype, xyzs.min().item(), xyzs.max().item())

        #plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())
        # query SDF and RGB
        density_outputs = self.density(xyzs.reshape(-1, 3))


        #sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)
        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():
                deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

                alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs['sigma'].squeeze(-1)) # [N, T]
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+1]
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T]

                # sample new z_vals
                z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1]) # [N, T-1]
                new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], upsample_steps, det=not self.training).detach() # [N, t]

                new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1) # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
                #Todo: clamp
                new_xyzs = clamp_xyz(new_xyzs, self.bound, self.bound_rate, self.bound_box_rate)

                if not self.nerfonly_mode:
                    # print("here?")
                    # raise ValueError("ee")
                    new_dirs = rays_d.view(-1, 1, 3).expand_as(new_xyzs)
                    if self.skeleton_mode and not self.pose_only:
                        if self.initiation:
                            gt_new_xyzs = new_xyzs.clone().detach()
                            gt_new_dirs = new_dirs.clone().detach()
                        new_xyzs, new_dirs, new_weight = self.cast(new_xyzs.reshape(-1, 3), new_dirs.reshape(-1, 3), skeleton_pose)

                    if joints is not None or self.pose_only:
                        # gt_new_xyzs = new_xyzs.clone().detach()
                        # gt_new_dirs = new_dirs.clone().detach()
                        # new_xyzs = cast_positions(new_xyzs.reshape(-1, 3), joints).reshape(-1, num_steps, 3)
                        if skeleton_pose is None:
                            poses = self.skeleton.get_listed_rotations()
                            poses2 = []
                            for ii in range(poses.shape[0]):
                                poses2.append(euler_to_quaternion(poses[ii]))
                            skeleton_pose = torch.stack(poses2, dim=0)
                        #switch
                        transforms = self.skeleton.rotations_to_invs_fast(skeleton_pose, type="quaternion")
                        # transforms = self.skeleton.rotations_to_invs(skeleton_pose, type="quaternion")
                        xyz_slice = new_xyzs.reshape(-1, 3).shape[0]
                        weights = compute_weights(new_xyzs.reshape(-1, 3), self.joints)
                        weights = torch.cat([weights, weights], dim=0)
                        tmp = torch.cat([new_xyzs.reshape(-1, 3),(new_xyzs-new_dirs).reshape(-1, 3)], dim=0)
                        tmp =  weighted_transformation(tmp, weights, transforms)
                        #debug
                        new_xyzs, new_dirs = tmp[:xyz_slice], tmp[:xyz_slice] - tmp[xyz_slice:]
                        # new_xyzs += torch.tensor([0,0,1]).to(new_xyzs.device)

                        del tmp, weights, transforms
                    new_xyzs = new_xyzs.reshape(-1, upsample_steps, 3)
                    new_dirs = new_dirs.reshape(-1, upsample_steps, 3)

            #Todo: clamp
            new_xyzs = clamp_xyz(new_xyzs, self.bound, self.bound_rate, self.bound_box_rate)    

            # only forward new points to save computation
            new_density_outputs = self.density(new_xyzs.reshape(-1, 3))
            #new_sigmas = new_density_outputs['sigma'].view(N, upsample_steps) # [N, t]
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(N, upsample_steps, -1)


            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1) # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=1)
            # print("z-vals2", z_vals.shape, z_index.shape)

            xyzs = torch.cat([xyzs, new_xyzs], dim=1) # [N, T+t, 3]
            xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))
            if not self.nerfonly_mode:
                if self.initiation:
                    gt_xyzs = torch.cat([gt_xyzs, gt_new_xyzs], dim=1) # [N, T+t, 3]
                    gt_xyzs = torch.gather(gt_xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(gt_xyzs))
                    gt_dirs = torch.cat([gt_dirs, gt_new_dirs], dim=1) # [N, T+t, 3]
                    gt_dirs = torch.gather(gt_dirs, dim=1, index=z_index.unsqueeze(-1).expand_as(gt_dirs))


                dirs = torch.cat([dirs, new_dirs], dim=1) # [N, T+t, 3]
                dirs = torch.gather(dirs, dim=1, index=z_index.unsqueeze(-1).expand_as(dirs))


            for k in density_outputs:
                tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))


        deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs['sigma'].squeeze(-1)) # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]


        mask = weights > 1e-4 # hard coded

        if self.nerfonly_mode:
            dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)


        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])


        rgbs = self.color(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), mask=mask.reshape(-1), **density_outputs)



        rgbs = rgbs.view(N, -1, 3) # [N, T+t, 3]
        # if self.heatmap:
        #     rgbs= torch.tensor([0,1,0], device=rgbs.device).unsqueeze(0).unsqueeze(0).repeat(rgbs.shape[0], rgbs.shape[1], 1)
        #     joint_weights = self.cast(xyzs.reshape(-1, 3), None, skeleton_pose, return_weight = True)
        #     # print(joint_weights.shape)

        #print(xyzs.shape, 'valid_rgb:', mask.sum().item())
        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1) # [N]
        # calculate depth 
        ori_z_vals = ((z_vals - near) / (far - near)).clamp(0, 1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) # [N, 3], in [0, 1]

        # mix background color
        if bg_color is None:
            bg_color = 1

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        image = image.view(*prefix, 3)

        if self.heatmap:
            joint_weights = self.weight(xyzs.reshape(-1, 3), skeleton_pose)
            return image, joint_weights.view(*prefix, num_steps+upsample_steps, -1)
            # print(joint_weights.shape)



        if self.training:
            depth=None
        else:
            depth = torch.sum(weights * ori_z_vals, dim=-1)
            depth = depth.view(*prefix)
        if torch.isnan(image).any():
            print("image", torch.isnan(image).any())
            print("weights", torch.isnan(weights).any())
            print("rgbs", torch.isnan(rgbs).any())
            print("dirs", torch.isnan(dirs).any())
            print("xyzs", torch.isnan(xyzs).any())
            raise ValueError("rgb-error")
        if self.skeleton_mode and self.training and self.initiation:
            return depth, image, xyzs, gt_xyzs, dirs, gt_dirs
        del xyzs,  dirs,  bg_color, ori_z_vals, weights_sum, density_outputs, deltas
        return depth, image


    def run_cuda(self, rays_o, rays_d, num_steps, upsample_steps, bg_color, perturb, joints = None, skeleton_pose=None, importance_sampling = False):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        if bg_color is None:
            bg_color = 1

        if self.training:
            # setup counter
            # step_start = time.perf_counter()
            counter = self.step_counter[self.local_step % 16]
            counter.zero_() # set to 0
            self.local_step += 1
            xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_grid, self.mean_density, self.iter_density, counter, self.mean_count, perturb, 128, False)

            if self.skeleton_mode:
                # casting_start = time.perf_counter()
                if self.initiation:
                    gt_xyzs = xyzs.clone().detach()
                    gt_dirs = dirs.clone().detach()

                # if torch.isnan(dirs).any():
                #     raise ValueError("dirs-error")
                xyzs, dirs, weight = self.cast(xyzs, dirs, skeleton_pose)
                xyzs = clamp_xyz(xyzs, self.bound, self.bound_rate)
                # dir_norm = torch.norm(dirs, dim=-1, keepdim=True)
                # dirs = torch.where( torch.logical_or(torch.isnan(dir_norm) , dir_norm < 1e-5) , dirs , dirs / dir_norm)


                # print("all_casting : ", time.perf_counter() - casting_start)
            # density_start = time.perf_counter()
            density_outputs = self.density(xyzs) # [M,], use a dict since it may include extra things, like geo_feat for rgb.
            sigmas = density_outputs['sigma']
            sigmas = self.density_scale * sigmas
            weights = raymarching.composite_weights_train(sigmas, deltas, rays) # [M,]
            
            # print("density: ", time.perf_counter() - density_start)
            # rgb_start = time.perf_counter()
            # masked rgb cannot accelerate cuda_ray training, disabled! (mask ratio is only ~50%, cannot beat the mask/unmask overhead.)
            mask = None # weights > 1e-4
            rgbs = self.color(xyzs, dirs, mask=mask, **density_outputs)

            if torch.isnan(rgbs).any():
                raise ValueError("rgb-error")

            #print(f'valid RGB query ratio: {mask.sum().item() / mask.shape[0]} (total = {mask.sum().item()})')

            weights_sum, image = raymarching.composite_rays_train(weights, rgbs, rays, self.bound)
            # print("rgb: ", time.perf_counter() - rgb_start)
            depth = None # currently training do not requires depth
            # print("step: ", time.perf_counter() - step_start)

        else:
            # allocate outputs 
            # if use autocast, must init as half so it won't be autocasted and lose reference.
            # dtype = torch.half if torch.is_autocast_enabled() else torch.float32
            # output should always be float32! only network inference uses half.
            dtype = torch.float32
            
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            if dry_black:
                return depth, image
            
            n_alive = N
            alive_counter = torch.zeros([1], dtype=torch.int32, device=device)

            rays_alive = torch.zeros(2, n_alive, dtype=torch.int32, device=device) # 2 is used to loop old/new
            rays_t = torch.zeros(2, n_alive, dtype=dtype, device=device)

            # pre-calculate near far
            near, far = near_far_from_bound(rays_o, rays_d, self.bound, type='cube')
            near = near.view(N)
            far = far.view(N)

            step = 0
            i = 0
            
            while step < 1024: # hard coded max step

                # count alive rays 
                if step == 0:
                    # init rays at first step.
                    torch.arange(n_alive, out=rays_alive[0])
                    rays_t[0] = near
                else:
                    alive_counter.zero_()
                    raymarching.compact_rays(n_alive, rays_alive[i % 2], rays_alive[(i + 1) % 2], rays_t[i % 2], rays_t[(i + 1) % 2], alive_counter)
                    n_alive = alive_counter.item() # must invoke D2H copy here
                
                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)
                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive[i % 2], rays_t[i % 2], rays_o, rays_d, self.bound, self.density_grid, self.mean_density, near, far, 128, perturb)
                if self.skeleton_mode:
                    xyzs, dirs, weight = self.cast(xyzs, dirs, skeleton_pose)
                    xyzs = clamp_xyz(xyzs, self.bound, self.bound_rate)
                    # dirs = self.cast(dirs, skeleton_pose)
                    # dirs = dirs/torch.norm(dirs, dim=-1, keepdim=True)

                if joints is not None:
                    weights = compute_weights(xyzs, self.joints)
                    poses = self.skeleton.get_listed_rotations()
                    poses2 = []
                    for ii in range(poses.shape[0]):
                        poses2.append(euler_to_quaternion(poses[ii]))
                    poses2 = torch.stack(poses2, dim=0)

                    transforms = self.skeleton.rotations_to_invs_fast(poses2, type="quaternion")

                    # print(transforms)

                    xyzs = weighted_transformation(xyzs, weights, transforms)
                    dirs = xyzs -  weighted_transformation(xyzs - dirs, weights, transforms)
                
                #sigmas, rgbs = self(xyzs, dirs)
                density_outputs = self.density(xyzs) # [M,], use a dict since it may include extra things, like geo_feat for rgb.
                sigmas = density_outputs['sigma']
                sigmas = self.density_scale * sigmas
                
                # no need for weights mask, since we already terminated those rays.
                
                rgbs = self.color(xyzs, dirs, **density_outputs)
                raymarching.composite_rays(n_alive, n_step, rays_alive[i % 2], rays_t[i % 2], sigmas, rgbs, deltas, weights_sum, depth, image)
                # print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')
                

                step += n_step
                i += 1


        
        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        image = image.view(*prefix, 3)


        if depth is not None:
            depth = torch.clamp(depth - near, min=0) / (far - near)
            depth = depth.view(*prefix)
        if self.skeleton_mode and self.training and self.initiation:
            return depth, image, xyzs, gt_xyzs, dirs, gt_dirs
        return depth, image

    @torch.no_grad()
    def mark_untrained_grid(self, poses, intrinsic, S=64):
        # poses: [B, 4, 4]
        # intrinsic: [3, 3]

        if not self.cuda_ray:
            return
        
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)

        B = poses.shape[0]
        
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
    
        resolution = self.density_grid.shape[0]

        half_grid_size = self.bound / resolution
        
        X = torch.linspace(-self.bound + half_grid_size, self.bound - half_grid_size, resolution).split(S)
        Y = torch.linspace(-self.bound + half_grid_size, self.bound - half_grid_size, resolution).split(S)
        Z = torch.linspace(-self.bound + half_grid_size, self.bound - half_grid_size, resolution).split(S)

        count = torch.zeros_like(self.density_grid)
        poses = poses.to(count.device)

        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        lx, ly, lz = len(xs), len(ys), len(zs)
                        # construct points
                        xx, yy, zz = custom_meshgrid(xs, ys, zs)
                        world_xyzs = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).unsqueeze(0).to(count.device) # [1, N, 3]

                        # split batch to avoid OOM
                        head = 0
                        while head < B:
                            tail = min(head + S, B)

                            # world2cam transform (poses is c2w, so we need to transpose)
                            cam_xyzs = world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
                            cam_xyzs = cam_xyzs @ poses[head:tail, :3, :3].transpose(1, 2) # [S, N, 3]
                            
                            # query if point is covered by any camera
                            mask_z = cam_xyzs[:, :, 2] > 0 # [S, N]
                            mask_x = torch.abs(cam_xyzs[:, :, 0]) < cx / fx * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask_y = torch.abs(cam_xyzs[:, :, 1]) < cy / fy * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask = (mask_z & mask_x & mask_y).sum(0).reshape(lx, ly, lz) # [N] --> [lx, ly, lz]

                            # update count 
                            count[xi * S: xi * S + lx, yi * S: yi * S + ly, zi * S: zi * S + lz] += mask
                            head += S
        
        # mark untrained grid as -1
        self.density_grid[count == 0] = -1

        #print(f'[mark untrained grid] {(count == 0).sum()} from {resolution ** 3}')

    @torch.no_grad()
    def update_extra_state(self, decay=0.9, S=128, joints=None, skeleton_pose=None):
        # call before each epoch to update extra states.
        if joints is not None:
            S//=4
        if not self.cuda_ray:
            return 
        
        ### update density grid
        resolution = self.density_grid.shape[0]

        half_grid_size = self.bound / resolution
        
        X = torch.linspace(-self.bound + half_grid_size, self.bound - half_grid_size, resolution).split(S)
        Y = torch.linspace(-self.bound + half_grid_size, self.bound - half_grid_size, resolution).split(S)
        Z = torch.linspace(-self.bound + half_grid_size, self.bound - half_grid_size, resolution).split(S)

        tmp_grid = torch.zeros_like(self.density_grid).to("cuda:0")
    
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    lx, ly, lz = len(xs), len(ys), len(zs)
                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    xyzs = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3]
                    
                    #hatsuki
                    
                    #hatsuki
                    # add noise in [-hgs, hgs]
                    sigmas = torch.zeros(lx,ly,lz).to(tmp_grid.device)
                    trial = 1
                    for i in range(trial):
                        xyzss = (xyzs + (torch.rand_like(xyzs) * 2 - 1) * half_grid_size).clone().detach()
                        if joints is not None:
                            xyzss = cast_positions(xyzss.to("cuda:0"), joints).to("cpu")
                        if skeleton_pose is not None:
                            xyzss, trash, trash = self.cast(xyzss.to("cuda:0"), None, skeleton_pose).to("cpu")
                        # query density
                        sigmas += self.density(xyzss.to(tmp_grid.device))['sigma'].reshape(lx, ly, lz).detach()/trial
                    # the magic scale number is from `scalbnf(MIN_CONE_STEPSIZE(), level)`, don't ask me why...
                    tmp_grid[xi * S: xi * S + lx, yi * S: yi * S + ly, zi * S: zi * S + lz] = sigmas * self.density_scale * 0.001691
        
        # maxpool to smooth
        if joints is not None:
            kernel = 4
            tmp_grid = F.pad(tmp_grid, (0, kernel-1, 0, kernel-1, 0, kernel-1))
            tmp_grid = F.max_pool3d(tmp_grid.unsqueeze(0).unsqueeze(0), kernel_size=kernel, stride=1).squeeze(0).squeeze(0)

        # ema update
        valid_mask = self.density_grid >= 0
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid[valid_mask]).item()
        self.iter_density += 1

        ### update step counter
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

        #print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, occ_rate={(self.density_grid > 0.01).sum() / (128**3):.3f} | [step counter] mean={self.mean_count}')


    def render(self, rays_o, rays_d, num_steps=128, upsample_steps=128, staged=False, max_ray_batch=4096, bg_color=None, perturb=False, joints = None, skeleton_pose=None, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]
        # upsample_steps = upsample_steps//4
        # num_steps = num_steps//4

        if self.cuda_ray:
            _run = self.run_cuda
        else:
            _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device
        xyzs = None
        gt_xyzs = None
        dirs = None
        gt_dirs = None
        depth=None
        image=None
        weight = None
        gt_weight = None
        # never stage when cuda_ray

        if staged and not self.cuda_ray and not (self.weight_only and self.training):
            if self.weight_only and self.training and False:
                weight= torch.empty((B, N, num_steps), device=device)
                image = torch.empty((B, N, num_steps), device=device)

                for b in range(B):
                    head = 0
                    while head < N:
                        tail = min(head + max_ray_batch, N)
                        depth_, image_ = _run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], num_steps, upsample_steps, bg_color, perturb, joints = joints, skeleton_pose=skeleton_pose)
                        if not self.training:
                            depth[b:b+1, head:tail] = depth_
                        image[b:b+1, head:tail] = image_
                        head += max_ray_batch
            else:

                depth = torch.empty((B, N), device=device)
                image = torch.empty((B, N, 3), device=device)

                for b in range(B):
                    head = 0
                    while head < N:
                        tail = min(head + max_ray_batch, N)
                        depth_, image_ = _run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], num_steps, upsample_steps, bg_color, perturb, joints = joints, skeleton_pose=skeleton_pose)
                        if not self.training:
                            depth[b:b+1, head:tail] = depth_
                        image[b:b+1, head:tail] = image_
                        head += max_ray_batch
        else:
            if self.skeleton_mode and self.training and self.initiation:
                depth, image, xyzs, gt_xyzs, dirs, gt_dirs= _run(rays_o, rays_d, num_steps, upsample_steps, bg_color, perturb, joints = joints, skeleton_pose=skeleton_pose)
            elif self.weight_only and self.training:
                weight, gt_weight =  _run(rays_o, rays_d, num_steps, upsample_steps, bg_color, perturb, joints = joints, skeleton_pose=skeleton_pose)
            else:
                depth, image = _run(rays_o, rays_d, num_steps, upsample_steps, bg_color, perturb, joints = joints, skeleton_pose=skeleton_pose)

        results = {}
        
        results['depth'] = depth
        results['rgb'] = image
        results["xyzs"] = xyzs
        results["gt_xyzs"] = gt_xyzs
            
        results["dirs"] = dirs
        results["gt_dirs"] = gt_dirs

        results["weight"] = weight
        results["gt_weight"] = gt_weight

        del depth, image, xyzs, gt_xyzs, dirs, gt_dirs
        return results


    def render_heat(self, rays_o, rays_d, num_steps=128, upsample_steps=128, staged=False, max_ray_batch=4096, bg_color=None, perturb=False, joints = None, skeleton_pose=None, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]
        upsample_steps = upsample_steps//4
        num_steps = num_steps//4

        if self.cuda_ray:
            _run = self.run_cuda
        else:
            _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device
        xyzs = None
        gt_xyzs = None
        dirs = None
        gt_dirs = None
        # never stage when cuda_ray

        if staged and not self.cuda_ray:
            weights = torch.empty((B, N, upsample_steps+num_steps, len(self.joints)), device=device)
            image = torch.empty((B, N, 3), device=device)

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    image_, weights_ = _run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], num_steps, upsample_steps, bg_color, perturb, joints = joints, skeleton_pose=skeleton_pose)
                    if not self.training:
                        weights[b:b+1, head:tail] = weights_
                    image[b:b+1, head:tail] = image_
                    head += max_ray_batch
        else:
            raise ValueError("dont come here")
            if self.skeleton_mode and self.training and self.initiation:
                depth, image, xyzs, gt_xyzs, dirs, gt_dirs= _run(rays_o, rays_d, num_steps, upsample_steps, bg_color, perturb, joints = joints, skeleton_pose=skeleton_pose)
            else:
                print("upsample_steps", upsample_steps)
                print("num_steps : ", num_steps)
                depth, image = _run(rays_o, rays_d, num_steps, upsample_steps, bg_color, perturb, joints = joints, skeleton_pose=skeleton_pose)

        results = {}
        
        results['weights'] = weights
        results['rgb'] = image
        return results