import torch
import torch.nn
import torch.nn.functional as F
from .sh import eval_sh_bases
import numpy as np
import time
from nerf.render_util import *
from models.sh_field import SphereHarmonicJoints
import glob


# def tv_loss_func_grid(image, weight = 0.01):
#     tv_h = ((image[:,1:,:,:] - image[:,:-1,:,:]).pow(2)).sum()
#     tv_w = ((image[:,:,1:,:] - image[:,:,:-1,:]).pow(2)).sum()    
#     return weight * (tv_h + tv_w) / (image.shape[0] * image.shape[1] * image.shape[2] * image.shape[3])
def tv_loss_func_plane(image, weight = 1.0):
    tv_h = (image[:,1:,:,:] - image[:,:-1,:,:])
    valid = torch.abs(tv_h) > 1e-8
    tv_h = torch.where(tv_h > 0, torch.ones_like(tv_h), -torch.ones_like(tv_h))
    # tv_h[valid] /= torch.abs(tv_h)[valid]
    tv_h = torch.abs(tv_h[:,1:,:,:] - tv_h[:,:-1,:,:]).sum()

    tv_w = (image[:,:,1:,:] - image[:,:,:-1,:])
    valid = torch.abs(tv_w) > 1e-8
    # tv_w[valid] /= torch.abs(tv_w)[valid]
    tv_w = torch.where(tv_w > 0, torch.ones_like(tv_w), -torch.ones_like(tv_w))
    tv_w = torch.abs(tv_w[:,:,1:,:] - tv_w[:,:,:-1,:]).sum()

    return weight * (tv_h + tv_w) * weight

def tv_loss_func_line(image, weight = 1.0):
    tv_h = (image[:,1:,:] - image[:,:-1,:])
    valid = torch.abs(tv_h) > 1e-8
    # tv_h[valid] /= torch.abs(tv_h)[valid]
    tv_h = torch.where(tv_h > 0, torch.ones_like(tv_h), -torch.ones_like(tv_h))
    
    tv_h = torch.abs(tv_h[:,1:,:] - tv_h[:,:-1,:]).sum()
    return weight * (tv_h) * weight



class BWCaster(nn.Module):
    def __init__(self, dim, gridSize, device):
        super().__init__()
        self.app_n_comp = [16,16,16]
        self.j_channel = dim
        self.gridSize = gridSize
        self.app_dim = dim
        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.aabb = None
        self.invaabbSize = None
        self.ray_aabb = None
        self.joints = None
        self.init_svd_volume(gridSize, device)
    def set_joints(self, joints):
        self.joints = joints

    def set_aabbs(self, aabb, rayaabb):
        self.aabb = aabb
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize

        self.ray_aabb = rayaabb
        self.rayaabbSize = self.ray_aabb[1] - self.ray_aabb[0]
        self.invrayaabbSize = 2.0/self.rayaabbSize

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.ray_aabb[0]) * self.invrayaabbSize - 1

    def set_skeleton(self, skeleton):
        self.skeleton = skeleton
    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.app_line, 'lr': lr_init_spatialxyz}, {'params': self.app_plane, 'lr': lr_init_spatialxyz}]
        return grad_vars

    def init_svd_volume(self, res, device):
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.032, device)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                # scale * torch.randn((self.j_channel, 1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
                # scale * torch.zeros((self.j_channel, 1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
                scale * torch.ones((self.j_channel, 1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            line_coef.append(
                # torch.nn.Parameter(scale * torch.randn((self.j_channel, 1, n_component[i], gridSize[vec_id], 1))))
                # torch.nn.Parameter(scale * torch.zeros((self.j_channel, 1, n_component[i], gridSize[vec_id], 1))))
                torch.nn.Parameter(scale * torch.ones((self.j_channel, 1, n_component[i], gridSize[vec_id], 1))))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)
        
    def forward(self, xyz_sampled, viewdirs, transforms, ray_valid):
        shape = xyz_sampled.shape
        xyz_slice = xyz_sampled.reshape(-1, 3).shape[0]
        self.joints = listify_skeleton(self.skeleton)    
        # # 1115
        weights = self.compute_weights(xyz_sampled.reshape(-1, 3), None, transforms, self.skeleton.get_listed_positions_first())
        self.weights = weights
        if(weights.isnan().any()):
            ValueError("weights is nan")
        # 1115
        # weights[...,3:] = 0.0
        # weights[...,:2] = 0.0
        tmp = torch.cat([xyz_sampled.reshape(-1, 3),(xyz_sampled-viewdirs).reshape(-1, 3)], dim=0)
        tmp =  weighted_transformation(tmp.reshape(-1, 3), torch.cat([weights, weights], dim=0), transforms)
        xyz_sampled = tmp[:xyz_slice]
        viewdirs = tmp[:xyz_slice] - tmp[xyz_slice:]
        xyz_sampled = xyz_sampled.reshape(shape[0],shape[1], 3)
        viewdirs = viewdirs.reshape(shape[0],shape[1], 3)
        return xyz_sampled, viewdirs

    def TV_loss_blendweights(self, reg, linear = False):
        total = 0
        for idx in range(len(self.app_line)):
            for i in range(self.app_line[idx].shape[0]):
                total = total + reg(self.app_line[idx][i].unsqueeze(0)) * 1e-4 + reg(self.app_plane[idx][i].unsqueeze(0)) * 1e-3
                if linear:
                    total = total + tv_loss_func_line(self.app_line[idx][i]) * 1e-4 + tv_loss_func_plane(self.app_plane[idx][i]) * 1e-3
        return total


    def get_weights(self):
        return self.weights

    def sample_BWfield(self, xyz_sampled):
        # plane + line basis
        feats = []
        for i in range(xyz_sampled.shape[0]):
            coordinate_plane = torch.stack((xyz_sampled[i, ..., self.matMode[0]], xyz_sampled[i,..., self.matMode[1]], xyz_sampled[i,..., self.matMode[2]])).detach().view(3, -1, 1, 2)
            coordinate_line = torch.stack((xyz_sampled[i,..., self.vecMode[0]], xyz_sampled[i,..., self.vecMode[1]], xyz_sampled[i,..., self.vecMode[2]]))
            coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
            sigma_feature = torch.zeros(xyz_sampled.shape[1], device=xyz_sampled.device)
            
            for idx_plane in range(len(self.app_plane)):
                plane_coef_point = F.grid_sample(self.app_plane[idx_plane][i,0].unsqueeze(0), coordinate_plane[[idx_plane]],
                                                    align_corners=True).view(-1, *xyz_sampled.shape[1:2])
                line_coef_point = F.grid_sample(self.app_line[idx_plane][i, 0].unsqueeze(0), coordinate_line[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[1:2])
                sigma_feature += torch.sum(plane_coef_point * line_coef_point, dim=0)

            feats.append(sigma_feature)
        # return torch.stack(feats)
        # return F.plus(torch.stack(feats) - 10)
        return F.relu(torch.stack(feats))
    
    def compute_weights(self, xyz, features, invs, locs):
        # return compute_weisghts(xyz, self.joints).to(torch.float32)

       
        xyz_new = torch.cat([xyz, torch.ones(xyz.shape[0]).unsqueeze(-1).to(xyz.device)], dim=-1)#[samples, 4]

        xyz = []
        for j in range(locs.shape[0]):
            xyz.append(torch.bmm(invs[j].unsqueeze(0).repeat(xyz_new.shape[0], 1, 1), xyz_new.unsqueeze(-1)).squeeze())#[samples, 4]
        xyz = torch.stack(xyz, dim=0)[:,:,:3]#[j,samples,3]
        xyz = self.normalize_coord(xyz)
        # viewdirs = locs.unsqueeze(-2).repeat(1,xyz.shape[1], 1) - xyz #(j, sample, 3)
        bwf = self.sample_BWfield(xyz) # [j,sample]

        # rad_sh = self.sh_feats.reshape(locs.shape[0], 1, sh_mult.shape[-1])#(j, 1, 9)
        # rads = torch.relu(torch.sum(sh_mult * rad_sh.repeat(1, xyz.shape[1], 1), dim=-1) + 0.5)#(j,sample,  1)
        #42
        return torch.transpose(bwf, 0 , 1)

    def get_kwargs(self):
        return {
            'skeleton': self.skeleton,
        }

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        torch.save(ckpt, path)




class shCaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.sh_feats = None
        self.use_distweight = False
    def set_skeleton(self, skeleton):
        self.skeleton = skeleton
    def set_aabbs(self, aabb, rayaabb):
        self.aabb = aabb
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize

        self.ray_aabb = rayaabb
        self.rayaabbSize = self.ray_aabb[1] - self.ray_aabb[0]
        self.invrayaabbSize = 2.0/self.rayaabbSize
        
    def set_usedistweight(self, use_distweight):
        self.use_distweight = use_distweight

    def forward(self, xyz_sampled, viewdirs, transforms, ray_valid):
        # return xyz_sampled, viewdirs
        # time_st = time.perf_counter()
        shape = xyz_sampled.shape
        xyz_slice = xyz_sampled.reshape(-1, 3).shape[0]
        tmp = torch.cat([xyz_sampled.reshape(-1, 3),(xyz_sampled-viewdirs).reshape(-1, 3)], dim=0)
        self.joints = listify_skeleton(self.skeleton)
        if self.use_distweight:
            # print(self.skeleton.get_listed_rotations())
            # exit()
            weights = compute_weights(xyz_sampled.reshape(-1,3), self.joints).to(torch.float32)
            # print("distweightsman")
        else:
            # print("not distweightsman")
            weights = self.get_SH_vals(xyz_sampled.reshape(-1, 3), self.sh_feats, transforms, self.skeleton.get_listed_positions_first())
            # exit("PP")
        if(weights.isnan().any()):
            ValueError("weights is nan")
        weights = torch.cat([weights, weights], dim=0)
        # time_en =  time.perf_counter()
        # print("weights time", time_en - time_st)

        # time_st = time.perf_counter()

        # taihi = tmp
        tmp =  weighted_transformation(tmp, weights, transforms)
        if(tmp.isnan().any() or tmp.isinf().any()):
            ValueError("tmp is nan")
        #debug
        xyz_sampled, viewdirs = tmp[:xyz_slice], tmp[:xyz_slice] - tmp[xyz_slice:]
        xyz_sampled = xyz_sampled.reshape(shape[0],shape[1], 3)
        viewdirs = viewdirs.reshape(shape[0],shape[1], 3)
        self.weights = weights
        # time_en =  time.perf_counter()
        # print("cast time", time_en - time_st)
        return xyz_sampled, viewdirs
            
    def set_SH_feats(self, feats):
        self.sh_feats = feats

    def set_allgrads(self, value):
        for param in self.sh_feats:
            param.requires_grad = value

    def get_weights(self):
        return None
        return self.weights

    def get_SH_vals(self, xyz, features, invs, locs):
        xyz_new = torch.cat([xyz, torch.ones(xyz.shape[0]).unsqueeze(-1).to(xyz.device)], dim=-1)#[samples, 4]

        xyz = []
        for j in range(locs.shape[0]):
            xyz.append(torch.bmm(invs[j].unsqueeze(0).repeat(xyz_new.shape[0], 1, 1), xyz_new.unsqueeze(-1)).squeeze())#[samples, 4]
        xyz = torch.stack(xyz, dim=0)[:,:,:3]#[j,samples,3]
        viewdirs = locs.unsqueeze(-2).repeat(1,xyz.shape[1], 1) - xyz #(j, sample, 3)
        lengths = torch.norm(viewdirs, dim=-1)
        viewdirs = viewdirs / lengths.unsqueeze(-1)
        sh_mult = eval_sh_bases(2, viewdirs)#(j,sample, 1)#[:, :,None].reshape(viewdirs.shape[0]*viewdirs.shape[1], 1, -1)#(sample*j, 1, 9)
        rad_sh = self.sh_feats.reshape(locs.shape[0], 1, sh_mult.shape[-1])#(j, 1, 9)
        rads = torch.relu(torch.sum(sh_mult * rad_sh.repeat(1, xyz.shape[1], 1), dim=-1) + 0.5)#(j,sample,  1)
        #42
        eps = 1e-6
        non_valid = rads < eps
        relative_distance = torch.relu(1.0 - lengths/rads)#(j, sample)
        relative_distance[non_valid] = 0.0
        relative_distance = torch.transpose(relative_distance, 0 , 1)#(j, sample) -> (sample, j)
        if torch.isnan(relative_distance).any() or torch.isinf(relative_distance).any():
            ValueError("nan or inf")

        return relative_distance#weights, (sample, j)(j, sample)では？
    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        torch.save(ckpt, path)