import torch
import torch.nn
import torch.nn.functional as F
from .sh import eval_sh_bases
import numpy as np
import time
from nerf.render_util import *
from models.sh_field import SphereHarmonicJoints
import glob

from torchngp.encoding import get_encoder
from torchngp.ffmlp import FFMLP

import tinycudann as tcnn

# from torchngp.nerf.renderer import NeRFRenderer

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

class CasterBase(nn.Module):
    def __init__(self, args = None):
        super().__init__()
        self.args = args
    
    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        torch.save(ckpt, path)

    def set_skeleton(self, skeleton):
        self.skeleton = skeleton
        self.joints = listify_skeleton(self.skeleton)

    # def set_joints(self, joints):
    #     self.joints = joints
    def set_args(self, args):
        self.args = args
    def set_aabbs(self, aabb, rayaabb):
        self.aabb = aabb
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize

        self.ray_aabb = rayaabb
        self.rayaabbSize = self.ray_aabb[1] - self.ray_aabb[0]
        self.invrayaabbSize = 2.0/self.rayaabbSize
    def get_weights(self):
        # return None
        return self.weights
    def get_kwargs(self):
        return {
            'skeleton': self.skeleton,
        }
    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.ray_aabb[0]) * self.invrayaabbSize - 1



class DistCaster(CasterBase):
    def __init__(self):
        super().__init__()

    def forward(self, xyz_sampled, viewdirs, transforms, ray_valid):
        shape = xyz_sampled.shape
        self.skeleton.apply_precomputed_localtransorms()

        xyz_slice = xyz_sampled.view(-1, 3).shape[0]
        tmp = torch.cat([xyz_sampled.view(-1, 3),(xyz_sampled-viewdirs).view(-1, 3)], dim=0)
        self.joints = listify_skeleton(self.skeleton)
        weights = compute_weights(xyz_sampled.view(-1,3), self.joints).to(torch.float32)
        if(weights.isnan().any()):
            ValueError("weights is nan")
        self.weights = weights
        weights = torch.cat([weights, weights], dim=0)
        tmp =  weighted_transformation(tmp, weights, transforms, if_transform_is_inv=self.args.use_indivInv)
        if(tmp.isnan().any() or tmp.isinf().any()):
            ValueError("tmp is nan")
        #debug
        xyz_sampled, viewdirs = tmp[:xyz_slice], tmp[:xyz_slice] - tmp[xyz_slice:]
        xyz_sampled = xyz_sampled.view(shape)
        viewdirs = viewdirs.view(shape)  

        return xyz_sampled, viewdirs    


# @torch.cuda.amp.autocast(enabled=True)
class MLPCaster(CasterBase):
    def __init__(self, dim, device, args = None):
        super().__init__(args = args)
        encoding="hashgrid"
        if self.args.free_opt1:
            encoding = "frequency"
        # encoding_dir="sphere_harmonics"
        self.num_layers=2
        self.hidden_dim=64
        # geo_feat_dim=15
        # num_layers_color=3
        # hidden_dim_color=64
        self.bound=1.0
        # bound /= 4.0
        # hidden_dim /= 4
        self.hidden_dim = int(self.hidden_dim)

        # sigma network
        self.geo_feat_dim = geo_feat_dim = 0
        self.interface_dim = 32
        
        # self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)
        
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * self.bound, multires = 5)
        self.interface_layer = None
        if self.args.free_opt1:
            self.interface_layer = nn.Linear(self.in_dim, self.interface_dim, bias=False).to(device)
        self.encoder = self.encoder.to(device)

        self.weight_nets = []
        for i in range(dim):

            self.weight_nets.append(
                FFMLP(
                    # input_dim=self.in_dim, 
                    input_dim=self.interface_dim, 
                    # input_dim=3, 
                    output_dim=1 + self.geo_feat_dim,
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                ).to(device)
            )
        self.weight_nets = nn.ModuleList(self.weight_nets)

    @torch.cuda.amp.autocast(enabled=True)
    def density(self, x):
        # x: [J, N, 3], in [-bound, bound]
        res = []
        for i in range(len(self.weight_nets)):
            tmp = self.encoder(x[i], bound=self.bound)
            if self.args.free_opt1:
                tmp = self.interface_layer(tmp)
            h = self.weight_nets[i](tmp)

            sigma = F.relu(h[..., 0])

            res.append(sigma)
        return torch.stack(res, dim=0)

    def forward(self, xyz_sampled, viewdirs, transforms, ray_valid):
        shape = xyz_sampled.shape
        xyz_slice = xyz_sampled.view(-1, 3).shape[0]
        # # 1115
        weights = self.compute_weights(xyz_sampled.view(-1, 3), transforms)
        self.weights = weights
        weights = torch.cat([weights, weights], dim=0)
        tmp = torch.cat([xyz_sampled.view(-1, 3),(xyz_sampled-viewdirs).view(-1, 3)], dim=0)
        tmp =  weighted_transformation(tmp, weights.to(torch.float32), transforms, if_transform_is_inv=self.args.use_indivInv)
        if(tmp.isnan().any() or tmp.isinf().any()):
            ValueError("tmp is nan")
        #debug
        xyz_sampled, viewdirs = tmp[:xyz_slice], tmp[:xyz_slice] - tmp[xyz_slice:]
        xyz_sampled = xyz_sampled.view(shape)
        viewdirs = viewdirs.view(shape)

        return xyz_sampled, viewdirs

    def compute_weights(self, xyz, transforms,  features=None, locs=None):
        # return compute_weisghts(xyz, self.joints).to(torch.float32)

       
        xyz_new = torch.cat([xyz, torch.ones(xyz.shape[0]).unsqueeze(-1).to(xyz.device)], dim=-1)#[samples, 4]
        if self.args.use_indivInv:
            invs = transforms
        else:
            invs = affine_inverse_batch(self.skeleton.precomp_forward_global_transforms)
        result = torch.matmul(invs, xyz_new.permute(1,0).unsqueeze(0).expand(invs.shape[0],4,-1)).squeeze().permute(0,2,1)#[j, samples, 4]
        result = self.normalize_coord(result[:,:,:3])
        bwf = self.density(result) # [j,sample]
        return bwf.permute(1,0)

class MLPCaster_integrate(MLPCaster):
    def __init__(self, dim, device, args = None):
        super().__init__(dim, device, args = args)
        encoding="frequency"

        # encoding_dir="sphere_harmonics"
        self.num_layers=2
        self.hidden_dim=64
        self.bound=1.0
        # bound /= 4.0
        # hidden_dim /= 4
        self.hidden_dim = int(self.hidden_dim)
        self.j_dim = dim
        self.input_dim = 3 * self.j_dim
        if_extra_dim = self.args.free_opt3
        extra_dim = 0
        if if_extra_dim:
            extra_dim = 1
        self.after_interface_dim = 16 - extra_dim

        # sigma network
        self.geo_feat_dim = 0
        self.interface_dim = 32
        
        # self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)
        
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * self.bound, multires = 6, input_dim=self.input_dim)
        # self.encoder, self.in_dim = get_encoder("hashgrid", desired_resolution=2048 * self.bound, multires = 5, input_dim=self.input_dim)

        self.interface_layer = nn.Linear(self.in_dim, self.interface_dim, bias=False).to(device)
        self.after_interface = nn.Linear(self.after_interface_dim, self.j_dim, bias=False).to(device)

        self.encoder = self.encoder.to(device)

        self.integrated_weight_net = FFMLP(
                    input_dim=self.interface_dim, 
                    output_dim=1*self.after_interface_dim + extra_dim + self.geo_feat_dim,
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                ).to(device)

    @torch.cuda.amp.autocast(enabled=True)    
    def density(self, x):
        # x: [J, N, 3], in [-bound, bound]
        x = x.permute(1,0,2)
        # x: [N, J, 3], in [-bound, bound]
        
        tmp = x.view(-1, 3*self.j_dim)
        tmp = self.encoder(tmp, bound=self.bound)
        tmp = self.interface_layer(tmp)
        h = self.integrated_weight_net(tmp)
        h = self.after_interface(h)
        sigma = F.relu(h)
        if self.args.free_opt3:
            self.bg_weights = sigma[...,-1].permute(1,0)
            return sigma.permute(1,0)[...,:-1]
        else:
            return sigma.permute(1,0)


class MLPCaster_integrate2(MLPCaster):
    def __init__(self, dim, device, args = None):
        super().__init__(dim, device, args = args)
        encoding="frequency"

        # encoding_dir="sphere_harmonics"
        self.num_layers=2
        self.hidden_dim=64
        self.bound=1.0
        # bound /= 4.0
        # hidden_dim /= 4
        self.hidden_dim = int(self.hidden_dim)
        self.j_dim = dim
        self.input_dim = 3
        if_extra_dim = self.args.free_opt3
        extra_dim = 0
        if if_extra_dim:
            extra_dim = 1
        self.after_interface_dim = 3 * self.j_dim

        # sigma network
        self.geo_feat_dim = 0
        self.interface_dim = 32
        
        # self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)
        
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * self.bound, multires = 6, input_dim=self.input_dim)
        # self.encoder, self.in_dim = get_encoder("hashgrid", desired_resolution=2048 * self.bound, multires = 5, input_dim=self.input_dim)

        self.interface_layer = nn.Linear(self.in_dim, self.interface_dim, bias=False).to(device)
        self.after_interface = nn.Linear(self.after_interface_dim, self.j_dim+extra_dim, bias=False).to(device)

        self.encoder = self.encoder.to(device)

        weight_nets = []
        for i in range(dim):

            weight_nets.append(
                FFMLP(
                    # input_dim=self.in_dim, 
                    input_dim=self.interface_dim, 
                    # input_dim=3, 
                    output_dim=1,
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                ).to(device)
            )
        self.weight_nets = nn.ModuleList(weight_nets)

    @torch.cuda.amp.autocast(enabled=True)    
    def density(self, x):
        # x: [J, N, 3], in [-bound, bound]
        # x = x.permute(1,0,2)
        # # x: [N, J, 3], in [-bound, bound]
        
        # x: [J, N, 3], in [-bound, bound]
        res = []
        for i in range(len(self.weight_nets)):
            tmp = self.encoder(x[i], bound=self.bound)
            tmp = self.interface_layer(tmp)
            h = self.weight_nets[i](tmp)
            res.append(h)
        # J, N, 3 :res
        res = torch.stack(res, dim=0)
        # res = res.permute(1,0,2).contiguous().view(-1, self.j_dim*3)
        # res = self.after_interface(res)
        # return F.relu(res).permute(1,0)
        return F.relu(res).squeeze()



class MLPCaster_net(MLPCaster):
    def __init__(self, dim, device):
        super().__init__(dim, device)
        encoding="hashgrid"
        encoding_dir="sphere_harmonics"
        num_layers=2
        hidden_dim=64
        geo_feat_dim=15
        num_layers_color=3
        hidden_dim_color=64
        bound=1.0
        # bound /= 4.0
        # hidden_dim /= 4
        hidden_dim = int(hidden_dim)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim = 0
        self.bound = bound
        # self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048//4 * bound, base_resolution=4, num_levels=4)
        # self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)

        # self.encoder = self.encoder.to(device)
        self.encoder, self.in_dim = get_encoder("frequency", desired_resolution=2048 * bound, multires = 6)
        self.in_dim_three = 3

        self.weight_nets = []
        for i in range(dim):
            sigma_net = []
            for l in range(num_layers):
                if l == 0:
                    # in_dim = self.in_dim_three
                    in_dim = self.in_dim
                else:
                    in_dim = hidden_dim
                
                if l == num_layers - 1:
                    out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
                else:
                    out_dim = hidden_dim
                
                sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))
            self.weight_nets.append(
                nn.ModuleList(sigma_net).to(device)
            )

    @torch.cuda.amp.autocast(enabled=True)
    def density(self, x):
        # x: [J, N, 3], in [-bound, bound]
        res = []
        for i in range(len(self.weight_nets)):
            h = self.encoder(x[i], bound=self.bound)
            # h = x[i]

            for l in range(self.num_layers):
                h = self.weight_nets[i][l](h)

                if l != self.num_layers - 1:
                    h = F.relu(h, inplace=True)

            sigma = F.relu(h[..., 0])
            res.append(sigma)
        return torch.stack(res, dim=0)



class BWCaster(CasterBase):
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


    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.ray_aabb[0]) * self.invrayaabbSize - 1
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
        xyz_slice = xyz_sampled.view(-1, 3).shape[0]
        # # 1115
        weights = self.compute_weights(xyz_sampled.view(-1, 3),transforms)
        self.weights = weights
        weights = torch.cat([weights, weights], dim=0)
        tmp = torch.cat([xyz_sampled.view(-1, 3),(xyz_sampled-viewdirs).view(-1, 3)], dim=0)
        tmp =  weighted_transformation(tmp, weights, transforms, if_transform_is_inv=self.args.use_indivInv)
        if(tmp.isnan().any() or tmp.isinf().any()):
            ValueError("tmp is nan")
        #debug
        xyz_sampled, viewdirs = tmp[:xyz_slice], tmp[:xyz_slice] - tmp[xyz_slice:]
        xyz_sampled = xyz_sampled.view(shape)
        viewdirs = viewdirs.view(shape)

        return xyz_sampled, viewdirs

    def TV_loss_blendweights(self, reg, linear = False):
        total = 0
        for idx in range(len(self.app_line)):
            for i in range(self.app_line[idx].shape[0]):
                total = total + reg(self.app_line[idx][i].unsqueeze(0)) * 1e-4 + reg(self.app_plane[idx][i].unsqueeze(0)) * 1e-3
                # if linear:
                #     total = total + (tv_loss_func_line(self.app_line[idx][i]) * 1e-4 + tv_loss_func_plane(self.app_plane[idx][i]) * 1e-3) * 1e-4
        return total
    def linear_loss(self):
        total = 0
        for idx in range(len(self.app_line)):
            for i in range(self.app_line[idx].shape[0]):
                total = total + (tv_loss_func_line(self.app_line[idx][i]) * 1e-4 + tv_loss_func_plane(self.app_plane[idx][i]) * 1e-3) * 1e-4
        return total


    # def sample_BWfield(self, xyz_sampled):
    #     # plane + line basis
    #     # plane : 3, self.j_channel, 1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]
    #     feats = []
    #     print(xyz_sampled.shape, len(self.app_plane), self.app_plane[0].shape)
        
    #     # for i in range(xyz_sampled.shape[0]):
    #     coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
    #     coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
    #     coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
    #     sigma_feature = torch.zeros(xyz_sampled.shape[:2], device=xyz_sampled.device)
    #     print(coordinate_plane.shape, coordinate_line.shape)
    #     # exit()
    #     for idx_plane in range(len(self.app_plane)):
    #         # self.app_plane[idx_plane][i,0].unsqueeze(0) : 1, C, W, H
    #         # coordinate_plane[[idx_plane]] : 1, H, W, 2
    #         print(self.app_plane[idx_plane][:,0].shape, coordinate_plane[idx_plane].unsqueeze(0).expand(self.j_channel,-1,-1,-1).shape)
            
    #         plane_coef_point = F.grid_sample(self.app_plane[idx_plane][:,0], coordinate_plane[idx_plane].unsqueeze(0).expand(self.j_channel,-1,-1,-1),
    #                                             align_corners=True).view(-1, *xyz_sampled.shape[1:2])
    #         line_coef_point = F.grid_sample(self.app_line[idx_plane][:, 0], coordinate_line[idx_plane].unsqueeze(0).expand(self.j_channel,-1,-1,-1),
    #                                         align_corners=True).view(-1, *xyz_sampled.shape[1:2])
    #         sigma_feature += torch.sum(plane_coef_point * line_coef_point, dim=0)

    #     # feats.append(sigma_feature)
    #     exit("here")
    #     return F.relu(torch.stack(feats))


    def sample_BWfield(self, xyz_sampled):
        # plane + line basis
        # plane : 3, self.j_channel, 1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]
        feats = []
        # print(xyz_sampled.shape, len(self.app_plane), self.app_plane[0].shape)
        
        for i in range(xyz_sampled.shape[0]):
            coordinate_plane = torch.stack((xyz_sampled[i, ..., self.matMode[0]], xyz_sampled[i,..., self.matMode[1]], xyz_sampled[i,..., self.matMode[2]])).detach().view(3, -1, 1, 2)
            coordinate_line = torch.stack((xyz_sampled[i,..., self.vecMode[0]], xyz_sampled[i,..., self.vecMode[1]], xyz_sampled[i,..., self.vecMode[2]]))
            coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
            sigma_feature = torch.zeros(xyz_sampled.shape[1], device=xyz_sampled.device)
            for idx_plane in range(len(self.app_plane)):
                # self.app_plane[idx_plane][i,0].unsqueeze(0) : 1, C, W, H
                # coordinate_plane[[idx_plane]] : 1, H, W, 2
                plane_coef_point = F.grid_sample(self.app_plane[idx_plane][i,0].unsqueeze(0), coordinate_plane[[idx_plane]],
                                                    align_corners=True).view(-1, *xyz_sampled.shape[1:2])
                line_coef_point = F.grid_sample(self.app_line[idx_plane][i, 0].unsqueeze(0), coordinate_line[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[1:2])
                sigma_feature += torch.sum(plane_coef_point * line_coef_point, dim=0)

            feats.append(sigma_feature)
        return F.relu(torch.stack(feats))
    
    def compute_weights(self, xyz, transforms,  features=None, locs=None):
        # return compute_weisghts(xyz, self.joints).to(torch.float32)

       
        xyz_new = torch.cat([xyz, torch.ones(xyz.shape[0]).unsqueeze(-1).to(xyz.device)], dim=-1)#[samples, 4]
        #xyz_new : [samples, 4]
        #invs: [j, 4, 4]

        # result = []
        # for j in range(transforms.shape[0]):
        #     if self.args.use_indivInv:
        #         inv = transforms[j]
        #     else:
        #         inv = affine_inverse(inv)
        #     # result.append(torch.bmm(invs[j].unsqueeze(0).repeat(xyz_new.shape[0], 1, 1), xyz_new.unsqueeze(-1)).squeeze())#[samples, 4]
        #     result.append(torch.matmul(inv, xyz_new.permute(1,0)).squeeze())#[samples, 4]
        # result = torch.stack(result, dim=0)[:,:,:3]#[j,samples,3]
        if self.args.use_indivInv:
            invs = transforms
        else:
            invs = affine_inverse_batch(self.skeleton.precomp_forward_global_transforms)
        result = torch.matmul(invs, xyz_new.permute(1,0).unsqueeze(0).expand(invs.shape[0],4,-1)).squeeze().permute(0,2,1)#[j, samples, 4]
        result = self.normalize_coord(result[:,:,:3])
        bwf = self.sample_BWfield(result) # [j,sample]
        # return torch.transpose(bwf, 0 , 1)
        return bwf.permute(1,0)

class shCaster(CasterBase):

    def __init__(self):
        super().__init__()
        self.sh_feats = None
        self.use_distweight = False


        
    # def set_usedistweight(self, use_distweight):
    #     self.use_distweight = use_distweight

    def forward(self, xyz_sampled, viewdirs, transforms, ray_valid):
        # return xyz_sampled, viewdirs
        # time_st = time.perf_counter()
        shape = xyz_sampled.shape
        xyz_slice = xyz_sampled.view(-1, 3).shape[0]
        # tmp = torch.cat([xyz_sampled.view(-1, 3),(xyz_sampled-viewdirs).view(-1, 3)], dim=0)
        tmp = torch.cat([xyz_sampled.view(-1, 3),(xyz_sampled-viewdirs).view(-1, 3)], dim=0)

        if torch.isnan(self.sh_feats).any() or torch.isinf(self.sh_feats).any():
            raise ValueError("shfeats"+"nan or inf in weights")
        
        if self.use_distweight:
            exit()
        else:
            weights = self.get_SH_vals(xyz_sampled.view(-1, 3), self.sh_feats, transforms, self.skeleton.get_listed_positions_first())

        if torch.isnan(weights).any() or torch.isinf(weights).any():
            raise ValueError("justaftergetweights"+"nan or inf in weights")
        
    
        self.weights = weights
        weights = torch.cat([weights, weights], dim=0)
        tmp =  weighted_transformation(tmp, weights, transforms, if_transform_is_inv=self.args.use_indivInv)
        if(tmp.isnan().any() or tmp.isinf().any()):
            ValueError("tmp is nan")
        #debug
        xyz_sampled, viewdirs = tmp[:xyz_slice], tmp[:xyz_slice] - tmp[xyz_slice:]
        xyz_sampled = xyz_sampled.view(shape)
        viewdirs = viewdirs.view(shape)
        

        return xyz_sampled, viewdirs
            
    def set_SH_feats(self, feats):
        self.sh_feats = feats

    def set_allgrads(self, value):
        for param in self.sh_feats:
            param.requires_grad = value





    def get_SH_vals(self, xyz, features, invs, locs):
        xyz_new = torch.cat([xyz, torch.ones(xyz.shape[0]).unsqueeze(-1).to(xyz.device)], dim=-1)#[samples, 4]
        # if torch.isnan(xyz).any() or torch.isinf(xyz).any():
        #     raise ValueError("nan or inf")
        # if torch.isnan(locs).any() or torch.isinf(locs).any():
        #     raise ValueError("nan or inf")
        # if torch.isnan(invs).any() or torch.isinf(invs).any():
        #     raise ValueError("nan or inf")
        xyz = []
        for j in range(locs.shape[0]):
            xyz.append(torch.bmm(invs[j].unsqueeze(0).repeat(xyz_new.shape[0], 1, 1), xyz_new.unsqueeze(-1)).squeeze())#[samples, 4]
            # xyz.append(torch.bmm(invs[j].unsqueeze(0).expand(xyz_new.shape[0], -1, -1), xyz_new.unsqueeze(-1)).squeeze())#[samples, 4]
        xyz = torch.stack(xyz, dim=0)[:,:,:3]#[j,samples,3]
        viewdirs = locs.unsqueeze(-2).repeat(1,xyz.shape[1], 1) - xyz #(j, sample, 3)
        # viewdirs = locs.unsqueeze(-2).expand(-1,xyz.shape[1], -1) - xyz #(j, sample, 3)
        # lengths = torch.norm(viewdirs, dim=-1)
        lengths = torch.nan_to_num(torch.linalg.norm(viewdirs, dim=-1))
        # if torch.isnan(lengths).any() or torch.isinf(lengths).any():
        #     raise ValueError("nan or inf")
        # viewdirs = viewdirs / lengths.unsqueeze(-1)
        # if torch.isnan(viewdirs).any() or torch.isinf(viewdirs).any():
        #     raise ValueError("nan or inf")
        viewdirs = torch.nn.functional.normalize(viewdirs, dim=-1)
        # if torch.isnan(viewdirs).any() or torch.isinf(viewdirs).any():
        #     raise ValueError("nan or inf")
        sh_mult = eval_sh_bases(2, viewdirs)#(j,sample, 1)#[:, :,None].view(viewdirs.shape[0]*viewdirs.shape[1], 1, -1)#(sample*j, 1, 9)
        rad_sh = self.sh_feats.view(locs.shape[0], 1, sh_mult.shape[-1])#(j, 1, 9)
        # if torch.isnan(rad_sh).any() or torch.isinf(rad_sh).any():
        #     raise ValueError("nan or inf")
        rads = torch.relu(torch.sum(sh_mult * rad_sh.repeat(1, xyz.shape[1], 1), dim=-1) + 0.5)#(j,sample,  1)
        # rads = torch.relu(torch.sum(sh_mult * rad_sh.expand(-1, xyz.shape[1], -1), dim=-1) + 0.5)#(j,sample,  1)
        # if torch.isnan(rads).any() or torch.isinf(rads).any():
        #     raise ValueError("nan or inf")
        #42
        eps = 1e-6
        non_valid = rads < eps
        relative_distance = torch.relu(1.0 - lengths/rads)#(j, sample)
        # if torch.isnan(relative_distance).any() or torch.isinf(relative_distance).any():
        #     raise ValueError("nan or inf")
        relative_distance[non_valid] = 0.0
        # if torch.isnan(relative_distance).any() or torch.isinf(relative_distance).any():
        #     raise ValueError("nan or inf")
        relative_distance = relative_distance.permute(1,0)#(j, sample) -> (sample, j)
        # if torch.isnan(relative_distance).any() or torch.isinf(relative_distance).any():
        #     raise ValueError("nan or inf")

        return relative_distance#weights, (sample, j)(j, sample)では？

class MLPCaster_tcnn(MLPCaster):
    def __init__(self, dim, device):
        super().__init__(dim, device)
        encoding="hashgrid"
        encoding_dir="sphere_harmonics"
        num_layers=2
        hidden_dim=64
        geo_feat_dim=15
        num_layers_color=3
        hidden_dim_color=64
        bound=1.0
        # bound /= 4.0
        # hidden_dim /= 4
        hidden_dim = int(hidden_dim)
        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim = 0
        self.bound = bound
        

        n_levels = 16//4
        bsae_res = 16
        leveldim = 2

        per_level_scale = np.exp2(np.log2(2048 * bound / bsae_res) / (n_levels - 1))

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": leveldim,
                "log2_hashmap_size": 19,
                "base_resolution": bsae_res,
                "per_level_scale": per_level_scale,
            },
        )

        self.weight_nets = []
        for i in range(dim):

            self.weight_nets.append(
                tcnn.Network(
                n_input_dims=leveldim*n_levels,
                n_output_dims=1 + self.geo_feat_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim,
                    "n_hidden_layers": num_layers - 1,
                },
                ).to(device)
            )

    @torch.cuda.amp.autocast(enabled=True)
    def density(self, x):
        # x: [J, N, 3], in [-bound, bound]
        res = []
        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        for i in range(len(self.weight_nets)):
            # sigma
            
            tmp = self.encoder(x[i])
            h = self.weight_nets[i](tmp)
            sigma = F.relu(h[..., 0])
            res.append(sigma)
        return torch.stack(res, dim=0)

