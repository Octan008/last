import torch
import torch.nn
import torch.nn.functional as F
from .sh import eval_sh_bases
import numpy as np
import time
from nerf.render_util import *
from models.sh_field import SphereHarmonicJoints
import glob
import os
import sys
# sys.path.append("./torch-ngp")
# from torchngp.nerf.network_ff import NeRFNetwork
from torchngp.nerf.network import NeRFNetwork
import time

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

def positional_encoding(positions, freqs):
    
        freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]


def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    # print("SHRender", rgb_sh.shape)
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled, viewdirs, features):

    rgb = features
    return rgb

class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb=aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1,1,*alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]]).to(self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1,-1,1,1,3), align_corners=True).view(-1)

        return alpha_vals
    def set_device(self, device):
        self.device = device
        self.alpha_volume = self.alpha_volume.to(self.device)
        self.aabb = self.aabb.to(self.device)
        self.invgridSize = self.invgridSize.to(self.device)

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1


class MLPRender_Fea(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = 2*viewpe*3 + 2*feape*inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPRender_PE(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (3+2*viewpe*3)+ (3+2*pospe*3)  + inChanel #
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPRender(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, featureC=128):
        super(MLPRender, self).__init__()

        self.in_mlpC = (3+2*viewpe*3) + inChanel
        self.viewpe = viewpe
        
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb



class TensorBase(torch.nn.Module):
    def __init__(self, aabb, gridSize, device, density_n_comp = 8, appearance_n_comp = 24, app_dim = 27,
                    shadingMode = 'MLP_PE', alphaMask = None, near_far=[2.0,6.0],
                    density_shift = -10, alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001,
                    pos_pe = 6, view_pe = 6, fea_pe = 6, featureC=128, step_ratio=2.0,
                    fea2denseAct = 'softplus'):
        super(TensorBase, self).__init__()

        self.device=device
        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        self.aabb = aabb.to(device)
        self.ray_aabb = aabb.to(device)
        self.alphaMask = alphaMask
        

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio

        self.use_gt_skeleton = False


        self.update_stepSize(gridSize)

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]

        self.gridSize_tmp = gridSize

        self.render_jointmask = False
        self.render_using_skeleton_quaternion = False
        self.render_using_skeleton_matrix = False

        self.extra = False
        self.init_svd_volume(gridSize[0], device)

        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = shadingMode, pos_pe, view_pe, fea_pe, featureC
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, device)

        self.data_preparation = True
        self.use_ngprender =False
        self.use_indivInv = True
        self.args = None
        self.tmp_animframe_index = None

        self.forward_caster_mode = False
        

            

        # self.sh_feats = nn.Parameter(torch.tensor([1.0], dtype=torch.float32).unsqueeze(0).repeat(20,9).to("cuda:0"), requires_grad=True)  # (j, dim, 1)
    def set_ngprender(self, if_use_ngprender):
        self.use_ngprender = if_use_ngprender
        if self.use_ngprender:
            self.ngprenderer = NeRFNetwork(bound=2)
    def set_args(self, args):
        self.args = args

    def set_render_flags(self, jointmask = False, using_skeleton_quaternion=True, using_skeleton_matrix=True):
        self.render_jointmask = jointmask
        self.render_using_skeleton_matrix = using_skeleton_matrix
        self.render_using_skeleton_quaternion = using_skeleton_quaternion

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        if shadingMode == 'MLP_PE':
            self.renderModule = MLPRender_PE(self.app_dim, view_pe, pos_pe, featureC).to(device)
        elif shadingMode == 'MLP_Fea':
            self.renderModule = MLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC).to(device)
        elif shadingMode == 'MLP':
            self.renderModule = MLPRender(self.app_dim, view_pe, featureC).to(device)
        elif shadingMode == 'SH':
            self.renderModule = SHRender
        elif shadingMode == 'RGB':
            assert self.app_dim == 3
            self.renderModule = RGBRender
        else:
            print("Unrecognized shading module")
            # exit()
        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)
        print(self.renderModule)
    

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.gridSize= torch.LongTensor(gridSize).to(self.device)
        self.units=self.aabbSize / (self.gridSize-1)
        self.stepSize=torch.mean(self.units)*self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples=int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def init_svd_volume(self, res, device):
        pass

    def compute_features(self, xyz_sampled):
        pass
    
    def compute_densityfeature(self, xyz_sampled):
        pass
    def compute_extrafeature(self, xyz_sampled):
        pass
    
    def compute_appfeature(self, xyz_sampled):
        pass
    
    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial = 0.02, lr_init_network = 0.001):
        pass
    def set_allgrads(self, value):
        pass
    def set_SH_feats(self, feats):
        self.sh_feats = feats

    def set_pointCaster(self, caster, caster_origin = None):
        self.caster = caster
        if caster_origin is not None:
            self.caster_origin = caster_origin
        else:
            self.caster_origin = caster
        self.caster_origin.set_aabbs(self.aabb, self.ray_aabb)
        self.caster_origin.set_args(self.args)
        
    def set_mimikCaster(self, caster):
        self.mimik_caster = caster
        self.mimik_caster.set_aabbs(self.aabb, self.ray_aabb)
        self.mimik_caster.set_args(self.args)

    def set_skeletonmode(self):
        self.data_preparation = False
        self.gridSize = self.gridSize_tmp

    def get_gridsize_as_list(self):
        if torch.is_tensor(self.gridSize):
            return self.gridSize.tolist()
        else:
            return self.gridSize


    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.get_gridsize_as_list(),
            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,

            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'fea2denseAct': self.fea2denseAct,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,

            'shadingMode': self.shadingMode,
            'pos_pe': self.pos_pe,
            'view_pe': self.view_pe,
            'fea_pe': self.fea_pe,
            'featureC': self.featureC
        }

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape':alpha_volume.shape})
            ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)
    # def save_sh(self, path):
    #     ckpt = {'state_dict': self.state_dict()}
    # def save_anim_ckpt(self, path):
    #     kwargs = self.get_kwargs()
    #     ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
    #     if self.alphaMask is not None:
    #         alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
    #         ckpt.update({'alphaMask.shape':alpha_volume.shape})
    #         ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape(-1))})
    #         ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
    #     torch.save(ckpt, path)

    def load(self, ckpt):
        if self.use_ngprender:
            tmp_load = torch.load(ckpt, map_location=self.device)
            # print(tmp_load)
            print(tmp_load["model"].keys())
            self.ngprenderer.load_state_dict(torch.load(ckpt, map_location=self.device)["model"], strict=False)
            # exit("load ngprenderer")
            print("load ngprenderer from", ckpt)
            
            return
            
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device), alpha_volume.float().to(self.device))
        self.load_state_dict(ckpt['state_dict'], strict = False)
    


    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        if self.data_preparation:
            mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        else:
            mask_outbbox = ((self.ray_aabb[0] > rays_pts) | (rays_pts > self.ray_aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def modify_aabb(self, rate):
        self.ray_aabb = self.aabb * rate;
    def temp_modify_aabb(self, aabb):
        self.ray_aabb = aabb;
    def temp_modify_all_aabb(self, aabb):
        self.ray_aabb = aabb;
        self.aabb = aabb;


    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples>0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        if self.data_preparation:
            rate_a = (self.aabb[1] - rays_o) / vec
            rate_b = (self.aabb[0] - rays_o) / vec
        else:
            rate_a = (self.ray_aabb[1] - rays_o) / vec
            rate_b = (self.ray_aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            rng += torch.rand_like(rng[:,[0]])
        step = stepsize * rng.to(rays_o.device)
        interpx = (t_min[...,None] + step)

        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        if self.data_preparation:
            mask_outbbox = ((self.aabb[0]>rays_pts) | (rays_pts>self.aabb[1])).any(dim=-1)
        else:
            mask_outbbox = ((self.ray_aabb[0]>rays_pts) | (rays_pts>self.ray_aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox

    def clamp_pts(self, pts):
        # pts : [N_rays, N_samples, 3]
        if self.data_preparation:
            return torch.stack([
                pts[...,0].clamp(min=self.aabb[0,0], max=self.aabb[1,0]),
                pts[...,1].clamp(min=self.aabb[0,1], max=self.aabb[1,1]),
                pts[...,2].clamp(min=self.aabb[0,2], max=self.aabb[1,2])
            ], dim=-1)
        else:
            return torch.stack([
                pts[...,0].clamp(min=self.ray_aabb[0,0], max=self.ray_aabb[1,0]),
                pts[...,1].clamp(min=self.ray_aabb[0,1], max=self.ray_aabb[1,1]),
                pts[...,2].clamp(min=self.ray_aabb[0,2], max=self.ray_aabb[1,2])
            ], dim=-1)


    def shrink(self, new_aabb, voxel_size):
        pass

    @torch.no_grad()
    def getDenseAlpha(self,gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        dense_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples

        # dense_xyz = dense_xyz
        # print(self.stepSize, self.distance_scale*self.aabbDiag)
        alpha = torch.zeros_like(dense_xyz[...,0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1,3), self.stepSize).view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):

        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0,2).contiguous()
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)

        valid_xyz = dense_xyz[alpha>0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f"%(total/total_voxels*100))
        return new_aabb
    
    def crop_rays(self, rays_d, rays_o):
        print

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240*5, bbox_only=False):
        print('========> filtering rays ...')
        tt = time.time()

        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1)#.clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)#.clamp(min=near, max=far)
                mask_inbbox = t_max > t_min

            else:
                xyz_sampled, _,_ = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)
                mask_inbbox= (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')
        return all_rays[mask_filtered], all_rgbs[mask_filtered]


    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features+self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)


    def compute_alpha(self, xyz_locs, length=1):

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)
            

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.compute_densityfeature(xyz_sampled)
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma
        

        alpha = 1 - torch.exp(-sigma*length).view(xyz_locs.shape[:-1])

        return alpha
    def set_skeleton(self, skeleton):
        self.skeleton = skeleton
    def set_framepose(self, pose):
        self.frame_pose = pose
    def set_posetype(self, posetype):
        self.posetype = posetype
        
    def set_tmp_animframe_index(self, index):
        self.tmp_animframe_index = index
    
    def mimik_mode(self, mode):
        self.mimik_mode = mode


    def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1, skeleton_props=None, is_render_only=False):
        if True:
            # N_samples *= 10
            viewdirs = rays_chunk[:, 3:6]
            if ndc_ray:
                xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
                dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
                rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
                dists = dists * rays_norm
                viewdirs = viewdirs / rays_norm
            else:
                xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
                dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
            
            if self.alphaMask is not None and self.data_preparation:
                self.alphaMask.set_device(self.device)
                alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
                alpha_mask = alphas > 0
                ray_invalid = ~ray_valid
                ray_invalid[ray_valid] |= (~alpha_mask)
                ray_valid = ~ray_invalid

            sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
            rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)


        shape = xyz_sampled.shape

        if self.forward_caster_mode:
            grid_res = 50
            box = self.get_grid_points([grid_res, grid_res, grid_res]).to(self.device)
            # box = self.get_grid_points([2,2,2]).to(self.device)
            box_shape = box.shape
            # box = box.view(-1,3)
            tmp_xyz_sampled = xyz_sampled
            xyz_sampled = box
            shape = xyz_sampled.shape
            tmp_shape = tmp_xyz_sampled.shape
            tmp_viewdirs = viewdirs





        #skeleton parsing -> transforms
        if not self.data_preparation:
            # transforms = self.skeleton.rotations_to_invs(gt_skeleton_pose)
            # print("a", self.skeleton.get_listed_rotations())
            if skeleton_props is not None:
                # print("frame_pose_set")
                self.frame_pose = skeleton_props["frame_pose"]
            # if (is_train or not is_render_only) and (not is_train or not self.use_gt_skeleton):
            # if (is_train or not is_render_only) and (not is_train):
            if True:
                # transforms = self.skeleton.rotations_to_invs_fast(self.frame_pose, type=self.posetype)
                if torch.isnan(self.frame_pose).any() or torch.isinf(self.frame_pose).any():
                    print(self.frame_pose)
                    print(self.skeleton.get_listed_names())
                    raise ValueError("justaftergetweights"+"nan or inf in weights")
        
                if self.args.use_indivInv:
                    transforms = self.skeleton.rotations_to_invs_fast(self.frame_pose, type=self.posetype)
                else:
                    if self.args.free_opt4:
                        transforms = self.skeleton.para_rotations_to_transforms_fast(self.frame_pose, type=self.posetype)
                        # transforms = self.skeleton.para_rotations_to_transforms(self.frame_pose, type=self.posetype)
                    else:
                        transforms = self.skeleton.rotations_to_transforms_fast(self.frame_pose, type=self.posetype)

                if torch.isnan(transforms ).any() or torch.isinf(transforms ).any():
                    raise ValueError("justaftergetweights"+"nan or inf in weights")
                    # print("not using indivInv")
                # exit("ff")
            else:
                for j in self.skeleton.get_children():
                    apply_animation(self.frame_pose, j)
                # gt_skeleton_pose = self.skeleton.get_listed_rotations()
                # self.skeleton.transformNet(gt_skeleton_pose)
                
                # transforms = self.skeleton.rotations_to_invs(gt_skeleton_pose)
                transforms = self.skeleton.get_invs()
                # print("using_gt_skeleton")
                # transforms = self.skeleton.rotations_to_invs(self.frame_pose)
            # print("b", self.skeleton.get_listed_rotations())
            draw_joints = self.render_jointmask
            if draw_joints:
                # self.skeleton.transformNet(self.frame_pose,type=self.posetype)     
                # self.skeleton.apply_transforms_top(self.frame_pose, use_precomp = False, type=self.posetype)
                draw_mask = self.skeleton.draw_mask_all_cached(rays_chunk[:, :3], rays_chunk[:, 3:6], 0.05)
                # draw_mask = self.skeleton.draw_mask_all(rays_chunk[:, :3], rays_chunk[:, 3:6], 0.05)
            # print("c", self.skeleton.get_listed_rotations())
        

        # Point Casting
        if not self.data_preparation:
            # xyz_slice = xyz_sampled.reshape(-1, 3).shape[0]
            # tmp = torch.cat([xyz_sampled.reshape(-1, 3),(xyz_sampled-viewdirs).reshape(-1, 3)], dim=0)
            # self.joints = listify_skeleton(self.skeleton)
            # weights = self.get_SH_vals(xyz_sampled.reshape(-1, 3), self.sh_feats, transforms, self.skeleton.get_listed_positions_first())


            # weights = torch.cat([weights, weights], dim=0)
            # # taihi = tmp
            # tmp =  weighted_transformation(tmp, weights, transforms)

            
            # #debug
            # xyz_sampled, viewdirs = tmp[:xyz_slice], tmp[:xyz_slice] - tmp[xyz_slice:]
            # xyz_sampled = xyz_sampled.reshape(shape[0],shape[1], 3)
            # viewdirs = viewdirs.reshape(shape[0],shape[1], 3)

            # dist weights
            # self.caster.set_joints(self.joints)
            if_cast = True
            torch.cuda.empty_cache()
            if if_cast:

                if self.forward_caster_mode:
                    viewdirs = torch.ones_like(xyz_sampled)
                xyz_sampled, viewdirs = self.caster(xyz_sampled, viewdirs, transforms, ray_valid, i_frame = self.tmp_animframe_index)
                if self.args.mimik == "cycle":
                    xyz_sampled, viewdirs = self.mimik_caster(xyz_sampled, viewdirs, transforms, ray_valid, i_frame = self.tmp_animframe_index)


                # print("casting", xyz_sampled.grad_fn)
                # exit()
                # self.clamp_pts(self, xyz_sampled)
                if not self.args.free_opt2:
                    self.caster_weights = self.caster_origin.get_weights()
                    weights_sum = torch.sum(self.caster_weights, dim=1)
                    self.bg_alpha = clip_weight(weights_sum, thresh = 1e-3).view(shape[0], -1).view(shape[0], -1)

                if self.forward_caster_mode:

                    # self.caster_weights = self.caster_origin.get_weights()
                    effectnum = 50
                    rays_d = rays_chunk[:, 3:6]
                    rays_o = rays_chunk[:, :3]
                    xyz_sampled = xyz_sampled.view(-1, 3)
                    k = torch.sum(xyz_sampled.view(1, -1, 3).expand(rays_d.shape[0], -1, -1) *  rays_d.view(-1, 1, 3).expand(-1, xyz_sampled.shape[0], -1), dim=-1)
                    op = xyz_sampled.view(1, -1, 3).expand(rays_d.shape[0], -1, -1) - rays_o.view(-1, 1, 3).expand(-1, xyz_sampled.shape[0], -1)
                    oh =  rays_d.view(-1, 1, 3).expand(-1, xyz_sampled.shape[0], -1) * k.view(-1, xyz_sampled.shape[0], 1).expand(-1, -1, 3)
                    effects =  -torch.norm(op - oh, dim=-1) # rayn, bN
                    effect_table, inds = torch.topk(effects, effectnum, dim=-1)# rayn, effectnum
                    new_box = torch.gather(xyz_sampled, 0, inds.view(-1, 1).expand(-1, 3)).view(-1, effectnum, 3) # rayn, 50, 3

                    refference_table = tmp_xyz_sampled.view(tmp_shape).unsqueeze(2).expand(-1,-1,effectnum,-1) #rayn,effectnum, sample, 3
                    # refference_table = refference_table.permute(2,0,1,3) #sample, rayn, effectnum, 3
                    refference_table = refference_table - new_box.unsqueeze(1).expand(tmp_shape[0], tmp_shape[1], effectnum, 3) #sample, rayn, effectnum, 3
                    
                    refference_table = torch.exp(-0.0005*torch.norm(refference_table, dim=-1)) #sample, rayn, effectnum
                    # refference_table = 1.0/torch.norm(refference_table, dim=-1) #sample, rayn, effectnum
                    # refference_table = refference_table.permute(1,0,2) #rayn, sample, effectnum
                    topk = 1
                    effect_table, tmp_inds = torch.topk(refference_table, topk, dim=-1) #rayn, sample, 3
                    inds = torch.gather(inds, 1, tmp_inds.view(rays_d.shape[0], -1)).view(rays_d.shape[0], -1, topk) #rayn, sample, 3
                    effect_table = effect_table / torch.sum(effect_table, dim=-1, keepdim=True)
                    effect_table = effect_table.view(-1,topk)
                    del refference_table, tmp_inds, new_box
                    

                    #     refference_table = tmp_xyz_sampled.view(-1,3).unsqueeze(1).expand(-1,box.view(-1,3).shape[0],-1) #N, bN, 3
                    #     self.caster_weights = self.caster_origin.get_weights()
                    #     refference_table = refference_table - xyz_sampled.view(-1,3).unsqueeze(0).expand(refference_table.shape[0],-1,-1) #N, bN, 3
                    #     refference_table = torch.exp(-torch.norm(refference_table, dim=-1)) #N, bN
                    #     # print("refference_table", refference_table.shape)
                    #     effect_table, inds = torch.topk(refference_table, 3, dim=-1) #N, 3
                    # del refference_table
                    torch.cuda.empty_cache()
                    xyz_sampled = self.normalize_coord(xyz_sampled)
                    sigma_feature = self.compute_densityfeature(xyz_sampled.view(-1, 3)) #bN, feats
                    validsigma = self.feature2density(sigma_feature)                    
                    ref_sigmas = torch.gather(validsigma, 0, inds.view(-1)).view(-1, topk)  #N, 3, feats

                    sigma = torch.sum(ref_sigmas * effect_table, dim=-1) #N
                    alpha, weight, bg_weight = raw2alpha(sigma.view(-1, 1), dists.view(-1, 1) * self.distance_scale)#N, 1

                    app_features = self.compute_appfeature(xyz_sampled.view(-1, 3))   #bN, feats
                    ref_app_features = torch.gather(app_features, 0, inds.view(-1).unsqueeze(-1).expand(-1, app_features.shape[1])).view(alpha.shape[0], -1, app_features.shape[1]) #N, 3, feats
                    interpolated_app_features = torch.sum(ref_app_features * effect_table.unsqueeze(-1).expand(-1,-1, app_features.shape[1]), dim=1) #N, feats
                    # app_mask = weight > self.rayMarch_weight_thres
                    app_mask = weight > -1
                    app_mask = app_mask.squeeze()
                    # print(torch.max(tmp_viewdirs.contiguous().view(-1, 3)), torch.min(tmp_viewdirs.contiguous().view(-1, 3)))
                    valid_rgbs = self.renderModule(None, tmp_viewdirs.contiguous().view(-1, 3)[app_mask], interpolated_app_features[app_mask])#.view(tmp_shape)
                    app_mask = app_mask.view(tmp_shape[:2])
                    rgb[app_mask] = valid_rgbs
                    if torch.isnan(rgb).any()or torch.isinf(rgb).any():
                        raise ValueError("rgb is nan")

                    weight = weight.view(tmp_shape[:2])

                    acc_map = torch.sum(weight, -1)
                    rgb_map = torch.sum(weight[..., None] * rgb, -2)

                    if white_bg or (is_train and torch.rand((1,))<0.5):
                        rgb_map = rgb_map + (1. - acc_map[..., None])

                    rgb_map = rgb_map.clamp(0,1)

                    with torch.no_grad():
                        depth_map = torch.sum(weight * z_vals, -1)
                        depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

                    if not self.data_preparation:
                        if draw_joints:
                            rgb_map[draw_mask] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=rgb_map.device)
                    if torch.isnan(rgb_map).any() or torch.isinf(rgb_map).any():
                        raise ValueError("rgb map is nan")

                    return rgb_map, depth_map # rgb, sigma, alpha, weight, bg_weight


                    

            save_npz = False;
            if save_npz:
                save_npz = {}
                save_npz["weights"] = self.caster_weights.cpu().numpy()
                save_npz["xyz_sampled"] = xyz_sampled.cpu().numpy()
            torch.cuda.empty_cache()

        if ray_valid.any():
                

            xyz_sampled = self.normalize_coord(xyz_sampled)

            xyz_sampled = xyz_sampled.reshape(shape[0],shape[1], 3)

            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])
            torch.cuda.empty_cache()

            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma
            
            if not self.data_preparation and save_npz:
                save_npz["sigma"] = sigma.cpu().numpy()
            # print("sigma", sigma.grad_fn)
            self.sigma = sigma
            # exit("amkingmasking")



        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)
        self.raw_sigma = weight
        if not self.args.free_opt2:
            weight = weight * self.bg_alpha
        torch.cuda.empty_cache()
        app_mask = weight > self.rayMarch_weight_thres
        # print("app_mask", app_mask.shape, app_mask.sum())

        # Compute_alpha
        if app_mask.any():
            app_features = self.compute_appfeature(xyz_sampled[app_mask])    
            valid_rgbs = self.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features)
            rgb[app_mask] = valid_rgbs
            # weight_slice =  weights.reshape(rgb.shape[0], -1, weights.shape[-1]).shape[1]//2
            # rgb[...,1:] = 0
            # rgb[...,0] = weights.reshape(rgb.shape[0], -1, weights.shape[-1])[:,:,2] * 1000
            # print("feats", app_features.grad_fn, valid_rgbs.grad_fn, xyz_sampled.grad_fn, viewdirs.grad_fn)
            # print("rgb", rgb.grad_fn)
            # exit()
            

            
            # rgb[inside][...,0] = 1.0;
            # rgb[inside][...,1:] = 0.0;
            # rgb[inside] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=torch.device('cuda:0'));
            if not self.data_preparation and save_npz:
                rgb[:,:,0] = weights[:,:,0];
                save_npz["rgb"] = rgb.cpu().numpy()
        


        if not self.data_preparation and save_npz:
            itr = 0;    
            files = glob.glob("./data_point_cloud_*.npz")
            if len(files) > 0:
                itr = int(files[-1].split(".")[1].split("_")[-1]) + 1
            np.savez("./data_point_cloud_"+str(itr)+".npz", **save_npz)
        

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        rgb_map = rgb_map.clamp(0,1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        if not self.data_preparation:
            if draw_joints:
                rgb_map[draw_mask] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=rgb_map.device)


        return rgb_map, depth_map # rgb, sigma, alpha, weight, bg_weight

    def get_density(self, xyz_sampled):
        #xyz_sampled = xyz_sampled.reshape(-1, 3)
        sigma_feature = self.compute_densityfeature(xyz_sampled)
        

        validsigma = self.feature2density(sigma_feature)
        return validsigma
    
    def get_grid_points(self, grid_sizes, box = None):
        if box is None:
            box = self.ray_aabb
            # box = self.aabb * 0.5
            gridsize = 200
        # box_width = box[1] - box[0]
        samples_x = torch.linspace(box[0][0], box[1][0], grid_sizes[0])
        samples_y = torch.linspace(box[0][1], box[1][1], grid_sizes[1])
        samples_z = torch.linspace(box[0][2], box[1][2], grid_sizes[2])
        box = torch.ones(grid_sizes[0],grid_sizes[1],grid_sizes[2] , 3)
        # print(box[...,0].shape, g.unsqueeze(0).unsqueeze(-1).shape)
        box[...,0] *= samples_x.unsqueeze(0).unsqueeze(0).repeat(grid_sizes[0], grid_sizes[1], 1)
        box[...,1] *= samples_y.unsqueeze(0).unsqueeze(-1).repeat(grid_sizes[0], 1, grid_sizes[2])
        box[...,2] *= samples_z.unsqueeze(-1).unsqueeze(-1).repeat(1, grid_sizes[1], grid_sizes[2])
        return box