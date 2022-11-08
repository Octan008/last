import torch
import torch.nn
import torch.nn.functional as F
from .sh import eval_sh_bases
import numpy as np
import time
from nerf.render_util import *
from models.sh_joints import SphereHarmonicJoints
import glob


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

        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        self.aabb = aabb
        self.ray_aabb = aabb
        self.alphaMask = alphaMask
        self.device=device

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio


        self.update_stepSize(gridSize)

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]

        self.gridSize = gridSize

        self.render_jointmask = False
        self.render_using_skeleton_quaternion = False
        self.render_using_skeleton_matrix = False


        self.init_svd_volume(gridSize[0], device)

        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = shadingMode, pos_pe, view_pe, fea_pe, featureC
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, device)

        self.data_preparation = True

        # self.sh_feats = nn.Parameter(torch.tensor([1.0], dtype=torch.float32).unsqueeze(0).repeat(20,9).to("cuda:0"), requires_grad=True)  # (j, dim, 1)

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
            exit()
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

    def set_pointCaster(self, caster):
        self.caster = caster

    # def get_SH_vals(self, xyz, features, invs, locs):
    #     #features :(j, 9)
    #     # xyz : (sample, 3)
    #     #locs : (j, 3)
    #     #invs : (j, 4, 4)
    #     # print("init")
    #     # print(features.shape, xyz.shape, locs.shape, invs.shape)
    #     # xyz_new = torch.transpose(torch.cat([xyz, torch.ones(xyz.shape[0]).unsqueeze(-1).to(xyz.device)], dim=-1), 0, 1)#[4, samples]
    #     xyz_new = torch.cat([xyz, torch.ones(xyz.shape[0]).unsqueeze(-1).to(xyz.device)], dim=-1)#[samples, 4]
    #     # print(xyz_new.shape, invs.shape)
    #     #40
    #     xyz = []
    #     for j in range(locs.shape[0]):
    #         # print(invs[j].unsqueeze(0).repeat(xyz_new.shape[0], 1, 1).shape, xyz_new.shape)
    #         xyz.append(torch.bmm(invs[j].unsqueeze(0).repeat(xyz_new.shape[0], 1, 1), xyz_new.unsqueeze(-1)).squeeze())#[samples, 4]
    #     # # exit("succeed")
    #     xyz = torch.stack(xyz, dim=0)[:,:,:3]#[j,samples,3]
    #     # xyz = xyz.unsqueeze(0).repeat(locs.shape[0], 1, 1)
    #     # print("succeed")
    #     # print(xyz.shape)
    #     # exit("succeed")
    #     #40

    #     # print("xyz", xyz.shape)
    #     # viewdirs = locs.unsqueeze(-2).repeat(1,xyz_new.shape[0], 1) - xyz_new.unsqueeze(0).repeat(locs.shape[0], 1, 1)#(j, sample, 3)
    #     # print("locs")
    #     # print(locs.shape,locs.unsqueeze(-2).shape, locs.unsqueeze(-2).repeat(1,xyz.shape[1], 1).shape)
    #     # exit("locs")
    #     viewdirs = locs.unsqueeze(-2).repeat(1,xyz.shape[1], 1) - xyz #(j, sample, 3)

    #     # viewdirs[:2,:,:] = 0.0
    #     # viewdirs[2,:,:] = 0.0
    #     # viewdirs[3:,:,:] = 0.0
    #     #変ではある
    #     # print("viewdirs", viewdirs.shape)
    #     # return torch.transpose(torch.exp(-(torch.sum(viewdirs*viewdirs, -1))*0.01), 0 , 1)#[sample, j]
    #     # print("viewdirs")
    #     # print(viewdirs.shape)
    #     # exit("viewdirs")
    #     # lengths = torch.sqrt(torch.sum(viewdirs*viewdirs, -1))
    #     lengths = torch.norm(viewdirs, dim=-1)
    #     # lengths[:2,:] = 0.0
    #     # viewdirs[2,:,:] = 0.0
    #     # lengths[3:,:] = 0.0
    #     # lengths = torch.sum(viewdirs*viewdirs, -1)
    #     # print("lengths", lengths.shape)
    #     viewdirs = viewdirs / lengths.unsqueeze(-1)
    #     # viewdirs_new = torch.zeros_like(viewdirs)#(j, sample, 3)
    #     # for j in range(locs.shape[0]):
    #     #     viewdirs_new[j] = torch.mm(invs[j].unsqueeze(0).repeat(xyz.shape[0], 1, 1), viewdirs)
        
        


    #     # sh_mult = eval_sh_bases(2, viewdirs)[:, :,None].reshape(viewdirs.shape[0]*viewdirs.shape[1], 1, -1)#(sample*j, 1, 9)
    #     sh_mult = eval_sh_bases(2, viewdirs)#(j,sample, 1)#[:, :,None].reshape(viewdirs.shape[0]*viewdirs.shape[1], 1, -1)#(sample*j, 1, 9)
    #     # sh_mult[:,:,1:] = 0.0
        
    #     # rad_sh = features.reshape(locs.shape[0], 1, sh_mult.shape[-1])#(j, 1, 9)
    #     # return sh_mult[...,0].reshape(xyz.shape[1], locs.shape[0])
    #     # self.sh_feats[1:,:] = 0.0;
    #     # print("ffd")
    #     # print(self.sh_feats.shape, self.sh_feats)
    #     # print("ff")
    #     # print(sh_mult.shape)
    #     # exit()
    #     rad_sh = self.sh_feats.reshape(locs.shape[0], 1, sh_mult.shape[-1])#(j, 1, 9)
    #     # print(rad_sh)#ここは間違ってない
    #     # print(rad_sh)
    #     # exit()
    #     # print(rad_sh)
    #     # print(rad_sh)
    #     # print()
    #     # print(rad_sh, self.sh_feats)
    #     # rad_sh = torch.ones(locs.shape[0], 1, sh_mult.shape[-1]).to("cuda:0")
    #     # print(sh_mult.shape, rad_sh.shape, sh_mult.device, rad_sh.device)
    #     # rads = torch.relu(torch.sum(sh_mult * rad_sh.repeat(xyz.shape[1], 1, 1), dim=-1))#(sample*j, 1)
    #     # rads = torch.relu(torch.sum(sh_mult * rad_sh.repeat(xyz.shape[1], 1, 1), dim=-1) + 0.5)#(sample*j, 1)
    #     rads = torch.relu(torch.sum(sh_mult * rad_sh.repeat(1, xyz.shape[1], 1), dim=-1) + 0.5)#(j,sample,  1)
    #     # rads = torch.relu(torch.sum(sh_mult * rad_sh.repeat(xyz.shape[1], 1, 1), dim=-1))#(sample*j, 1)
    #     # rads =  torch.sum(sh_mult * rad_sh.repeat(xyz.shape[1], 1, 1), dim=-1) #(sample*j, 1)
    #     # rads = torch.max(0.0, 1.0 - rads.reshape(rads.shape[0]) / lengths.reshape(rads.shape[0]))#(sample*j)
    #     # rads = torch.relu(1.0 - rads.reshape(rads.shape[0]) / lengths.reshape(rads.shape[0]))#(sample*j)
    #     #42
    #     #keepout
    #     # lengths = lengths.reshape(rads.shape[0])
    #     # rads = rads.reshape(rads.shape[0])
        

    #     #rads = rads.reshape(lengths.shape[0], lengths.shape[1])#(j, sample)
    #     # rads = torch.transpose(rads.reshape(lengths.shape[1], lengths.shape[0]), 0, 1)#(j, sample)
    #     # print(rads.shape)
    #     # print(rads[:,0])
    #     # exit("rads")
    #     #keepout
    #     #42
    #     eps = 1e-6
    #     non_valid = rads < eps
    #     relative_distance = torch.relu(1.0 - lengths/rads)#(j, sample)
    #     # relative_distance = lengths/rads
    #     # relative_distance = lengths/8.9628
    #     # #38
    #     # out_of_sphere = lengths/rads > 1.0
    #     # relative_distance[out_of_sphere] = 1.0
    #     # #38
    #     # relative_distance = torch.exp(-lengths*0.01)
    #     test = relative_distance < eps
    #     # print(relative_distance.shape, relative_distance[test].shape, torch.mean(rads), torch.mean(lengths))
    #     # exit()
    #     relative_distance[non_valid] = 0.0
    #     #keepout#
    #     # relative_distance = relative_distance.reshape(xyz.shape[1], locs.shape[0])#(sample, j)
    #     relative_distance = torch.transpose(relative_distance, 0 , 1)#(j, sample) -> (sample, j)
    #     #keepout
    #     # relative_distance_sum = torch.sum(relative_distance, dim=-1)
    #     # eps = 1e-5
    #     # valid_dist = relative_distance_sum > eps
    #     # print(rads.shape, rads_sum.shape, valid_rads.shape)
    #     # relative_distance[valid_dist] = relative_distance[valid_dist] / relative_distance_sum[valid_dist].unsqueeze(-1)
    #     # exit("shval")
    #     if torch.isnan(relative_distance).any() or torch.isinf(relative_distance).any():
    #         ValueError("nan or inf")
    #     # relative_distance[:,3:] = 0.0;
    #     # relative_distance[:,2] = 0.0;
    #     # relative_distance[:,:2] = 0.0;
    #     # relative_distance[:,1:] = 0.0;
    #     #4

    #     return relative_distance#weights, (sample, j)(j, sample)では？


    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
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
        # mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
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
        # rate_a = (self.aabb[1] - rays_o) / vec
        # rate_b = (self.aabb[0] - rays_o) / vec
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
        # mask_outbbox = ((self.aabb[0]>rays_pts) | (rays_pts>self.aabb[1])).any(dim=-1)
        mask_outbbox = ((self.ray_aabb[0]>rays_pts) | (rays_pts>self.ray_aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox


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


    def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1, skeleton_props=None, is_render_only=False):
        
        # <sample points> -> xyz, viewdirs
        if True:
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
            
            # if self.alphaMask is not None:
            #     alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            #     alpha_mask = alphas > 0
            #     ray_invalid = ~ray_valid
            #     ray_invalid[ray_valid] |= (~alpha_mask)
            #     ray_valid = ~ray_invalid

            sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
            rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)


        
        # for j in self.skeleton.get_children():
        #     apply_animation(self.framepose, j)
        # gt_skeleton_pose = self.skeleton.get_listed_rotations()
        # self.skeleton.transformNet(gt_skeleton_pose)

        #skeleton parsing -> transforms
        if not self.data_preparation:
            # transforms = self.skeleton.rotations_to_invs(gt_skeleton_pose)
            if skeleton_props is not None:
                # print("frame_pose_set")
                self.frame_pose = skeleton_props["frame_pose"]
            if is_train or not is_render_only:
                transforms = self.skeleton.rotations_to_invs_fast(self.frame_pose, type="quaternion")
            else:
                for j in self.skeleton.get_children():
                    apply_animation(self.frame_pose, j)
                gt_skeleton_pose = self.skeleton.get_listed_rotations()
                self.skeleton.transformNet(gt_skeleton_pose)
        
                transforms = self.skeleton.rotations_to_invs(gt_skeleton_pose)
                # transforms = self.skeleton.rotations_to_invs(self.frame_pose)

            draw_joints = self.render_jointmask
            if draw_joints:
                self.skeleton.transformNet(self.frame_pose,type="quaternion")       
                mask = self.skeleton.draw_mask_all(rays_chunk[:, :3], rays_chunk[:, 3:6], 0.05)
        shape = xyz_sampled.shape
        # Point Casting
        if not self.data_preparation:
            # xyz_slice = xyz_sampled.reshape(-1, 3).shape[0]
            # tmp = torch.cat([xyz_sampled.reshape(-1, 3),(xyz_sampled-viewdirs).reshape(-1, 3)], dim=0)
            # self.joints = listify_skeleton(self.skeleton)
            # weights = self.get_SH_vals(xyz_sampled.reshape(-1, 3), self.sh_feats, transforms, self.skeleton.get_listed_positions_first())

            # if(weights.isnan().any()):
            #     ValueError("weights is nan")
            # weights = torch.cat([weights, weights], dim=0)
            # # taihi = tmp
            # tmp =  weighted_transformation(tmp, weights, transforms)
            # if(tmp.isnan().any() or tmp.isinf().any()):
            #     ValueError("tmp is nan")
            
            # #debug
            # xyz_sampled, viewdirs = tmp[:xyz_slice], tmp[:xyz_slice] - tmp[xyz_slice:]
            # xyz_sampled = xyz_sampled.reshape(shape[0],shape[1], 3)
            # viewdirs = viewdirs.reshape(shape[0],shape[1], 3)
            
            xyz_sampled, viewdirs = self.caster(xyz_sampled, viewdirs, transforms)
            weights = self.caster.get_weights()

            save_npz = False;
            if save_npz:
                save_npz = {}
                save_npz["weights"] = weights.cpu().numpy()
                save_npz["xyz_sampled"] = xyz_sampled.cpu().numpy()

        
        # Compute_sigma
        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            xyz_sampled = xyz_sampled.reshape(shape[0],shape[1], 3)

            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])
            

            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma

            # weights = weights[:weights.shape[0]//2,:].reshape(xyz_sampled.shape[0], xyz_sampled.shape[1], weights.shape[-1])

            # outside  = weights.sum(dim=-1) < 0.001
            # inside = ~ outside

            # sigma[outside] = -0.000
            # sigma[inside] = 0.2;
            if not self.data_preparation and save_npz:
                save_npz["sigma"] = sigma.cpu().numpy()



        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        app_mask = weight > self.rayMarch_weight_thres

        # Compute_alpha
        if app_mask.any():
            app_features = self.compute_appfeature(xyz_sampled[app_mask])
            # print(app_features.shape)
            
            valid_rgbs = self.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features)
            # exit()
            rgb[app_mask] = valid_rgbs
            # print("rgb化")
            # print(rgb.shape, weights.shape)
            # exit("rgb化")
            
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
                rgb_map[mask] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=torch.device('cuda:0'))

        return rgb_map, depth_map # rgb, sigma, alpha, weight, bg_weight

