import torch
import torch.nn
import torch.nn.functional as F
from .sh import eval_sh_bases
import numpy as np
import time
from nerf.render_util import *
from models.sh_joints import SphereHarmonicJoints
import glob


class shCaster(nn.Module):
    def __init(self):
        self.sh_feats = None
    def set_skeleton(self, skeleton):
        self.skeleton = skeleton

    def forward(self, xyz_sampled, viewdirs, transforms):
        shape = xyz_sampled.shape
        xyz_slice = xyz_sampled.reshape(-1, 3).shape[0]
        tmp = torch.cat([xyz_sampled.reshape(-1, 3),(xyz_sampled-viewdirs).reshape(-1, 3)], dim=0)
        self.joints = listify_skeleton(self.skeleton)
        weights = self.get_SH_vals(xyz_sampled.reshape(-1, 3), self.sh_feats, transforms, self.skeleton.get_listed_positions_first())
        if(weights.isnan().any()):
            ValueError("weights is nan")
        weights = torch.cat([weights, weights], dim=0)
        # taihi = tmp
        tmp =  weighted_transformation(tmp, weights, transforms)
        if(tmp.isnan().any() or tmp.isinf().any()):
            ValueError("tmp is nan")
        #debug
        xyz_sampled, viewdirs = tmp[:xyz_slice], tmp[:xyz_slice] - tmp[xyz_slice:]
        xyz_sampled = xyz_sampled.reshape(shape[0],shape[1], 3)
        viewdirs = viewdirs.reshape(shape[0],shape[1], 3)
        self.weights = weights
        return xyz_sampled, viewdirs
            
    def set_SH_feats(self, feats):
        self.sh_feats = feats
    def get_weights(self):
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