from tkinter import N
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from .renderer import NeRFRenderer
from .render_util import *
import time

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                encoding="hashgrid",
                encoding_dir="sphere_harmonics",
                num_layers=2,
                hidden_dim=64,
                geo_feat_dim=15,
                num_layers_color=3,
                hidden_dim_color=64,
                bound=1,
                weight_prediction=False,
                num_joints=None,
                type="euler",
                **kwargs,
                ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.weight_prediction = weight_prediction
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)
        self.type=type
        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_color = get_encoder(encoding_dir)
        self.in_dim_color += self.geo_feat_dim
        self.num_layers_cast = 3

        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_color
            else:
                in_dim = hidden_dim
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)


        self.in_dim_cast = (num_joints+1) * 3
        if type == "quaternion":
            self.in_dim_cast = num_joints*4 + 3
        elif type=="matrix":
            self.in_dim_cast = num_joints*4*4 + 3
            
        cast_net = []
        for l in range(self.num_layers_cast):
            if l == 0:
                in_dim = self.in_dim_cast
                # in_dim = 66
            else:
                in_dim = hidden_dim


            
            if l == self.num_layers_cast - 1:
                if weight_prediction:
                    out_dim=num_joints
                # out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
                else:
                    out_dim = 3
            else:
                out_dim = hidden_dim
            
            cast_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.cast_net = nn.ModuleList(cast_net)

    
    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = self.encoder(x, bound=self.bound)

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = F.relu(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color

    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        # if torch.isnan(x).any():
        #     raise ValueError("isnannan")

        x = self.encoder(x, bound=self.bound)
        h = x
        # if torch.isnan(h).any():
        #     raise ValueError("h0")

        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            # if torch.isnan(h).any():
            #     print(torch.isnan(x).any())
            #     raise ValueError("h1")

            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = F.relu(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]
        d = self.encoder_dir(d)

        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        h = torch.sigmoid(h)


        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs
    def weight(self, x, skeleton_pose):

        n = x.shape[0]
        h = torch.cat([x, skeleton_pose.flatten().unsqueeze(0).expand(n, -1)], dim=-1)

        for l in range(self.num_layers_cast):
            h = self.cast_net[l](h)
            if l != self.num_layers_cast - 1 or True:
                h = F.relu(h, inplace=True)
        return h

    def cast(self, x, dir, skeleton_pose):
        # weight_start = time.perf_counter()


        n = x.shape[0]
        h = torch.cat([x, skeleton_pose.flatten().unsqueeze(0).expand(n, -1)], dim=-1)

        for l in range(self.num_layers_cast):
            h = self.cast_net[l](h)
            if l != self.num_layers_cast - 1 or True:
                h = F.relu(h, inplace=True)
        # print("weight_net", time.perf_counter() - weight_start)
        if self.weight_prediction:
            # lbs_start = time.perf_counter()
            # inv_start = time.perf_counter()
            # transforms = self.skeleton.rotations_to_invs(skeleton_pose, type=self.type)
            transforms = self.skeleton.rotations_to_invs_fast(skeleton_pose, type=self.type)
            # print("inv : ", time.perf_counter() - inv_start)
            # cast_start = time.perf_counter()
            # h = torch.where(h < 1e-6, torch.zeros_like(h), h)
            # tmp = h.sum(dim=1)
            # if (tmp < 1e-6).any():
            #     print(h.shape, tmp.shape)
            #     raise ValueError("minus weight")
            if dir is not None:

                xyz_slice = x.shape[0]
                tmp = torch.cat([x,x-dir], dim=0)
                # h = h.repeat(2, 1)
                h2 = torch.cat([h,h],dim=0)
                new_tmp = weighted_transformation(tmp, h2, transforms)
                # print("cast : ", time.perf_counter() - cast_start)
                return new_tmp[:xyz_slice], new_tmp[:xyz_slice] - new_tmp[xyz_slice:], h

            new_dir = None
            new_xyz = weighted_transformation(x, h, transforms)
            # if dir is not None:
            #     new_dir = new_xyz - weighted_transformation(x - dir, h, transforms)
            # # print("cast : ", time.perf_counter() - cast_start)
            # print("lbs : ", time.perf_counter() - lbs_start)
            # if torch.isnan(new_xyz).any() or torch.isinf(new_xyz).any():
            #     print(new_xyz)
            #     raise ValueError("Nan or inf input x Found")
            return new_xyz, new_dir, h

        return h