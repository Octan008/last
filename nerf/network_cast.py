import numpy as np
import trimesh

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching
from .utils import custom_meshgrid
from .render_util import *
import copy
from ffmlp import FFMLP

# import tinycudann as tcnn



class LocationCaster(nn.Module):
    def __init__(self,
                cuda=False,
                num_joints=None
                ):
        super().__init__()

        assert num_joints is not None
        self.num_joints = num_joints
        self.cuda = cuda

    
    def forward(self, x, pose):
        raise NotImplementedError()

    def cast(self, x, pose):
        raise NotImplementedError()

            

class CastNetwork(LocationCaster):
    def __init__(self,
                cuda=False,
                num_joints=None,
                num_layers=2,
                hidden_dim=64,
                ):
        super().__init__(cuda, num_joints)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.in_dim = (num_joints+1)*3
        #padding?
        self.out_dim=3
        #https://programming-surgeon.com/script/euler-python-script/

        # if self.in_dim%16!=0:
        #     self.in_dim = (self.in_dim//16 + 1) * 16
            

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # self.geo_feat_dim = geo_feat_dim
        # self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)

        cast_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim


            
            if l == num_layers - 1:
                # out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
                out_dim = 3
            else:
                out_dim = hidden_dim
            
            cast_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.cast_net = nn.ModuleList(cast_net)



        # self.cast_net = FFMLP(
        #     input_dim=self.in_dim, 
        #     output_dim=self.out_dim,
        #     hidden_dim=self.hidden_dim,
        #     num_layers=self.num_layers
        # )

        # self.cast_net = tcnn.Network(
        #     n_input_dims=63,
        #     n_output_dims=3,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": hidden_dim,
        #         "n_hidden_layers": self.num_layers - 1,
        #     },
        # )
        
    def forward(self, x, skeleton_pose):
        #pose : [N, j, 3]
        #x: [N, 3] -> [N,1,3]
        h = torch.cat([x, skeleton_pose], dim=-1)

        return h

    def cast(self, x, skeleton_pose):
        n = x.shape[0]
        skeleton_pose = skeleton_pose.flatten().unsqueeze(0).expand(n, -1)

        h = torch.cat([x, skeleton_pose], dim=-1)

        for l in range(self.num_layers):
            h = self.cast_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        return h