import torch
import torch.nn as nn
import torch.nn.functional as F
from .sh import eval_sh_bases
import numpy as np
import time
from nerf.render_util import *
from models.sh_field import SphereHarmonicJoints
import glob

from torchngp.encoding import get_encoder
from torchngp.ffmlp import FFMLP

# import tinycudann as tcnn

from .rigidbody import *
from functools import partial
import functorch
from .py_ffmlp import *


class PoseVector(nn.Module):
    def __init__(self, num_frames, num_dims, device, args=None):
        super().__init__()
        # poses = torch.randn(num_frames, num_joints*2)
        self.device = device
        # self.pose_params = nn.Parameter(poses, requires_grad=True)
        self.pose_params = nn.Embedding(num_frames, num_dims)

    def forward(self, i):
        return self.pose_params(torch.tensor([i], device=self.device))

    def init_weights(self, distribution):
        torch.nn.init.uniform_(
            self.pose_params.weight, -distribution, distribution)


class CasterBase(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.print_time = False
        self.matmul_func = partial(lambda x, y: torch.matmul(x, y))

    def torch_mlp_net(self, input_sequencial, _in_dim, _out_dim, num_layers, hidden_dim, device, if_use_bias=False):

        for l in range(num_layers):
            if l == 0:
                # in_dim = self.in_dim_three
                in_dim = _in_dim
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = _out_dim
            else:
                out_dim = hidden_dim
            input_sequencial.add_module("main_linear_%d" % l, nn.Linear(
                in_dim, out_dim, bias=if_use_bias).to(device))

    def forward(self, xyz_sampled, viewdirs, transforms, ray_valid, i_frame=None):
        pass

    def recoverty_skip(self, xyz_sampled, shape, padding):
        dim = xyz_sampled.shape[-1]
        xyz_sampled = xyz_sampled.view(shape[0], -1, dim)
        res = xyz_sampled[:, :-2,
                          :].unsqueeze(-2).expand(shape[0], -1, self.skip_rate+1, dim)
        res2 = xyz_sampled[:, 1:-1,
                           :].unsqueeze(-2).expand(shape[0], -1, self.skip_rate+1, dim)
        rate = torch.linspace(0, 1, self.skip_rate+1).view(1, 1, -1, 1).expand(
            shape[0], -1, self.skip_rate+1, 1).to(xyz_sampled.device)
        tmp = res * (1-rate) + res2 * rate
        add = xyz_sampled[:, -2:-1,
                          :].unsqueeze(-2).expand(shape[0], -1, padding, dim)
        add2 = xyz_sampled[:, -1:,
                           :].unsqueeze(-2).expand(shape[0], -1, padding, dim)
        rate_add = torch.linspace(0, 1, padding).view(
            1, 1, -1, 1).expand(shape[0], -1, padding, 1).to(xyz_sampled.device)
        tmp_add = add * (1-rate_add) + add2 * rate_add
        result = torch.cat([tmp[:, :, :-1, :].reshape(shape[0], -1, dim),
                           tmp_add[:, :, :, :].reshape(shape[0], -1, dim)], dim=1)
        return result

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        torch.save(ckpt, path)

    def set_all_grads(self, flag):
        for p in self.parameters():
            p.requires_grad = flag

    def set_skeleton(self, skeleton):
        self.skeleton = skeleton
        self.joints = listify_skeleton(self.skeleton)

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

    def scale_loss(self, sq_residual, scale=0.03, alpha=-2, eps=1e-6):
        unit = sq_residual/(scale**2)
        loss = (2*(unit))/(unit + 4)
        return loss * scale


class DistCaster(CasterBase):
    def __init__(self):
        super().__init__()

    def forward(self, xyz_sampled, viewdirs, transforms, ray_valid, i_frame=None):
        shape = xyz_sampled.shape
        self.skeleton.apply_precomputed_localtransorms()

        xyz_slice = xyz_sampled.view(-1, 3).shape[0]
        tmp = torch.cat(
            [xyz_sampled.view(-1, 3), (xyz_sampled-viewdirs).view(-1, 3)], dim=0)
        self.joints = listify_skeleton(self.skeleton)
        weights = compute_weights(
            xyz_sampled.view(-1, 3), self.joints).to(torch.float32)
        if (weights.isnan().any()):
            ValueError("weights is nan")
        self.weights = weights
        weights = torch.cat([weights, weights], dim=0)
        tmp = weighted_transformation(
            tmp, weights, transforms, if_transform_is_inv=self.args.use_indivInv)
        if (tmp.isnan().any() or tmp.isinf().any()):
            ValueError("tmp is nan")
        # debug
        xyz_sampled, viewdirs = tmp[:xyz_slice], tmp[:xyz_slice] - \
            tmp[xyz_slice:]
        xyz_sampled = xyz_sampled.view(shape)
        viewdirs = viewdirs.view(shape)

        return xyz_sampled, viewdirs


# @torch.cuda.amp.autocast(enabled=True)
class MLPCaster(CasterBase):
    def __init__(self, dim, device, args=None, use_ffmlp=False,
                 encoding="frequency", num_layers=2, hidden_dim=64, bound=1.0, use_bias=False, interface_dim=32, use_interface=True
                 ):
        super().__init__(args=args)

        self.encoding = encoding
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.bound = bound
        # self.hidden_dim = int(self.hidden_dim)

        self.if_use_ffmlp = use_ffmlp

        # sigma network
        self.geo_feat_dim = 0
        self.use_bias = use_bias
        self.interface_dim = interface_dim
        self.use_interface = use_interface
        self.skeleton_dim = dim
        self.output_dim = 1

        self.encoder, self.in_dim = get_encoder(
            self.encoding, desired_resolution=2048 * self.bound, multires=5)
        self.encoder = self.encoder.to(device)
        # self.encoder = torch.jit.script(self.encoder)

        if use_interface:
            # self.interface_layer = nn.Linear(self.in_dim, self.interface_dim, bias=True).to(device)
            self.interface_layer = nn.Linear(
                self.in_dim, self.interface_dim, bias=args.free_opt8).to(device)
            # exit()
            if args.free_opt8:
                torch.nn.init.uniform_(self.interface_layer.bias, -0.01, 0.01)
        else:
            self.interface_dim = self.in_dim

        if args.free_opt8:
            # self.bound = 8
            # self.encoder, self.in_dim = get_encoder("hashgrid", desired_resolution=2048 * self.bound, multires=6)
            # self.encoder = self.encoder.to(device)
            self.interface_dim = self.in_dim * self.skeleton_dim
            self.output_dim = self.output_dim * self.skeleton_dim

        if self.args.free_opt8:
            self.weight_nets = py_FFMLP(
                # input_dim=self.in_dim,
                input_dim=self.interface_dim,
                # input_dim=3,
                output_dim=self.output_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                device=device,
                # std=0.01,
                bias=True,
            ).to(device)
            # self.weight_nets = torch.jit.script(self.weight_nets)
        else:

            if self.if_use_ffmlp:
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
            else:
                self.weight_nets = []
                for i in range(dim):
                    pyffmlp = py_FFMLP(
                        # input_dim=self.in_dim,
                        input_dim=self.interface_dim,
                        # input_dim=3,
                        output_dim=1 + self.geo_feat_dim,
                        hidden_dim=self.hidden_dim,
                        num_layers=self.num_layers,
                        device=device
                    ).to(device)
                    # pyffmlp = torch.jit.script(pyffmlp)
                    self.weight_nets.append(
                        pyffmlp
                    )
                self.weight_nets = nn.ModuleList(self.weight_nets)

    @torch.cuda.amp.autocast(enabled=True)
    def mlp_branch(self, x, i=None, func=None):
        x = x.view(-1, 3)
        # tmp = self.encoder(x, bound=self.bound)
        tmp = self.encoder(x)
        if self.use_interface:
            tmp = self.interface_layer(tmp)

        if i is not None:
            h = self.weight_nets[i](tmp)
        else:
            h = func(tmp)

        sigma = F.relu(h[..., 0])

        return sigma

    @torch.cuda.amp.autocast(enabled=True)
    def concate_mlp(self, x):
        # x: sample, j, 3
        tmp = self.encoder(x, bound=self.bound)  # sample, j, in_dim
        # sample, j*in_dim
        return self.concate_mlp_sub(tmp.view(-1, self.interface_dim))

    def concate_mlp_sub(self, tmp):

        h = self.weight_nets(tmp)
        sigma = F.relu(h)
        sigma = sigma.view(-1, self.skeleton_dim).permute(1, 0)  # j, sample
        return sigma

    @torch.cuda.amp.autocast(enabled=True)
    def mlp(self, x):
        # x: [J, N, 3], in [-bound, bound]
        # x = x.view(self.skeleton_dim, -1, 3)
        x = x.view(-1, self.skeleton_dim, 3)
        res = []
        for i in range(len(self.weight_nets)):
            res.append(self.mlp_branch(x[:, i], i))
        return torch.stack(res, dim=0)

    @torch.cuda.amp.autocast(enabled=True)
    def forward(self, xyz_sampled, viewdirs, transforms, ray_valid, i_frame=None):
        shape = xyz_sampled.shape
        xyz_slice = xyz_sampled.view(-1, 3).shape[0]
        xyz_sampled, viewdirs = self.warp_points(
            xyz_sampled,  transforms, viewdirs=viewdirs)
        return xyz_sampled.view(shape), viewdirs.view(shape)
        tmp = self.warp_points(xyz_sampled,  transforms, viewdirs=viewdirs)

        # after care
        xyz_sampled, viewdirs = tmp[:xyz_slice], tmp[:xyz_slice] - \
            tmp[xyz_slice:]
        xyz_sampled = xyz_sampled.view(shape)
        viewdirs = viewdirs.view(shape)

        return xyz_sampled, viewdirs

    @torch.cuda.amp.autocast(enabled=True)
    def warp_points(self, points, transforms, viewdirs=None):
        # points = torch.cat([points, torch.ones(points.shape[:-1], device = points.device).unsqueeze(-1)], dim=-1)#[samples, 4]
        weights = self.compute_weights(
            points.view(-1, 3), transforms)  # [samples, J]
        self.weights = weights
        self.cache_transforms = transforms
        self.restore_xyz = points

        if self.print_time:
            start = time.time()
        if viewdirs is not None:
            result, viewdirs = weighted_transformation(points.view(-1, 3), weights.to(
                torch.float32), transforms, viewdirs.contiguous().view(-1, 3), if_transform_is_inv=self.args.use_indivInv)
            return result, viewdirs
        else:
            subjects = points.reshape(-1, 3)

            result = weighted_transformation(subjects, weights.to(
                torch.float32), transforms, if_transform_is_inv=self.args.use_indivInv)
            return result

    # @torch.cuda.amp.autocast(enabled=True)
    # def _warp_points(self, points, transforms, viewdirs = None):
        if self.print_time:
            start = time.time()
        # points = torch.cat([points, torch.ones(points.shape[:-1], device = points.device).unsqueeze(-1)], dim=-1)#[samples, 4]
        weights = self.compute_weights(points.view(-1, 3), transforms)
        if self.print_time:
            print('compute_weights', time.time() - start)
        self.weights = weights
        self.cache_transforms = transforms
        self.restore_xyz = points

        if self.print_time:
            start = time.time()
        if viewdirs is not None:
            if self.print_time:
                start = time.time()
            weights = torch.cat([weights, weights], dim=0)
            if self.print_time:
                print('weights', time.time() - start)
            if self.print_time:
                start = time.time()
            # viewdirs = torch.cat([viewdirs, torch.zeros(viewdirs.shape[:-1], device = points.device).unsqueeze(-1)], dim=-1)#[samples, 4]
            if self.print_time:
                print('viewdirs', time.time() - start)
            if self.print_time:
                start = time.time()
            subjects = torch.cat(
                [points.view(-1, 3), (points-viewdirs).view(-1, 3)], dim=0)
            if self.print_time:
                print('subjects', time.time() - start)
            # transforms = torch.cat([transforms, transforms], dim=0)
        else:
            subjects = points.reshape(-1, 3)
        if self.print_time:
            print('subjects', time.time() - start)

        if self.print_time:
            start = time.time()
        result = weighted_transformation(subjects, weights.to(
            torch.float32), transforms, if_transform_is_inv=self.args.use_indivInv)
        if self.print_time:
            print('weighted_transformation', time.time() - start)
        return result

    def compute_rest_positions(self, xyz, transforms,  features=None, locs=None):
        xyz = xyz.view(-1, 3)
        xyz = torch.cat([xyz, torch.ones(
            xyz.shape[0], device=xyz.device).unsqueeze(-1)], dim=-1)  # [samples, 4]
        assert (not torch.isnan(transforms).any())
        if self.args.use_indivInv:
            invs = transforms
        else:
            invs = affine_inverse_batch(
                self.skeleton.precomp_forward_global_transforms)

        result = functorch.vmap(self.matmul_func, in_dims=(
            None, 0), out_dims=0)(invs, xyz.unsqueeze(-1)).squeeze(-1)  # [sample, j, 4]
        # print(result.shape)
        self.cache_transformed_pos = result.permute(1, 0, 2)  # [j, sample, 4]
        self.mapped_pos = result
        dists = torch.sum(result[:, :, :3] * result[:, :, :3], dim=-1)
        alphas = torch.exp(-dists)  # [sample, j]

        result = self.normalize_coord(result[:, :, :3])
        result = (result+1)/2

        return result, alphas
        # result : N, J, 3
        # alphas : N, J

    @torch.cuda.amp.autocast(enabled=True)
    def compute_weights(self, xyz, transforms,  features=None, locs=None):
        result, alphas = self.compute_rest_positions(
            xyz, transforms, features, locs)
        if self.args.free_opt8:
            bwf = self.concate_mlp(result)  # sample, j -> j, sample
        else:
            bwf = self.mlp(result)  # sample, j -> j, sample
        bwf = bwf * alphas.permute(1, 0)  # [j, sample]
        return bwf.permute(1, 0)  # [sample, j]

    @torch.cuda.amp.autocast(enabled=True)
    def compute_warp_jacobian(self, points):
        func = partial(self.warp_points, transforms=self.cache_transforms)
        return functorch.vmap(functorch.jacfwd(func))(points)

    @torch.cuda.amp.autocast(enabled=True)
    def compute_weights_grad(self, points):
        if not self.args.free_opt8 or True:
            func = partial(self.compute_weights,
                           transforms=self.cache_transforms)
            tmp = functorch.vmap(functorch.jacfwd(
                func), in_dims=0, out_dims=0)(points).squeeze(1)
            tmp = torch.matmul(tmp, points.unsqueeze(-1)).squeeze(-1)
            return tmp
        else:
            tmp, _ = self.compute_rest_positions(points, self.cache_transforms)
            encoded_pos = self.encoder(tmp)
            func = partial(self.concate_mlp_sub)
            tmp = functorch.vmap(functorch.jacfwd(func), in_dims=0, out_dims=0)(
                encoded_pos.view(-1, 1, self.interface_dim)).squeeze(1)
            tmp = torch.matmul(
                tmp, encoded_pos.view(-1, self.interface_dim, 1)).squeeze(-1)
            return tmp

        #
        # return functorch.vmap(functorch.vjp(func), in_dims = 0)(points)

    @torch.cuda.amp.autocast(enabled=True)
    def compute_elastic_loss(self, num_sample=-1):
        points = self.restore_xyz[:, :].reshape(-1, 3)
        points_shape = points.shape
        idx = None
        if num_sample > 0:
            idx = torch.randint(0, points_shape[0], size=[
                                num_sample], device=points.device)
            points = torch.index_select(points, 0, idx)

        jacobian = self.compute_warp_jacobian(points.contiguous())
        __, svals, __ = torch.svd(jacobian.to(torch.float32), compute_uv=False)
        log_svals = torch.log(svals.to(torch.float16)+1e-4)
        residual = torch.sum(log_svals**2, dim=-1)

        return self.scale_loss(residual, scale=0.03), idx

    @torch.cuda.amp.autocast(enabled=True)
    def compute_weight_elastic_loss(self, num_sample=-1):
        points = self.restore_xyz[:, :].reshape(-1, 3)
        points_shape = points.shape
        idx = None
        if num_sample > 0:
            idx = torch.randint(0, points_shape[0], size=[
                                num_sample], device=points.device)
            points = torch.index_select(points, 0, idx)

        grad = self.compute_weights_grad(points.contiguous())
        if torch.isnan(grad).any():
            print("nan in residual")
            raise ValueError("gradval")
        grad = grad ** 2
        log_grad = torch.log(1+grad)
        residual = torch.sum(log_grad, dim=-1)

        return self.scale_loss(residual, scale=0.5), idx


class MLPCaster_integrate(MLPCaster):
    def __init__(self, dim, device, args=None):
        super().__init__(dim, device, args=args)
        encoding = "frequency"
        self.num_layers = 2
        self.hidden_dim = 64
        self.bound = 1.0
        self.hidden_dim = int(self.hidden_dim)

        # sigma network
        self.geo_feat_dim = geo_feat_dim = 0
        self.interface_dim = 32
        self.after_interface_dim = 3
        self.j_dim = dim

        # self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)

        self.encoder, self.in_dim = get_encoder(
            encoding, desired_resolution=2048 * self.bound)
        self.interface_layer = None
        self.interface_layer = nn.Linear(
            self.in_dim, self.interface_dim, bias=False).to(device)
        self.after_layer = nn.Linear(
            self.after_interface_dim*self.j_dim, self.j_dim, bias=False).to(device)
        self.encoder = self.encoder.to(device)

        weight_nets = []
        for i in range(self.j_dim):

            weight_nets.append(
                FFMLP(
                    # input_dim=self.in_dim,
                    input_dim=self.interface_dim,
                    # input_dim=3,
                    output_dim=self.after_interface_dim,
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                ).to(device)
            )
        self.weight_nets = nn.ModuleList(weight_nets)

    @torch.cuda.amp.autocast(enabled=True)
    def density(self, x):
        # x: [J, N, 3], in [-bound, bound]
        res = []
        for i in range(len(self.weight_nets)):
            tmp = self.encoder(x[i], bound=self.bound)
            tmp = self.interface_layer(tmp)
            h = self.weight_nets[i](tmp)
            res.append(h)
            # sigma = F.relu(h[..., 0])

            # res.append(sigma)
        res = torch.stack(res, dim=0)  # j,n,3
        res = res.permute(1, 0, 2).contiguous().view(-1,
                                                     self.after_interface_dim*self.j_dim)  # n,j,3->n,j*3
        res = self.after_layer(res)  # n,j*3->n,j

        return F.relu(res.permute(1, 0))


class DirectMapCaster(CasterBase):
    def __init__(self, num_frames, device, args=None):
        super().__init__(args=args)
        encoding = "frequency"
        # encoding = "hashgrid"
        self.num_layers = 2
        self.hidden_dim = 64
        self.bound = 1.0
        self.hidden_dim = int(self.hidden_dim)
        self.num_frames = num_frames

        # sigma network
        self.geo_feat_dim = geo_feat_dim = 0
        self.interface_dim = 32
        self.trunk_dim = 16
        self.interface_layer = None
        self.if_use_bias = True

        self.encoder, self.in_dim = get_encoder(
            encoding, desired_resolution=2048 * self.bound, multires=5)

        self.encoder = self.encoder
        self.pose_dim = 16
        self.distribution = 0.15

        self.pose_params = PoseVector(
            num_frames, self.pose_dim, device).to(device)

        self.map_nets = []
        self.interface_layer = nn.Linear(
            self.in_dim+self.pose_dim, self.interface_dim, bias=self.if_use_bias).to(device)

        self.map_nets = nn.Sequential().to(device)
        self.map_nets.add_module('linear1', nn.Linear(
            self.interface_dim, self.trunk_dim, bias=self.if_use_bias).to(device))
        for i in range(len(self.map_nets)):
            torch.nn.init.uniform_(
                self.map_nets[i].weight, -self.distribution, self.distribution)
            torch.nn.init.uniform_(
                self.map_nets[i].bias, -self.distribution, self.distribution)
        # self.map_nets = FFMLP(
        #     input_dim=self.interface_dim,
        #     # input_dim=self.interface_dim,
        #     # input_dim=3,
        #     output_dim= self.trunk_dim,
        #     hidden_dim=self.hidden_dim,
        #     num_layers=self.num_layers,
        #     std=self.distribution
        # ).to(device)
        self.central_map = py_FFMLP(
            input_dim=self.in_dim,
            output_dim=1,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            std=self.distribution*10,
            device=device
        )

        self.branch_w = nn.Linear(
            self.trunk_dim, 3, bias=self.if_use_bias).to(device)
        self.branch_v = nn.Linear(
            self.trunk_dim, 3, bias=self.if_use_bias).to(device)

        torch.nn.init.uniform_(
            self.interface_layer.weight, -self.distribution, self.distribution)
        torch.nn.init.uniform_(
            self.interface_layer.bias, -self.distribution, self.distribution)

        torch.nn.init.uniform_(self.branch_w.weight, -
                               self.distribution, self.distribution)
        torch.nn.init.uniform_(self.branch_w.bias, -
                               self.distribution, self.distribution)
        torch.nn.init.uniform_(self.branch_v.weight, -
                               self.distribution, self.distribution)
        torch.nn.init.uniform_(self.branch_v.bias, -
                               self.distribution, self.distribution)
        self.pose_params.init_weights(self.distribution)

        # self.map_nets = nn.ModuleList(self.map_nets)
        # self.interface_layer = nn.ModuleList(self.interface_layer)
        self.weights = None

    def set_Cycle_pose(self, pose):
        self.pose_params = pose
        for param in self.pose_params.parameters():
            param.requires_grad = False

    def central_weight(self, x):
        x = self.normalize_coord(x)
        x = self.encoder(x, bound=self.bound)
        x = self.central_map(x)
        x = torch.sigmoid(x*100)
        return x

    def central_loss(self, num_sample=-1):
        points = self.restore_xyz.view(-1, 3)
        points_shape = points.shape
        idx = None
        if num_sample > 0:
            idx = torch.randint(0, points_shape[0], size=[
                                num_sample], device=points.device)
            points = torch.index_select(points, 0, idx)
        points_shape = points.shape
        old_points = points
        points = self.warp_points(
            points, pose_params=self.pose_params(self.i_frame))

        central = self.central_weight(points.view(-1, 3))  # N
        central = 1 - torch.exp(-central*3)
        central_weight_loss = torch.mean(central)

        center_mask = central > 0.9  # num_center
        num_centers = torch.sum(center_mask)
        # num_center, 3
        center_points = points.view(-1, 3)[center_mask.squeeze(-1)]
        dists = points.view(-1, 1, 3).expand(-1, num_centers, -1) - center_points.view(
            1, -1, 3).expand(points_shape[0], -1, -1)  # N, num_center, 3
        dists = torch.norm(dists, dim=2)  # N, num_center
        dist_weights = torch.exp(-dists)  # N, num_center
        dist_weights = dist_weights / \
            torch.sum(dist_weights, dim=1).view(-1,
                                                1).expand(-1, num_centers)  # N, num_center

        # num_center, 6
        reference_so3 = self.screw_axis.view(-1, 6)[center_mask.squeeze(-1)]

        meaned_refference = torch.sum(reference_so3.view(1, -1, 6).expand(
            points_shape[0], -1, -1) * dist_weights.view(-1, num_centers, 1).expand(-1, -1, 6), dim=1)  # N,6

        reconstruct_so3 = self.screw_axis.view(-1, 6) * \
            central + meaned_refference * (1-central)  # N, 6

        reconstruxt_loss = torch.mean(torch.norm(
            reconstruct_so3 - self.screw_axis.view(-1, 6), dim=1))
        return central_weight_loss, reconstruxt_loss

    def central_grad_loss(self, num_sample=-1):
        points = self.warp_points[:, :].reshape(-1, 3)
        points_shape = points.shape
        idx = None
        if num_sample > 0:
            idx = torch.randint(0, points_shape[0], size=[
                                num_sample], device=points.device)
            points = torch.index_select(points, 0, idx)
        # central = self.central_weight(points.view(-1,3))
        central_grad_loss = functorch.vmap(
            functorch.grad(self.central_weight))(points.view(-1, 3))
        return torch.mean(central_grad_loss ** 2)

    def set_reqires_grad(self, flag):
        for param in self.pose_params.parameters():
            param.requires_grad = flag

    @torch.cuda.amp.autocast(enabled=True)
    def density(self, x, pose_params):
        x = self.normalize_coord(x)
        tmp = self.encoder(x, bound=self.bound)
        embed = pose_params.expand(tmp.shape[0], -1)
        tmp = torch.cat([tmp, embed], dim=-1)
        tmp = self.interface_layer(tmp)
        tunk = self.map_nets(tmp)
        w = self.branch_w(tunk)

        v = self.branch_v(tunk)
        return w, v

    def warp_points(self, points, pose_params, viewdirs=None):
        new_transforms = self.compute_transforms(
            points.reshape(-1, 3), pose_params)
        self.warp_transforms = new_transforms
        if viewdirs is not None:
            subjects = points.reshape(-1, 3)
            subjects = torch.cat([subjects, torch.ones(
                subjects.shape[0], device=subjects.device).unsqueeze(-1)], dim=-1)  # [N,4]
            new_p = torch.matmul(
                new_transforms, subjects.unsqueeze(-1)).squeeze()[..., :3]
            new_v = torch.matmul(
                new_transforms[..., :3, :3], viewdirs.reshape(-1, 3).unsqueeze(-1)).squeeze()[..., :3]
            return new_p, new_v
        else:
            subjects = points.reshape(-1, 3)
            subjects = torch.cat([subjects, torch.ones(
                subjects.shape[0], device=subjects.device).unsqueeze(-1)], dim=-1)  # [N,4]
            return torch.matmul(new_transforms, subjects.unsqueeze(-1)).squeeze()[..., :3]

        subjects = torch.cat([subjects, torch.ones(
            subjects.shape[0]).unsqueeze(-1).to(subjects.device)], dim=--1)  # [N,4]
        return torch.matmul(new_transforms, subjects.unsqueeze(-1)).squeeze()[..., :3]

    def forward(self, xyz_sampled, viewdirs, transforms, ray_valid, i_frame=None):
        shape = xyz_sampled.shape
        xyz_slice = xyz_sampled.view(-1, 3).shape[0]
        self.restore_xyz = xyz_sampled
        self.i_frame = i_frame
        # tmp = self.warp_points(xyz_sampled, self.pose_params(i_frame), viewdirs)
        # xyz_sampled, viewdirs = tmp[:xyz_slice], tmp[:xyz_slice] - tmp[xyz_slice:]
        xyz_sampled, viewdirs = self.warp_points(
            xyz_sampled, self.pose_params(i_frame), viewdirs)
        self.warped_point = xyz_sampled
        # xyz_sampled, __ = tmp[:xyz_slice], tmp[:xyz_slice] - tmp[xyz_slice:]
        xyz_sampled = xyz_sampled.view(shape)
        viewdirs = viewdirs.view(shape)

        return xyz_sampled, viewdirs

    @torch.cuda.amp.autocast(enabled=True)
    def compute_transforms(self, xyz, pose_params, features=None, locs=None):
        w, v = self.density(xyz, pose_params)  # sample, 6
        eps = 1e-6
        theta = torch.sum(w*w, axis=-1)
        theta = torch.clamp(theta, min=eps).sqrt()

        v = v/theta.unsqueeze(-1)
        w = w/theta.unsqueeze(-1)
        screw_axis = torch.cat([w, v], axis=-1)
        self.screw_axis = screw_axis
        transform = exp_se3(screw_axis.permute(1, 0), theta).permute(2, 0, 1)

        return transform

    def compute_jacobian(self, points):
        func = partial(self.warp_points,
                       pose_params=self.pose_params(self.i_frame))
        return functorch.vmap(functorch.jacfwd(func))(points)
        # return torch.autograd.functional.jacobian(func, points, create_graph=True, vectorize=True)

    @torch.cuda.amp.autocast(enabled=True)
    def compute_elastic_loss(self, num_sample=-1, scale=0.5):
        points = self.restore_xyz[:, :].reshape(-1, 3)
        points_shape = points.shape
        idx = None
        if num_sample > 0:
            idx = torch.randint(0, points_shape[0], size=[
                                num_sample], device=points.device)
            points = torch.index_select(points, 0, idx)

        jacobian = self.compute_jacobian(points.contiguous())
        # print(jacobian.shape)
        __, svals, __ = torch.svd(jacobian.to(torch.float32), compute_uv=False)
        # print(svals.shape)
        # exit()
        log_svals = torch.log(svals.to(torch.float16)+1e-6)
        residual = torch.sum(log_svals**2, dim=-1)

        return self.scale_loss(residual, scale=scale), idx


class MapCaster(CasterBase):
    def __init__(self, num_frames, device, args=None):
        super().__init__(args=args)
        encoding = "frequency"
        # encoding = "hashgrid"
        self.num_layers = 2
        self.hidden_dim = 64
        self.bound = 1.0
        self.hidden_dim = int(self.hidden_dim)
        self.num_frames = num_frames

        # sigma network
        self.geo_feat_dim = geo_feat_dim = 0
        self.interface_dim = 32
        self.interface_dim = 32
        self.interface_layer = None

        self.encoder, self.in_dim = get_encoder(
            encoding, desired_resolution=2048 * self.bound, multires=6)
        # self.interface_layer = nn.Linear(self.in_dim, self.interface_dim, bias=False).to(device)
        self.interface_layer = []
        # self.interface_layer.weight.data.fill_(0.00)
        # self.interface_layer = nn.Linear(self.in_dim, self.interface_dim, bias=False).to(device)
        self.encoder = self.encoder
        self.pose_params = PoseVector(num_frames, 20, device).to(device)

        self.map_nets = []
        self.interface_layer = nn.Linear(
            self.in_dim+20*2, self.interface_dim, bias=False).to(device)

        self.map_nets = FFMLP(
            input_dim=self.interface_dim,
            # input_dim=self.interface_dim,
            # input_dim=3,
            output_dim=3+3,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        ).to(device)
        # torch_mlp_net(self.interface_dim, 3+3, self.hidden_dim, self.num_layers, device).to(device)
        # self.map_nets = nn.ModuleList(self.map_nets)
        # self.interface_layer = nn.ModuleList(self.interface_layer)
        self.weights = None

    def sub_encoding(self, xyzs):
        xyzs = xyzs.view(-1, 3)
        xyzs1 = xyzs.clone()
        xyzs2 = xyzs.clone()
        for i in range(6):
            xyzs1 += torch.sin(xyzs * torch.exp(torch.tensor(i)))
            xyzs2 += torch.cos(xyzs * torch.exp(torch.tensor(i)))
        return xyzs1, xyzs2

    @torch.cuda.amp.autocast(enabled=True)
    def density(self, x, i_frame):
        # if i_frame == 0 or i_frame == 1:
        #     print(i_frame, self.pose_params(i_frame))
        tmp = self.encoder(x, bound=self.bound)
        tmp = torch.cat([tmp, self.pose_params(
            i_frame).expand(tmp.shape[0], -1)], dim=-1)
        tmp = self.interface_layer(tmp)
        # tmp = F.relu(tmp)
        h = self.map_nets(tmp)
        return h

    def forward(self, xyz_sampled, viewdirs, transforms, ray_valid, i_frame=None):
        shape = xyz_sampled.shape
        xyz_slice = xyz_sampled.view(-1, 3).shape[0]
        # # 1115
        transforms = self.compute_transforms(
            xyz_sampled.view(-1, 3), transforms, i_frame)
        # self.weights = weights
        transforms = torch.cat([transforms, transforms], dim=0)
        tmp = torch.cat(
            [xyz_sampled.view(-1, 3), (xyz_sampled-viewdirs).view(-1, 3)], dim=0)
        tmp = torch.cat(
            [tmp, torch.ones(tmp.shape[0]).unsqueeze(-1).to(tmp.device)], dim=--1)  # [N,4]
        tmp = torch.matmul(transforms, tmp.unsqueeze(-1)).squeeze()[..., :3]
        if (tmp.isnan().any() or tmp.isinf().any()):
            ValueError("tmp is nan")
        # debug
        xyz_sampled, viewdirs = tmp[:xyz_slice], tmp[:xyz_slice] - \
            tmp[xyz_slice:]
        xyz_sampled = xyz_sampled.view(shape)
        viewdirs = viewdirs.view(shape)

        return xyz_sampled, viewdirs

    def compute_transforms(self, xyz, transforms,  i_frame, features=None, locs=None):
        euler_scaling = 1
        trans_scaling = 1
        map_features = self.density(xyz, i_frame)  # sample, 6
        correction_euler, correction_trans = self.sub_encoding(xyz)
        map_features = map_features + \
            torch.cat([correction_euler, correction_trans], dim=-1) * 0.03

        # map_features = map_features + torch.tensor([3.0, -4.0, 3.0, -0.1, 0.2, -0.15]).to(map_features.device).unsqueeze(0).expand(map_features.shape[0], -1)
        mats = euler_to_matrix_batch(
            map_features[:, :3]*euler_scaling, top_batching=True)  # sample, 3, 3
        # mats = torch.eye(4).unsqueeze(0).expand(xyz.shape[0], -1, -1).to(xyz.device)
        mats[:, :3, 3] = map_features[:, 3:]*trans_scaling
        return mats


class BWCaster(CasterBase):
    def __init__(self, dim, gridSize, device, skip_rate=-1, args=None):
        super().__init__()
        self.app_n_comp = [16, 16, 16]
        self.j_channel = dim
        self.gridSize = gridSize
        self.app_dim = dim
        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]
        self.aabb = None
        self.invaabbSize = None
        self.ray_aabb = None
        self.joints = None
        self.init_svd_volume(gridSize, device)
        self.skip_rate = skip_rate
        self.args = args

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.ray_aabb[0]) * self.invrayaabbSize - 1

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001, wd=1e-6):
        grad_vars = [{'params': self.app_line, 'lr': lr_init_spatialxyz, 'weight_decay': wd}, {
            'params': self.app_plane, 'lr': lr_init_spatialxyz, 'weight_decay': wd}]
        return grad_vars

    def init_svd_volume(self, res, device):
        self.app_plane, self.app_line = self.init_one_svd(
            self.app_n_comp, self.gridSize, 1, device)
        self.basis_mat = torch.nn.Linear(
            sum(self.app_n_comp), self.app_dim, bias=False).to(device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                -(scale * 0.5) + 0.1 + scale * torch.randn((self.j_channel, 1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            line_coef.append(
                torch.nn.Parameter(
                    -(scale * 0.5) + 0.1 + scale * torch.randn((self.j_channel,
                                                                1, n_component[i], gridSize[vec_id], 1))
                ))
        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

    def forward(self, xyz_sampled, viewdirs, transforms, ray_valid, i_frame=None):
        shape = xyz_sampled.shape

        if self.skip_rate > 0:
            xyz_sampled, viewdirs = xyz_sampled[:,
                                                ::self.skip_rate, :], viewdirs[:, ::self.skip_rate, :]
            padding = 1
            if shape[1] % self.skip_rate != 1:
                padding = shape[1] % self.skip_rate
                xyz_sampled = torch.cat(
                    [xyz_sampled, xyz_sampled[:, -1:, :]], dim=1)
                viewdirs = torch.cat([viewdirs, viewdirs[:, -1:, :]], dim=1)

        # xyz_slice = xyz_sampled.view(-1, 3).shape[0]
        weights = self.compute_weights(xyz_sampled.view(-1, 3), transforms)
        self.weights, self.cache_transforms, self.restore_xyz = weights, transforms, xyz_sampled

        xyz_sampled, viewdirs = weighted_transformation(xyz_sampled.view(-1, 3), weights.to(
            torch.float32), transforms, viewdirs.contiguous().view(-1, 3), if_transform_is_inv=self.args.use_indivInv)
        xyz_sampled, viewdirs = xyz_sampled.view(
            shape[0], -1, 3), viewdirs.view(shape[0], -1, 3)

        if self.skip_rate > 0:
            xyz_sampled = self.recoverty_skip(xyz_sampled, shape, padding)
            viewdirs = self.recoverty_skip(viewdirs, shape, padding)
            self.weights = self.recoverty_skip(
                self.weights, shape, padding).view(shape[0]*shape[1], -1)
        return xyz_sampled, viewdirs

    # def TV_loss_blendweights(self, reg, linear = False):
    #     total = 0
    #     for idx in range(len(self.app_line)):
    #         for i in range(self.app_line[idx].shape[0]):
    #             total = total + reg(self.app_line[idx][i].unsqueeze(0)) * 1e-4 + reg(self.app_plane[idx][i].unsqueeze(0)) * 1e-3
    #     return total

    def TV_loss_bwf(self, reg, linear=False):
        total = 0
        for idx in range(len(self.app_line)):
            total = total + reg(self.app_line[idx][0].unsqueeze(0)) * 1e-3
        return total

    def linear_loss(self):
        total = 0
        for idx in range(len(self.app_line)):
            for i in range(self.app_line[idx].shape[0]):
                total = total + (tv_loss_func_line(self.app_line[idx][i]) * 1e-4 + tv_loss_func_plane(
                    self.app_plane[idx][i]) * 1e-3) * 1e-4
        return total

    def L1_loss_bwf(self):
        total = 0
        for idx in range(len(self.app_line)):
            # + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
            total = total + \
                torch.mean(
                    torch.abs(self.app_line[idx])) + torch.mean(torch.abs(self.app_plane[idx]))
        return total

    def sample_BWfield(self, xyz_sampled):
        # plane + line basis
        # plane : 3, self.j_channel, 1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]
        feats = []
        for i in range(xyz_sampled.shape[0]):
            coordinate_plane = torch.stack(
                (xyz_sampled[i, ..., self.matMode[0]], xyz_sampled[i, ..., self.matMode[1]], xyz_sampled[i, ..., self.matMode[2]])).view(3, -1, 1, 2)
            coordinate_line = torch.stack(
                (xyz_sampled[i, ..., self.vecMode[0]], xyz_sampled[i, ..., self.vecMode[1]], xyz_sampled[i, ..., self.vecMode[2]]))
            coordinate_line = torch.stack(
                (torch.zeros_like(coordinate_line), coordinate_line), dim=-1).view(3, -1, 1, 2)
            sigma_feature = torch.zeros(
                xyz_sampled.shape[1], device=xyz_sampled.device)
            for idx_plane in range(len(self.app_plane)):
                # self.app_plane[idx_plane][i,0].unsqueeze(0) : 1, C, W, H
                # coordinate_plane[[idx_plane]] : 1, H, W, 2
                plane_coef_point = F.grid_sample(self.app_plane[idx_plane][i, 0].unsqueeze(0), coordinate_plane[[idx_plane]],
                                                 align_corners=True).view(-1, *xyz_sampled.shape[1:2])
                line_coef_point = F.grid_sample(self.app_line[idx_plane][i, 0].unsqueeze(0), coordinate_line[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[1:2])
                sigma_feature = sigma_feature + \
                    torch.sum(plane_coef_point * line_coef_point, dim=0)

            feats.append(sigma_feature)
        return F.relu(torch.stack(feats))

    def compute_weights(self, xyz, transforms,  features=None, locs=None):
        xyz = torch.cat([xyz, torch.ones(
            xyz.shape[0], device=xyz.device).unsqueeze(-1)], dim=-1)  # [samples, 4]
        # xyz_new : [samples, 4]
        # invs: [j, 4, 4]

        if self.args.use_indivInv:
            invs = transforms
        else:
            invs = affine_inverse_batch(
                self.skeleton.precomp_forward_global_transforms)
        # result = torch.matmul(invs, xyz_new.permute(1,0).unsqueeze(0).expand(invs.shape[0],4,-1)).squeeze().permute(0,2,1)#[j, samples, 4]
        result = functorch.vmap(self.matmul_func, in_dims=(
            None, 0), out_dims=1)(invs, xyz.unsqueeze(-1)).squeeze(-1)
        self.cache_transformed_pos = result
        dists = torch.sum(result[:, :, :3] * result[:, :, :3], dim=-1)
        alphas = torch.exp(-dists)
        result = self.normalize_coord(result[:, :, :3])
        bwf = self.sample_BWfield(result)  # [j,sample]
        self.test_rad = alphas[0]
        bwf = bwf * alphas
        return bwf.permute(1, 0)


class Map_BWCaster(CasterBase):
    def __init__(self, dim, gridSize, device):
        super().__init__()
        self.app_n_comp = [16, 16, 16]
        self.j_channel = dim
        self.gridSize = gridSize
        self.app_dim = dim
        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]
        self.aabb = None
        self.invaabbSize = None
        self.ray_aabb = None
        self.joints = None
        self.se3_dim = 6
        self.init_svd_volume(gridSize, device)

    def compute_jacobian(self, points):
        func = partial(self.warp_points)
        return functorch.vmap(functorch.jacfwd(func))(points)
        # return torch.autograd.functional.jacobian(func, points, create_graph=True, vectorize=True)

    @torch.cuda.amp.autocast(enabled=True)
    def compute_elastic_loss(self, num_sample=-1):
        points = self.restore_xyz[:, :].reshape(-1, 3)
        points_shape = points.shape
        idx = None
        if num_sample > 0:
            idx = torch.randint(0, points_shape[0], size=[
                                num_sample], device=points.device)
            points = torch.index_select(points, 0, idx)

        jacobian = self.compute_jacobian(points.contiguous())
        __, svals, __ = torch.svd(jacobian.to(torch.float32), compute_uv=False)
        log_svals = torch.log(svals.to(torch.float16)+1e-6)
        residual = torch.sum(log_svals**2, dim=-1)

        return self.scale_loss(residual, scale=0.5), idx

    def warp_points(self, xyz_sampled, viewdirs=None):

        # # 1115
        xyz_normalized = self.normalize_coord(xyz_sampled)
        bwf = self.sample_BWfield(
            xyz_normalized.view(-1, 3)).permute(1, 0)  # [j,sample]
        w = bwf[..., :3]
        v = bwf[..., 3:]
        eps = 1e-6
        theta = torch.sum(w*w, axis=-1)
        theta = torch.clamp(theta, min=eps).sqrt()

        v = v/theta.unsqueeze(-1)
        w = w/theta.unsqueeze(-1)
        screw_axis = torch.cat([w, v], axis=-1)

        transform = exp_se3(screw_axis.permute(1, 0), theta).permute(2, 0, 1)
        # print(screw_axis.shape, theta.shape, xyz_sampled.shape, transform.shape)
        xyz_sampled = torch.cat([xyz_sampled.view(-1, 3), torch.ones(xyz_sampled.view(-1,
                                3).shape[0], device=xyz_sampled.device).unsqueeze(-1)], dim=--1)  # [N,4]
        xyz_sampled = torch.matmul(
            transform, xyz_sampled.view(-1, 4).unsqueeze(-1)).squeeze()[..., :3]

        xyz_sampled = xyz_sampled[..., :3]

        self.weights = transform
        self.cache_transforms = transform
        self.restore_xyz = xyz_sampled
        if viewdirs is not None:
            viewdirs = torch.matmul(
                transform[..., :3, :3], viewdirs.reshape(-1, 3).unsqueeze(-1)).squeeze()[..., :3]
            return xyz_sampled, viewdirs
        return xyz_sampled

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.ray_aabb[0]) * self.invrayaabbSize - 1

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001, wd=1e-6):
        grad_vars = [{'params': self.app_line, 'lr': lr_init_spatialxyz, 'weight_decay': wd}, {
            'params': self.app_plane, 'lr': lr_init_spatialxyz, 'weight_decay': wd}]
        return grad_vars

    def init_svd_volume(self, res, device):
        self.app_plane, self.app_line = self.init_one_svd(
            self.app_n_comp, self.gridSize, 0.1, device)
        self.basis_mat = torch.nn.Linear(
            sum(self.app_n_comp), self.app_dim, bias=False).to(device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                # scale * torch.randn((self.j_channel, 1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
                # scale * torch.zeros((self.j_channel, 1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
                scale * torch.randn((self.se3_dim, 1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            # scale * torch.ones((self.j_channel, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            line_coef.append(
                # torch.nn.Parameter(scale * torch.randn((self.j_channel, 1, n_component[i], gridSize[vec_id], 1))))
                # torch.nn.Parameter(scale * torch.zeros((self.j_channel, 1, n_component[i], gridSize[vec_id], 1))))
                torch.nn.Parameter(scale * torch.randn((self.se3_dim, 1, n_component[i], gridSize[vec_id], 1))))
            # torch.nn.Parameter(scale * torch.ones((self.j_channel, n_component[i], gridSize[vec_id], 1))))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

    def forward(self, xyz_sampled, viewdirs, transforms, ray_valid, i_frame=None):
        shape = xyz_sampled.shape
        xyz_slice = xyz_sampled.view(-1, 3).shape[0]
        xyz_sampled, viewdirs = self.warp_points(xyz_sampled, viewdirs)

        # # # 1115
        # xyz_normalized = self.normalize_coord(xyz_sampled)
        # bwf = self.sample_BWfield(xyz_normalized.view(-1, 3)).permute(1,0) # [j,sample]
        # w = bwf[...,:3]
        # v = bwf[...,3:]
        # eps = 1e-6
        # theta  = torch.sum(w*w, axis=-1)
        # theta = torch.clamp(theta, min = eps).sqrt()

        # v = v/theta.unsqueeze(-1)
        # w = w/theta.unsqueeze(-1)
        # screw_axis = torch.cat([w, v], axis=-1)

        # transform = exp_se3(screw_axis.permute(1,0), theta).permute(2,0,1)
        # # print(screw_axis.shape, theta.shape, xyz_sampled.shape, transform.shape)
        # xyz_sampled = torch.cat([xyz_sampled.view(-1,3), torch.ones(xyz_sampled.view(-1,3).shape[0], device = xyz_sampled.device).unsqueeze(-1)], dim=--1)#[N,4]
        # xyz_sampled = torch.matmul(transform, xyz_sampled.view(-1, 4).unsqueeze(-1)).squeeze()[...,:3]
        # viewdirs = torch.matmul(transform[...,:3,:3], viewdirs.reshape(-1, 3).unsqueeze(-1)).squeeze()[...,:3]
        # xyz_sampled = xyz_sampled[...,:3]

        # self.weights = transform
        # self.cache_transforms = transforms
        # self.restore_xyz = xyz_sampled

        # xyz_sampled, viewdirs = weighted_transformation(xyz_sampled.view(-1, 3), weights.to(torch.float32), transforms, viewdirs.contiguous().view(-1, 3), if_transform_is_inv=self.args.use_indivInv)
        # print(shape, xyz_sampled.shape, viewdirs.shape)
        xyz_sampled = xyz_sampled.view(shape)
        viewdirs = viewdirs.view(shape)
        return xyz_sampled, viewdirs

    def TV_loss_blendweights(self, reg, linear=False):
        total = 0
        for idx in range(len(self.app_line)):
            for i in range(self.app_line[idx].shape[0]):
                total = total + reg(self.app_line[idx][i].unsqueeze(0)) * 1e-4 + reg(
                    self.app_plane[idx][i].unsqueeze(0)) * 1e-3
        return total

    def TV_loss_bwf(self, reg, linear=False):
        total = 0

        for idx in range(len(self.app_line)):
            # print("tvloss", total)
            total = total + reg(self.app_line[idx][0].unsqueeze(0)) * 1e-3
        return total

    def linear_loss(self):
        total = 0
        for idx in range(len(self.app_line)):
            for i in range(self.app_line[idx].shape[0]):
                total = total + (tv_loss_func_line(self.app_line[idx][i]) * 1e-4 + tv_loss_func_plane(
                    self.app_plane[idx][i]) * 1e-3) * 1e-4
        return total

    def sample_BWfield(self, xyz_sampled):
        # plane + line basis
        # plane : 3, self.j_channel, 1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]
        feats = []
        for i in range(self.se3_dim):
            coordinate_plane = torch.stack(
                (xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).view(3, -1, 1, 2)
            coordinate_line = torch.stack(
                (xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
            coordinate_line = torch.stack(
                (torch.zeros_like(coordinate_line), coordinate_line), dim=-1).view(3, -1, 1, 2)
            sigma_feature = torch.zeros(
                xyz_sampled.shape[0], device=xyz_sampled.device)
            for idx_plane in range(len(self.app_plane)):
                # self.app_plane[idx_plane][i,0].unsqueeze(0) : 1, C, W, H
                # coordinate_plane[[idx_plane]] : 1, H, W, 2
                plane_coef_point = F.grid_sample(self.app_plane[idx_plane][i, 0].unsqueeze(0), coordinate_plane[[idx_plane]],
                                                 align_corners=True).view(-1, *xyz_sampled.shape[:1])
                line_coef_point = F.grid_sample(self.app_line[idx_plane][i, 0].unsqueeze(0), coordinate_line[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])
                sigma_feature = sigma_feature + \
                    torch.sum(plane_coef_point * line_coef_point, dim=0)

            feats.append(sigma_feature)
        return F.relu(torch.stack(feats))

    def compute_weights(self, xyz, transforms,  features=None, locs=None):
        # return compute_weisghts(xyz, self.joints).to(torch.float32)

        xyz = torch.cat([xyz, torch.ones(
            xyz.shape[0], device=xyz.device).unsqueeze(-1)], dim=-1)  # [samples, 4]
        # xyz_new : [samples, 4]
        # invs: [j, 4, 4]

        if self.args.use_indivInv:
            invs = transforms
        else:
            invs = affine_inverse_batch(
                self.skeleton.precomp_forward_global_transforms)
        result = torch.matmul(invs, xyz_new.permute(1, 0).unsqueeze(0).expand(
            invs.shape[0], 4, -1)).squeeze().permute(0, 2, 1)  # [j, samples, 4]
        # result = functorch.vmap(self.matmul_func, in_dims = (None, 0), out_dims = 1)(invs, xyz.unsqueeze(-1)).squeeze(-1)
        self.cache_transformed_pos = result
        result = self.normalize_coord(result[:, :, :3])
        bwf = self.sample_BWfield(result)  # [j,sample]
        # return torch.transpose(bwf, 0 , 1)
        return bwf.permute(1, 0)

    # def sample_BWfield(self, xyz_sampled):
    #     # plane + line basis
    #     # plane : 3, self.j_channel, 1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]
    #     feats = []
    #     print(xyz_sampled.shape, len(self.app_plane), self.app_plane[0].shape)

    #     # for i in range(xyz_sampled.shape[0]):
    #     coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).view(3, self.j_channel, -1, 1, 2)
    #     coordinate_line  = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
    #     coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).view(3, self.j_channel, -1, 1, 2)
    #     sigma_feature = torch.zeros(xyz_sampled.shape[:2], device=xyz_sampled.device)
    #     print(coordinate_plane.shape, coordinate_line.shape)
    #     # exit()
    #     for idx_plane in range(len(self.app_plane)):
    #         # self.app_plane[idx_plane][i,0].unsqueeze(0) : 1, C, W, H
    #         # coordinate_plane[[idx_plane]] : 1, H, W, 2
    #         # print(self.app_plane[idx_plane][:,0].shape, coordinate_plane[idx_plane].unsqueeze(0).expand(self.j_channel,-1,-1,-1).shape)

    #         plane_coef_point = F.grid_sample(self.app_plane[idx_plane], coordinate_plane[idx_plane],
    #                                             align_corners=True).view(self.j_channel, 16, -1)
    #         line_coef_point = F.grid_sample(self.app_line[idx_plane], coordinate_line[idx_plane],
    #                                         align_corners=True).view(self.j_channel, 16, -1)
    #         sigma_feature += torch.sum(plane_coef_point * line_coef_point, dim=1)

    #     # feats.append(sigma_feature)
    #     return F.relu(sigma_feature)


class shCaster(CasterBase):

    def __init__(self):
        super().__init__()
        self.sh_feats = None
        self.use_distweight = False

    def forward(self, xyz_sampled, viewdirs, transforms, ray_valid, i_frame=None):
        # return xyz_sampled, viewdirs
        # time_st = time.perf_counter()
        shape = xyz_sampled.shape
        xyz_slice = xyz_sampled.view(-1, 3).shape[0]
        # tmp = torch.cat([xyz_sampled.view(-1, 3),(xyz_sampled-viewdirs).view(-1, 3)], dim=0)
        tmp = torch.cat(
            [xyz_sampled.view(-1, 3), (xyz_sampled-viewdirs).view(-1, 3)], dim=0)

        # if torch.isnan(self.sh_feats).any() or torch.isinf(self.sh_feats).any():
        #     raise ValueError("shfeats"+"nan or inf in weights")

        if self.use_distweight:
            exit()
        else:
            # a = self.skeleton.get_listed_positions(use_cached=True)[...,:3]
            # b = self.skeleton.get_listed_positions()
            # print(a.shape, b.shape)
            # exit()
            weights = self.get_SH_vals(xyz_sampled.view(-1, 3), self.sh_feats, transforms,
                                       self.skeleton.get_listed_positions(use_cached=True)[..., :3])

        # if torch.isnan(weights).any() or torch.isinf(weights).any():
        #     raise ValueError("justaftergetweights"+"nan or inf in weights")

        self.weights = weights
        weights = torch.cat([weights, weights], dim=0)
        tmp = weighted_transformation(
            tmp, weights, transforms, if_transform_is_inv=self.args.use_indivInv)
        if (tmp.isnan().any() or tmp.isinf().any()):
            ValueError("tmp is nan")
        # debug
        xyz_sampled, viewdirs = tmp[:xyz_slice], tmp[:xyz_slice] - \
            tmp[xyz_slice:]
        xyz_sampled = xyz_sampled.view(shape)
        viewdirs = viewdirs.view(shape)

        return xyz_sampled, viewdirs

    def set_SH_feats(self, feats):
        self.sh_feats = feats

    def set_allgrads(self, value):
        for param in self.sh_feats:
            param.requires_grad = value

    def get_SH_vals(self, xyz, features, invs, locs):
        xyz_new = torch.cat([xyz, torch.ones(
            xyz.shape[0]).unsqueeze(-1).to(xyz.device)], dim=-1)  # [samples, 4]
        # if torch.isnan(xyz).any() or torch.isinf(xyz).any():
        #     raise ValueError("nan or inf")
        # if torch.isnan(locs).any() or torch.isinf(locs).any():
        #     raise ValueError("nan or inf")
        # if torch.isnan(invs).any() or torch.isinf(invs).any():
        #     raise ValueError("nan or inf")
        xyz = []
        for j in range(locs.shape[0]):
            xyz.append(torch.bmm(invs[j].unsqueeze(0).repeat(
                xyz_new.shape[0], 1, 1), xyz_new.unsqueeze(-1)).squeeze())  # [samples, 4]
            # xyz.append(torch.bmm(invs[j].unsqueeze(0).expand(xyz_new.shape[0], -1, -1), xyz_new.unsqueeze(-1)).squeeze())#[samples, 4]
        xyz = torch.stack(xyz, dim=0)[:, :, :3]  # [j,samples,3]
        # (j, sample, 3)
        viewdirs = locs.unsqueeze(-2).repeat(1, xyz.shape[1], 1) - xyz
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
        # (j,sample, 1)#[:, :,None].view(viewdirs.shape[0]*viewdirs.shape[1], 1, -1)#(sample*j, 1, 9)
        sh_mult = eval_sh_bases(2, viewdirs)
        rad_sh = self.sh_feats.view(
            locs.shape[0], 1, sh_mult.shape[-1])  # (j, 1, 9)
        # if torch.isnan(rad_sh).any() or torch.isinf(rad_sh).any():
        #     raise ValueError("nan or inf")
        rads = torch.relu(torch.sum(
            sh_mult * rad_sh.repeat(1, xyz.shape[1], 1), dim=-1) + 0.5)  # (j,sample,  1)
        # rads = torch.relu(torch.sum(sh_mult * rad_sh.expand(-1, xyz.shape[1], -1), dim=-1) + 0.5)#(j,sample,  1)
        # if torch.isnan(rads).any() or torch.isinf(rads).any():
        #     raise ValueError("nan or inf")
        # 42
        eps = 1e-6
        # non_valid = rads < eps
        rads = torch.clamp(rads, min=eps)
        relative_distance = torch.relu(1.0 - lengths/rads)  # (j, sample)
        # if torch.isnan(relative_distance).any() or torch.isinf(relative_distance).any():
        #     raise ValueError("nan or inf")
        # relative_distance[non_valid] = relative_distance[non_valid]*0.0
        # if torch.isnan(relative_distance).any() or torch.isinf(relative_distance).any():
        #     raise ValueError("nan or inf")
        relative_distance = relative_distance.permute(
            1, 0)  # (j, sample) -> (sample, j)
        # if torch.isnan(relative_distance).any() or torch.isinf(relative_distance).any():
        #     raise ValueError("nan or inf")

        return relative_distance  # weights, (sample, j)(j, sample)?????????


def tv_loss_func_plane(image, weight=1.0):
    tv_h = (image[:, 1:, :, :] - image[:, :-1, :, :])
    valid = torch.abs(tv_h) > 1e-8
    tv_h = torch.where(tv_h > 0, torch.ones_like(tv_h), -torch.ones_like(tv_h))
    # tv_h[valid] /= torch.abs(tv_h)[valid]
    tv_h = torch.abs(tv_h[:, 1:, :, :] - tv_h[:, :-1, :, :]).sum()

    tv_w = (image[:, :, 1:, :] - image[:, :, :-1, :])
    valid = torch.abs(tv_w) > 1e-8
    # tv_w[valid] /= torch.abs(tv_w)[valid]
    tv_w = torch.where(tv_w > 0, torch.ones_like(tv_w), -torch.ones_like(tv_w))
    tv_w = torch.abs(tv_w[:, :, 1:, :] - tv_w[:, :, :-1, :]).sum()

    return weight * (tv_h + tv_w) * weight


def tv_loss_func_line(image, weight=1.0):
    tv_h = (image[:, 1:, :] - image[:, :-1, :])
    valid = torch.abs(tv_h) > 1e-8
    # tv_h[valid] /= torch.abs(tv_h)[valid]
    tv_h = torch.where(tv_h > 0, torch.ones_like(tv_h), -torch.ones_like(tv_h))

    tv_h = torch.abs(tv_h[:, 1:, :] - tv_h[:, :-1, :]).sum()
    return weight * (tv_h) * weight


# class MapCaster_grid(CasterBase):
#     def __init__(self, dim, gridSize, device, num_frames, args = None):
#         super().__init__()
#         self.app_n_comp = [16,16,16]
#         self.j_channel = dim
#         self.gridSize = gridSize
#         self.app_dim = dim
#         self.matMode = [[0,1], [0,2], [1,2]]
#         self.vecMode =  [2, 1, 0]
#         self.aabb = None
#         self.invaabbSize = None
#         self.ray_aabb = None
#         self.joints = None
#         self.num_frames = num_frames
#         self.grids_plane = []
#         self.grids_line = []
#         self.feats_dim = 6
#         self.args = args
#         for i in range(num_frames):
#             plane, line = self.init_svd_volume(gridSize, device)
#             self.grids_plane.append(plane)
#             self.grids_line.append(line)


#     def normalize_coord(self, xyz_sampled):
#         return (xyz_sampled-self.ray_aabb[0]) * self.invrayaabbSize - 1
#     def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
#         grad_vars = []
#         for i in range(self.num_frames):
#             grad_vars += [{'params': self.grids_plane[i], 'lr': lr_init_spatialxyz}, {'params': self.grids_line[i], 'lr': lr_init_spatialxyz}]
#         return grad_vars

#     def init_svd_volume(self, res, device):
#         self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.032, device)
#         self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)
#         return self.app_plane, self.app_line

#     def init_one_svd(self, n_component, gridSize, scale, device):
#         plane_coef, line_coef = [], []
#         for i in range(len(self.vecMode)):
#             vec_id = self.vecMode[i]
#             mat_id_0, mat_id_1 = self.matMode[i]
#             plane_coef.append(torch.nn.Parameter(
#                 # scale * torch.randn((self.j_channel, 1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
#                 # scale * torch.zeros((self.j_channel, 1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
#                 scale * torch.ones((self.feats_dim, 1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
#             line_coef.append(
#                 # torch.nn.Parameter(scale * torch.randn((self.j_channel, 1, n_component[i], gridSize[vec_id], 1))))
#                 # torch.nn.Parameter(scale * torch.zeros((self.j_channel, 1, n_component[i], gridSize[vec_id], 1))))
#                 torch.nn.Parameter(scale * torch.ones((self.feats_dim, 1, n_component[i], gridSize[vec_id], 1))))

#         return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

#     def forward(self, xyz_sampled, viewdirs, transforms, ray_valid, i_frame = None):
#         shape = xyz_sampled.shape
#         xyz_slice = xyz_sampled.view(-1, 3).shape[0]
#         # # 1115
#         transforms = self.compute_transforms(xyz_sampled.view(-1, 3), transforms, i_frame)
#         # self.weights = weights
#         transforms = torch.cat([transforms, transforms], dim=0)
#         tmp = torch.cat([xyz_sampled.view(-1, 3),(xyz_sampled-viewdirs).view(-1, 3)], dim=0)
#         tmp = torch.cat([tmp, torch.ones(tmp.shape[0]).unsqueeze(-1).to(tmp.device)], dim=--1)#[N,4]
#         tmp = torch.matmul(transforms, tmp.unsqueeze(-1)).squeeze()[...,:3]
#         if(tmp.isnan().any() or tmp.isinf().any()):
#             ValueError("tmp is nan")
#         #debug
#         xyz_sampled, viewdirs = tmp[:xyz_slice], tmp[:xyz_slice] - tmp[xyz_slice:]
#         xyz_sampled = xyz_sampled.view(shape)
#         viewdirs = viewdirs.view(shape)

#         return xyz_sampled, viewdirs

#     def compute_transforms(self, xyz, transforms,  i_frame, features=None, locs=None):
#         euler_scaling = 1
#         trans_scaling = 1
#         map_features = self.sample_BWfield(self.normalize_coord(xyz), i_frame) #sample, 6
#         mats = euler_to_matrix_batch(map_features[:,:3]*euler_scaling, top_batching=True) # sample, 3, 3
#         # mats = torch.eye(4).unsqueeze(0).expand(xyz.shape[0], -1, -1).to(xyz.device)
#         mats[:,:3,3] = map_features[:,3:]*trans_scaling
#         return mats

#     def TV_loss_blendweights(self, reg, linear = False):
#         total = 0
#         for idx in range(len(self.app_line)):
#             for i in range(self.app_line[idx].shape[0]):
#                 total = total + reg(self.app_line[idx][i].unsqueeze(0)) * 1e-4 + reg(self.app_plane[idx][i].unsqueeze(0)) * 1e-3
#                 # if linear:
#                 #     total = total + (tv_loss_func_line(self.app_line[idx][i]) * 1e-4 + tv_loss_func_plane(self.app_plane[idx][i]) * 1e-3) * 1e-4
#         return total
#     def linear_loss(self):
#         total = 0
#         for idx in range(len(self.app_line)):
#             for i in range(self.app_line[idx].shape[0]):
#                 total = total + (tv_loss_func_line(self.app_line[idx][i]) * 1e-4 + tv_loss_func_plane(self.app_plane[idx][i]) * 1e-3) * 1e-4
#         return total


#     def sample_BWfield(self, xyz_sampled, i_frame):
#         # plane + line basis
#         # plane : 3, self.j_channel, 1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]
#         feats = []
#         # print(xyz_sampled.shape, len(self.app_plane), self.app_plane[0].shape)
#         app_plane = self.grids_plane[i_frame]
#         app_line = self.grids_line[i_frame]
#         for i in range(self.feats_dim):
#             coordinate_plane = torch.stack((xyz_sampled[i, ..., self.matMode[0]], xyz_sampled[i,..., self.matMode[1]], xyz_sampled[i,..., self.matMode[2]])).detach().view(3, -1, 1, 2)
#             coordinate_line = torch.stack((xyz_sampled[i,..., self.vecMode[0]], xyz_sampled[i,..., self.vecMode[1]], xyz_sampled[i,..., self.vecMode[2]]))
#             coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
#             sigma_feature = torch.zeros(xyz_sampled.shape[1], device=xyz_sampled.device)
#             for idx_plane in range(len(app_plane)):
#                 # self.app_plane[idx_plane][i,0].unsqueeze(0) : 1, C, W, H
#                 # coordinate_plane[[idx_plane]] : 1, H, W, 2
#                 plane_coef_point = F.grid_sample(app_plane[idx_plane][i,0].unsqueeze(0), coordinate_plane[[idx_plane]],
#                                                     align_corners=True).view(-1, *xyz_sampled.shape[1:2])
#                 line_coef_point = F.grid_sample(app_line[idx_plane][i, 0].unsqueeze(0), coordinate_line[[idx_plane]],
#                                                 align_corners=True).view(-1, *xyz_sampled.shape[1:2])
#                 sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

#             feats.append(sigma_feature)

#         return torch.stack(feats)


# class MLPCaster_tcnn(MLPCaster):
#     def __init__(self, dim, device):
#         super().__init__(dim, device)
#         encoding="hashgrid"
#         encoding_dir="sphere_harmonics"
#         num_layers=2
#         hidden_dim=64
#         geo_feat_dim=15
#         num_layers_color=3
#         hidden_dim_color=64
#         bound=1.0
#         # bound /= 4.0
#         # hidden_dim /= 4
#         hidden_dim = int(hidden_dim)
#         # sigma network
#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
#         self.geo_feat_dim = geo_feat_dim = 0
#         self.bound = bound


#         n_levels = 16//4
#         bsae_res = 16
#         leveldim = 2

#         per_level_scale = np.exp2(np.log2(2048 * bound / bsae_res) / (n_levels - 1))

#         self.encoder = tcnn.Encoding(
#             n_input_dims=3,
#             encoding_config={
#                 "otype": "HashGrid",
#                 "n_levels": n_levels,
#                 "n_features_per_level": leveldim,
#                 "log2_hashmap_size": 19,
#                 "base_resolution": bsae_res,
#                 "per_level_scale": per_level_scale,
#             },
#         )

#         self.weight_nets = []
#         for i in range(dim):

#             self.weight_nets.append(
#                 tcnn.Network(
#                 n_input_dims=leveldim*n_levels,
#                 n_output_dims=1 + self.geo_feat_dim,
#                 network_config={
#                     "otype": "FullyFusedMLP",
#                     "activation": "ReLU",
#                     "output_activation": "None",
#                     "n_neurons": hidden_dim,
#                     "n_hidden_layers": num_layers - 1,
#                 },
#                 ).to(device)
#             )

#     @torch.cuda.amp.autocast(enabled=True)
#     def density(self, x):
#         # x: [J, N, 3], in [-bound, bound]
#         res = []
#         x = (x + self.bound) / (2 * self.bound) # to [0, 1]
#         for i in range(len(self.weight_nets)):
#             # sigma

#             tmp = self.encoder(x[i])
#             h = self.weight_nets[i](tmp)
#             sigma = F.relu(h[..., 0])
#             res.append(sigma)
#         return torch.stack(res, dim=0)


# class MLPCaster_net(MLPCaster):
#     def __init__(self, dim, device):
#         super().__init__(dim, device)
#         encoding="hashgrid"
#         encoding_dir="sphere_harmonics"
#         num_layers=2
#         hidden_dim=64
#         geo_feat_dim=15
#         num_layers_color=3
#         hidden_dim_color=64
#         bound=1.0
#         # bound /= 4.0
#         # hidden_dim /= 4
#         hidden_dim = int(hidden_dim)

#         # sigma network
#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
#         self.geo_feat_dim = geo_feat_dim = 0
#         self.bound = bound
#         # self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048//4 * bound, base_resolution=4, num_levels=4)
#         # self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)

#         # self.encoder = self.encoder.to(device)
#         self.encoder, self.in_dim = get_encoder("frequency", desired_resolution=2048 * bound, multires = 6)
#         self.in_dim_three = 3

#         self.weight_nets = []
#         for i in range(dim):
#             sigma_net = []
#             for l in range(num_layers):
#                 if l == 0:
#                     # in_dim = self.in_dim_three
#                     in_dim = self.in_dim
#                 else:
#                     in_dim = hidden_dim

#                 if l == num_layers - 1:
#                     out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
#                 else:
#                     out_dim = hidden_dim

#                 sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))
#             self.weight_nets.append(
#                 nn.ModuleList(sigma_net).to(device)
#             )

#     @torch.cuda.amp.autocast(enabled=True)
#     def density(self, x):
#         # x: [J, N, 3], in [-bound, bound]
#         res = []
#         for i in range(len(self.weight_nets)):
#             h = self.encoder(x[i], bound=self.bound)
#             # h = x[i]

#             for l in range(self.num_layers):
#                 h = self.weight_nets[i][l](h)

#                 if l != self.num_layers - 1:
#                     h = F.relu(h, inplace=True)

#             sigma = F.relu(h[..., 0])
#             res.append(sigma)
#         return torch.stack(res, dim=0)
