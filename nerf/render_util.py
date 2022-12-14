# from cv2 import transform
from multiprocessing.sharedctypes import Value
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# import raymarching
import math
import json
import copy
from .provider import *
from functools import partial
import functorch

import time
from .rigidbody import * 

# from pytorch3d import transforms as p3d_transforms

my_torch_device = "cuda"

def clip_weight(val, thresh = 1e-4, min=0, max=1):
    return 1-torch.clamp((1-val*(1.0/thresh)), min=min, max=max)

# #@torch.jit.script
# def affine_inverse_batch(bmatrix, device="cuda"):
#     # inv = torch.eye(4, device=bmatrix.device).unsqueeze(0).repeat(bmatrix.shape[0], 1, 1)
#     # inv[:,:3,:3] = torch.transpose(bmatrix[:,:3,:3], 1, 2)
#     # inv[:,:3, 3] = torch.bmm(torch.transpose(bmatrix[:,:3,:3], 1, 2), -bmatrix[:,:3,3].unsqueeze(-1)).squeeze()
#     eye = torch.eye(4, device=bmatrix.device).unsqueeze(0).repeat(bmatrix.shape[0], 1, 1)

#     rot = bmatrix[:,:3,:3].permute(0,2,1)
#     trans = torch.matmul(rot, -bmatrix[:,:3,3].unsqueeze(-1))

#     inv = torch.cat([torch.cat([rot, trans], dim=-1), eye[:,3:,:]], dim=1)
#     return inv

@torch.jit.script
def affine_inverse_batch(bmatrix):
    # inv = torch.eye(4, device=bmatrix.device).unsqueeze(0).repeat(bmatrix.shape[0], 1, 1)
    # inv[:,:3,:3] = torch.transpose(bmatrix[:,:3,:3], 1, 2)
    # inv[:,:3, 3] = torch.bmm(torch.transpose(bmatrix[:,:3,:3], 1, 2), -bmatrix[:,:3,3].unsqueeze(-1)).squeeze()
    # eye = torch.eye(4, device = "cuda:0").unsqueeze(0).repeat(bmatrix.shape[0], 1, 1)

    rot = bmatrix[:,:3,:3].permute(0,2,1)
    trans = torch.matmul(rot, -bmatrix[:,:3,3].unsqueeze(-1))

    inv = torch.cat([torch.cat([rot, trans], dim=-1), torch.tensor([0,0,0,1], device = "cuda:0").view(1,1,-1).expand(rot.shape[0], -1, -1)], dim=1)
    return inv

#@torch.jit.script
def affine_inverse(matrix, device="cuda"):
    # inv = torch.eye(4, device=matrix.device)
    # # inv[:3,:3] = torch.transpose(matrix[:3,:3], 0, 1)
    # inv[:3,:3] = matrix[:3,:3].permute(1,0)
    # inv[:3, 3] = torch.mv(matrix[:3,:3].permute(1,0), -matrix[:3,3])
    rot = matrix[:3,:3].permute(1,0)
    trans = torch.mv(rot, -matrix[:3,3])
    inv = torch.cat([torch.cat([rot, trans], dim=-1), torch.tensor([0,0,0,1], device = matrix.device)], dim=1)
    return inv
#https://qiita.com/harmegiddo/items/96004f7c8eafbb8a45d0#%E3%82%AA%E3%82%A4%E3%83%A9%E3%83%BC%E8%A7%92

#@torch.jit.script
def quaternion_to_matrix_batch(q):
    #[J, 4]
    w, x, y, z= torch.transpose(q, 0, 1)
    #[J, J, J, J]
    return torch.cat([
    torch.stack([
        torch.stack([-(1- 2*w*w-2*x*x),   2*x*y-2*w*z,   2*x*z+2*w*y],dim=0),
        torch.stack([   2*x*y+2*w*z, -(1-2*y*y-2*w*w),   2*y*z-2*w*x],dim=0),
        torch.stack([   2*x*z-2*w*y,   2*y*z+2*w*x, -(1-2*z*z-2*w*w)],dim=0),
        torch.zeros(3, w.shape[0], device=q.device)
        # torch.tensor([0.0,0.0,0.0], device=q.device).unsqueeze(-1).repeat(1, w.shape[0])
    ], dim=0), 
    torch.cat(
        [torch.zeros(3, w.shape[0], device=q.device),
        torch.ones(1, w.shape[0], device=q.device)]
    ,dim=0).unsqueeze(1)
    ], dim=1)
    #[4,4,J]

#@torch.jit.script
def quaternion_to_matrix(q):
    w, x, y, z= q
    return torch.cat([
    torch.stack([
        torch.stack([-(1- 2*w*w-2*x*x),   2*x*y-2*w*z,   2*x*z+2*w*y],dim=0),
        torch.stack([   2*x*y+2*w*z, -(1-2*y*y-2*w*w),   2*y*z-2*w*x],dim=0),
        torch.stack([   2*x*z-2*w*y,   2*y*z+2*w*x, -(1-2*z*z-2*w*w)],dim=0),
        torch.tensor([0.0,0.0,0.0], device=q.device)
    ], dim=0), torch.tensor(
        [0.0,0.0,0.0,1.0], device=q.device
    ).unsqueeze(1)], dim=1)

#@torch.jit.script
def euler_to_quaternion(r):
    sx, sy, sz = torch.sin(torch.deg2rad(r/2.0))
    cx, cy, cz = torch.cos(torch.deg2rad(r/2.0))
    return torch.stack([
        cx*cy*cz - sx*sy*sz,
        cy*sx*cz + sy*cx*sz,
        sy*cx*cz - cy*sx*sz,
        cy*cx*sz + sy*sx*cz
    ],dim=0)
    
#@torch.jit.script
def compute_weights(xyzs, joints):
    #xyzs, reshaped(-1,3)
    weights_list = []
    for j in joints:
        # tmp_xyzs = torch.cat([xyzs, torch.ones(xyzs.shape[0], 1, device=xyzs.device)], dim=1)
        # xyzs = torch.matmul(j.get_inv().unsqueeze(0), tmp_xyzs.permute(0,1).unsqueeze(0).unsqueeze(-1)).squeeze(-1).squeeze(0)[...,:3]
        weights = (xyzs - j.global_pos(ngp=True)).to(torch.float32).to(xyzs.device)
        weights = (weights*weights).sum(dim=1, keepdim=True).squeeze()
        weights = torch.exp(-weights*10)
        # if j.name in ['root', 'Sharkthroat',  'SharkHead', 'SharkHead_tail', 'SharkJaw', 'SharkJaw_tail', 'Sharktail0', 'Sharktail1', 'Sharktail2', 'Sharktail3', 'Sharktail4', 'Sharktail4_tail', 'Righhfena1', 'Rightfena2', 'Rightfena3', 'Rightfena3_tail', 'Leftfena1', 'Leftfena2', 'Leftfena3', 'Leftfena3_tail']:
        #     weights = weights*0.0
    
        # if j.name in ['root', 'Sharkthroat',  'SharkHead', 'SharkHead_tail', 'SharkJaw']:
        #     weights = weights*0.0
            # 'Righhfena1', 'Rightfena2', 'Rightfena3', 'Rightfena3_tail', 'Leftfena1', 'Leftfena2', 'Leftfena3', 'Leftfena3_tail'
        if j.tail:
            weights = weights*0.0
        weights_list.append(weights)
    
    weights_list = torch.stack(weights_list, dim=0)
    # exit("render_util, compute_weights")
    return torch.transpose(weights_list, 0, 1)


@torch.cuda.amp.autocast(enabled=True)
def weighted_transformation(xyzs, weights, transforms, viewdir = None, if_transform_is_inv:bool = True):
    #xyzs -> [N, 3]
    #weights -> [N, J]
    #transforms -> [J, 4, 4]
    #https://qiita.com/tand826/items/9e1b6a4de785097fe6a5
    if viewdir is not None:
        return weighted_transformation_vd(xyzs, weights, transforms, viewdir, if_transform_is_inv)
    else:
        return weighted_transformation_novd(xyzs, weights, transforms, if_transform_is_inv)

    print_time = False
    
    # if print_time  : start = time.time()
    weights_sum = weights.sum(dim=1)
    n_sample = xyzs.shape[0]
    # n_joint = transforms.shape[0]
    eps = 1e-7

    weights_sum = torch.clamp(weights_sum, min=eps)
    weights = weights/weights_sum.unsqueeze(1)
    # if print_time  : print("time1", time.time() - start)

    # if print_time  : start = time.time()
    xyzs = torch.cat([xyzs, torch.ones(n_sample, device = xyzs.device).unsqueeze(-1)], dim=-1)#[N,4]
    
    # transforms : [J, 4, 4]
    # weights : [N, J]
    # xyzs : [N, 4]

    
    if if_transform_is_inv:
        tmp = torch.matmul(torch.matmul(weights, transforms.view(transforms.shape[0], -1)).view(n_sample, 4, 4), xyzs.unsqueeze(-1))
    else:
        # if print_time  : start = time.time()
        tmp = torch.matmul(weights, transforms.view(transforms.shape[0], -1)).view(n_sample, 4, 4)
        # if print_time  : print("time2", time.time() - start)
        # if print_time  : start = time.time()
        tmp = affine_inverse_batch(tmp)
        # if print_time  : print("time3", time.time() - start)
        # if print_time  : start = time.time()
        if viewdir is not None:
            tmp_viewdir = torch.matmul(tmp[:,:3,:3], viewdir.unsqueeze(-1))
        tmp = torch.matmul(tmp, xyzs.unsqueeze(-1))
        # if print_time  : print("time4", time.time() - start)
    result = tmp.squeeze()[...,:3]

    if viewdir is not None:
        return result, tmp_viewdir

    return result
# @torch.jit.script
def weighted_transformation_novd(xyzs, weights, transforms, if_transform_is_inv:bool = True):
    #xyzs -> [N, 3]
    #weights -> [N, J]
    #transforms -> [J, 4, 4]
    #https://qiita.com/tand826/items/9e1b6a4de785097fe6a5
    print_time = False
    
    # if print_time  : start = time.time()
    weights_sum = weights.sum(dim=1)
    n_sample = xyzs.shape[0]
    # n_joint = transforms.shape[0]
    eps = 1e-7

    weights_sum = torch.clamp(weights_sum, min=eps)
    weights = weights/weights_sum.unsqueeze(1)
    # if print_time  : print("time1", time.time() - start)

    # if print_time  : start = time.time()
    xyzs = torch.cat([xyzs, torch.ones(n_sample, device = xyzs.device).unsqueeze(-1)], dim=-1)#[N,4]
    
    # transforms : [J, 4, 4]
    # weights : [N, J]
    # xyzs : [N, 4]

    
    if if_transform_is_inv:
        tmp = torch.matmul(torch.matmul(weights, transforms.view(transforms.shape[0], -1)).view(n_sample, 4, 4), xyzs.unsqueeze(-1))
    else:
        # if print_time  : start = time.time()
        tmp = torch.matmul(weights, transforms.view(transforms.shape[0], -1)).view(n_sample, 4, 4)
        # if print_time  : print("time2", time.time() - start)
        # if print_time  : start = time.time()
        tmp = affine_inverse_batch(tmp)
        # if print_time  : print("time3", time.time() - start)
        # if print_time  : start = time.time()
        # if viewdir is not None:
        #     tmp_viewdir = torch.matmul(tmp[:,:3,:3], viewdir.unsqueeze(-1))
        tmp = torch.matmul(tmp, xyzs.unsqueeze(-1))
        # if print_time  : print("time4", time.time() - start)
    result = tmp.squeeze()[...,:3]

    # if viewdir is not None:
    #     return result, tmp_viewdir

    return result
# @torch.jit.script
def weighted_transformation_vd(xyzs, weights, transforms, viewdir = None, if_transform_is_inv:bool = True):
    #xyzs -> [N, 3]
    #weights -> [N, J]
    #transforms -> [J, 4, 4]
    #https://qiita.com/tand826/items/9e1b6a4de785097fe6a5
    print_time = False
    
    # if print_time  : start = time.time()
    weights_sum = weights.sum(dim=1)
    n_sample = xyzs.shape[0]
    # n_joint = transforms.shape[0]
    eps = 1e-7

    weights_sum = torch.clamp(weights_sum, min=eps)
    weights = weights/weights_sum.unsqueeze(1)
    # if print_time  : print("time1", time.time() - start)

    # if print_time  : start = time.time()
    xyzs = torch.cat([xyzs, torch.ones(n_sample, device = xyzs.device).unsqueeze(-1)], dim=-1)#[N,4]
    
    # transforms : [J, 4, 4]
    # weights : [N, J]
    # xyzs : [N, 4]

    
    if if_transform_is_inv:
        tmp = torch.matmul(torch.matmul(weights, transforms.view(transforms.shape[0], -1)).view(n_sample, 4, 4), xyzs.unsqueeze(-1))
        tmp_viewdir = None
    else:
        # if print_time  : start = time.time()
        tmp = torch.matmul(weights, transforms.view(transforms.shape[0], -1)).view(n_sample, 4, 4)
        # if print_time  : print("time2", time.time() - start)
        # if print_time  : start = time.time()
        tmp = affine_inverse_batch(tmp)
        # if print_time  : print("time3", time.time() - start)
        # if print_time  : start = time.time()
        # if viewdir is not None:
        tmp_viewdir = torch.matmul(tmp[:,:3,:3], viewdir.unsqueeze(-1))
        tmp = torch.matmul(tmp, xyzs.unsqueeze(-1))
        # if print_time  : print("time4", time.time() - start)
    result = tmp.squeeze()[...,:3]


    return result, tmp_viewdir

    # return result


def listify_position(data, data_pos):
    res = []

    def listify_position_rec(data, data_pos):
        name = data["name"]
        # res = [(data["name"], data["head_local"], data["head"],data["tail_local"], data["tail"]) ]
        res = [data_pos[name]["head"]]
        for c in data["children"]:
            res.extend(listify_position_rec(c, data_pos))
        if len(data["children"]) == 0:
            res.append(data_pos[name]["tail"])
        return res
    for c in data["children"]:
        res.extend(listify_position_rec(c, data_pos))
    return res




def apply_animation(animation, skeleton):
    name = skeleton.name
    if "_tail" in name:
        return
    transform = animation[name]["matrix_local"]
    skeleton.apply_transform(transform=transform)
    for c in skeleton.get_children():
        apply_animation(animation, c)

def listify_skeleton(skeleton):
    res = [skeleton]
    for c in skeleton.get_children():
        res.extend(listify_skeleton(c))
    return res

def make_joints_from_blender(file_path, device="cuda"):
    def make_joint_rec(data, data_pos, parent_joint, obj_bind):
        name = data["name"]
        # obj_bind = torch.tensor(obj_bind)
        bind_pose = torch.mm(obj_bind, torch.tensor(data["bind_transform"]))
        joint = Joint(
            bind_pose=bind_pose,
            parent_bind=parent_joint.global_transform(),
            name = data["name"],
            device=device
        )
        if len(data["children"]) == 0:
            translate = torch.tensor(np.array(data_pos[name]["tail"]) - np.array(data_pos[name]["head"])).float()
            translate = torch.mv(obj_bind[:3,:3], translate)
            bind_pose[:3,3] = bind_pose[:3,3] + translate
            joint.add_child(
                Joint(
                    bind_pose=bind_pose,
                    parent_bind=joint.global_transform(),
                    name = data["name"]+"_tail",
                    tail=True,
                    device=device
                )
            )
        else:
            for c in data["children"]:
                joint.add_child(make_joint_rec(c, data_pos, joint, obj_bind))
        return joint

    f = open(file_path, 'r')
    data = json.load(f)
    root = data["initial_state"][0]
    data_pos = data["initial_bone"]
    bind = torch.tensor(root["bind_transform"])
    # bind = bind @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).astype(np.float32)
    # bind = torch.mm(nerf_to_ngp_caster(device="cpu"), bind)
    # bind = bind.double()
    bind = bind.to(torch.float32)
    skeleton = Joint(
            bind_pose = bind,
            parent_bind=torch.eye(4, dtype=torch.float32, device=torch.device(device)),
            name = root["name"],
            device=device
        )
    for c in root["children"]:
        skeleton.add_child(make_joint_rec(c, data_pos, skeleton, bind))
    return skeleton




def euler_to_matrix(angle = None, translate = None):
    if True:
        eps = 1e-6
        w = angle.unsqueeze(-1)
        theta  = torch.sum(w*w, axis=-1)
        theta = torch.clamp(theta, min = eps).sqrt()
        
        w = w/theta.unsqueeze(-1)
        transform = exp_so3(w, theta).permute(2,0,1)
        return torch.cat([
        torch.stack([
            transform,
            torch.tensor([0.0,0.0,0.0], device=angle.device)
        ], dim=0), torch.tensor(
            [0.0,0.0,0.0,1.0], device=angle.device
        ).unsqueeze(1)], dim=1)
        return transform
    mat = torch.eye(4, device=torch.device(angle.device))
    
    if translate is not None:
        mat[:3, 3] = translate
    if angle is not None:
        # mat = p3d_transforms.euler_angles_to_matrix(torch.deg2rad(angle), convention="XYZ")
        # mat = torch.cat([mat, torch.tensor([0.0,0.0,0.0], device=angle.device).unsqueeze(0)], dim=0)
        # mat = torch.cat([mat, torch.tensor([0.0,0.0,0.0,1.0], device=angle.device).unsqueeze(1)], dim=1)
        # return mat
        sx, sy, sz = torch.sin(torch.deg2rad(angle))
        cx, cy, cz = torch.cos(torch.deg2rad(angle))

        return torch.cat([
        torch.stack([
            torch.stack([cy*cz,  -cy*sz,   sy],dim=0),
            torch.stack([cx*sz+cz*sx*sy, cx*cz-sx*sy*sz,   -cy*sx],dim=0),
            torch.stack([sx*sz-cx*cz*sy,   cz*sx+cx*sy*sz, cx*cy],dim=0),
            torch.tensor([0.0,0.0,0.0], device=angle.device)
        ], dim=0), torch.tensor(
            [0.0,0.0,0.0,1.0], device=angle.device
        ).unsqueeze(1)], dim=1)


def euler_to_matrix_batch(angle = None, top_batching=False):
    mat = torch.eye(4, device=torch.device(my_torch_device))
    
    # if translate is not None:
    #     print(mat[:3, 3].shape, translate.shape)
    #     mat[:3, 3] = translate
    if True:
        if not top_batching:
            angle = angle.permute(1,0)
        eps = 1e-6
        w = angle.contiguous().view(-1,3)
        theta  = torch.sum(w*w, axis=-1)
        theta = torch.clamp(theta, min = eps).sqrt()


        w = w/theta.unsqueeze(-1)
        # print(theta)
        # print(w)
        # exit()
        transform = exp_so3(w.permute(1,0), theta)

        result =   torch.cat([
        torch.cat([
            # torch.stack([cy*cz,  -cy*sz,   sy],dim=0),
            # torch.stack([cx*sz+cz*sx*sy, cx*cz-sx*sy*sz,   -cy*sx],dim=0),
            # torch.stack([sx*sz-cx*cz*sy,   cz*sx+cx*sy*sz, cx*cy],dim=0),
            transform,
            torch.tensor([0.0,0.0,0.0], device=angle.device).unsqueeze(1).unsqueeze(0).repeat(1, 1, angle.shape[0])
            # torch.tensor([0.0,0.0,0.0], device=angle.device).unsqueeze(1).expand(-1, angle.shape[1])
        ], dim=0), torch.tensor(
            [0.0,0.0,0.0,1.0], device=angle.device
        ).unsqueeze(1).unsqueeze(1).repeat(1,1,angle.shape[0])], dim=1)

        if top_batching:
            result = result.permute(2,0,1)
        return result

    if top_batching:
        angle = angle.permute(1,0)
    
    if angle is not None:
        sx, sy, sz = torch.sin(torch.deg2rad(angle))
        cx, cy, cz = torch.cos(torch.deg2rad(angle))
        result = torch.cat([
        torch.stack([
            torch.stack([cy*cz,  -cy*sz,   sy],dim=0),
            torch.stack([cx*sz+cz*sx*sy, cx*cz-sx*sy*sz,   -cy*sx],dim=0),
            torch.stack([sx*sz-cx*cz*sy,   cz*sx+cx*sy*sz, cx*cy],dim=0),
            torch.tensor([0.0,0.0,0.0], device=angle.device).unsqueeze(1).repeat(1, angle.shape[1])
            # torch.tensor([0.0,0.0,0.0], device=angle.device).unsqueeze(1).expand(-1, angle.shape[1])
        ], dim=0), torch.tensor(
            [0.0,0.0,0.0,1.0], device=angle.device
        ).unsqueeze(1).unsqueeze(1).repeat(1,1,angle.shape[1])], dim=1)
        #??????????????????????????????????????????

        if top_batching:
            result = result.permute(2,0,1)
        return result


def make_transform(angle=None, translate = None):
    # return euler_to_matrix_old(angle, translate)
    return euler_to_matrix(angle, translate)
#

def each_dot(a,b):
    return torch.sum(a*b, dim=-1)


class Joint():
    def __init__(self, bind_pose = None, parent_bind = None, name = "NotSet", ngp_space=True, device=torch.device("cuda"), tail=False):
        self.ngp_space=ngp_space
        self.device = device
        self.torchtype=torch.float32
        self.tail = tail


        # self.bind_pose = torch.tensor(bind_pose,  dtype=torch.float32, device=torch.device(my_torch_device))
        bind_pose = self.mytensor(bind_pose).to(self.device)

        self.bind_inv = torch.linalg.pinv(bind_pose.float())

        self.R = self.myeye(4)
        self.T = self.myeye(4)
        self.S = self.myeye(4)

        # self.parent_transform = torch.tensor(parent_bind,  dtype=torch.float32, device=torch.device(my_torch_device))
        self.parent_transform = self.mytensor(parent_bind)
        self.initial_transform = torch.mm(torch.linalg.pinv(self.parent_transform.float()), bind_pose)

        # if torch.isnan(self.initial_transform).any():
        #     raise ValueError("test")

        self.name = name
        self.children = []
        self.markers = []
        self.local_transform_cached = self.local_transform()
        self.root = False

        self.markers = [
            torch.mv(self.global_transform(), self.mytensor([0.05,    0,    0, 1])),
            torch.mv(self.global_transform(), self.mytensor([   0, 0.05,    0, 1])),
            torch.mv(self.global_transform(), self.mytensor([   0,    0, 0.05, 1]))
        ]
        self.first_pos = self.global_pos();

    def set_as_root(self):
        self.root = True
        self.joints = listify_skeleton(self)


    def compute_depth(self, parent_depth):
        self.depth = parent_depth+1
        res_depth = self.depth
        for c in self.children:
            res_depth = max(res_depth, c.compute_depth(self.depth))
        return res_depth

    def compute_transform_ids(self, ids):
        ids.append(self.id)
        self.ids = copy.deepcopy(ids)
        res = [self.ids]
        for c in self.children:
            res.extend(c.compute_transform_ids(copy.deepcopy(ids)))
        return res
        

    def euler_to_matrix(self, euler):
        #[N,j,3]->[N,j,4,4]
        return make_transform(euler)

    def myeye(self, n, dtype=torch.float32, device = None):
        if device is None:
            return  torch.eye(n,  dtype=self.torchtype, device=self.device)
        else:
            return  torch.eye(n,  dtype=self.torchtype, device=device)

    def mytensor(self, input, dtype=torch.float32, device = None):
        if device is None:
            device = self.device

        if not torch.is_tensor(input):
            input = torch.tensor(input, device=device)
        return input

    def matrix_to_euler_pos(self, matrix, debug_p3d = False):
        if debug_p3d:
            raise ValueError("debug_p3d")
        
            # return torch.rad2deg(p3d_transforms.matrix_to_euler_angles(matrix[...,:3,:3], convention="XYZ"))
        # matrix = matrix.t()
        r11, r12, r13 = matrix[0,:3]
        r21, r22, r23 = matrix[1,:3]
        r31, r32, r33 = matrix[2,:3]
        if True:
            # print(r11+r22+r33)
            theta = torch.acos((r11+r22+r33-1)/2.0)
            # print(theta)
            assert(torch.isnan(theta).any() == False)
            theta = torch.acos((r11+r22+r33-1)/2.0)
            eps = 1e-5
            nx = (r32-r23)/(2*torch.sin(theta)+eps)
            ny = (r13-r31)/(2*torch.sin(theta)+eps)
            nz = (r21-r12)/(2*torch.sin(theta)+eps)
            # print(nx, ny, nz)
            assert(torch.isnan(nx).any() == False)
            assert(torch.isnan(ny).any() == False)
            assert(torch.isnan(nz).any() == False)
            return torch.stack([nx, ny, nz], dim=0) * theta

        # theta1 = torch.arctan(-(r23 / r33))
        # theta2 = torch.arctan(r13 * torch.cos(theta1) / r33)
        # theta3 = torch.arctan(-(r12 / r11))
        theta1 = torch.atan2(-r23 , r33)
        # theta2 = torch.atan2(r13 * torch.cos(theta1), r33)
        theta2 = torch.asin(r13)
        theta3 = torch.atan2(-r12, r11)
        # theta2 = torch.asin(-r31)

        theta1 = theta1 * 180 / np.pi
        theta2 = theta2 * 180 / np.pi
        theta3 = theta3 * 180 / np.pi # as euler
        return torch.stack([theta1, theta2, theta3], dim=0)

    def matrix_to_euler_pos_batch(self, matrix, top_batching=False, debug_p3d = False):
        if debug_p3d:
            raise ValueError("debug_p3d")
            # if top_batching:
            #     return torch.rad2deg(p3d_transforms.matrix_to_euler_angles(matrix[...,:3,:3], convention="XYZ"))
            # else:
            #     return torch.rad2deg(p3d_transforms.matrix_to_euler_angles(matrix.permute(2,0,1)[...,:3,:3], convention="XYZ")).permute(1,2,0)
        if top_batching:
            matrix = matrix.permute(1,2,0)
        # matrix = matrix.permute(1,0,2)
        r11, r12, r13 = matrix[0,:3]
        r21, r22, r23 = matrix[1,:3]
        r31, r32, r33 = matrix[2,:3]
        if True:
            eps = 1e-5
            theta = torch.acos(torch.clamp((r11+r22+r33-1)/2.0, -1, 1))
            assert(not torch.isnan(theta).any())
            nx = (r32-r23)/(2*torch.sin(theta)+eps)
            ny = (r13-r31)/(2*torch.sin(theta)+eps)
            nz = (r21-r12)/(2*torch.sin(theta)+eps)
            result = torch.stack([nx, ny, nz], dim=0) * theta
            assert(not torch.isnan(result).any())
            if top_batching:
                return result.permute(1,0)
            return result
        # theta1 = torch.arctan(-r23 / r33)
        # theta2 = torch.arctan(r13 * torch.cos(theta1) / r33)
        # theta3 = torch.arctan(-r12 / r11)
        theta1 = torch.atan2(-r23 , r33) #(cy*sx) / (cy*cx)
        # theta2 = torch.atan2(r13 * torch.cos(theta1), r33)
        theta2 = torch.asin(r13)
        theta3 = torch.atan2(-r12, r11)

        theta1 = theta1 * 180 / np.pi
        theta2 = theta2 * 180 / np.pi
        theta3 = theta3 * 180 / np.pi # as euler
        if top_batching:
            return torch.stack([theta1, theta2, theta3], dim=-1)
        return torch.stack([theta1, theta2, theta3], dim=0)


    def precompute_id(self, next_id):
        self.id = next_id
        tmp_id = self.id
        for c in self.children:
            tmp_id = c.precompute_id(tmp_id+1)
        return tmp_id

    def transformNet(self, poses, type = "euler"):
        #poses: [j, 3]
        if torch.is_tensor(poses[self.id]):
            pose = self.pose_converter(poses[self.id], type)
        else:
            pose = self.pose_converter(self.mytensor(poses[self.id]), type)
        self.apply_transform(pose)
        del pose
        res = [self.matrix_to_euler_pos(self.global_transform())]
        for c in self.children:
            res.extend(c.transformNet(poses, type=type))
        return torch.stack(res, dim=0)

    def pose_converter(self, input, type):
        if type=="quaternion":
            # mat = quaternion_to_matrix(input)
            # eu = self.matrix_to_euler_pos(mat)
            # input = euler_to_quaternion(eu)
            # print(self.name, eu)
            if "_tail" in self.name:
                return self.myeye(4)
            return quaternion_to_matrix(input)
        elif type=="matrix":
            return input
        else:
            return self.euler_to_matrix(input)

    def get_inv(self, parent=None):
        if parent is not None:
            t = torch.matmul(parent, self.local_transform())
        else:
            t = torch.matmul(self.parent_transform, self.local_transform())

        return affine_inverse(torch.matmul(t, self.bind_inv), self.device)

    def get_invs(self, parent=None):
        if parent is not None:
            t = torch.matmul(parent, self.local_transform())
        else:
            t = torch.matmul(self.parent_transform, self.local_transform())

        res = [affine_inverse(torch.matmul(t, self.bind_inv), self.device)]

        # print("compute_inv : ", time.perf_counter() - comp_start)

        for c in self.children:
            res.extend(c.get_invs(parent=t))
        if parent is not None:
            return res
        else:
            return torch.stack(res, dim=0)
    
    def rotations_to_invs(self, poses, parent=None, type="euler"):
        return affine_inverse_batch(self.rotations_to_transforms(poses, parent=parent, type=type))

    def para_precompute(self, nj):
        self.precomp_num_joints = nj
        self.precomp_depth = self.compute_depth(0)
        self.precomp_trans_ids = self.compute_transform_ids([])
        # depth???????????????????????????joint ???id?????????
        self.precomp_ids = self.get_listed_ids()
        self.precomp_trans = self.get_listed_transforms_initial()
        # initial_transform = bind ????????????
        self.precomp_bindinvs = self.get_listed_transforms_bindinv()



    def para_rotations_to_transforms(self, poses, type="euler"):
        if torch.is_tensor(poses[self.id]):
            pose = self.pose_converter(poses[self.id][:3], type) # as radian
        else:
            pose = self.pose_converter(torch.tensor(poses[self.id][:3], dtype=torch.float32, device=torch.device(poses.device)), type)
        translates = poses[self.id][3:]
        pose[...,:3,3] = translates

        self.precomp_forward_global_transform = torch.matmul(pose, self.bind_inv)
        res = [self.precomp_forward_global_transform]
        
        for c in self.children:
            res.extend(c.para_rotations_to_transforms(poses, type=type))
        self.precomp_forward_global_transforms = torch.stack(res, dim=0)

        return torch.stack(res, dim=0)

    def para_rotations_to_transforms_fast(self, poses, type="quaternion"):
        translates = poses[...,3:]
        poses = poses[...,:3]

        # exit("rotations_to_invs_fast")
        if type=="quaternion":
            animations = quaternion_to_matrix_batch(poses).permute(2,0,1)
        elif type=="euler":
            animations = euler_to_matrix_batch(poses, top_batching=True)

        animations[:,:3,3] = translates
        assert(not torch.isnan(animations).any())
        self.precomp_forward_global_transforms = animations
        return torch.bmm(animations, self.precomp_bindinvs)


    def get_listed_precomp_global_transforms(self):
        tmp_precomp = [self.precomp_forward_global_transform]
        for c in self.children:
            tmp_precomp.extend(c.get_listed_precomp_global_transforms())
        return tmp_precomp
    

    def rotations_to_transforms(self, poses, parent=None, type="euler"):
        # etm_start = time.perf_counter()

        
        if torch.is_tensor(poses[self.id]):
            pose = self.pose_converter(poses[self.id], type) # as radian
        else:
            pose = self.pose_converter(torch.tensor(poses[self.id], dtype=torch.float32, device=torch.device(poses.device)), type)
        self.apply_transform(pose, only_self=True)
        if parent is not None:
            t = torch.matmul(parent, self.local_transform())
        else:
            t = torch.matmul(self.parent_transform, self.local_transform())
        self.precomp_forward_global_transform = t

        res = [torch.matmul(t, self.bind_inv)]


        for c in self.children:
            res.extend(c.rotations_to_transforms(poses, parent=t, type=type))
        if parent is not None:
            return res
        else:
            self.precomp_forward_global_transforms = torch.stack(self.get_listed_precomp_global_transforms(), dim=0)
            return torch.stack(res, dim=0)

    def __get_pose_mat(self):
        return torch.matmul(self.T, torch.matmul(self.R, self.S))

    def local_transform(self, only_rotation = True):
        if only_rotation:
            self.local_transform_cached = torch.mm(self.initial_transform, self.R.float())
            return self.local_transform_cached
        transform = torch.matmul(self.T, torch.matmul(self.R, self.S))
        self.local_transform_cached = torch.mm(self.initial_transform, transform)
        return self.local_transform_cached

    def global_transform(self):
        return torch.matmul(self.parent_transform.float(), self.local_transform())

    def global_pos(self, ngp=False):
        seed = torch.zeros(4, dtype=torch.float32, device=torch.device(self.device))
        seed[3] = 1
        result =  torch.mv(self.global_transform(), seed)[:3]

        del seed
        return result

    def inv_totalTransform(self, ngp=False):
        t = torch.matmul(self.global_transform(), self.bind_inv)
        return affine_inverse(t, self.device)

    def add_child(self, c):
        self.children.append(c)

    def __decode_transform(self, transform):
        # r = torch.eye(4, dtype=torch.float32, device=torch.device(my_torch_device))
        # t = torch.eye(4, dtype=torch.float32, device=torch.device(my_torch_device))
        # s = torch.eye(4, dtype=torch.float32, device=torch.device(my_torch_device))
        r, t, s = self.myeye(4), self.myeye(4), self.myeye(4)
        r[:3,:3] = transform[:3,:3]
        t[:,3] = transform[:,3]
        return r,t,s

    def apply_transform(self, transform, only_rotation=True, only_self=False):
        if not torch.is_tensor(transform):
            # transform = torch.tensor(transform,  dtype=torch.float32, device=torch.device(my_torch_device))
            transform = self.mytensor(transform)
        if only_rotation:
            self.R = transform
        else:
            self.R, dummy, self.S = self.__decode_transform(transform)
            del transform,dummy
        if only_self:
            return
        self.__apply_descender(self.parent_transform)

    def __apply_descender(self, parent_transform):
        self.parent_transform = parent_transform
        del parent_transform
        for c in self.children:
            c.__apply_descender(torch.matmul(self.parent_transform.float(), self.local_transform()))

    def get_listed_positions(self, use_cached=False):
        if use_cached:
            return torch.matmul(self.precomp_forward_global_transforms, torch.tensor([0.0,0.0,0.0,1.0], device = self.precomp_forward_global_transforms.device).unsqueeze(0).repeat(self.precomp_forward_global_transforms.shape[0],1).unsqueeze(-1)).squeeze()
        res = [self.global_pos()]
        for c in self.children:
            res.extend(c.get_listed_positions())
        return torch.stack(res, dim=0)

    def get_listed_parents(self, parent_id = None):
        if parent_id is None:
            parent_id = self.id
        self.parent_id = parent_id
        res = [self.parent_id]
        for c in self.children:
            res.extend(c.get_listed_parents(self.id))
        self.parent_id_list = torch.tensor(res, device = self.device)
        return res

    def get_listed_positions_center(self, use_cached=False):
        if use_cached:
            pos = torch.matmul(self.precomp_forward_global_transforms, torch.tensor([0.0,0.0,0.0,1.0], device = self.precomp_forward_global_transforms.device).unsqueeze(0).repeat(self.precomp_forward_global_transforms.shape[0],1).unsqueeze(-1)).squeeze()
        res = [self.global_pos()]
        for c in self.children:
            res.extend(c.get_listed_positions())
        pos = torch.stack(res, dim=0)
        parents = torch.gather(pos, 0, self.parent_id_list.unsqueeze(-1).repeat(1,3))
        return (pos + parents)/2

    def get_listed_names(self):
        res=[self.name]
        for c in self.children:
            res.extend(c.get_listed_names())
        return res

    def get_listed_ids(self):
        res=[self.id]
        for c in self.children:
            res.extend(c.get_listed_ids())
        return res

    def get_listed_rotations(self, type = "euler"):
        if type=="quaternion":
            res = [euler_to_quaternion(self.matrix_to_euler_pos(self.__get_pose_mat()))]
        elif type == "matrix":
            res = [self.__get_pose_mat()]
        else:
            res = [self.matrix_to_euler_pos(self.__get_pose_mat())]
        for c in self.children:
            res.extend(c.get_listed_rotations(type=type))
        return torch.stack(res, dim=0)

    def refresh(self):
        self.apply_transform(self.myeye(4), only_rotation=False)
        for c in self.children:
            c.refresh()
        

    def get_listed_transforms(self):
        res = [self.__get_pose_mat()]
        for c in self.children:
            res.extend(c.get_listed_transforms())
        return torch.stack(res, dim=0)

    def get_listed_global_transforms(self):
        res = [self.global_transform()]
        for c in self.children:
            res.extend(c.get_listed_global_transforms())
        return torch.stack(res, dim=0)

    def get_listed_transforms_initial(self):
        res = [self.initial_transform]
        for c in self.children:
            res.extend(c.get_listed_transforms_initial())
        return torch.stack(res, dim=0)

    def get_listed_transforms_bindinv(self):
        res = [self.bind_inv]
        for c in self.children:
            res.extend(c.get_listed_transforms_bindinv())
        return torch.stack(res, dim=0)

    def get_id_array(self):
        res = [self.id]
        for c in self.children:
            res.extend(c.get_id_array())
        return res

    def get_children(self):
        return self.children

    def get_tail_ids(self):
        res = []
        if self.tail:
            res.append(self.id)
        for c in self.children:
            res.extend(c.get_tail_ids())
        return res

        
    def set_tails(self, tails):
        self.tails = tails
    
    def draw_mask(self, rays_o, rays_d, radius, point = None):
        if point is None:
            center = self.global_pos()
        else:
            center = point
        if not self.ngp_space:
            center = ngp_vector_to_nerf(center, tensor=True)
        #here already ngp
        


        v2 = each_dot(rays_d, rays_d)
        xc = rays_o - center
        r2 = radius ** 2
        d = each_dot(rays_d, xc) ** 2 -(v2 * (each_dot(xc,xc) - r2))
        mask = d >= 0.0
        t = each_dot(-rays_d,xc) - torch.sqrt(d)

        mask2 = t >= 0.0
        
            
        return torch.logical_and(mask, mask2)

    def draw_mask_batched(self, rays_o, rays_d, radius, centers):
        #centers: J, 3
        #rays_o: B, 3
        # rays_o = rays_o.unsqueeze(0).repeat(centers.shape[0], 1, 1)
        # rays_d = rays_d.unsqueeze(0).repeat(centers.shape[0], 1, 1)
        rays_o = rays_o.unsqueeze(0).expand(centers.shape[0], -1, -1)
        rays_d = rays_d.unsqueeze(0).expand(centers.shape[0], -1, -1)
        v2 = each_dot(rays_d, rays_d) #J, B
        xc = rays_o - centers.unsqueeze(1).expand(rays_o.shape) #J, B, 3
        r2 = radius ** 2
        d = each_dot(rays_d, xc) ** 2 -(v2 * (each_dot(xc,xc) - r2))
        mask = d >= 0.0
        t = each_dot(-rays_d,xc) - torch.sqrt(d)

        mask2 = t >= 0.0
        return torch.logical_and(mask, mask2).sum(dim=0) > 0

    def draw_mask_all_cached(self, rays_o, rays_d, radius):
        # print(rays_o.shape, rays_d.shape)
        tmp = self.precomp_forward_global_transforms[:,:3,3]
        # tmp = self.get_listed_positions(/)
        # print(tmp.shape)
        return self.draw_mask_batched(rays_o, rays_d, radius, tmp)


      
    def draw_mask_all(self, rays_o, rays_d, radius):
        # names = ['root', 'Sharkthroat', 'SharkHead', 'SharkHead_tail', 'SharkJaw', 'SharkJaw_tail', 'Sharktail0', 'Sharktail1', 'Sharktail2', 'Sharktail3', 'Sharktail4', 'Sharktail4_tail', 'Righhfena1', 'Rightfena2', 'Rightfena3', 'Rightfena3_tail', 'Leftfena1', 'Leftfena2', 'Leftfena3', 'Leftfena3_tail']

        mask = self.draw_mask(rays_o, rays_d, radius)
        for i in range(len(self.markers)):
            p2 = self.markers[i]
            mask2 = self.draw_mask(rays_o, rays_d, 0.005, point = torch.matmul(self.global_transform(), torch.matmul(self.bind_inv, p2))[:3])
            mask = torch.logical_or(mask, mask2)
        # if self.name in ['root', 'SharkHead_tail', 'SharkJaw_tail', 'Sharktail4_tail', 'Rightfena3_tail', 'Leftfena3_tail']:
        #     mask = rays_o[...,0] > 1000000000000
        for c in self.children:
            mask2 = c.draw_mask_all(rays_o, rays_d, radius)
            mask = torch.logical_or(mask, mask2)
        return mask

    def set_inv_precomputations(self, nj):
        self.precomp_num_joints = nj
        self.precomp_depth = self.compute_depth(0)
        self.precomp_trans_ids = self.compute_transform_ids([])
        # depth???????????????????????????joint ???id?????????
        self.precomp_ids = self.get_listed_ids()
        self.precomp_trans = self.get_listed_transforms_initial()
        # initial_transform = bind ????????????
        self.precomp_bindinvs = self.get_listed_transforms_bindinv()
        #bind_inv????????????
        for i, id_array in enumerate(self.precomp_trans_ids):
            while len(id_array) < self.precomp_depth:
                self.precomp_trans_ids[i].append(-1)
        mats_ids = self.mytensor(self.precomp_trans_ids, dtype=torch.uint8)
        '''
         - j -
         0  0  0  0
         1 -1  1 -1
        -1 -1  2 -1
        '''
        self.precomp_mats = self.myeye(4).repeat(self.precomp_num_joints, self.precomp_depth, 1, 1)
        # self.precomp_mats = self.myeye(4).expand(self.precomp_num_joints, self.precomp_depth, -1, -1)
        for i in range(len(self.precomp_ids)):# for J
            self.precomp_mats = torch.where(
                (mats_ids==self.precomp_ids[i]).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 4), # ??????inds??????????????????joint??????????????????4*4???
                self.precomp_trans[self.precomp_ids[i]].expand_as(self.precomp_mats),             # ??????????????????initial_transform?????????????????????eye
                self.precomp_mats)
        
        self.precomp_masks = []
        for i in range(len(self.precomp_ids)):# for J
            self.precomp_masks.append((mats_ids==self.precomp_ids[i]).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 4))
        # print(len(self.precomp_masks), self.precomp_masks[0].shape)
        # joint ???????????????????????????

    def tree_mul(self, mat, depth):
        if depth == 1:
            return mat.squeeze()
        form = depth//2
        return torch.bmm(self.tree_mul(mat[:, :form], form), self.tree_mul(mat[:, form:], depth-form))

    def apply_transforms_top(self, poses, use_precomp = True, type="euler"):
        if type=="quaternion":
            poses = quaternion_to_matrix_batch(poses).permute(2,0,1)
        elif type=="euler":
            poses = euler_to_matrix_batch(torch.transpose(poses, 0, 1)).permute(2,0,1)
        if use_precomp:
            for i, j in enumerate(self.joints):
                j.apply_transform(self.precomp_forward_global_transforms[i], only_self=True)
            return
        for i, j in enumerate(self.joints):
            j.apply_transform(poses[i], only_self=False)

    def rotations_to_invs_fast(self, poses, type="quaternion"):
        return affine_inverse_batch(self.rotations_to_transforms_fast(poses, type=type))


    #@torch.jit.script
    def rotations_to_transforms_fast(self, poses, type="quaternion"):
        # print(poses.shape, type)
        # print(euler_to_matrix_batch(torch.transpose(poses, 0, 1)).permute(2,0,1).shape)

        # exit("rotations_to_invs_fast")
        if type=="quaternion":
            animations = quaternion_to_matrix_batch(poses).permute(2,0,1)
        elif type=="euler":
            animations = euler_to_matrix_batch(poses, top_batching=True)
        #[J, 4, 4]
        mats2 = self.myeye(4).repeat(self.precomp_num_joints, self.precomp_depth, 1, 1)
        # mats2 = self.myeye(4).expand(self.precomp_num_joints, self.precomp_depth, -1, -1)
        #[J, D, 4, 4] / D????????????skeleton?????????
        for i in range(len(self.precomp_ids)): # for J
            # if i in self.tails:
            #     continue
                # tail ????????????animation???????????????
            mats2 = torch.where(self.precomp_masks[i], animations[self.precomp_ids[i]].expand_as(mats2), mats2)
            # joint ????????????transform ???????????????????????? precomp_mask
        mat = torch.bmm(self.precomp_mats.view(self.precomp_num_joints*self.precomp_depth, 4, 4), mats2.view(self.precomp_num_joints*self.precomp_depth, 4, 4)).view(self.precomp_num_joints, self.precomp_depth, 4, 4)
        # ??????id??????????????????????????????????????????initial_transform ??? ?????????????????????????????????
        # ??????????????????????????????????????????joint ??? transform????????????????????????
        # self.precomp_forward_global_transforms = torch.bmm(self.tree_mul(mat, self.precomp_depth), self.precomp_bindinvs)
        self.precomp_forward_global_transforms = self.tree_mul(mat, self.precomp_depth)
        self.precomp_poses = animations
        
        return torch.bmm(self.precomp_forward_global_transforms, self.precomp_bindinvs)
        # return affine_inverse_batch(torch.bmm(self.tree_mul(mat, self.precomp_depth), self.precomp_bindinvs), self.device)

    def apply_precomputed_localtransorms(self):
        for i, j in enumerate(self.joints):
            j.apply_transform(self.precomp_poses[i], only_self=False)


