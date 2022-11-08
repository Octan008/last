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

import time

my_torch_device = "cuda:0"

def affine_inverse_batch(bmatrix):
    inv = torch.eye(4, device="cuda:0").unsqueeze(0).repeat(bmatrix.shape[0], 1, 1)
    inv[:,:3,:3] = torch.transpose(bmatrix[:,:3,:3], 1, 2)
    inv[:,:3, 3] = torch.bmm(torch.transpose(bmatrix[:,:3,:3], 1, 2), -bmatrix[:,:3,3].unsqueeze(-1)).squeeze()
    return inv

def affine_inverse(matrix):
    inv = torch.eye(4, device="cuda:0")
    inv[:3,:3] = torch.transpose(matrix[:3,:3], 0, 1)
    inv[:3, 3] = torch.mv(torch.transpose(matrix[:3,:3], 0, 1), -matrix[:3,3])
    return inv
#https://qiita.com/harmegiddo/items/96004f7c8eafbb8a45d0#%E3%82%AA%E3%82%A4%E3%83%A9%E3%83%BC%E8%A7%92

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

def euler_to_quaternion(r):
    sx, sy, sz = torch.sin(torch.deg2rad(r/2.0))
    cx, cy, cz = torch.cos(torch.deg2rad(r/2.0))
    return torch.stack([
        cx*cy*cz - sx*sy*sz,
        cy*sx*cz + sy*cx*sz,
        sy*cx*cz - cy*sx*sz,
        cy*cx*sz + sy*sx*cz
    ],dim=0)
    

def compute_weights(xyzs, joints):
    weights_list = []
    for j in joints:
        weights = (xyzs - j.global_pos(ngp=True)).to(torch.float16).to(my_torch_device)
        weights = (weights*weights).sum(dim=1, keepdim=True).squeeze()
        weights = torch.exp(-weights*0.01)
        weights_list.append(weights)
        
    weights_list = torch.stack(weights_list, dim=0)
    return torch.transpose(weights_list, 0, 1)


def weighted_transformation(xyzs, weights, transforms):
    #xyzs -> [N, 3]
    #weights -> [N, J]
    #transforms -> [J, 4, 4]
    #https://qiita.com/tand826/items/9e1b6a4de785097fe6a5
    
    weights_sum = weights.sum(dim=1)
    n_sample = xyzs.shape[0]
    n_joint = transforms.shape[0]
    eps = 1e-7
    non_valid = (weights_sum < eps).unsqueeze(-1).expand(n_sample, 3)
    valid = ~non_valid
    validT = torch.transpose(valid, 0, 1);
    # print("weights_sum_zerotest")
    # print(torch.sum(weights_sum < eps), weights_sum.shape)
    # exit("weights_sum_zerotest")

    weights_sum =  torch.where(weights_sum < eps, torch.ones_like(weights_sum), weights_sum)
    # print("weights_sum")
    # print(weights_sum.shape)
    # print(torch.sum(weights))
    # print(torch.sum(weights_sum))
    # exit("weight_sum")

    weights = weights/weights_sum.unsqueeze(1)

    
    xyzs = torch.transpose(xyzs, 0, 1) #[4, N]
    
    xyzs = torch.cat([xyzs, torch.ones(n_sample).unsqueeze(0).to(xyzs.device)], dim=-0)#[4, N]
    
    # print(weights.shape, transforms.shape)
    # result = torch.zeros_like(xyzs) #[N, 3]
    result = torch.nan_to_num(torch.transpose(torch.matmul(transforms[0], xyzs), 0, 1)  * weights[:,0].unsqueeze(1))[:,:3]
    # print(result.shape)
    # print(valid.unsqueeze(-1).expand(result.shape[0], result.shape[1]).shape)
    # print(result[valid.unsqueeze(-1).expand(result.shape[0], result.shape[1])].reshape(-1, 3).shape)
    # exit()
    tmp_ =  validT[0].unsqueeze(0).expand(4, -1)
    for i in range(1, transforms.shape[0]):
        if result[valid].shape[0] == 0:
            break
        # print(xyzs[validT[0].unsqueeze(0).expand(4, validT.shape[-1])].shape)
        # result[valid] += torch.nan_to_num(torch.transpose(torch.matmul(transforms[i], xyzs[validT[0].unsqueeze(0).expand(4, -1)].reshape(4, -1)), 0, 1)[...,:3].reshape(n_sample*3)  * weights[:,i][valid[:,0]].unsqueeze(1))
        result += torch.nan_to_num(torch.transpose(torch.matmul(transforms[i], xyzs), 0, 1)[...,:3]  * weights[:,i].unsqueeze(1))[...,:3]
    # print(result.shape, xyzs.shape, non_valid.shape, non_valid.unsqueeze(-1).expand(result.shape[0], result.shape[1]).shape)
    result[non_valid] = torch.transpose(xyzs, 0, 1)[...,:3][non_valid]
    # print("non_valid")
    # print(result[non_valid].shape, result.shape)
    # exit("non_valid")
    return result


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

def make_joints_from_blender(file_path):
    def make_joint_rec(data, data_pos, parent_joint, obj_bind):
        name = data["name"]
        # obj_bind = torch.tensor(obj_bind)
        bind_pose = torch.mm(obj_bind, torch.tensor(data["bind_transform"]))
        joint = Joint(
            bind_pose=bind_pose,
            parent_bind=parent_joint.global_transform(),
            name = data["name"]
        )
        if len(data["children"]) == 0:
            translate = torch.tensor(np.array(data_pos[name]["tail"]) - np.array(data_pos[name]["head"])).float()
            translate = torch.mv(obj_bind[:3,:3], translate)
            bind_pose[:3,3] += translate
            joint.add_child(
                Joint(
                    bind_pose=bind_pose,
                    parent_bind=joint.global_transform(),
                    name = data["name"]+"_tail",
                    tail=True
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
            parent_bind=torch.eye(4, dtype=torch.float16, device=torch.device(my_torch_device)),
            name = root["name"]
        )
    for c in root["children"]:
        skeleton.add_child(make_joint_rec(c, data_pos, skeleton, bind))
    return skeleton




def cast_positions(positions, joints, weights_list = None):
    if weights_list is None:
        weights_list = []
        for j in joints:
            weights = (positions - j.global_pos(ngp=True)).to(torch.float16).to(my_torch_device)
            weights = (weights*weights).sum(dim=1, keepdim=True).squeeze()
            weights = torch.exp(-weights*10)
            weights_list.append(weights)
        weights_list = torch.stack(weights_list, dim=0)
        weights_sum = weights_list.sum(dim=0)
        weights_sum =  torch.where(weights_sum < 1e-5, torch.ones_like(weights_sum), weights_sum)
    else:
        weights_sum = weights_list.sum(dim=0)
        weights_sum =  torch.where(weights_sum < 1e-5, torch.ones_like(weights_sum), weights_sum)
 
    ones = torch.ones(positions.shape[0], device=my_torch_device,  dtype=torch.float32).unsqueeze(1)
    casted = torch.zeros(positions.shape[0],4, device=my_torch_device,  dtype=torch.float32).unsqueeze(2)
    weights_list = weights_list/weights_sum
    positions = torch.cat([positions, ones], dim=1)
    
    for i, j in enumerate(joints):
        mat = j.inv_totalTransform(ngp=True).unsqueeze(0).expand(positions.shape[0],-1,-1)

        casted += torch.matmul(mat, positions[...,None]) * weights_list[i].unsqueeze(1).unsqueeze(2)
        # if torch.isnan(casted).any() or torch.isinf(casted).any():
        #     raise ValueError("error!")
    del ones, weights_list, positions, mat
    return torch.squeeze(casted)[...,:3]



def euler_to_matrix(angle = None, translate = None):
    mat = torch.eye(4, device=torch.device(my_torch_device))
    
    if translate is not None:
        print(mat[:3, 3].shape, translate.shape)
        mat[:3, 3] = translate
    if angle is not None:
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


def make_transform(angle=None, translate = None):
    # return euler_to_matrix_old(angle, translate)
    return euler_to_matrix(angle, translate)
#

def each_dot(a,b):
    tmp = torch.zeros_like(a[...,0])
    for i in range(a.shape[-1]):
        tmp += a[...,i] * b[...,i]
    return tmp


class Joint():
    def __init__(self, bind_pose = None, parent_bind = None, name = "NotSet", ngp_space=True, device=torch.device("cuda:0"), tail=False):
        self.ngp_space=ngp_space
        self.device = device
        self.torchtype=torch.float32
        self.tail = tail


        # self.bind_pose = torch.tensor(bind_pose,  dtype=torch.float32, device=torch.device(my_torch_device))
        bind_pose = self.mytensor(bind_pose)

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

        self.markers = [
            torch.mv(self.global_transform(), self.mytensor([0.05,    0,    0, 1])),
            torch.mv(self.global_transform(), self.mytensor([   0, 0.05,    0, 1])),
            torch.mv(self.global_transform(), self.mytensor([   0,    0, 0.05, 1]))
        ]
        self.first_pos = self.global_pos();

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
            return  torch.tensor(input,  dtype=self.torchtype, device=self.device)
        else: 
            return  torch.tensor(input,  dtype=self.torchtype, device=device)

    def matrix_to_euler_pos(self, matrix):
        r11, r12, r13 = matrix[0,:3]
        r21, r22, r23 = matrix[1,:3]
        r31, r32, r33 = matrix[2,:3]
        theta1 = torch.arctan(-r23 / r33)
        theta2 = torch.arctan(r13 * torch.cos(theta1) / r33)
        theta3 = torch.arctan(-r12 / r11)
        theta1 = theta1 * 180 / np.pi
        theta2 = theta2 * 180 / np.pi
        theta3 = theta3 * 180 / np.pi # as euler


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

    def rotations_to_invs(self, poses, parent=None, type="euler"):
        # etm_start = time.perf_counter()
        if torch.is_tensor(poses[self.id]):
            pose = self.pose_converter(poses[self.id], type) # as radian
        else:
            pose = self.pose_converter(torch.tensor(poses[self.id], dtype=torch.float32, device=torch.device(my_torch_device)), type)
        # print("etm: ", time.perf_counter() - etm_start)
        # apt_start = time.perf_counter()
        self.apply_transform(pose, only_self=True)
        # del pose
        # print("apt : ", time.perf_counter() - apt_start)
        # comp_start = time.perf_counter()
        
        if parent is not None:
            t = torch.matmul(parent, self.local_transform())
        else:
            t = torch.matmul(self.parent_transform, self.local_transform())

        res = [affine_inverse(torch.matmul(t, self.bind_inv))]

        # print("compute_inv : ", time.perf_counter() - comp_start)

        for c in self.children:
            res.extend(c.rotations_to_invs(poses, parent=t, type=type))
        if parent is not None:
            return res
        else:
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

    # def inv_list(self, parent=None):
    #     if parent is None:
    #         parent = self.parent_transform
    #     t = torch.mm(parent, self.local_transform_cached)
    #     res = [affine_inverse(torch.mm(t, self.bind_inv))]
    #     for c in self.children:
    #         res.extend(c.inv_list(t))
    #     return res

    def inv_totalTransform(self, ngp=False):
        t = torch.matmul(self.global_transform(), self.bind_inv)
        return affine_inverse(t)


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

    def get_listed_positions(self):
        res = [self.global_pos()]
        for c in self.children:
            res.extend(c.get_listed_positions())
        return torch.stack(res, dim=0)

    def get_listed_positions_first(self):
        res = [self.first_pos]
        for c in self.children:
            res.extend(c.get_listed_positions())
        return torch.stack(res, dim=0)

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

    def get_listed_transforms(self):
        res = [self.__get_pose_mat()]
        for c in self.children:
            res.extend(c.get_listed_transforms())
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
    def set_tails(self, tails):
        self.tails = tails          
      
    def draw_mask_all(self, rays_o, rays_d, radius):
        mask = self.draw_mask(rays_o, rays_d, radius)
        for i in range(len(self.markers)):
            p2 = self.markers[i]
            mask2 = self.draw_mask(rays_o, rays_d, 0.005, point = torch.matmul(self.global_transform(), torch.matmul(self.bind_inv, p2))[:3])
            mask = torch.logical_or(mask, mask2)
        for c in self.children:
            mask2 = c.draw_mask_all(rays_o, rays_d, radius)
            mask = torch.logical_or(mask, mask2)
        return mask

    def set_inv_precomputations(self, nj):
        self.precomp_num_joints = nj
        self.precomp_depth = self.compute_depth(0)
        self.precomp_trans_ids = self.compute_transform_ids([])
        # depthに沿って計算すべきjoint のidが並ぶ
        self.precomp_ids = self.get_listed_ids()
        self.precomp_trans = self.get_listed_transforms_initial()
        # initial_transform = bind のリスト
        self.precomp_bindinvs = self.get_listed_transforms_bindinv()
        #bind_invのリスト
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
        for i in range(len(self.precomp_ids)):# for J
            self.precomp_mats = torch.where(
                (mats_ids==self.precomp_ids[i]).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 4), # 関係inds行列から該当jointをとりだし、4*4化
                self.precomp_trans[self.precomp_ids[i]].expand_as(self.precomp_mats),             # 該当部分にはinitial_transformを入れる残りはeye
                self.precomp_mats)
        
        self.precomp_masks = []
        for i in range(len(self.precomp_ids)):# for J
            self.precomp_masks.append((mats_ids==self.precomp_ids[i]).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 4))
        # print(len(self.precomp_masks), self.precomp_masks[0].shape)
        # joint ごとにマスクを作成

    def tree_mul(self, mat, depth):
        if depth == 1:
            return mat.squeeze()
        form = depth//2
        return torch.bmm(self.tree_mul(mat[:, :form], form), self.tree_mul(mat[:, form:], depth-form))

    def rotations_to_invs_fast(self, poses, type="quaternion"):
        animations = torch.transpose(torch.transpose(quaternion_to_matrix_batch(poses), 0,2), 1, 2)
        #[J, 4, 4]
        mats2 = self.myeye(4).repeat(self.precomp_num_joints, self.precomp_depth, 1, 1)
        #[J, D, 4, 4] / Dは最大のskeletonの深さ
        for i in range(len(self.precomp_ids)): # for J
            # if i in self.tails:
            #     continue
                # tail の場合、animationを当てない
            mats2 = torch.where(self.precomp_masks[i], animations[self.precomp_ids[i]].expand_as(mats2), mats2)
            # joint ごとに、transform が入るべき場所が precomp_mask
        mat = torch.bmm(self.precomp_mats.reshape(self.precomp_num_joints*self.precomp_depth, 4, 4), mats2.reshape(self.precomp_num_joints*self.precomp_depth, 4, 4)).reshape(self.precomp_num_joints, self.precomp_depth, 4, 4)
        # 計算id行列上の同じ位置のもの同士、initial_transform と 回転行列をかけて一定化
        # 最終的には縦に掛けていけば各joint の transformが手に入る状態に

        return affine_inverse_batch(torch.bmm(self.tree_mul(mat, self.precomp_depth), self.precomp_bindinvs))



    # def make_child(self, p=None, name="noName"):
    #     bind_pose = torch.eye( 4, dtype=torch.float32, device=torch.device(my_torch_device))
    #     bind_pose[:3,3] = torch.tensor(p, dtype=torch.float32, device=torch.device(my_torch_device))
    #     self.children.append(
    #         Joint(
    #             bind_pose=bind_pose,
    #             parent_bind=self.global_transform(),
    #             name=name
    #         )
    #     )







    


# class Joint():
#     def __init__(self, p=None, parent_t=None, local_ts=None, bind_matrix=None, relative_transform=None, name=None) -> None:
#         if bind_matrix is None:

#             #Global Marker Positions
#             p.append(1)
#             self.p = torch.tensor(p,  dtype=torch.float32, device=torch.device(my_torch_device))
#             self.markers = [
#                 self.p+torch.tensor([0.06, 0, 0,0],  dtype=torch.float32, device=torch.device(my_torch_device)),
#                 self.p+torch.tensor([0, 0.06, 0,0],  dtype=torch.float32, device=torch.device(my_torch_device)),
#                 self.p+torch.tensor([0, 0, 0.06,0],  dtype=torch.float32, device=torch.device(my_torch_device))
#             ]

#             self.T = torch.eye(4,  dtype=torch.float32, device=torch.device(my_torch_device))
#             self.T[:,3] = self.p

#             self.R = torch.eye(4,  dtype=torch.float32, device=torch.device(my_torch_device))
#             self.S = torch.eye(4,  dtype=torch.float32, device=torch.device(my_torch_device))
#             #local transforms
#             if local_ts is not None:
                
#                 self.T[:3,3] = self.p[:3] - local_ts["global_position"][:3]

#                 self.R = local_ts["rotation"]

#             #bind pose
#             self.bind = torch.eye(4,  dtype=torch.float32, device=torch.device(my_torch_device))
#             self.bind[:,3] = self.p
#             self.bind_inv = torch.linalg.pinv(self.bind.float()).to(torch.float16)

#             #parent_transform
#             self.parent_transform = parent_t.to(my_torch_device)
#             self.child = []
#             self.localTransform = self.local_transform()
#         else:
#             self.bind = torch.tensor(bind_matrix,  dtype=torch.float32, device=torch.device(my_torch_device))
            
#             #Global Marker Positions
#             self.p = self.bind[:,3]
#             self.markers = [
#                 torch.mv(self.bind, torch.tensor([0.06, 0, 0,0],  dtype=torch.float32, device=torch.device(my_torch_device))),
#                 torch.mv(self.bind, torch.tensor([0, 0.06, 0,0],  dtype=torch.float32, device=torch.device(my_torch_device))),
#                 torch.mv(self.bind, torch.tensor([0, 0, 0, 0.06],  dtype=torch.float32, device=torch.device(my_torch_device)))
#             ]
#             relative_transform = torch.tensor(relative_transform , dtype=torch.float32, device=torch.device(my_torch_device))
#             self.T = torch.eye(4,  dtype=torch.float32, device=torch.device(my_torch_device))
#             self.T[:,3] = self.p    

#             self.R = torch.eye(4,  dtype=torch.float32, device=torch.device(my_torch_device))
#             self.R[:3,:3] = torch.mm(relative_transform[:3,:3], self.bind[:3,:3])
#             self.S = torch.eye(4,  dtype=torch.float32, device=torch.device(my_torch_device))

#             #bind pose
#             self.bind_inv = torch.linalg.pinv(self.bind.float()).to(torch.float16)

#             #parent_transform
#             self.parent_transform = torch.tensor(parent_t,  dtype=torch.float32, device=torch.device(my_torch_device))
#             self.child = []
#             self.localTransform = self.local_transform()
#         if name is not None:
#             self.name = name

#     def get_listed_positions(self):
#         res = [self.global_pos()]
#         # print(self.name, res)
#         for c in self.get_childs():
#             res.extend(c.get_listed_positions())
#         return torch.stack(res, dim=0)



#     def inv_totalmatrix(self):
#         return torch.linalg.pinv(torch.matmul(self.global_matrix(), self.bind_inv).float()).to(torch.float16)




#     def local_transform(self):
#         return torch.matmul(self.T, torch.matmul(self.R, self.S))
#     def make_child(self, p):
#         self.child.append(
#             Joint(p, torch.matmul(self.parent_transform, self.localTransform), {
#                 "global_position": self.p,
#                 "rotation": torch.eye(4, dtype=torch.float32, device=torch.device(my_torch_device)),
#             })
#         )
#     def get_child(self, i):
#         return self.child[i]
#     def get_childs(self):
#         return self.child
#     def get_bind(self):
#         return self.bind
#     def get_bind_inv(self):
#         return torch.linalg.pinv(self.bind.float()).to(torch.float16)
    
#     def decode_transform(self, transform):
#         r = torch.eye(4, dtype=torch.float32, device=torch.device(my_torch_device))
#         t = torch.eye(4, dtype=torch.float32, device=torch.device(my_torch_device))
#         s = torch.eye(4, dtype=torch.float32, device=torch.device(my_torch_device))
#         r[:3,:3] = copy.deepcopy(transform[:3,:3])
#         t[:,3] = transform[:,3]
#         return r,t,s



#     def add_child(self, child):
#         self.child.append(child)

#     def apply_transform(self, rot=None, translate = None, scale = None, transform=None):
#         if rot is not None:
#             self.R = rot
#         if translate is not None:
#             self.T = translate
#         if scale is not None:
#             self.S = scale
#         if transform is not None:
#             self.R, dummy, self.S = self.decode_transform(torch.tensor(transform,  dtype=torch.float32, device=torch.device(my_torch_device)))

        
#         self.localTransform = copy.deepcopy(self.local_transform())
#         self.apply_descender(self.parent_transform)

#     def apply_descender(self, parent_t):
#         self.parent_transform = parent_t
#         for c in self.child:
#             c.apply_descender(torch.matmul(self.parent_transform, self.localTransform))

#     def draw_mask_all(self, rays_o, rays_d, radius):
#         mask = self.draw_mask(rays_o, rays_d, radius)
#         for i in range(3):
#             p2 = self.markers[i]
#             mask2 = self.draw_mask(rays_o, rays_d, 0.01, point = torch.matmul(self.global_matrix(), torch.matmul(self.bind_inv, p2))[:3])
#             mask = torch.logical_or(mask, mask2)
#         for c in self.child:
#             mask2 = c.draw_mask_all(rays_o, rays_d, radius)
#             mask = torch.logical_or(mask, mask2)
#         return mask
#     def local_pos(self):
#         return self.p
#     def global_pos(self, affine = False):
        
#         root = torch.zeros(4, dtype=torch.float32, device=torch.device(my_torch_device))
#         root[3] = 1
#         return torch.mv(self.global_matrix(), root).to(torch.float32)[:3]

#     def global_matrix(self):
#         return torch.matmul(self.parent_transform, self.localTransform)
#     def global_matrix_inv(self):

#         return torch.linalg.pinv(self.global_matrix().float()).to(torch.float16)
    

#     def draw_mask(self, rays_o, rays_d, radius, point = None):
 
#         if point is None:
#             center = self.global_pos()
#         else:
#             center = point

#         v2 = each_dot(rays_d, rays_d)
#         xc = rays_o - center
#         r2 = radius ** 2
#         d = each_dot(rays_d, xc) ** 2 -(v2 * (each_dot(xc,xc) - r2))
#         mask = d >= 0.0
#         t = each_dot(-rays_d,xc) - torch.sqrt(d)

#         mask2 = t >= 0.0
#         return torch.logical_and(mask, mask2)

# def sphere_mask(rays_o, rays_d, center, radius):
#     center = torch.tensor(center,  dtype=torch.float16, device=torch.device(my_torch_device))
#     v2 = each_dot(rays_d, rays_d)
#     xc = rays_o - center
#     r2 = radius ** 2
#     d = each_dot(rays_d, xc) ** 2 -(v2 * (each_dot(xc,xc) - r2))
#     mask = d >= 0.0
#     t = each_dot(-rays_d,xc) - torch.sqrt(d)

#     mask2 = t >= 0.0
#     return torch.logical_and(mask, mask2)
# #     -v(x-c)±√D
# # t = ---------------
# #          |v|2

# #     D = {v(x-c)}2 - |v|2(|x-c|2-r2)


        
# def box_mask(rays_o, rays_d, center, width):
#         i=0
#         min_x = (center[i]+(width/2.0) - rays_o[...,i]) / rays_d[...,i]
#         max_x = (center[i]-(width/2.0) - rays_o[...,i]) / rays_d[...,i]
#         swap = min_x > max_x
#         min_x[swap], max_x[swap] = max_x[swap], min_x[swap]
#         i=1
#         min_y = (center[i]+(width/2.0) - rays_o[...,i]) / rays_d[...,i]
#         max_y = (center[i]-(width/2.0) - rays_o[...,i]) / rays_d[...,i]
#         swap = min_y > max_y
#         min_y[swap], max_y[swap] = max_y[swap], min_y[swap]
#         min_swap=min_y > min_x
#         min_x[min_swap],  min_y[min_swap] = min_y[min_swap], min_x[min_swap]
#         max_swap = max_y < max_x
#         max_x[max_swap] , max_y[max_swap] = max_y[max_swap], max_x[max_swap]
#         i=2
#         min_y = (center[i]+(width/2.0) - rays_o[...,i]) / rays_d[...,i]
#         max_y = (center[i]-(width/2.0) - rays_o[...,i]) / rays_d[...,i]
#         swap = min_y > max_y
#         min_y[swap],  max_y[swap] =  max_y[swap], min_y[swap]
#         min_swap=min_y > min_x
#         min_x[min_swap], min_y[min_swap] = min_y[min_swap], min_x[min_swap]
#         max_swap = max_y < max_x
#         max_x[max_swap] ,max_y[max_swap] = max_y[max_swap], max_x[max_swap]

#         intersect = max_x > min_x
#         return intersect 





# def make_joints_from_blender(file_path):
#     def make_joint_rec(data, data_pos, j):
#         name = data["name"]
        
#         bind_transform = torch.tensor(data["bind_transform"],  dtype=torch.float32, device=torch.device(my_torch_device))

#         relative_transform = data["relative_transform"]
#         parent_transform=j.global_matrix()

#         # print(data["name"],"&", [[round(data["bind_transform"][j][i], 2) for i in range(4)] for j in range(4)], ";", end="")
#         joint = Joint(
#             data_pos[name]["head"],
#             parent_transform,
#             {
#                 "global_position": j.p,
#                 "rotation": torch.eye(4, dtype=torch.float32, device=torch.device(my_torch_device)),
#             },
#             name=data["name"]
#         )
#         if len(data["childs"]) == 0:
#             joint.add_child(
#                 Joint(
#                     data_pos[name]["tail"],
#                     joint.global_matrix(),
#                     {
#                         "global_position": joint.p,
#                         "rotation": torch.eye(4, dtype=torch.float32, device=torch.device(my_torch_device)),
#                     },
#                     name=data["name"]+"_tail"
#                 )
#             )

#         else:
#             for c in data["childs"]:
#                 joint.add_child(make_joint_rec(c,data_pos,joint))
        
#         return joint
        
#     f = open(file_path, 'r')
#     data = json.load(f)
#     root = data["initial_state"][0]
#     data_pos = data["initial_bone"]
#     print(root["name"])
#     bind_transform = root["bind_transform"]
#     relative_transform = root["relative_transform"]
#     skeleton = Joint(
#         parent_t = torch.eye(4, dtype=torch.float16, device=torch.device(my_torch_device)),
#         bind_matrix = bind_transform,
#         relative_transform=relative_transform,
#         name=root["name"]
#         )
#     for c in root["childs"]:
#         skeleton.add_child(make_joint_rec(c, data_pos, skeleton))
#     return skeleton

    
# def mask_origin_box(p):
#     half_width = 0.05
#     length = p[...,0] ** 2 + p[...,1] ** 2 + p[...,2] ** 2
#     mask = length < (half_width ** 2)
#     return mask

# def mask_point(src, ref_point, half_width):
#     p = src - torch.tensor(ref_point, dtype=torch.float16, device=torch.device(my_torch_device))
#     length = p[...,0] ** 2 + p[...,1] ** 2 + p[...,2] ** 2
#     mask = length < (half_width ** 2)
#     return mask


# def euler_to_matrix_old(angle = None, translate = None):
#     mat = torch.eye(4, device=torch.device(my_torch_device))
    
#     if translate is not None:
#         print(mat[:3, 3].shape, translate.shape)
#         mat[:3, 3] = translate
#     if angle is not None:
#         for i, an in enumerate(angle):
#             cos = math.cos(math.radians(an))
#             sin = math.sin(math.radians(an))
#             angle = torch.eye(4, device=torch.device(my_torch_device))
#             # print(cos,sin, an, i, angle)
#             if i== 0:
#                 angle[1,1] = cos
#                 angle[2,2] = cos
#                 angle[2,1] = sin
#                 angle[1,2] = -sin
#             elif i == 1:
#                 angle[0,0] = cos
#                 angle[2,2] = cos
#                 angle[0,2] = sin
#                 angle[2,0] = -sin

#             else:
#                 angle[0,0] = cos
#                 angle[1,1] = cos
#                 angle[1,0] = sin
#                 angle[0,1] = -sin
#             mat = torch.matmul(angle, mat)
#     return mat