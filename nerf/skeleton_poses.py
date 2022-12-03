import torch
import torch.nn as nn
# from ..utils.lie_group_helper import make_c2w
from .render_util import *


class LearnSkeletonPose(nn.Module):
    def __init__(self, num_frames, num_joints, learn=True, init_c2w=None, type="euler"):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnSkeletonPose, self).__init__()
        self.num_frames = num_frames
        self.init_c2w = None
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)
        self.if_pin = False
        self.pins = []
        self.pins_dict = {}
        self.type = type
        if type=="euler":
            self.pose = nn.Parameter(torch.zeros(size=(num_frames, num_joints, 3), dtype=torch.float32), requires_grad=learn)  # (N, j, 3)
            # exit("not implemented")
        elif type=="quaternion":
            # exit("not implemented")
            #Todo: switch
            # tmp = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(num_frames, num_joints, 1)
            tmp = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(num_frames, num_joints, 1)
            # self.pose = nn.Parameter(torch.zeros(size=(num_frames, num_joints, 4), dtype=torch.float32), requires_grad=learn)  # (N, j, 3)
            self.pose = nn.Parameter(tmp, requires_grad=learn)  # (N, j, 3)
            # exit("not implemented2")
        elif type=="matrix":
            tmp = torch.eye(4, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(num_frames, num_joints, 4, 4)
            self.pose = nn.Parameter(tmp, requires_grad=learn)  # (N, j, 3)
        # self.t = nn.Parameter(torch.zeros(size=(num_frames, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

    def pin_mode(self, flag):
        self.if_pin = flag

    def set_pin(self, id, pose):
        self.pins.append(id)
        self.pins_dict[id] = pose
        print("set pin : ", id, pose)

    def set_tails(self, tails):
        self.tails = tails
        return 
    def save(self, path):
        # kwargs = self.get_kwargs()
        ckpt = {'state_dict': self.state_dict()}
        # if self.alphaMask is not None:
        #     alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
        #     ckpt.update({'alphaMask.shape':alpha_volume.shape})
        #     ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape(-1))})
        #     ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)
    def load(self, ckpt):
        self.load_state_dict(ckpt['state_dict'], strict = False)

    def forward(self, frame_id):
        # if self.if_pin and frame_id in self.pins:
        #     return self.pins_dict[frame_id.item()].squeeze()
        if self.type == "euler":
            res = self.pose[frame_id].squeeze()
            with torch.no_grad():
                for t in self.tails:
                    res[t] *= 0
            return res  # (j, 3, ) axis-angle
        if True:
            quat = self.pose[frame_id, :, :].squeeze() #j, 3
            # with torch.no_grad():
            #     for t in self.tails:
            #         quat[t] *= 0
            norm2 = torch.sum(quat*quat, dim=-1) # j positive value
            ws = (1 - norm2).unsqueeze(-1) #j
            ws = torch.where(ws < 0.0, torch.zeros_like(ws), ws)
            ws = torch.sqrt(ws)
            q = torch.cat([ws, quat], dim=-1)
            return q
        else:
            quat = self.pose[frame_id, :, :].squeeze() #j, 3
            q = euler_to_quaternion(quat.T).T
            # print(q.shape)
            # exit("ff")
            return q
            

