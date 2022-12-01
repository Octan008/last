from turtle import forward
import torch
import torch.nn as nn

class SphereHarmonicJoints(nn.Module):
    def __init__(self, num_joints, dim):
        self.dim = dim
        self.num_joints = num_joints
        super(SphereHarmonicJoints, self).__init__()
        feats = torch.zeros(num_joints, dim)
        feats[:,0] = 30.0
        self.feats = nn.Parameter(feats, requires_grad=True)
        # self.feats = nn.Parameter(torch.tensor([30.0], dtype=torch.float32).unsqueeze(0).repeat(num_joints, dim), requires_grad=True)#.to("cuda:0")  # (j, dim, 1)
        # print("init", self.state_dict().keys())
    def set_allgrads(self, value):
        # grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz},
        #         {'params': self.app_line, 'lr': lr_init_spatialxyz},
        #         {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        for param in self.feats:
            if not value:
                param = param.detach()
            # param.requires_grad = value
    def get_feats(self):
        return self.feats[...]
    def forward(self):
        return self.feats
    def save(self, path):
        # kwargs = self.get_kwargs()
        ckpt = {'state_dict': self.state_dict()}
        # print("save", self.state_dict().keys())
        # print(self.state_dict().keys())
        # if self.alphaMask is not None:
        #     alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
        #     ckpt.update({'alphaMask.shape':alpha_volume.shape})
        #     ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape(-1))})
        #     ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)