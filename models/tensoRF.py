from .tensorBase import *
from torchngp.encoding import get_encoder
from torchngp.ffmlp import FFMLP
from .py_ffmlp import *

class TensorVM(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVM, self).__init__(aabb, gridSize, device, **kargs)
        

    def init_svd_volume(self, res, device):
        self.plane_coef = torch.nn.Parameter(
            0.1 * torch.randn((3, self.app_n_comp + self.density_n_comp, res, res), device=device))
        self.line_coef = torch.nn.Parameter(
            0.1 * torch.randn((3, self.app_n_comp + self.density_n_comp, res, 1), device=device))
        self.basis_mat = torch.nn.Linear(self.app_n_comp * 3, self.app_dim, bias=False, device=device)

    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.line_coef, 'lr': lr_init_spatialxyz}, {'params': self.plane_coef, 'lr': lr_init_spatialxyz},
                         {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def compute_features(self, xyz_sampled):

        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach()
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach()

        plane_feats = F.grid_sample(self.plane_coef[:, -self.density_n_comp:], coordinate_plane, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        line_feats = F.grid_sample(self.line_coef[:, -self.density_n_comp:], coordinate_line, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        sigma_feature = torch.sum(plane_feats * line_feats, dim=0)
        
        
        plane_feats = F.grid_sample(self.plane_coef[:, :self.app_n_comp], coordinate_plane, align_corners=True).view(3 * self.app_n_comp, -1)
        line_feats = F.grid_sample(self.line_coef[:, :self.app_n_comp], coordinate_line, align_corners=True).view(3 * self.app_n_comp, -1)
        
        
        app_features = self.basis_mat((plane_feats * line_feats).T)
        
        return sigma_feature, app_features

    def compute_densityfeature(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_feats = F.grid_sample(self.plane_coef[:, -self.density_n_comp:], coordinate_plane, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        line_feats = F.grid_sample(self.line_coef[:, -self.density_n_comp:], coordinate_line, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        sigma_feature = torch.sum(plane_feats * line_feats, dim=0)
        
        
        return sigma_feature
    
    def compute_appfeature(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        
        plane_feats = F.grid_sample(self.plane_coef[:, :self.app_n_comp], coordinate_plane, align_corners=True).view(3 * self.app_n_comp, -1)
        line_feats = F.grid_sample(self.line_coef[:, :self.app_n_comp], coordinate_line, align_corners=True).view(3 * self.app_n_comp, -1)
        
        
        app_features = self.basis_mat((plane_feats * line_feats).T)
        
        
        return app_features
    

    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            # print(self.line_coef.shape, vector_comps[idx].shape)
            n_comp, n_size = vector_comps[idx].shape[:-1]
            
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            # print(vector_comps[idx].shape, vector_comps[idx].view(n_comp,n_size).transpose(-1,-2).shape, dotp.shape)
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            # print(vector_comps[idx].shape, vector_comps[idx].view(n_comp,n_size).transpose(-1,-2).shape, dotp.shape,non_diagonal.shape)
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        
        return self.vectorDiffs(self.line_coef[:,-self.density_n_comp:]) + self.vectorDiffs(self.line_coef[:,:self.app_n_comp])
    
    
    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        # plane_coef[0] = torch.nn.Parameter(
        #     F.interpolate(plane_coef[0].data, size=(res_target[1], res_target[0]), mode='bilinear',
        #                   align_corners=True))
        # line_coef[0] = torch.nn.Parameter(
        #     F.interpolate(line_coef[0].data, size=(res_target[2], 1), mode='bilinear', align_corners=True))
        # plane_coef[1] = torch.nn.Parameter(
        #     F.interpolate(plane_coef[1].data, size=(res_target[2], res_target[0]), mode='bilinear',
        #                   align_corners=True))
        # line_coef[1] = torch.nn.Parameter(
        #     F.interpolate(line_coef[1].data, size=(res_target[1], 1), mode='bilinear', align_corners=True))
        # plane_coef[2] = torch.nn.Parameter(
        #     F.interpolate(plane_coef[2].data, size=(res_target[2], res_target[1]), mode='bilinear',
        #                   align_corners=True))
        # line_coef[2] = torch.nn.Parameter(
        #     F.interpolate(line_coef[2].data, size=(res_target[0], 1), mode='bilinear', align_corners=True))

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        # self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        # self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

        scale = res_target[0]/self.line_coef.shape[2] #assuming xyz have the same scale
        plane_coef = F.interpolate(self.plane_coef.detach().data, scale_factor=scale, mode='bilinear',align_corners=True)
        line_coef  = F.interpolate(self.line_coef.detach().data, size=(res_target[0],1), mode='bilinear',align_corners=True)
        self.plane_coef, self.line_coef = torch.nn.Parameter(plane_coef), torch.nn.Parameter(line_coef)
        self.compute_stepSize(res_target)
        print(f'upsamping to {res_target}')


class TensorVMSplit(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVMSplit, self).__init__(aabb, gridSize, device, **kargs)
        


    def init_svd_volume(self, res, device):
        print("TensoRF")
        print(self.density_n_comp, self.app_n_comp)
        # exit()
        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)
        #
        if self.extra: 
            self.extra_plane, self.extra_line = self.init_one_svd_extra(self.density_n_comp, self.gridSize, 0.1, device)


    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

    def init_one_svd_extra(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.ones((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0])).to(torch.float32)))  #
            line_coef.append(
                torch.nn.Parameter(scale * torch.ones((1, n_component[i], gridSize[vec_id], 1))))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)
    
    

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz}, {'params': self.density_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz}, {'params': self.app_plane, 'lr': lr_init_spatialxyz},
                         {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def set_allgrads(self, value):
        # grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz},
        #         {'params': self.app_line, 'lr': lr_init_spatialxyz},
        #         {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        for param in self.density_line:
            param.requires_grad = value
        for param in self.app_line:
            param.requires_grad = value
        for param in self.basis_mat.parameters():
            param.requires_grad = value

        for param in self.density_plane:
            param.requires_grad = value
        for param in self.app_plane:
            param.requires_grad = value
        
        # if isinstance(self.renderModule, torch.nn.Module):
        #     for param in self.renderModule.parameters():
        #         param.requires_grad = value


    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]
            
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)
    
    def L1_loss_bwf(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + torch.mean(torch.abs(self.density_plane[idx])) + torch.mean(torch.abs(self.density_line[idx]))# + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        return total
    
    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2 + reg(self.density_line[idx]) * 1e-3
        return total
        
    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2 + reg(self.app_line[idx]) * 1e-3
        return total

    def compute_densityfeature(self, xyz_sampled):

        # plane + line basis
        # coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        # coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).view(3, -1, 1, 2)
        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)

        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)
        return sigma_feature

    def compute_appfeature(self, xyz_sampled):

        # plane + line basis
        # coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        # coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).view(3, -1, 1, 2)
        # print("compute_appfeature", coordinate_plane.grad_fn, coordinate_line.grad_fn)

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            # print("for", plane_coef_point.shape, line_coef_point.shape)
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)


        return self.basis_mat((plane_coef_point * line_coef_point).T)

    def compute_extrafeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.extra_plane)):
            plane_coef_point = F.grid_sample(self.extra_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(self.extra_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature




    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))


        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, second_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = second_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units

        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )


        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(second_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", second_aabb, "\ncorrect aabb", correct_aabb)
            second_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = second_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))


class TensorCP(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorCP, self).__init__(aabb, gridSize, device, **kargs)


    def init_svd_volume(self, res, device):
        self.density_line = self.init_one_svd(self.density_n_comp[0], self.gridSize, 0.2, device)
        self.app_line = self.init_one_svd(self.app_n_comp[0], self.gridSize, 0.2, device)
        self.basis_mat = torch.nn.Linear(self.app_n_comp[0], self.app_dim, bias=False).to(device)


    def init_one_svd(self, n_component, gridSize, scale, device):
        line_coef = []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component, gridSize[vec_id], 1))))
        return torch.nn.ParameterList(line_coef).to(device)

    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars


    def compute_densityfeature(self, xyz_sampled):

        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)


        line_coef_point = F.grid_sample(self.density_line[0], coordinate_line[[0]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.density_line[1], coordinate_line[[1]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.density_line[2], coordinate_line[[2]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])
        sigma_feature = torch.sum(line_coef_point, dim=0)
        
        
        return sigma_feature
    
    def compute_appfeature(self, xyz_sampled):

        coordinate_line = torch.stack(
            (xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)


        line_coef_point = F.grid_sample(self.app_line[0], coordinate_line[[0]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.app_line[1], coordinate_line[[1]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.app_line[2], coordinate_line[[2]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])

        return self.basis_mat(line_coef_point.T)
    

    @torch.no_grad()
    def up_sampling_Vector(self, density_line_coef, app_line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            density_line_coef[i] = torch.nn.Parameter(
                F.interpolate(density_line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))
            app_line_coef[i] = torch.nn.Parameter(
                F.interpolate(app_line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        return density_line_coef, app_line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.density_line, self.app_line = self.up_sampling_Vector(self.density_line, self.app_line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, second_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = second_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units

        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)


        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )

        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(second_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", second_aabb, "\ncorrect aabb", correct_aabb)
            second_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = second_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

    def density_L1(self):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + torch.mean(torch.abs(self.density_line[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + reg(self.density_line[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_line)):
            total = total + reg(self.app_line[idx]) * 1e-3
        return total


# class TensoRFnGP(TensorBase):
#     def __init__(self, aabb, gridSize, device, **kargs):
#         super(TensoRFnGP, self).__init__(aabb, gridSize, device, **kargs)
#         # self.density_shift = -1

#         # self.fea2denseAct = "relu"
#         # self.alphaMask_thres = 0.01
#         # print(self.fea2denseAct)
#         # exit(
        
#     def feature2density(self, density_features):
#         # return torch.relu(density_features)
#         return super().feature2density(density_features)

#     def init_svd_volume(self, res, device):
#         print("TensoRFnGP")
#         # encoding="hashgrid"
#         encoding = "frequency"
#         encoding_dir="sphere_harmonics"
#         num_layers=2
#         hidden_dim=256
#         geo_feat_dim=256
#         num_layers_color=3
#         hidden_dim_color=64
#         bound= 15
#         self.bound = bound
#         # sigma network
#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
#         self.geo_feat_dim = geo_feat_dim
#         self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)
#         self.encoder = self.encoder.to(device)
#         # self.encoder = self.encoder

#         # self.sigma_net = py_FFMLP(
#         #     input_dim=self.in_dim, 
#         #     output_dim=1 + self.geo_feat_dim,
#         #     hidden_dim=self.hidden_dim,
#         #     num_layers=self.num_layers,
#         #     device=device,
#         #     bias = True
#         # ).to(device)
#         self.sigma_net1 = py_FFMLP(
#             input_dim=self.in_dim, 
#             output_dim=self.hidden_dim,
#             hidden_dim=self.hidden_dim,
#             num_layers=5,
#             device=device,
#             bias = True
#         ).to(device)

#         self.sigma_net2 = py_FFMLP(
#             input_dim=self.hidden_dim+self.in_dim, 
#             output_dim=1 + self.geo_feat_dim,
#             hidden_dim=self.hidden_dim,
#             num_layers=3,
#             device=device,
#             bias = True
#         ).to(device)

#         self.sigma_net = nn.ModuleList([self.sigma_net1, self.sigma_net2])
        

#         # color network
#         self.num_layers_color = 2     
#         self.hidden_dim_color = hidden_dim_color
#         self.encoder_dir, self.in_dim_color = get_encoder(encoding_dir)
#         # print("in_dim", self.in_dim)
#         self.encoder_dir = self.encoder_dir.to(device)
#         # self.encoder_dir = self.encoder_dir.half()
#         self.in_dim_color += self.geo_feat_dim + 1 # a manual fixing to make it 32, as done in nerf_network.h#178
        
#         self.color_net = py_FFMLP(
#             input_dim=self.in_dim_color, 
#             output_dim=3,
#             hidden_dim=128,
#             num_layers=self.num_layers_color,
#             device=device,
#             bias = True
#         ).to(device)

#     def normalize_coord(self, xyz_sampled):
#         # print("not normalizing")
#         return xyz_sampled
        
#     @torch.cuda.amp.autocast(enabled=True)
#     def density(self, x):
#         # x = self.normalize_coord(x)
#         # x = x.half()
#         # x: [N, 3], in [-bound, bound]

#         x = self.encoder(x, bound=self.bound)
#         # print("x_embed in forward", torch.max(x), torch.min(x))

#         if torch.isnan(x).any():
#             raise ValueError("x is nan")
#         h = self.sigma_net1(x)
#         h = torch.cat([h, x], dim=-1)
#         h = self.sigma_net2(h)
#         if torch.isnan(h).any():
#             raise ValueError("h is nan")

#         # sigma = F.relu(h[..., 0])
#         sigma = h[..., 0]
#         # print("sigma in forward", torch.max(sigma), torch.min(sigma))
#         if torch.isnan(sigma).any():
#             raise ValueError("sigma is nan")
#         geo_feat = h[..., 1:]

#         return {
#             'sigma': sigma,
#             'geo_feat': geo_feat,
#         }

#     # allow masked inference
#     @torch.cuda.amp.autocast(enabled=True)
#     def color(self, x, d, mask=None, geo_feat=None, **kwargs):
#         # x = self.normalize_coord(x)
#         # x: [N, 3] in [-bound, bound]
#         # mask: [N,], bool, indicates where we actually needs to compute rgb.
#         # x = x.half()
#         # d = d.half()
#         if geo_feat is not None:
#             # geo_feat = geo_feat.half()
#             geo_feat = geo_feat.view(-1, self.geo_feat_dim)

#         if mask is not None:
#             rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
#             # in case of empty mask
#             if not mask.any():
#                 return rgbs
#             x = x[mask]
#             d = d[mask]
#             geo_feat = geo_feat[mask]


#         d = self.encoder_dir(d)

#         p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
#         h = torch.cat([d, geo_feat, p], dim=-1)
#         h = self.color_net(h)
        
#         # sigmoid activation for rgb
#         h = torch.sigmoid(h)

#         if mask is not None:
#             # print(h.shape, mask.shape, rgbs.shape)
#             rgbs[mask] = h.to(rgbs.dtype)
#         else:
#             rgbs = h

#         return rgbs



#     def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
#         # grad_vars = [
#         # {'name': 'encoding', 'params': list(self.encoder.parameters()), 'lr': lr_init_spatialxyz, 'weight_decay': 1e-6},
#         # {'name': 'net', 'params': list(self.sigma_net.parameters()) + list(self.color_net.parameters()), 'weight_decay': 1e-6, 'lr': lr_init_network}
#         # ]
#         grad_vars = [
#         {'name': 'encoding', 'params': list(self.encoder.parameters()), 'lr': lr_init_spatialxyz},
#         {'name': 'net', 'params': list(self.sigma_net.parameters()) + list(self.color_net.parameters()), 'lr': lr_init_network}
#         ]
#         return grad_vars

#     def set_allgrads(self, value):
#         # grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz},
#         #         {'params': self.app_line, 'lr': lr_init_spatialxyz},
#         #         {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
#         for param in self.density_line:
#             param.requires_grad = value
#         for param in self.app_line:
#             param.requires_grad = value
#         for param in self.basis_mat.parameters():
#             param.requires_grad = value

#         for param in self.density_plane:
#             param.requires_grad = value
#         for param in self.app_plane:
#             param.requires_grad = value
        



    
#     def density_L1(self):
#         total = 0
#         # print(self.state_dict().keys())
#         for param in self.sigma_net.parameters():
#             total = total + torch.mean(torch.abs(param))
#             # print(param.shape, torch.mean(torch.abs(param)))
#         # exit()
#         return total
#         # return torch.tensor(total)
#         total = 0
#         for idx in range(len(self.density_plane)):
#             total = total + torch.mean(torch.abs(self.density_plane[idx])) + torch.mean(torch.abs(self.density_line[idx]))# + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
#         return total

#     def compute_densityfeature(self, xyz_sampled):
#         # print("compute_desityfeature lll")
#         return self.density(xyz_sampled)["sigma"]

#     def compute_appfeature(self, xyz_sampled):
#         pass



#     @torch.cuda.amp.autocast(enabled=True)
#     def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1, skeleton_props=None, is_render_only=False):        
#         viewdirs = rays_chunk[:, 3:6]
#         # N_samples = N_samples//2
#         if ndc_ray:
#             xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
#             dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
#             rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
#             dists = dists * rays_norm
#             viewdirs = viewdirs / rays_norm
#         else:
#             xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
#             dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
#         viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        
#         if self.alphaMask is not None and self.data_preparation:
#             self.alphaMask.set_device(self.device)
#             alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
#             alpha_mask = alphas > 0
#             ray_invalid = ~ray_valid
#             ray_invalid[ray_valid] |= (~alpha_mask)
#             ray_valid = ~ray_invalid

#         # sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)

#         shape = xyz_sampled.shape
#         near, far = self.near_far
#         sample_dist = (far - near) / N_samples
#         N, num_steps = shape[0], shape[1]
#         rays_o = rays_chunk[:, :3]#.unsqueeze(1).repeat(1, upsample_steps, 1) # [N, t, 3]
#         rays_d = rays_chunk[:, 3:6]#.unsqueeze(1).repeat(1, upsample_steps, 1) # [N, t, 3]
#         self.density_scale = 1
#         use_ngp_code = True
#         xyz_sampled = self.clamp_pts(xyz_sampled)
#         sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
#         rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
        

#         if False and use_ngp_code:
#             self.bound_box_rate = torch.transpose(torch.tensor([
#                     # [-0.5, 0.8],[-0.1, 0.5],[-0.55, 0.55]
#                     [-1.0, 1.0],[-1.0, 1.0],[-1.0, 1.0]
#                 ], device = torch.device("cuda:0")
#             ), 0, 1)
            
#             # # extra state for cuda raymarching
#             # self.cuda_ray = cuda_ray
#             # self.skeleton_mode = skeleton_mode
#             # self.initiation = initiation
#             # self.mix_render = mix_render
#             # # self.bound_rate = torch.tensor([0.7, 0.6, 0.6], device=torch.device("cuda:0"))
#             self.bound_rate = torch.tensor([1.0, 1.0, 1.0], device=torch.device("cuda:0"))
#             near, far = near_far_from_bound(rays_o, rays_d, 1, type='cube', bound_rate = self.bound_rate, bound_box=self.ray_aabb)
#             # near, far = near_far_from_bound(rays_o, rays_d, self.bound, type='cube', bound_rate = self.bound_rate)

#             z_vals = torch.linspace(0.0, 1.0, num_steps, device=self.device).unsqueeze(0) # [1, T]
#             z_vals = z_vals.expand((N, num_steps)) # [N, T]
#             z_vals = near + (far - near) * z_vals # [N, T], in [near, far]

#             # perturb z_vals
#             sample_dist = (far - near) / num_steps
#             # # if perturb:
#             # z_vals = z_vals + (torch.rand(z_vals.shape, device=self.device) - 0.5) * sample_dist
#             # z_vals = z_vals.clamp(near, far) # avoid out of bounds xyzs.

#             # generate xyzs
#             xyz_sampled = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
#             xyz_sampled = self.clamp_pts(xyz_sampled)
#             # tmp = xyzs.clone().detach()
#         #skeleton parsing -> transforms
#         if not self.data_preparation:
#             if skeleton_props is not None:
#                 self.frame_pose = skeleton_props["frame_pose"]
#             if self.args.use_indivInv:
#                 transforms = self.skeleton.rotations_to_invs_fast(self.frame_pose, type=self.posetype)
#             else:
#                 transforms = self.skeleton.rotations_to_transforms_fast(self.frame_pose, type=self.posetype)
#             draw_joints = self.render_jointmask
#             if draw_joints:
#                 draw_mask = self.skeleton.draw_mask_all_cached(rays_chunk[:, :3], rays_chunk[:, 3:6], 0.05)
        
            

#         with torch.cuda.amp.autocast(enabled=True):

#             #################
#             # Point Casting : 1 #
#             #################
#             if not self.data_preparation:       
#                 exit()            

#                 xyz_sampled, viewdirs = self.caster(xyz_sampled, viewdirs, transforms, ray_valid)
                
#                 if self.args.mimik == "cycle":
#                     xyz_sampled, viewdirs = self.mimik_caster(xyz_sampled, viewdirs, transforms, ray_valid, i_frame = self.tmp_animframe_index)
#                 caster_weights = self.caster_origin.get_weights().view(N, num_steps, -1)



#             #################
#             # Density : 1 #
#             #################

#             density_outputs = self.density(xyz_sampled.reshape(-1, 3))
#             # for k, v in density_outputs.items():
#             #     density_outputs[k] = v.view(N, num_steps, -1)
#             sigma = self.feature2density(density_outputs['sigma']).view(N, num_steps, -1)
#             # sigma[ray_valid] = self.feature2density(density_outputs['sigma'][ray_valid]).view(N, num_steps, -1)
#             density_feats = density_outputs['geo_feat'].view(N, num_steps, -1)

#             coarsefine = True
            
#             #################
#             # Resample #
#             #################

#             deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T-1] #zvalsはある
#             deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
#             if use_ngp_code:
            
#                 # alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs['sigma'].squeeze(-1)) # [N, T]
#                 alphas = 1 - torch.exp(-deltas * self.density_scale * sigma.squeeze(-1)) # [N, T]
#                 alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-4], dim=-1) # [N, T+1]
#                 weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T]
#             else:
#                 alphas, weights, bg_weight = raw2alpha(sigma.view(shape[:-1]), dists * self.distance_scale)
#             if coarsefine:


#                 # # sample new z_vals
#                 N_samples = N_samples if N_samples>0 else self.nSamples
#                 # print("steps", N_samples, N, num_steps)
#                 upsample_steps = N_samples
#                 z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1]) # [N, T-1]
#                 second_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], upsample_steps, det=not self.training).detach() # [N, t]


#                 second_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * second_z_vals.unsqueeze(-1) # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
#                 second_dirs = rays_d.view(-1, 1, 3).expand_as(second_xyzs)
#                 second_xyzs = self.clamp_pts(second_xyzs)
#                 if not self.data_preparation:
#                     second_xyzs, second_dirs = self.caster(second_xyzs, second_dirs, transforms, ray_valid)
#                     second_caster_weights = self.caster_origin.get_weights().view(N, upsample_steps, -1)

#                 #################
#                 # Density : 2 #
#                 #################
#                 # xyz_sampled = self.clamp_pts(second_xyzs)
#                 second_density_outputs = self.density(second_xyzs.reshape(-1, 3))
#                 #second_sigmas = second_density_outputs['sigma'].view(N, upsample_steps) # [N, t]
#                 # for k, v in second_density_outputs.items():
#                 #     second_density_outputs[k] = v.view(N, upsample_steps, -1)
#                 second_sigma = self.feature2density(second_density_outputs['sigma']).view(N, upsample_steps, -1)
#                 second_density_feats = second_density_outputs['geo_feat'].view(N, upsample_steps, -1)


#                 #################
#                 # Concat #
#                 #################
#                 # re-order
#                 #zvals
#                 z_vals = torch.cat([z_vals, second_z_vals], dim=1) # [N, T+t]
#                 z_vals, z_index = torch.sort(z_vals, dim=1)

#                 #pos
#                 xyzs = torch.cat([xyz_sampled, second_xyzs], dim=1) # [N, T+t, 3]
#                 xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))

#                 # #density
#                 # for k in density_outputs:
#                 #     tmp_output = torch.cat([density_outputs[k], second_density_outputs[k]], dim=1)
#                 #     density_outputs[k] = torch.gather(tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))

#                 #sigma
#                 sigma = torch.cat([sigma, second_sigma], dim=1) # [N, T+t]
#                 sigma = torch.gather(sigma, dim=1, index=z_index.unsqueeze(-1))

#                 # density_feats
#                 density_feats = torch.cat([density_feats, second_density_feats], dim=1) # [N, T+t, 3]
#                 density_feats = torch.gather(density_feats, dim=1, index=z_index.unsqueeze(-1).expand_as(density_feats))


#                 if use_ngp_code:
#                     deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
#                     deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
#                     # alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs['sigma'].squeeze(-1)) # [N, T+t]
#                     alphas = 1 - torch.exp(-deltas * self.density_scale * sigma.squeeze(-1)) # [N, T+t]
#                     alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-4], dim=-1) # [N, T+t+1]
#                     weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]
#                 else:
#                     # print(dists.shape, sigma.shape, N)
#                     deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
#                     deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
#                     dists = torch.cat([dists, deltas], dim=1)
#                     dists = torch.gather(dists, dim=1, index=z_index)
#                     alphas, weights, bg_weight = raw2alpha(sigma.view(N, -1), dists * self.distance_scale)


#                 #cast_point_
#                 if not self.data_preparation:
#                     self.caster_weights = torch.cat([caster_weights, second_caster_weights], dim=1) # [N, T+t, 3]
#                     self.caster_weights = torch.gather(xyzs, dim=-1, index=z_index.unsqueeze(-1).expand_as(self.caster_weights))
#                     weights_sum = torch.sum(self.caster_weights, dim=-1)
#                     self.bg_alpha = clip_weight(weights_sum, thresh = 0.2).view(shape[0], -1)
#                     castweight_mask = self.bg_alpha.squeeze(-1) > 0.8
#                     self.occupancy = 1 - alphas
#                     self.raw_sigma = weights

#                     if self.args.free_opt9:
#                         id_weight_render = torch.argmin(torch.abs(xyzs[...,2]-1), dim = -1)
#                         weights_sum = torch.clamp(weights_sum, min=1e-7)
#                         self.weights_sum = weights_sum
#                         weights_color =  self.caster_weights/weights_sum.unsqueeze(1)
#                         weights_color = weights_color.view(shape[0], shape[1], weights_color.shape[-1])
#                         weights_color = weights_color * self.bg_alpha.unsqueeze(-1)
#                         weights_color = torch.gather(weights_color, 1, id_weight_render.view(-1,1,1).expand(-1,-1,weights_color.shape[-1]))
#                         self.render_weights = weights_color.squeeze(1)


#                 dirs = torch.cat([viewdirs, second_dirs], dim=1) # [N, T+t, 3]
#                 dirs = torch.gather(dirs, dim=1, index=z_index.unsqueeze(-1).expand_as(dirs))
#             else:
#                 xyzs = xyz_sampled.contiguous()
#                 dirs = viewdirs.contiguous()
#                 density_feats = density_feats.contiguous()
#                 weights = weights.contiguous()

#             mask = weights > 1e-4 # hard coded
#             if not self.data_preparation:
#                 aabb_mask = self.aabb_mask(xyzs)
#                 alpha_mask = self.alpha_mask(xyzs).view(shape[0],shape[1])
#                 mask = mask & castweight_mask & aabb_mask & alpha_mask


#             #################
#             # RGB Render #
#             #################


#             # rgbs = self.color(xyzs.view(-1,3), dirs.view(-1,3), mask=mask.view(-1), **density_outputs)
#             rgbs = self.color(xyzs.view(-1,3), dirs.view(-1,3), mask=mask.view(-1), geo_feat = density_feats.view(-1,1))


#             rgbs = rgbs.view(N, -1, 3) # [N, T+t, 3]
#             # alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

#             acc_map = torch.sum(weights, -1)
#             rgb_map = torch.sum(weights[..., None] * rgbs, -2)

#             if white_bg or (is_train and torch.rand((1,))<0.5):
#                 rgb_map = rgb_map + (1. - acc_map[..., None])

#             rgb_map = rgb_map.clamp(0,1)

#             with torch.no_grad():
#                 depth_map = torch.sum(weights * z_vals, -1)
#                 depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

#             if not self.data_preparation:
#                 if draw_joints:
#                     rgb_map[draw_mask] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=rgb_map.device)

#             return rgb_map, depth_map # rgb, sigma, alpha, weight, bg_weight







#     @torch.no_grad()
#     def shrink(self, second_aabb):
#         print("====> shrinking ...")
#         xyz_min, xyz_max = second_aabb
#         t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units

#         t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
#         b_r = torch.stack([b_r, self.gridSize]).amin(0)



#         if not torch.all(self.alphaMask.gridSize == self.gridSize):
#             t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
#             correct_aabb = torch.zeros_like(second_aabb)
#             correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
#             correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
#             print("aabb", second_aabb, "\ncorrect aabb", correct_aabb)
#             second_aabb = correct_aabb

#         newSize = b_r - t_l
#         self.aabb = second_aabb
#         self.update_stepSize((newSize[0], newSize[1], newSize[2]))




#     @torch.no_grad()
#     def up_sampling_VM(self, plane_coef, line_coef, res_target):

#         for i in range(len(self.vecMode)):
#             vec_id = self.vecMode[i]
#             mat_id_0, mat_id_1 = self.matMode[i]
#             plane_coef[i] = torch.nn.Parameter(
#                 F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
#                               align_corners=True))
#             line_coef[i] = torch.nn.Parameter(
#                 F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))


#         return plane_coef, line_coef

#     @torch.no_grad()
#     def upsample_volume_grid(self, res_target):
#         pass
#         # self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
#         # self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

#         self.update_stepSize(res_target)
#         print(f'upsamping to {res_target}')


#     def vectorDiffs(self, vector_comps):
#         total = 0
        
#         for idx in range(len(vector_comps)):
#             n_comp, n_size = vector_comps[idx].shape[1:-1]
            
#             dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
#             non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
#             total = total + torch.mean(torch.abs(non_diagonal))
#         return total

#     def vector_comp_diffs(self):
#         return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)
#         # if ray_valid.any():
                    
#     def TV_loss_density(self, reg):
#         total = 0
#         for idx in range(len(self.density_plane)):
#             total = total + reg(self.density_plane[idx]) * 1e-2 + reg(self.density_line[idx]) * 1e-3
#         return total
        
#     def TV_loss_app(self, reg):
#         total = 0
#         for idx in range(len(self.app_plane)):
#             total = total + reg(self.app_plane[idx]) * 1e-2 + reg(self.app_line[idx]) * 1e-3
#         return total




        #     xyz_sampled = self.normalize_coord(xyz_sampled)

        #     xyz_sampled = xyz_sampled.reshape(shape[0],shape[1], 3)
        #     # print(xyz_sampled.shape, xyz_sampled.reshape(-1,3).shape)
        #     # print(xyz_sampled[ray_valid].shape)
        #     sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])
            

        #     validsigma = self.feature2density(sigma_feature)
        #     sigma[ray_valid] = validsigma

        #     # weights = weights[:weights.shape[0]//2,:].reshape(xyz_sampled.shape[0], xyz_sampled.shape[1], weights.shape[-1])

        #     # outside  = weights.sum(dim=-1) < 0.001
        #     # inside = ~ outside

        #     # sigma[outside] = -0.000
        #     # sigma[inside] = 0.2;
        #     if not self.data_preparation and save_npz:
        #         save_npz["sigma"] = sigma.cpu().numpy()
        #     self.sigma = sigma
        #     # exit("amkingmasking")



        # alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)
        # weight = weight * bg_alpha

        # app_mask = weight > self.rayMarch_weight_thres

        # # Compute_alpha
        # if app_mask.any():
        #     app_features = self.compute_appfeature(xyz_sampled[app_mask])    
        #     valid_rgbs = self.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features)
        #     rgb[app_mask] = valid_rgbs
        #     # weight_slice =  weights.reshape(rgb.shape[0], -1, weights.shape[-1]).shape[1]//2
        #     # rgb[...,1:] = 0
        #     # rgb[...,0] = weights.reshape(rgb.shape[0], -1, weights.shape[-1])[:,:,2] * 1000
            

            
        #     # rgb[inside][...,0] = 1.0;
        #     # rgb[inside][...,1:] = 0.0;
        #     # rgb[inside] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=torch.device('cuda:0'));
        #     if not self.data_preparation and save_npz:
        #         rgb[:,:,0] = weights[:,:,0];
        #         save_npz["rgb"] = rgb.cpu().numpy()
        


        # if not self.data_preparation and save_npz:
        #     itr = 0;    
        #     files = glob.glob("./data_point_cloud_*.npz")
        #     if len(files) > 0:
        #         itr = int(files[-1].split(".")[1].split("_")[-1]) + 1
        #     np.savez("./data_point_cloud_"+str(itr)+".npz", **save_npz)
        

        # acc_map = torch.sum(weight, -1)
        # rgb_map = torch.sum(weight[..., None] * rgb, -2)

        # if white_bg or (is_train and torch.rand((1,))<0.5):
        #     rgb_map = rgb_map + (1. - acc_map[..., None])

        # rgb_map = rgb_map.clamp(0,1)

        # with torch.no_grad():
        #     depth_map = torch.sum(weight * z_vals, -1)
        #     depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        # if not self.data_preparation:
        #     if draw_joints:
        #         rgb_map[draw_mask] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=rgb_map.device)


        # return rgb_map, depth_map # rgb, sigma, alpha, weight, bg_weight

    # def init_one_svd(self, n_component, gridSize, scale, device):
    #     plane_coef, line_coef = [], []
    #     for i in range(len(self.vecMode)):
    #         vec_id = self.vecMode[i]
    #         mat_id_0, mat_id_1 = self.matMode[i]
    #         plane_coef.append(torch.nn.Parameter(
    #             scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
    #         line_coef.append(
    #             torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

    #     return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

    # def init_one_svd_extra(self, n_component, gridSize, scale, device):
    #     plane_coef, line_coef = [], []
    #     for i in range(len(self.vecMode)):
    #         vec_id = self.vecMode[i]
    #         mat_id_0, mat_id_1 = self.matMode[i]
    #         plane_coef.append(torch.nn.Parameter(
    #             scale * torch.ones((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0])).to(torch.float32)))  #
    #         line_coef.append(
    #             torch.nn.Parameter(scale * torch.ones((1, n_component[i], gridSize[vec_id], 1))))

    #     return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)
