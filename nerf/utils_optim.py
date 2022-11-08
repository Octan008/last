from operator import truth
import os
import glob
from tabnanny import check
from this import d
import tqdm
import random
import warnings
import tensorboardX

import numpy as np
import gc
import pandas as pd

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version

from .provider import nerf_matrix_to_ngp

from .render_util import *
from .utils import * 
import math

from .log_utils import * 

animate = True


class OptimTrainer(object):
    def __init__(self, 
                name, # name of this experiment
                conf, # extra conf
                model, # network 
                criterion=None, # loss function, if None, assume inline implementation in train_step
                optimizer=None, # optimizer
                ema_decay=None, # if use EMA, set the decay
                lr_scheduler=None, # scheduler
                metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                local_rank=0, # which GPU am I
                world_size=1, # total num of GPUs
                device=None, # device to use, usually setting to None is OK. (auto choose device)
                mute=False, # whether to mute all print
                fp16=False, # amp optimize level
                eval_interval=1, # eval once every $ epoch
                max_keep_ckpt=2, # max num of saved ckpts in disk
                workspace='workspace', # workspace to save logs & ckpts
                best_mode='min', # the smaller/larger result, the better
                use_loss_as_metric=True, # use loss as the first metric
                report_metric_at_train=False, # also report metrics at training
                use_checkpoint="latest", # which ckpt to use at init time
                use_tensorboardX=True, # whether to use tensorboard for logging
                scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                cast_model=None,
                pose_model=None,
                skeleton=None,
                use_posenet=False,
                old_checkpoint = 0
                ):
        
        self.name = name
        self.conf = conf
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.use_posenet = use_posenet

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        self.pose_model = None
        if pose_model is not None:
            pose_model.to(self.device)
            self.pose_model = pose_model
        self.skeleton=skeleton
        self.joints = listify_skeleton(self.skeleton)

        self.from_scratch = True

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            if pose_model is not None:
                self.optimizer = optimizer(self.model, self.pose_model)
                # for params in self.pose_model.parameters():
                #     params.requires_grad = True

                print(self.optimizer)
            else:
                self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            if pose_model is not None:
                self.ema = ExponentialMovingAverage(list(self.model.parameters()) + list(self.pose_model.parameters()), decay=ema_decay)
            else: 
                self.ema = ExponentialMovingAverage(list(self.model.parameters()), decay=ema_decay)
            # if self.model.pose_only:
            #     self.ema = ExponentialMovingAverage(list(self.pose_model.parameters()), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        for param in model.sigma_net.parameters():
            param.requires_grad = False
        for param in model.color_net.parameters():
            param.requires_grad = False

        # variable init
        self.epoch = 1
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "valid_loss_pose_local": [],
            "valid_loss_pose_global": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        animation_conf = self.conf["path"]+"/transforms.json"
        
        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")
            self.skel_ckpt_path = os.path.join(self.workspace, 'skel_checkpoints')
            
            if conf["heatmap"]:
                self.skel_ckpt_path = os.path.join(self.workspace, 'olds/old_skel_checkpoints_'+str(old_checkpoint))
            self.skel_best_path = f"{self.skel_ckpt_path}/skel_{self.name}.pth.tar"
            os.makedirs(self.skel_ckpt_path, exist_ok=True)

        self.nerf_ckpt_path = "shark/checkpoints/ngp_ep0300.pth.tar"
        self.nerf_best_path = f"{self.nerf_ckpt_path}/{self.name}.pth.tar"
        self.log("[INFO] Loading NeRF checkpoin ...")
        self.load_nerf_checkpoint(self.nerf_ckpt_path)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "best":
                if os.path.exists(self.skel_best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.skel_best_path)
                else:
                    self.log(f"[INFO] {self.skel_best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	

    def train_step(self, data):
        images = data["image"] # [B, H, W, 3/4]
        poses = data["pose"] # [B, 4, 4]
        intrinsics = data["intrinsic"] # [B, 3, 3]
        frame_id = data["frame_id"]
        skeleton_pose=None
        skeleton_pose = self.pose_model(frame_id)
        use_posenet = self.use_posenet
        if use_posenet:
            skeleton_pose = self.skeleton.transformNet(skeleton_pose)
        # update grid
        if self.model.cuda_ray and not self.model.initiation:
            with torch.cuda.amp.autocast(enabled=self.fp16):
                self.model.update_extra_state(skeleton_pose=skeleton_pose)
        

        # sample rays 
        B, H, W, C = images.shape
        rays_o, rays_d, inds = get_rays(poses, intrinsics, H, W, self.conf['num_rays'])
        images = torch.gather(images.reshape(B, -1, C), 1, torch.stack(C*[inds], -1)) # [B, N, 3/4]

        # train with random background color if using alpha mixing
        bg_color = torch.ones(3, device=self.device) # [3], fixed white background
        #bg_color = torch.rand(3, device=self.device) # [3], frame-wise random.
        # bg_color = torch.rand_like(images[..., :3]) # [N, 3], pixel-wise random.

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=True, skeleton_pose=skeleton_pose, **self.conf)
        if self.conf["weight_only"]:
            w = outputs["weight"]
            gw = outputs["gt_weight"]
            loss = self.criterion(w, gw)
            return w, gw, loss


        pred_rgb = outputs['rgb']
        if outputs["xyzs"] is not None:
            loss = self.criterion(outputs['gt_xyzs'], outputs["xyzs"]) + self.criterion(outputs['gt_dirs'], outputs["dirs"])
        else:
            loss = self.criterion(pred_rgb, gt_rgb)
        del outputs

        return pred_rgb, gt_rgb, loss

    def eval_step(self, data, draw_skeleton=False, heatmap=False):
        images = data["image"] # [B, H, W, 3/4]
        poses = data["pose"] # [B, 4, 4]
        intrinsics = data["intrinsic"] # [B, 3, 3]
        frame_id = data["frame_id"]
        skeleton_pose = self.pose_model(frame_id)
        use_posenet = self.use_posenet
        if use_posenet:
            skeleton_pose = self.skeleton.transformNet(skeleton_pose)
            # update grid
        if self.model.cuda_ray and not self.model.initiation:
            with torch.cuda.amp.autocast(enabled=self.fp16):
                self.model.update_extra_state(skeleton_pose=skeleton_pose)

        # sample rays 
        B, H, W, C = images.shape
        rays_o, rays_d, _ = get_rays(poses, intrinsics, H, W, -1)

        bg_color = torch.ones(3, device=self.device) # [3]
        # eval with fixed background color
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        if heatmap:
            outputs = self.model.render_heat(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, skeleton_pose=skeleton_pose, **self.conf)
            pred_rgb = outputs['rgb'].reshape(B, H, W, -1)
            loss = self.criterion(pred_rgb, gt_rgb)
            return pred_rgb, gt_rgb, outputs['weights'].reshape(B, H, W, -1), loss
            
        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, skeleton_pose=skeleton_pose, **self.conf)


        pred_rgb = outputs['rgb'].reshape(B, H, W, -1)
        pred_depth = outputs['depth'].reshape(B, H, W)

        loss = self.criterion(pred_rgb, gt_rgb)


        del outputs, rays_o, rays_d, bg_color

        return pred_rgb, pred_depth, gt_rgb, loss


    def joint_mask(self, data, radius = 0.01, joint = None):
        images = data["image"] # [B, H, W, 3/4]
        poses = data["pose"] # [B, 4, 4]
        intrinsics = data["intrinsic"] # [B, 3, 3]

        B, H, W, C = images.shape


        rays_o, rays_d, _ = get_rays(poses, intrinsics, H, W, -1)
        prefix = rays_o.shape[:-1]
        if joint is None:
            intersect = self.skeleton.draw_mask_all(rays_o, rays_d, radius)
        else:
            intersect = joint.draw_mask(rays_o, rays_d, radius)
        
        intersect = intersect.view(*prefix, 1)
        intersect = intersect.reshape(B, H, W)
        return intersect
    def box_mask(self, data):
        images = data["image"] # [B, H, W, 3/4]
        poses = data["pose"] # [B, 4, 4]
        intrinsics = data["intrinsic"] # [B, 3, 3]

        B, H, W, C = images.shape


        rays_o, rays_d, _ = get_rays(poses, intrinsics, H, W, -1)
        prefix = rays_o.shape[:-1]

        intersect = self.model.bound_box_mask(rays_o, rays_d)
        
        intersect = intersect.view(*prefix, 1)
        intersect = intersect.reshape(B, H, W)
        return intersect
    def composite_heatmaps(self, weights):
        # print(weights.shape)
        weights = weights[0]
        weights = weights.reshape(weights.shape[0], weights.shape[1], len(self.joints), -1)
        # print(weights.shape)
        images = torch.sum(weights, dim=-1)
        print(weights.shape, images.shape)
        return images


    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):  
        poses = data["pose"] # [B, 4, 4]
        intrinsics = data["intrinsic"] # [B, 3, 3]
        H, W = int(data['H'][0]), int(data['W'][0]) # get the target size...

        B = poses.shape[0]

        rays_o, rays_d, _ = get_rays(poses, intrinsics, H, W, -1)
        prefix = rays_o.shape[:-1]


        if bg_color is not None:
            bg_color = bg_color.to(self.device)
        
        # outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb, **self.conf)
        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb, joints = self.joints, **self.conf)
        radius = 0.01
        if self.skeleton is not None or False:
            intersect = self.skeleton.draw_mask_all(rays_o, rays_d, radius)
            intersect = intersect.view(*prefix, 1)
            intersect = intersect.reshape(B, H, W)



        pred_rgb = outputs['rgb'].reshape(B, H, W, -1)

        if self.skeleton is not None:
            pred_rgb[intersect] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=torch.device('cuda:0'))

        pred_depth = outputs['depth'].reshape(B, H, W)

        return pred_rgb, pred_depth


    def save_mesh(self, save_path=None, resolution=256, threshold=10):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sdfs = self.model.density(pts.to(self.device))
            return sdfs

        bounds_min = torch.FloatTensor([-self.model.bound] * 3)
        bounds_max = torch.FloatTensor([self.model.bound] * 3)

        vertices, triangles = extract_geometry(bounds_min, bounds_max, resolution=resolution, threshold=threshold, query_func=query_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        if self.conf["heatmap"]:
            print("heatmap")
            self.heatmap_one_epoch(valid_loader, frames=[i for i in range(30)])
            exit()

        if self.model.cuda_ray:
            self.model.mark_untrained_grid(train_loader.dataset.poses, train_loader.dataset.intrinsic)
        
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            
            # if epoch==1 or epoch == 251:
            #     # self.train_one_epoch(train_loader)
            #     if self.ema is not None:
            #         self.ema.update()
            #     self.evaluate_one_epoch(valid_loader)
            #     exit()

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0 and self.epoch % self.eval_interval  == 0:
                # exit()
                self.save_optim_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                # exit()
                self.evaluate_one_epoch(valid_loader)
                self.save_optim_checkpoint(full=False, best=True)

            # gc.collect()
            # torch.cuda.empty_cache()

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()


    def evaluate(self, loader):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader)
        self.use_tensorboardX = use_tensorboardX


    def render(self, loader, save_path=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'render_results')

        os.makedirs(save_path, exist_ok=True)
        animation_conf = self.conf["path"]+"/transforms.json"
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        #rendering settings
        num_render = self.conf["num_renderpose"]
        width = 180*num_render/40
        if self.conf["circle_render"]:
            start = -width
            end = width
        else:
            start = -180+9*5
            end = -180+9*5
        
        render_poses = torch.stack(
            [
                torch.from_numpy(pose_spherical(angle, -30.0, 10.0))
                for angle in np.linspace(start, end, num_render + 1)[:-1]
            ],
            0,
        ).to(torch.float32)
        
        with torch.no_grad():
            if self.conf["skeleton"]:
                # update grid
                if self.model.cuda_ray:
                    with torch.cuda.amp.autocast(enabled=self.fp16):
                        # pass
                        self.model.reset_extra_state()
                        self.model.update_extra_state()

            if self.conf["skeleton"]:
                frames = json.load(open(animation_conf, 'r'))["frames"]

            

            for i, pose in enumerate(render_poses):
                data = {"pose":torch.from_numpy(nerf_matrix_to_ngp(pose)).unsqueeze(0), "intrinsic": loader.__iter__().next()["intrinsic"],
                        "H": loader.__iter__().next()["H"], "W":loader.__iter__().next()["W"]}# update grid

                if not (i in [j for j in range(65,75)]):
                    continue
                data = self.prepare_data(data)

                if self.conf["skeleton"]:
                    # #lego-animation
                    # rot = make_transform(angle = [0,mod_angles[i], 0])
                    # skeleton.get_children()[0].apply_transform(rot.cpu().numpy())
                    for j in self.skeleton.get_children():
                        apply_animation(frames[i]["animation"], j)


                data["joints" ]= self.joints
                data["skeleton"] = self.skeleton

                if self.conf["skeleton"]:
                    if self.model.cuda_ray:
                        with torch.cuda.amp.autocast(enabled=self.fp16):
                            # self.model.reset_extra_state()
                            self.model.update_extra_state(decay=0.9, joints=self.joints)

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth = self.test_step(data)     
                    # preds, preds_depth = self.eval_step(data)     
                data["image"] = preds
                intersect = self.joint_mask(data)
                preds[intersect] = preds[intersect]* 0.5 + torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=torch.device('cuda:0')) * 0.5    
                intersect = self.box_mask(data)
                preds[intersect] = preds[intersect]* 0.8 + torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32, device=torch.device('cuda:0')) * 0.2

                path = os.path.join(save_path, f'{i:04d}.png')
                path_depth = os.path.join(save_path, f'{i:04d}_depth.png')

                #self.log(f"[INFO] saving test image to {path}")

                cv2.imwrite(path, cv2.cvtColor((preds[0].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                cv2.imwrite(path_depth, (preds_depth[0].detach().cpu().numpy() * 255).astype(np.uint8))

                pbar.update(loader.batch_size)

        self.log(f"==> Finished Test.")

    def test(self, loader, save_path=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()
        with torch.no_grad():

            # update grid
            if self.model.cuda_ray:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            for i, data in enumerate(loader):
                
                data = self.prepare_data(data)

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth = self.test_step(data)                
                
                path = os.path.join(save_path, f'{i:04d}.png')
                path_depth = os.path.join(save_path, f'{i:04d}_depth.png')

                #self.log(f"[INFO] saving test image to {path}")

                cv2.imwrite(path, cv2.cvtColor((preds[0].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                cv2.imwrite(path_depth, (preds_depth[0].detach().cpu().numpy() * 255).astype(np.uint8))

                pbar.update(loader.batch_size)

        self.log(f"==> Finished Test.")
    

    def prepare_data(self, data):
        if isinstance(data, list):
            for i, v in enumerate(data):
                if isinstance(v, np.ndarray):
                    data[i] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[i] = v.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[k] = v.to(self.device, non_blocking=True)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device, non_blocking=True)
        else: # is_tensor, or other similar objects that has `to`
            data = data.to(self.device, non_blocking=True)

        return data

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # # update grid
        # if self.model.cuda_ray:
        #     with torch.cuda.amp.autocast(enabled=self.fp16):
        #         self.model.update_extra_state()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            # if self.local_step == 3:
            #     exit()
            
            self.local_step += 1
            self.global_step += 1
            
            data = self.prepare_data(data)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)
                # if torch.isnan(loss).any():
                #     self.save_optim_checkpoint(full=True, best=False)
                #     raise ValueError("loss error")
                
            self.scaler.scale(loss).backward()

            if self.model.skeleton_mode:
                self.scaler.step(self.optimizer)
                self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            del loss
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        #ここでupdate
        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)
        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")
        del preds
        # gc.collect()
        # torch.cuda.empty_cache()
        



    def evaluate_one_epoch(self, loader, joint = None, log_image_summery=False, heatmap=False):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        total_loss = 0
        total_loss_pose_local, total_loss_pose_global = 0, 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0
            
            # update grid
            if self.model.cuda_ray:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            for i, data in enumerate(loader):
                # if not i in [0, 10, 16]:
                #     continue
                self.local_step += 1
                data = self.prepare_data(data)

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    #skeleton preparation
                    
                    for j in self.skeleton.get_children():
                        apply_animation(data["frame_pose"], j)

                    gt_skeleton_pose = self.skeleton.get_listed_rotations()
                    # gt_skeleton_transform = self.skeleton.transformNet(gt_skeleton_pose)
                    data["skeleton"] = self.skeleton
                    intersect = self.joint_mask(data)
                    tmp = []
                    for ii in range(gt_skeleton_pose.shape[0]):
                        tmp.append(euler_to_quaternion(gt_skeleton_pose[ii]))
                    gt_skeleton_pose = torch.stack(tmp, dim=0)

                    optim_skeleton_pose = self.pose_model(data["frame_id"]) # [J, 4]
                    self.skeleton.transformNet(optim_skeleton_pose, type="quaternion")
                    data["skeleton"] = self.skeleton
                    intersect2 = self.joint_mask(data)

                    num_joints = gt_skeleton_pose.shape[0]

                    gap_skeleton_pose = gt_skeleton_pose - optim_skeleton_pose
            
                    loss_pose_local = torch.sum(torch.flatten(gap_skeleton_pose*gap_skeleton_pose),dim=0)
                    loss_pose_local = loss_pose_local.item() / num_joints

                    
                    #skeleton preparation

                    preds, preds_depth, truths, loss = self.eval_step(data)


                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)
                
                loss_val = loss.item()
                total_loss += loss_val
                total_loss_pose_local += loss_pose_local
                # total_loss_pose_global += loss_pose_global

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    for metric in self.metrics:
                        metric.update(preds, truths)

                    # save image
                    save_path = os.path.join(self.workspace, "validation", f'{self.name}_{self.epoch:04d}_{self.local_step:04d}.png')
                    save_path_depth = os.path.join(self.workspace, "validation", f'{self.name}_{self.epoch:04d}_{self.local_step:04d}_depth.png')
                    #save_path_gt = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch:04d}_{self.local_step:04d}_gt.png')

                    #self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    #truths が gt
                    if log_image_summery:
                        cv2.imwrite(save_path, cv2.cvtColor((preds[0].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                        cv2.imwrite(save_path_depth, (preds_depth[0].detach().cpu().numpy() * 255).astype(np.uint8))
                    else:
                        props={
                            "title": f'epoch: {self.epoch:04d} frame: {self.local_step:04d}',
                            "loss": f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})",
                            "loss_pose_local": loss_pose_local,
                            # "loss_pose_global": loss_pose_global,
                        }

                        preds[intersect] = preds[intersect]* 0.5 + torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=torch.device('cuda:0')) * 0.5
                        preds[intersect2] = preds[intersect2]* 0.5 + torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=torch.device('cuda:0')) * 0.5
                        intersect = self.box_mask(data)
                        # preds[intersect] = preds[intersect]* 0.7 + torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32, device=torch.device('cuda:0')) * 0.3 


                        save_log_im(tensor2imnp(preds[0]), tensor2imnp(truths[0]), save_path, props=props)
                        save_log_im(tensor2imnp(preds_depth[0]), tensor2imnp(truths[0]), save_path_depth, props=props)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)

                # del preds, preds_depth, truths, data, loss
                # gc.collect()
                # torch.cuda.empty_cache()

            


        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)
        self.stats["valid_loss_pose_local"].append(total_loss_pose_local / self.local_step)
        # self.stats["valid_loss_pose_global"].append(total_loss_pose_global / self.local_step)
        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")
        del total_loss, total_loss_pose_local
        # gc.collect()
        # torch.cuda.empty_cache()
        
    def heatmap_one_epoch(self, loader, joint = None, log_image_summery=False, frames = [0]):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        total_loss = 0
        total_loss_pose_local, total_loss_pose_global = 0, 0


        self.model.eval()

        with torch.no_grad():
            self.local_step = 0
            
            # update grid
            if self.model.cuda_ray:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            for i, data in enumerate(loader):
                if not i in frames:
                    continue
                # for jj, joint in enumerate(self.joints):
                # self.local_step += 1
                data = self.prepare_data(data)

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    #skeleton preparation
                    
                    for j in self.skeleton.get_children():
                        apply_animation(data["frame_pose"], j)

                    gt_skeleton_pose = self.skeleton.get_listed_rotations()
                    # gt_skeleton_transform = self.skeleton.transformNet(gt_skeleton_pose)
                    data["skeleton"] = self.skeleton
                    intersect = self.joint_mask(data)
                    tmp = []
                    for ii in range(gt_skeleton_pose.shape[0]):
                        tmp.append(euler_to_quaternion(gt_skeleton_pose[ii]))
                    gt_skeleton_pose = torch.stack(tmp, dim=0)

                    optim_skeleton_pose = self.pose_model(data["frame_id"]) # [J, 4]
                    self.skeleton.transformNet(optim_skeleton_pose, type="quaternion")
                    data["skeleton"] = self.skeleton
                    intersect2 = self.joint_mask(data)

                    num_joints = gt_skeleton_pose.shape[0]

                    gap_skeleton_pose = gt_skeleton_pose - optim_skeleton_pose
            
                    loss_pose_local = torch.sum(torch.flatten(gap_skeleton_pose*gap_skeleton_pose),dim=0)
                    loss_pose_local = loss_pose_local.item() / num_joints
                    
                    #skeleton preparation

                    preds, truths, weights, loss = self.eval_step(data, heatmap = True)
                    heats = self.composite_heatmaps(weights)
                    heats_min = torch.min(heats)
                    heats_max = torch.max(heats)
                
                loss_val = loss.item()
                total_loss += loss_val
                total_loss_pose_local += loss_pose_local
                # total_loss_pose_global += loss_pose_global

                    # only rank = 0 will perform evaluation.
                for jj, joint in enumerate(self.joints):
                    if self.local_rank == 0:
                        # save image
                        self.local_step += 1
                        # _preds= preds.clone().detach()
                        save_path = os.path.join(self.workspace, "heatmap", f'heat_{self.name}_{self.epoch:04d}_{i:04d}_{jj:04d}.png')
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        #truths が gt
                        props={
                            "title": f'epoch: {self.epoch:04d} frame: {self.local_step:04d}',
                            "joint_name" : f"{joint.name}",
                            "loss": f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})",
                            "loss_pose_local": f"loss_pose_local = {loss_pose_local: 4f}",
                            # "loss_pose_global": loss_pose_global,
                        }
                        heatmap = heats[:,:,jj].unsqueeze(-1)
                        # print(torch.max(heatmap), torch.min(heatmap))
                        heatmap = (heatmap - heats_min)/(heats_max - heats_min)
                        # print(torch.max(heatmap), torch.min(heatmap))
                        heat_color = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=torch.device('cuda:0')).unsqueeze(0).unsqueeze(0).repeat(heatmap.shape[0], heatmap.shape[1], 1)
                        bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=torch.device('cuda:0')).unsqueeze(0).unsqueeze(0).repeat(heatmap.shape[0], heatmap.shape[1], 1)
                        heatmap = heatmap * heat_color + bg_color*(1-heatmap)
                        heat_rate = 0.7
                        heatmap = heatmap * heat_rate + preds[0] * (1.0 - heat_rate)
                        heatmap[intersect[0]] = heatmap[intersect[0]]* 0.5 + torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=torch.device('cuda:0')) * 0.5
                        heatmap[intersect2[0]] = heatmap[intersect2[0]]* 0.5 + torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=torch.device('cuda:0')) * 0.5
                        intersect3 = self.joint_mask(data, radius = 0.05, joint=joint)

                        heatmap[intersect3[0]] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=torch.device('cuda:0'))
                        # save_log_im(tensor2imnp(_preds[0]), tensor2imnp(truths[0]), save_path, props=props)
                        save_log_im(tensor2imnp(heatmap), tensor2imnp(truths[0]), save_path, props=props)



    
        # average_loss = total_loss / self.local_step
        # self.stats["valid_loss"].append(average_loss)
        # self.stats["valid_loss_pose_local"].append(total_loss_pose_local / self.local_step)
        # # self.stats["valid_loss_pose_global"].append(total_loss_pose_global / self.local_step)
        # if self.local_rank == 0:
        #     pbar.close()
        #     if not self.use_loss_as_metric and len(self.metrics) > 0:
        #         result = self.metrics[0].measure()
        #         self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
        #     else:
        #         self.stats["results"].append(average_loss) # if no metric, choose best by min loss

        #     for metric in self.metrics:
        #         self.log(metric.report(), style="blue")
        #         if self.use_tensorboardX:
        #             metric.write(self.writer, self.epoch, prefix="evaluate")
        #         metric.clear()

        # if self.ema is not None:
        #     self.ema.restore()

        # self.log(f"++> Evaluate epoch {self.epoch} Finished.")
        # del total_loss, total_loss_pose_local
        # # gc.collect()
        # # torch.cuda.empty_cache()
    
    def save_optim_checkpoint(self, full=False, best=False):

        state = {
            'epoch': self.epoch,
            'stats': self.stats,
        }

        # if self.model.cuda_ray:
        #     state['mean_count'] = self.model.mean_count
        #     state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()
            state['pose_model'] = self.pose_model.state_dict()
            # state['cast_model'] = self.cast_model.state_dict()

            file_path = f"{self.skel_ckpt_path}/skel_{self.name}_ep{self.epoch:04d}.pth.tar"

            self.stats["checkpoints"].append(file_path)

            # if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
            #     old_ckpt = self.stats["checkpoints"].pop(0)
            #     if os.path.exists(old_ckpt):
            #         os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()
                    state['pose_model'] = self.pose_model.state_dict()
                    # state['cast_model'] = self.cast_model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.skel_best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

            
    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.skel_ckpt_path}/skel_{self.name}_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'pose_model' not in checkpoint_dict :
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return
        if self.pose_model is not None:
            missing_keys, unexpected_keys = self.pose_model.load_state_dict(checkpoint_dict['pose_model'], strict=False)
            self.log("[INFO] loaded model.")
            if len(missing_keys) > 0:
                self.log(f"[WARN] missing keys: {missing_keys}")
            if len(unexpected_keys) > 0:
                self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']+1

        # if self.model.cuda_ray:
        #     if 'mean_count' in checkpoint_dict:
        #         self.model.mean_count = checkpoint_dict['mean_count']
        #     if 'mean_density' in checkpoint_dict:
        #         self.model.mean_density = checkpoint_dict['mean_density']
        
        if self.optimizer and  'optimizer' in checkpoint_dict and (not self.from_scratch):
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer, use default.")
        
        # strange bug: keyerror 'lr_lambdas'
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler, use default.")
        
        if 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler, use default.")

    def load_nerf_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.nerf_ckpt_path}/{self.name}_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return
        for key in list(checkpoint_dict['model'].keys()):
            if "cast_model" in key or "pose_model" in key or "cast_net"  in key:
                checkpoint_dict["model"].pop(key)
        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded NeRF model. : "+checkpoint )
        
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        # if self.ema is not None and 'ema' in checkpoint_dict and False:
        #     self.ema.load_state_dict(checkpoint_dict['ema'], strict=False)

        # self.stats = checkpoint_dict['stats']
        # self.epoch = checkpoint_dict['epoch']

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
        
        # if self.optimizer and  'optimizer' in checkpoint_dict:
        #     try:
        #         self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
        #         self.log("[INFO] loaded optimizer.")
        #     except:
        #         self.log("[WARN] Failed to load optimizer, use default.")
        
        # # strange bug: keyerror 'lr_lambdas'
        # if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
        #     try:
        #         self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
        #         self.log("[INFO] loaded scheduler.")
        #     except:
        #         self.log("[WARN] Failed to load scheduler, use default.")
        
        # if 'scaler' in checkpoint_dict:
        #     try:
        #         self.scaler.load_state_dict(checkpoint_dict['scaler'])
        #         self.log("[INFO] loaded scaler.")
        #     except:
        #         self.log("[WARN] Failed to load scaler, use default.")
