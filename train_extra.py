
import os
import tensorflow
from torch import tensor, unsqueeze
from tqdm.auto import tqdm
from opt import config_parser
from models.sh_field import SphereHarmonicJoints
from models.pointCaster import *


import json, random
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime

from dataLoader import dataset_dict
import traceback,sys

from nerf.render_util import *
from nerf.skeleton_poses import *

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import random

parallel = False
# dist_test = True
indivInv = False
rank_criteria = 0
mix_precision = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


@torch.no_grad()
def export_mesh(args):

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha,_ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply', bbox=tensorf.aabb.cpu(), level=0.005)

npz_point_cloud = False


def Mimic_optim(rank, args, n_gpu = 1):

    rank_diff = rank - rank_criteria
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    if parallel:
        dist.init_process_group("gloo", rank=rank, world_size=n_gpu)
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False, data_preparation = True)
    test_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True, data_preparation = True)
    # test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True, data_preparation = True)

    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    args.use_indivInv = False

    
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'
    

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)



    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    device = "cuda:"+str(rank)
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
    print(nSamples, args.nSamples, cal_n_samples(reso_cur,args.step_ratio))

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.step_ratio = args.step_ratio
        tensorf.update_stepSize(tensorf.gridSize_tmp)
        tensorf.load(ckpt)
        tensorf.modify_aabb(torch.tensor([2.5, 2.5, 1]).to(device))
        
    else:
        tensorf = eval(args.model_name)(aabb, reso_cur, device,
                    density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,
                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct)

    tensorf.set_args(args)
    tensorf.set_posetype(args.pose_type)
    tensorf.set_allgrads(False)
    tensorf.use_gt_skeleton = args.use_gt_skeleton



    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    

    #linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]

    PSNRs,PSNRs_test = [],[0]

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    w, h = train_dataset.img_wh
    print(w, h)


    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")


    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    
    animation_conf=args.datadir+"/transforms.json"
    skeleton =  make_joints_from_blender(animation_conf, device = device)
    skeleton.precompute_id(0)
    joints = listify_skeleton(skeleton)
    if args.free_opt4:
        skeleton.para_precompute(len(joints))
    skeleton.set_inv_precomputations(len(joints))
    skeleton.set_as_root()
    skeleton.set_tails(skeleton.get_tail_ids())
    train_dataset.compute_skeleton_poses(skeleton)
    test_dataset.compute_skeleton_poses(skeleton)
    tensorf.set_skeleton(skeleton)
    tensorf.set_skeletonmode()

    # frames = json.load(open(animation_conf, 'r'))["frames"]

    num_frames = len(train_dataset.frame_poses)

    ray_per_img = test_dataset.rays_per_img
    itr = 0
    allrays = allrays.reshape(num_frames, -1, 6)
    num_animFrames = len(train_dataset.unique_animFrames)
    allanimframes = train_dataset.all_animFrames
    allrgbs = allrgbs.reshape(num_frames, -1, 3)


    
    skeleton_dataset = LearnSkeletonPose(num_animFrames, len(joints), type=args.pose_type)
    if args.free_opt4:
        skeleton.refresh()
        tfs = skeleton.get_listed_global_transforms()
        translates = tfs[...,:3, 3]
        rotations = skeleton.matrix_to_euler_pos_batch(tfs[...,:3,:3], top_batching=True)
        para_six_pose =  torch.cat([rotations, translates], dim=-1)
        skeleton_dataset = LearnSkeletonPose(num_animFrames, len(joints), type="para_six", para_pose = para_six_pose)
    skeleton_dataset.to(device)
    skeleton_dataset.set_tails(skeleton.get_tail_ids())


    if args.mimik == "test1":
        ckpt = torch.load(args.ckpt_mimik)
        mimik_pCaster = MLPCaster(len(joints), device, args = args , use_ffmlp = True)
        mimik_pCaster.set_skeleton(skeleton)
        mimik_pCaster.load_state_dict(ckpt["state_dict"])
        mimik_pCaster.set_all_grads(False)
        tensorf.set_mimikCaster(mimik_pCaster)
            
            # exit()

    elif args.mimik == "test2":
        ckpt = torch.load(args.ckpt_mimik)
        mimik_pCaster = DirectMapCaster(num_animFrames, device, args = args)
        mimik_pCaster.set_skeleton(skeleton)
        mimik_pCaster.load_state_dict(ckpt["state_dict"])
        mimik_pCaster.set_all_grads(False)
        tensorf.set_mimikCaster(mimik_pCaster)
        # exit()
    elif args.mimik == "cycle":
        ckpt = torch.load(args.ckpt_mimik)
        mimik_pCaster = DirectMapCaster(num_animFrames, device, args = args)
        mimik_pCaster.set_skeleton(skeleton)
        mimik_pCaster.load_state_dict(ckpt["state_dict"])
        mimik_pCaster.set_all_grads(False)
        tensorf.set_mimikCaster(mimik_pCaster)
    else:
        raise ValueError("mimik type not supported")
    
    

    sh_field = SphereHarmonicJoints(len(joints), 9)
    sh_field.to(device)

    if args.sh_feats is not None:
        feats = torch.load(args.sh_feats)
        #sh_field.set_feats(feats)
        # print(feats["state_dict"].keys())
        sh_field.load_state_dict(feats["state_dict"])

    if args.caster == "sh":
        pCaster_origin  = shCaster()
        pCaster_origin.set_SH_feats(sh_field())
        grad_vars_sh_field = sh_field.parameters()
    elif args.caster == "bwf":
        reso_cur_2 = reso_cur
        reso_cur_2[0] = reso_cur_2[0]//4
        reso_cur_2[1] = reso_cur_2[1]//4
        reso_cur_2[2] = reso_cur_2[2]//4
        pCaster_origin = BWCaster(len(joints), reso_cur_2 ,device)
        if args.ckpt_pcaster is not None:
            ckpt = torch.load(args.ckpt_pcaster, map_location=device)
            pCaster_origin.load_state_dict(ckpt["state_dict"])
    elif args.caster == "dist":
        pCaster_origin  = DistCaster()
    elif args.caster == "mlp":
        if args.free_opt1:
            pCaster_origin = MLPCaster_integrate(len(joints), device, args=args)
        else:
            pCaster_origin = MLPCaster(len(joints), device, args = args)
        # pCaster_origin = MLPCaster_tcnn(len(joints), device)
        # pCaster_origin = MLPCaster_net(len(joints), device)
    elif args.caster == "forward":
        tensorf.forward_caster_mode = True
        pCaster_origin = DistCaster()
    elif args.caster == "map":
        # if args.free_opt3:
        if False:
            reso_cur_2 = reso_cur
            reso_cur_2[0] = reso_cur_2[0]//4
            reso_cur_2[1] = reso_cur_2[1]//4
            reso_cur_2[2] = reso_cur_2[2]//4
            pCaster_origin = MapCaster_grid(len(joints), reso_cur_2 , device, num_animFrames,  args=args)
        else:
            pCaster_origin = MapCaster(num_animFrames, device, args=args)
    elif  args.caster == "direct_map":
        pCaster_origin = DirectMapCaster(num_animFrames, device, args=args)
    else:
        try:
            x = 1 / 0
        except:
            traceback.print_exc()
            exit("caster not found")

        


    pCaster_origin.set_skeleton(skeleton)
    

    
   
    # for grad in grad_vars:
    #     for param in grad["params"]:
    #         param.requires_grad = False
    

    # skeleton_dataset.save(f'{logfolder}/{args.expname}_skeleton.th')
    if args.ckpt_skeleton is not None:
        skeleton_dataset.load(torch.load(args.ckpt_skeleton, map_location=device))

    lr_grid = 1e-4
    lr_sh = args.lr_sh
    if args.pose_type == "euler":
        lr_skel = args.lr_skel
    else:
        lr_skel = 1e-4


    if parallel:
        # tensorf = tensorf.to("cuda:0")
        # tensorf_ddp =  DDP(tensorf, device_ids=[rank])
        pCaster = DDP(pCaster_origin, device_ids=[rank])
        if args.caster == "bwf":
            grad_vars_pcaster = pCaster_origin.get_optparam_groups(lr_init_spatialxyz = lr_grid )
        tensorf.set_pointCaster(pCaster, pCaster_origin)

        # tensorf_ddp = tensorf

    else:
        tensorf_ddp = tensorf
        pCaster = pCaster_origin
        if args.caster == "bwf":
            grad_vars_pcaster = pCaster.get_optparam_groups(lr_init_spatialxyz = lr_grid )
        tensorf.set_pointCaster(pCaster)

    if parallel:
        num_frames = num_frames // n_gpu
        if(rank < num_frames % n_gpu):
            num_frames += 1

    grad_vars_skeletonpose = list(skeleton_dataset.parameters())


    
    

    #<Training> Setup Optimizer
    # optimizer = torch.optim.Adam([{'params': grad_vars_pcaster, 'lr': 1e-1}], betas=(0.9,0.99))
    if mix_precision:
        scaler = torch.cuda.amp.GradScaler() # for mixed precision
    if args.caster == "sh":
        params =  [{'params': grad_vars_sh_field, 'lr': lr_sh}]
        if not args.use_gt_skeleton:
            params.append({'params': grad_vars_skeletonpose, 'lr': lr_skel})
        optimizer = torch.optim.Adam(params, betas=(0.9,0.99))
    elif args.caster == "bwf":
        params = grad_vars_pcaster
        if not args.use_gt_skeleton:
            params.append({'params': grad_vars_skeletonpose, 'lr': lr_skel})
        optimizer = torch.optim.Adam(params, betas=(0.9,0.99))

    elif args.caster == "dist":
        optimizer = torch.optim.Adam( [{'params': grad_vars_skeletonpose, 'lr': lr_skel}], betas=(0.9,0.99))

    elif args.caster == "mlp":
        wd = 1e-6
        params =  [
            {'name':'weight_nets','params': list(pCaster_origin.weight_nets.parameters()), 'weight_decay': wd, 'lr':1e-4},
            {'name':'encoder','params': list(pCaster_origin.encoder.parameters()),  'lr': 2e-2}
        ]

        params.append({'name':'interface_layer','params': list(pCaster_origin.interface_layer.parameters()), 'weight_decay': wd, 'lr': 1e-4})
        if args.free_opt1:
            params.append({'name':'after_layer','params': list(pCaster_origin.after_layer.parameters()), 'weight_decay': wd, 'lr': 1e-4})
        if not args.use_gt_skeleton:
            params.append({'name':'skeleton', 'params': grad_vars_skeletonpose, 'lr': lr_skel})
        optimizer = torch.optim.Adam( params, betas=(0.9,0.99))

    elif args.caster == "map":
        # if args.free_opt3:
        if False:
            wd = 1e-6
            params =  pCaster_origin.get_optparam_groups()
            optimizer = torch.optim.Adam( params, betas=(0.9,0.99))
        else:
            wd = 1e-6
            params =  [
                {'name':'map_nets','params': list(pCaster_origin.map_nets.parameters()), 'weight_decay': wd, 'lr':1e-4},
                {'name':'encoder','params': list(pCaster_origin.encoder.parameters()),  'lr': 2e-2}
            ]
            params.append({'name':'pose_params','params': list(pCaster_origin.pose_params.parameters()), 'weight_decay': wd, 'lr':1e-4})
            params.append({'name':'interface_layer','params': list(pCaster_origin.interface_layer.parameters()), 'weight_decay': wd, 'lr': 1e-4})
            optimizer = torch.optim.Adam( params, betas=(0.9,0.99))
    elif args.caster == "direct_map":
            wd = 1e-6
            lr = 1e-3
            params =  [
                {'name':'map_nets','params': list(pCaster_origin.map_nets.parameters()), 'weight_decay': wd, 'lr':lr},
                {'name':'encoder','params': list(pCaster_origin.encoder.parameters()),  'lr': 2e-2}
            ]
            params.append({'name':'pose_params','params': list(pCaster_origin.pose_params.parameters()), 'weight_decay': wd, 'lr':lr})
            params.append({'name':'interface_layer','params': list(pCaster_origin.interface_layer.parameters()), 'weight_decay': wd, 'lr': lr})
            params.append({'name':'branch_w','params': list(pCaster_origin.branch_w.parameters()), 'weight_decay': wd, 'lr': lr})
            params.append({'name':'branch_v','params': list(pCaster_origin.branch_v.parameters()), 'weight_decay': wd, 'lr': lr})
            optimizer = torch.optim.Adam( params, betas=(0.9,0.99))
    elif args.caster == "forward":
        optimizer = torch.optim.Adam( [{'params': grad_vars_skeletonpose, 'lr': lr_skel}], betas=(0.9,0.99))
    else:
        try:
            x = 1 / 0
        except:
            traceback.print_exc()
            exit("caster not found")

    tensorf = tensorf.to(device)
    
    for iteration in pbar:
        # # JOKE skeleton_optim
        if args.JOKE:
            print("JOKE RENDER extra===")
            if iteration == 0 or (iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0):
                skeleton_props ={"skeleton_dataset": skeleton_dataset}
                PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                prtx=f'{iteration:06d}_', N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False, skeleton_props=skeleton_props, device=device)
                summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)
                print("JOKE")
            exit("JOKEEXIT")
        # # JOKE skeleton_optim
        
        # ray_idx = trainingSampler.nextids()
        # rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)
        itr = itr % num_frames
        tensorf.set_render_flags()
        torch.cuda.empty_cache()
        batching = args.local_batching
        batch_len = 1
        if batching:
            arr = torch.arange(0, ray_per_img)
            l = list(range(ray_per_img))
            random.shuffle(l)
            rand_idxs = torch.tensor(l, device=device)
            batch_len = ray_per_img // args.batch_size
            batch_len = min(20, batch_len)
        else:
            idx = torch.randint(0, ray_per_img, size=[args.batch_size], device=device)
            rays_train = torch.index_select(allrays[itr+num_frames*rank_diff].to(device), 0, idx)
            rgb_train = torch.index_select(allrgbs[itr+num_frames*rank_diff].to(device), 0, idx)
        
        if args.use_gt_skeleton:
            skel = test_dataset.frame_skeleton_pose[itr+num_frames*rank_diff]
        else:
            skel = skeleton_dataset(allanimframes[itr+num_frames*rank_diff])
        tensorf.set_framepose(skel)
        skeleton_props ={"frame_pose": skel}

        for i_batch in range(batch_len):
            if batching:
                print("batching", i_batch, "/", batch_len)  
                idx = rand_idxs[i_batch*args.batch_size:(i_batch+1)*args.batch_size]
                # print(allrays[itr+num_frames*rank_diff].shape, idx.shape)
                # exit()
                rays_train = torch.index_select(allrays[itr+num_frames*rank_diff].to(device), 0, idx)
                rgb_train = torch.index_select(allrgbs[itr+num_frames*rank_diff].to(device), 0, idx)
            #rgb_map, alphas_map, depth_map, weights, uncertainty
            with torch.cuda.amp.autocast(): 
                tensorf.set_tmp_animframe_index(allanimframes[itr+num_frames*rank_diff])
        
                
                points = tensorf.get_grid_points([100,100,100])
                points = points.to(device)
                points = points.view(-1,3)
                if args.mimik == "cycle":
                    with torch.no_grad():
                        _viewdirs = torch.tensor([1,0,0], device=device).repeat(points.shape[0],1)
                        casted_xyzs, casted_viewdirs = mimik_pCaster(points, _viewdirs, None, None, i_frame = allanimframes[itr+num_frames*rank_diff])
                    newxyzs, newviewdirs = pCaster_origin(casted_xyzs,  casted_viewdirs, None, None, i_frame = allanimframes[itr+num_frames*rank_diff])
                    loss = torch.mean((newxyzs - points.reshape(-1,3))**2) + torch.mean((newviewdirs - _viewdirs)**2)
                else:
                    with torch.no_grad():
                        gt_skel_transforms = tensorf.skeleton.rotations_to_transforms_fast(test_dataset.frame_skeleton_pose[itr+num_frames*rank_diff], type=args.pose_type)
                    if args.mimik == "test1":
                        with torch.no_grad():
                            # gt_skel_transforms = tensorf.skeleton.rotations_to_transforms_fast(test_dataset.frame_skeleton_pose[itr+num_frames*rank_diff], type=args.pose_type)
                            ref_weights = mimik_pCaster.compute_weights(points.view(-1, 3), gt_skel_transforms)
                            ref_transforms = torch.matmul(ref_weights, gt_skel_transforms.view(gt_skel_transforms.shape[0], -1)).view(points.shape[0], 4, 4)
                            # ref_inv = affine_inverse_batch(tmp)
                        

                    elif args.mimik == "test2":
                        with torch.no_grad():
                            ref_transforms = mimik_pCaster.compute_transforms(points, mimik_pCaster.pose_params(allanimframes[itr+num_frames*rank_diff]))
                        # transforms = tensorf.skeleton.rotations_to_transforms_fast(skel, type=args.pose_type)
                    else:
                        raise ValueError("mimik not found")
                    

                    if args.caster == "direct_map":
                        inf_transform = pCaster_origin.compute_transforms(points, pCaster_origin.pose_params(allanimframes[itr+num_frames*rank_diff]))
                        # exit("test2")
                    elif args.caster == "mlp":
                        transforms = tensorf.skeleton.rotations_to_transforms_fast(skel, type=args.pose_type)
                        inf_weights = pCaster_origin.compute_weights(points.view(-1, 3), transforms)
                        inf_transform = torch.matmul(inf_weights, transforms.view(transforms.shape[0], -1)).view(points.shape[0], 4, 4)

                    else:
                        raise ValueError("mimik not found")
                    

                    loss = torch.mean((inf_transform - ref_transforms) ** 2)

                # rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(rays_train, tensorf, chunk=args.batch_size,
                #                         N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True, skeleton_props=skeleton_props)
                # loss = torch.mean((rgb_map - rgb_train) ** 2)

                if torch.isnan(loss):
                    raise ValueError("Loss is NaN")

                total_loss = 0
                tvloss = 0
                linearloss = 0
                rest_loss = 0
                eloss=0
                if args.caster == "bwf":
                    total_loss += tvloss  + linearloss

                if args.caster == "sh" and not args.use_gt_skeleton:
                    pass

                if args.caster == "direct_map":
                    pass

                # loss
                total_loss = total_loss + loss
                if Ortho_reg_weight > 0:
                    loss_reg = tensorf.vector_comp_diffs()
                    total_loss += Ortho_reg_weight*loss_reg
                    summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
                if L1_reg_weight > 0:
                    loss_reg_L1 = tensorf.density_L1()
                    total_loss += L1_reg_weight*loss_reg_L1
                    summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

                if TV_weight_density>0:
                    TV_weight_density *= lr_factor
                    loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
                    total_loss = total_loss + loss_tv
                    summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
                if TV_weight_app>0:
                    TV_weight_app *= lr_factor
                    loss_tv = loss_tv + tensorf.TV_loss_app(tvreg)*TV_weight_app
                    total_loss = total_loss + loss_tv
                    summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

                optimizer.zero_grad()

                if mix_precision:
                    scaler.scale(total_loss).backward()
                else:
                    # print(total_loss)
                    total_loss.backward()
                
                

                loss = loss.detach().item()

                if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                    print("isnanisany", torch.isnan(total_loss).any(),  torch.isinf(total_loss).any())
                    raise ValueError("nan or inf")
                
                PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
                summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
                summary_writer.add_scalar('train/mse', loss, global_step=iteration)


                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * lr_factor

                # Print the current values of the losses.
                if iteration % args.progress_refresh_rate == 0:
                    pbar.set_description(
                        f'Iteration {iteration:05d}:'
                        # + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                        # + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                        + f' mse = {loss:.6f}'
                        + f' tvloss = {tvloss:.6f}'
                        # + f' linearloss = {linearloss:.6f}'
                        # + f' restloss = {rest_loss:.6f}'
                        + f' elastic_loss = {eloss:.6f}'

                    )
                    PSNRs = []
                
                # del rgb_map, alphas_map, depth_map, weights, uncertainty, loss, total_loss, tvloss, linearloss
                if mix_precision:
                    scaler.step(optimizer) 
                
                    optimizer.step()
                    
                
                    scaler.update() 
                else:
                    optimizer.step()
                    # scheduler.step ()



                if args.free_opt2 and itr == 1 and False:
                    with torch.no_grad():
                        rays_train = allrays[itr+num_frames*rank_diff]
                        rgb_train = allrgbs[itr+num_frames*rank_diff]
                        rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(rays_train, tensorf, chunk=args.test_batch_size,
                                        N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True, skeleton_props=skeleton_props)
                        rgb_map = (rgb_map.cpu().numpy() * 255).astype('uint8')
                        gt_rgb = (rgb_train.cpu().numpy() * 255).astype('uint8')
                        H,W = train_dataset.img_wh
                        rgb_map = np.concatenate((rgb_map.reshape(H,W,3), gt_rgb.reshape(H,W,3)), axis=1)
                        imageio.imwrite(f'{logfolder}/{args.expname}_dist_test{iteration:03d}.png', rgb_map)
                
                #<Training> Test and Save the model.
                with torch.no_grad():
                    if (iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0) and i_batch == batch_len-1:
                        if rank == rank_criteria:
                            skeleton_props ={"skeleton_dataset": skeleton_dataset}
                            PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis, prtx=f'{iteration:06d}_', N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False, skeleton_props=skeleton_props , device=device)
                            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)
                            # tensorf.save(f'{logfolder}/{args.expname}.th')
                            # exit("EXITING")
                            # if rank == 0:
                            skeleton_dataset.save(f'{logfolder}/{args.expname}_skeleton_it{iteration}.th')
                            sh_field.save(f'{logfolder}/{args.expname}_sh_it{iteration}.th')
                            if not args.caster == "sh":
                                pCaster_origin.save( f'{logfolder}/{args.expname}_pCaster_it{iteration}.th')
                            tensorf.save(f'{logfolder}/{args.expname}_it{iteration}.th')
        itr += 1

    if rank == rank_criteria:
        skeleton_dataset.save(f'{logfolder}/{args.expname}_skeleton.th')
        sh_field.save(f'{logfolder}/{args.expname}_sh.th')
        pCaster.save( f'{logfolder}/{args.expname}_pCaster.th')
        tensorf.save(f'{logfolder}/{args.expname}.th')
        


    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

def cast_invert(rank, args, n_gpu = 1):
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'
    

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.step_ratio = args.step_ratio
        tensorf.update_stepSize(tensorf.gridSize_tmp)
        tensorf.load(ckpt)
        tensorf.modify_aabb(torch.tensor([2.5, 2.5, 1]).to(device))
    tensorf.set_args(args)
    tensorf.set_posetype(args.pose_type)
    tensorf.set_allgrads(False)
    tensorf.use_gt_skeleton = args.use_gt_skeleton

    animation_conf=args.datadir+"/transforms.json"
    skeleton =  make_joints_from_blender(animation_conf, device = device)
    skeleton.precompute_id(0)
    joints = listify_skeleton(skeleton)
    if args.free_opt4:
        skeleton.para_precompute(len(joints))
    skeleton.set_inv_precomputations(len(joints))
    skeleton.set_as_root()
    skeleton.set_tails(skeleton.get_tail_ids())
    train_dataset.compute_skeleton_poses(skeleton)
    test_dataset.compute_skeleton_poses(skeleton)
    tensorf.set_skeleton(skeleton)
    tensorf.set_skeletonmode()

    # frames = json.load(open(animation_conf, 'r'))["frames"]

    num_frames = len(train_dataset.frame_poses)

    ray_per_img = test_dataset.rays_per_img
    itr = 0
    allrays = allrays.reshape(num_frames, -1, 6)
    num_animFrames = len(train_dataset.unique_animFrames)
    allanimframes = train_dataset.all_animFrames
    allrgbs = allrgbs.reshape(num_frames, -1, 3)

    rank_diff = rank - rank_criteria
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    if parallel:
        dist.init_process_group("gloo", rank=rank, world_size=n_gpu)
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False, data_preparation = True)
    test_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True, data_preparation = True)
    num_animFrames = len(train_dataset.unique_animFrames)
    print(f"num_animFrames: {num_animFrames}")
    # pcasterの初期化
    if args.caster == "map":
        pCaster_origin = MapCaster(num_animFrames, device, args=args)
        pCaster_cycle = MapCaster(num_animFrames, device, args=args)
        wd = 1e-6
        params =  [
            {'name':'map_nets','params': list(pCaster_cycle.map_nets.parameters()), 'weight_decay': wd, 'lr':1e-4},
            {'name':'encoder','params': list(pCaster_cycle.encoder.parameters()),  'lr': 2e-2}
        ]
        params.append({'name':'pose_params','params': list(pCaster_cycle.pose_params.parameters()), 'weight_decay': wd, 'lr':1e-4})
        params.append({'name':'interface_layer','params': list(pCaster_cycle.interface_layer.parameters()), 'weight_decay': wd, 'lr': 1e-4})
        optimizer = torch.optim.Adam( params, betas=(0.9,0.99))

    elif  args.caster == "direct_map":
        pCaster_origin = DirectMapCaster(num_animFrames, device, args=args)
        pCaster_cycle = DirectMapCaster(num_animFrames, device, args=args)            
        wd = 1e-6
        lr = 1e-3
        params =  [
            {'name':'map_nets','params': list(pCaster_cycle.map_nets.parameters()), 'weight_decay': wd, 'lr':lr},
            {'name':'encoder','params': list(pCaster_cycle.encoder.parameters()),  'lr': 2e-2}
        ]
        params.append({'name':'pose_params','params': list(pCaster_cycle.pose_params.parameters()), 'weight_decay': wd, 'lr':lr})
        params.append({'name':'interface_layer','params': list(pCaster_cycle.interface_layer.parameters()), 'weight_decay': wd, 'lr': lr})
        params.append({'name':'branch_w','params': list(pCaster_cycle.branch_w.parameters()), 'weight_decay': wd, 'lr': lr})
        params.append({'name':'branch_v','params': list(pCaster_cycle.branch_v.parameters()), 'weight_decay': wd, 'lr': lr})
        optimizer = torch.optim.Adam( params, betas=(0.9,0.99))
    else:
        try:
            x = 1 / 0
        except:
            traceback.print_exc()
            exit("caster not found")

    # pcaster_inv の初期化
    pCaster_cycle.set_Cycle_pose(pCaster_origin.pose_params)
    pCaster_origin.set_reqires_grad(False)


    skeleton_dataset = LearnSkeletonPose(num_animFrames, len(joints), type=args.pose_type)
    skeleton_dataset.to(device)
    skeleton_dataset.set_tails(skeleton.get_tail_ids())
    


    if args.ckpt_pCaster is not None:
        pass
    else:
        exit()
    max_iteration = 10000
    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    
    box_points = tensorf.get_grid_points([50,50,50])
    shape = box_points.shape
    for iteration in pbar:
        casted_xyzs = pCaster_origin(box_points.reshape(-1,3))
        xyzs = pCaster_cycle(casted_xyzs)
        loss = torch.mean((xyzs - box_points.reshape(-1,3))**2)
        optimizer.zero_grad()
        loss.backward()
        loss = loss.detach().item()


        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', loss, global_step=iteration)
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                # + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                # + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {loss:.6f}'

            )
            PSNRs = []
        optimizer.step()
        with torch.no_grad():
            if (iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0):
                if rank == rank_criteria:
                    skeleton_props ={"skeleton_dataset": skeleton_dataset}
                    PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis, prtx=f'{iteration:06d}_', N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False, skeleton_props=skeleton_props , device=device)
                    summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)
                    # tensorf.save(f'{logfolder}/{args.expname}.th')
                    # exit("EXITING")
                    # if rank == 0:
                    skeleton_dataset.save(f'{logfolder}/{args.expname}_skeleton_it{iteration}.th')
                    # sh_field.save(f'{logfolder}/{args.expname}_sh_it{iteration}.th')
                    # if not args.caster == "sh":
                    #     pCaster_origin.save( f'{logfolder}/{args.expname}_pCaster_it{iteration}.th')
                    tensorf.save(f'{logfolder}/{args.expname}_it{iteration}.th')



if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    torch.autograd.set_detect_anomaly(True)
    np.random.seed(20211202)

    args = config_parser()
    print(args)
    rank_criteria = args.rank_criteria

    if  args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path):
        # render_test(args)
        pass
    else:
        if args.cyclic:
            cast_invert(rank_criteria, args)
            exit()
        if parallel:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            n_gpu = 2
            mp.spawn(Mimic_optim,
                args=(args, n_gpu),
                nprocs=n_gpu,
                join=True)
        else:
            Mimic_optim(rank_criteria, args)


