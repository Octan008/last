import torch,os,imageio,sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from models.tensoRF import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask
from utils import *
from dataLoader.ray_utils import ndc_rays_blender
from nerf.render_util import *

npz_point_cloud = False
n_vis_offset = 1

def crop_rays(ray, w, h, crop_box, rgb=None, filter = 1):
    print(ray.shape)
    print()
    ray = ray.reshape(w, h, 6)
    print(ray.shape)
    ray = ray[crop_box[0][0]:crop_box[0][1]:filter, crop_box[1][0]:crop_box[1][1]:filter, :]
    print(ray.shape)
    ray = ray.reshape(-1, 6)
    print(ray.shape)
    if rgb is not None:
        
        return ray, rgb
    else:
        return ray
def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda', skeleton_props=None, is_render_only=False):


    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
    
        rgb_map, depth_map = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples, skeleton_props = skeleton_props, is_render_only=is_render_only)

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
    
    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None
    # return torch.cat(rgbs), None, None, None, None

@torch.no_grad()
def evaluation(test_dataset,tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda', skeleton_props=None, is_render_only = False):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)
    test_evaluation = False
    try:
        tqdm._instances.clear()
    except Exception:
        pass
    skeleton_dataset = None
    if skeleton_props is not None:
        print("set_skeleton_dataset")
        skeleton_dataset = skeleton_props["skeleton_dataset"]

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    # print(test_dataset.all_rays.shape[0])
    # exit("exit_test")
    idxs = list(range(n_vis_offset, test_dataset.all_rays.shape[0], img_eval_interval))
    tensorf.set_render_flags(jointmask = True)
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[n_vis_offset::img_eval_interval].to(device)), file=sys.stdout):
        # samples = test_dataset.all_rays[n_vis_offset].to(device)
        # print("idx", idx)
        # continue
        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])
        # exit("crop_ray")
        if npz_point_cloud:
            rays = crop_rays(rays, W, H, [[200, 600], [200, 600]], filter = 10)
            W, H = (600-200)//10, (600-200)//10
        # exit("crop_ray")

        if not args.data_preparation:
            print("animframe", test_dataset.all_animFrames[idxs[idx]], idxs[idx])
            # continue
            #tmp
            if args.free_opt2:
                tensorf.set_tmp_animframe_index(test_dataset.all_animFrames[idxs[idx]])
                
            if skeleton_dataset is not None and not args.use_gt_skeleton:
                scale = idx / 15.0
                skeleton_props = {"frame_pose": skeleton_dataset(test_dataset.all_animFrames[idxs[idx]])}
                # for j in tensorf.skeleton.get_children():
                #     apply_animation(test_dataset.frame_poses[idxs[idx]], j)
                gt_skeleton_pose = tensorf.skeleton.get_listed_rotations(type=args.pose_type)
                # # exit("こっち")
            else:
                # for j in tensorf.skeleton.get_children():
                #     apply_animation(test_dataset.frame_poses[idxs[idx]], j)
                # gt_skeleton_pose = tensorf.skeleton.get_listed_rotations(type=args.pose_type)
                gt_skeleton_pose = test_dataset.frame_skeleton_pose[idxs[idx]]
                if args.free_opt4:
                    tensorf.skeleton.rotations_to_transforms_fast(gt_skeleton_pose, type = args.pose_type)
                    # tensorf.skeleton.rotations_to_transforms(gt_skeleton_pose, type = args.pose_type)
                    tfs = tensorf.skeleton.precomp_forward_global_transforms
                    # print(tfs.shape)
                    translates = tfs[...,:3, 3]
      
                    # rotations = tensorf.skeleton.matrix_to_euler_pos(tfs[...,:3,:3])
                    rotations = tensorf.skeleton.matrix_to_euler_pos_batch(tfs[...,:3,:3], top_batching = True)
                    gt_skeleton_pose = torch.cat([rotations, translates], dim=-1)
                    # gt_skeleton_pose = tfs
                skeleton_props = {"frame_pose": gt_skeleton_pose}
                # print(gt_skeleton_pose.shape)
            
            # rgb_map_gtpose, _, depth_map_gtpose, _, _ = renderer(rays, tensorf, chunk=args.test_batch_size, N_samples=N_samples,
            #             ndc_ray=ndc_ray, white_bg = white_bg, device=device, skeleton_props={"frame_pose": gt_skeleton_pose}, is_render_only=is_render_only)
            # rgb_map_gtpose = rgb_map_gtpose.clamp(0.0, 1.0)
            # rgb_map_gtpose = rgb_map_gtpose.reshape(H, W, 3).cpu()
            # rgb_map_gtpose = (rgb_map_gtpose.numpy() * 255).astype('uint8')
            #tmp

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=args.test_batch_size, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device, skeleton_props=skeleton_props, is_render_only=is_render_only)

        rgb_map = rgb_map.clamp(0.0, 1.0)
        

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()
        

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        if len(test_dataset.all_rgbs) and test_evaluation:
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        
        #GT と inf を一緒に保存
        gt_rgb = (test_dataset.all_rgbs[idxs[idx]].view(H, W, 3).numpy() * 255).astype('uint8')
        # gt_rgb = (test_dataset.all_rgbs[n_vis_offset].view(H, W, 3).numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, gt_rgb), axis=1)
        rgb_map = np.concatenate((rgb_map, gt_rgb), axis=1)

        #tmp
        # if not args.data_preparation:
        #     rgb_map = np.concatenate((rgb_map, rgb_map_gtpose), axis=1)
        #tmp
        # exit()

        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)
        if npz_point_cloud:
            raise ValueError("interrupt");

    # imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    # imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    # if PSNRs:
    #     psnr = np.mean(np.asarray(PSNRs))
    #     if compute_extra_metrics:
    #         ssim = np.mean(np.asarray(ssims))
    #         l_a = np.mean(np.asarray(l_alex))
    #         l_v = np.mean(np.asarray(l_vgg))
    #         np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
    #     else:
    #         np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

@torch.no_grad()
def evaluation_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)
        # exit()

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

