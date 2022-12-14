import torch,cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T
from torchngp.nerf.provider import nerf_matrix_to_ngp

import sys
from .ray_utils import *
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from nerf.render_util import *


class BlenderDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1, data_preparation = True):

        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.img_wh = (int(800/downsample),int(800/downsample))
        self.define_transforms()

        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        # if not  data_preparation:
        self.scene_bbox = self.scene_bbox * 10.0#shark
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.define_proj_mat()

        self.white_bg = True
       
        self.near_far = [2.0,18.0]#shark
        # self.near_far = [2.0,6.0]
        if not data_preparation:
             self.near_far = [2.0,6.0]
        
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.downsample=downsample
        self.focus_mode = None
        self.focus_vec = None

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth
    def compute_skeleton_poses(self, skeleton):
        self.frame_skeleton_pose = []
        for i, pose in enumerate(self.frame_poses):
            for j in skeleton.get_children():
                apply_animation(pose, j)
            self.frame_skeleton_pose.append(skeleton.get_listed_rotations().clone())
        # exit()

    def set_focus_mode(self, vec, focus_mode=True):
        if vec is None:
            vec = [0,0,0]
        self.focus_mode = focus_mode
        self.focus_vec = torch.tensor(vec).float().view(1, 1, 3)
        


        
    
    def read_meta(self):

        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        self.focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh


        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal,self.focal])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal,0,w/2],[0,self.focal,h/2],[0,0,1]]).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []
        self.all_animFrames = []
        self.unique_animFrames = set()
        #here
        self.frame_poses = []
        self.downsample=1.0
        self.num_frames = len(self.meta['frames'])
        self.rays_per_img = h * w

        img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
        idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):#img_list:#

            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            # pose = nerf_matrix_to_ngp(np.array(frame['transform_matrix']))
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]
            self.frame_poses += [frame['animation']]

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)
            
            if self.downsample!=1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs += [img]
            anim_frame = frame["anim_frame"]
            self.all_animFrames += [anim_frame]
            self.unique_animFrames.add(anim_frame)
            # print(frame['file_path'], anim_frame)
            # if(int(frame['file_path'].split(".png")[0].split("_")[-1]) == int(frame["anim_frame"])):
            #     print("same")
            # else:
            #     print("not same")
            #     exit("not ok")
            


            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            # mouse_pos = torch.tensor([0,0,0])
            # mosuse_dir = mouse_pos - rays_o[0]
            # mosuse_dir = mosuse_dir / torch.norm(mosuse_dir, dim=-1, keepdim=True)
            # dot = torch.sum(rays_d * mosuse_dir, dim=-1, keepdim=True).view(w,h)
            # id = torch.argmax(dot, keepdim=True)
            # print(dot.shape, id.shape, id)
            # minih, miniw = 100,100
            # exit()
            # l_minih = torch.max(0, id[0]-minih)
            # r_minih = torch.min(h, id[0]+minih)
            # l_miniw = torch.max(0, id[1]-miniw)
            # r_miniw = torch.min(w, id[1]+miniw)

            # a = torch.meshgrid(torch.arange(l_minih, r_minih), torch.arange(l_miniw, r_miniw))
            # mouse_ray = rays_d.select_index(a)
            

            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        # exit("blender")
        self.poses = torch.stack(self.poses)
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)

#             self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)


    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
        
    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx] # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img,
                      'mask': mask}
        return sample
