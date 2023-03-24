import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import math
import os
import cv2
import json
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as T
from utils import get_ray_weight, SimpleSampler
import time
from .ray_utils import *


class BRICSDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1, is_stack=False, n_frames=100,
                 tmp_path='memory', scene_box=[-3.0, -3.0, -3.0], temporal_variance_threshold=1000,
                 frame_start=0, near=0.1, far=15.0, diffuse_kernel=0, n_cam=53, render_views=10):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        self.root_dir = datadir
        print("Root:", self.root_dir)
        self.split = split
        print("Split:", self.split)
        self.is_stack = is_stack
        print("Is stack:", is_stack)
        self.downsample = downsample
        print("Downsample:", downsample)
        self.diffuse_kernel = diffuse_kernel
        print("Diffuse:", diffuse_kernel)
        self.define_transforms()
        self.tmp_path = tmp_path
        print("Tmp path:", tmp_path)
        self.temporal_variance_threshold = temporal_variance_threshold
        print("Temporal var threshold:", temporal_variance_threshold)
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        print("blender2opencv:", self.blender2opencv)
        self.n_frames = n_frames
        print("N frames:", n_frames)
        self.frame_start = frame_start
        print("Frame start:", frame_start)
        self.render_views = render_views
        self.n_cam = n_cam
        self.near_far = [near, far]
        print("Near-far:", self.near_far)
        self.scene_bbox = torch.tensor([scene_box, list(map(lambda x: -x, scene_box))])
        print(self.scene_bbox)
        self.read_meta()
        #self.define_proj_mat()
        self.white_bg = True
       
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

    def read_meta(self):
        
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)
        w, h = int(self.meta['frames'][0]['w']/self.downsample), int(self.meta['frames'][0]['h']/self.downsample)
        self.img_wh = [w,h]
        assert(self.downsample == 1.0)
        
        self.video_paths = sorted(glob.glob(os.path.join(self.root_dir, 'frames_1/*')))
        # TODO: CHANGE FILE PATH
        _calc_std(os.path.join(self.root_dir, 'frames_1'),
                  os.path.join(self.root_dir, 'stds_1'),
                  frame_start=self.frame_start, 
                  n_frame=self.n_frames, 
                  n_cam=self.n_cam)

       # ray directions for all pixels, same for all images (same H, W, focal)
        #self.directions = get_ray_directions(h, w, [self.focal_x,self.focal_y], center=[self.cx, self.cy])  # (h, w, 3)
        #self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        #self.intrinsics = torch.tensor([[self.focal_x,0,self.cx],[0,self.focal_y,self.cy],[0,0,1]]).float()
        
        self.directions = []
        self.intrinsics = []
        self.focal_x = []
        self.focal_y = []
        self.cx = []
        self.cy = []
        for i in range(0, len(self.meta['frames'])):
            self.focal_x.append(self.meta['frames'][i]['fl_x'])
            self.focal_y.append(self.meta['frames'][i]['fl_y'])
            self.cx.append(self.meta['frames'][i]['cx'])
            self.cy.append(self.meta['frames'][i]['cy'])
            self.directions.append(get_ray_directions(h, w, [self.focal_x[i], self.focal_y[i]], center=[self.cx[i], self.cy[i]]))
            self.intrinsics.append(torch.tensor([[self.focal_x[i],0,self.cx[i]],[0,self.focal_y[i],self.cy[i]],[0,0,1]]).float())
        #print(self.directions.shape)
        #print(self.intrinsics.shape)
        
        self.all_rays = []
        self.all_rgbs = []
        self.all_stds_without_diffusion = []
        self.all_rays_weight = []
        self.all_stds = []
        self.poses = []
        
        idxs = list(range(0, len(self.meta['frames'])))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):
            # get c2w 
            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]
            # load frames
            cam_id = frame['file_path'].split('/')[-1].split('_')[0]
            frames_paths = []
            for j in range(self.frame_start+1, self.frame_start+self.n_frames+1, 1):
                frames_paths.append(os.path.join(self.root_dir, 'frames_1', cam_id, f"{j:04d}.png"))
                #frames_paths.append(os.path.join(self.root_dir, 'images', f"{cam_id}_{i:04d}.png"))
            frames = [Image.open(frame_path) for frame_path in frames_paths]
            assert (self.downsample==1.0)
            frames = [self.transform(img) for img in frames]  # (T, 4, h, w)
            if frames[0].shape[0] == 4:
                frames = [img.view(4, -1).permute(1, 0) for img in frames]  # (T, h*w, 4) RGBA
                for j in range(len(frames)):
                    frames[j] = frames[j][:, :3] * frames[j][:, -1:] + (1 - frames[j][:, -1:])  # blend A to RGB
            else:
                frames = [img.view(3, -1).permute(1, 0) for img in frames]  # (T, h*w, 3) RGB
            frames = torch.stack(frames, dim=1) # hw T 3
            # preprocess ray importance sampling
            std_path = os.path.join(self.root_dir, 'stds_1', f'{cam_id}_std.npy')
            if self.diffuse_kernel > 0:
                std_frames_without_diffuse = np.load(std_path)
                std_frames = diffuse(std_frames_without_diffuse, self.diffuse_kernel)
            else:
                std_frames_without_diffuse = None
                std_frames = np.load(std_path)
            std_frames = torch.from_numpy(std_frames).reshape(-1)
            if std_frames_without_diffuse is not None:
                std_frames_without_diffuse = torch.from_numpy(std_frames_without_diffuse).reshape(-1)

            rays_weight = get_ray_weight(frames)
            
            self.all_rays_weight.append(rays_weight.half())
            self.all_rgbs += [frames.half()]
            self.all_stds += [std_frames.half()]
            if std_frames_without_diffuse is not None:
                self.all_stds_without_diffusion += [std_frames_without_diffuse.half()]

            #rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            rays_o, rays_d = get_rays(self.directions[i], c2w)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
        
        self.poses = torch.stack(self.poses)
        
        center = torch.mean(self.scene_bbox, dim=0)
        radius = torch.norm(self.scene_bbox[1]-center)*1.2
        up = torch.mean(self.poses[:, :3, 1], dim=0).tolist()
        pos_gen = circle(radius=radius, h=-0.2*up[1], axis='y')
        self.render_path = gen_path(pos_gen, up=up,frames=self.render_views)
        self.render_path[:, :3, 3] += center
        
        # print(self.poses.shape)
        # print(self.render_views)
        # N_views, N_rots = self.render_views, 2
        # tt = self.poses[:, :3, 3]
        # up = normalize(self.poses[:, :3, 1].sum(0))
        # rads = np.percentile(np.abs(tt), 90, 0)
        # self.render_path = get_spiral(self.poses, self.near_far, N_views=N_views)
        
        if not self.is_stack:
            self.all_rays_weight = torch.cat(self.all_rays_weight, dim=0) # (Nr)
            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, T, 3)
            self.all_stds = torch.cat(self.all_stds, 0)
            if len(self.all_stds_without_diffusion) > 0:
                self.all_stds_without_diffusion = torch.cat(self.all_stds_without_diffusion, 0)
            # calc the dynamic data
            dynamic_mask = self.all_stds > self.temporal_variance_threshold
            self.dynamic_rays = self.all_rays[dynamic_mask]
            self.dynamic_rgbs = self.all_rgbs[dynamic_mask]
            self.dynamic_stds = self.all_stds[dynamic_mask]
        else:
            self.all_rays_weight = torch.stack(self.all_rays_weight, dim=0) # (Nr)
            self.all_rays = torch.stack(self.all_rays, 0)   # (len(self.meta['frames]),h,w, 3)
            T = self.all_rgbs[0].shape[1]
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], T, 3)  # (len(self.meta['frames]),h,w, T, 3)
            self.all_stds = torch.stack(self.all_stds, 0).reshape(-1,*self.img_wh[::-1])
            if len(self.all_stds_without_diffusion) > 0:
                self.all_stds_without_diffusion = torch.stack(self.all_stds_without_diffusion, 0).reshape(-1,*self.img_wh[::-1])

    #def define_proj_mat(self):
    #    self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth

    def shift_stds(self):
        self.all_stds = self.all_stds_without_diffusion
        return self.all_stds

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        sample = {'rays': self.all_rays[idx],
                  'rgbs': self.all_rgbs[idx]}

        return sample


def _calc_std(frame_path_root, std_path_root, frame_start=0, n_frame=300, n_cam=53):
    if os.path.exists(std_path_root):
        return
    os.makedirs(std_path_root)
    for i in range(n_cam):
        # load frames
        frame_paths = []
        for j in range(frame_start+1, frame_start+n_frame+1, 1):
            frame_paths.append(os.path.join(frame_path_root, f'cam{i:02d}', f"{j:04d}.png"))
        frames = []
        for fp in frame_paths:
            frame = Image.open(fp).convert('RGB')
            frame = np.array(frame, dtype=float) / 255.
            frames.append(frame)
        frame = np.stack(frames, axis=0)
        # compute std
        std_map = frame.std(axis=0).mean(axis=-1)
        std_map_blur = (cv2.GaussianBlur(std_map, (31, 31), 0)).astype(float)
        np.save(os.path.join(std_path_root, f'cam{i:02d}_std.npy'), std_map_blur)


def diffuse(std, kernel):
    h, w = std.shape
    oh, ow = h, w
    add_h = kernel - (h % kernel)
    add_w = kernel - (w % kernel)
    if add_h > 0:
        std = np.concatenate((std, np.zeros((add_h , w))), axis=0)
    if add_w > 0:
        std = np.concatenate((std, np.zeros((h+add_h, add_w))), axis=1)
    h, w = std.shape
    std = std.reshape(h//kernel, kernel, w//kernel, kernel).transpose(0, 2, 1, 3).max(axis=-1).max(axis=-1)
    std = std.reshape(h//kernel, 1, w//kernel, 1).repeat(kernel, axis=1).repeat(kernel, axis=3)
    std = std.reshape(h, w)[:oh, :ow]
    return std
    
def circle(radius=3.5, h=0.0, axis='z', t0=0, r=1):
    if axis == 'z':
        return lambda t: [radius * np.cos(r * t + t0), radius * np.sin(r * t + t0), h]
    elif axis == 'y':
        return lambda t: [radius * np.cos(r * t + t0), h, radius * np.sin(r * t + t0)]
    else:
        return lambda t: [h, radius * np.cos(r * t + t0), radius * np.sin(r * t + t0)]

def normalize(x, axis=-1, order=2):
    if isinstance(x, torch.Tensor):
        l2 = x.norm(p=order, dim=axis, keepdim=True)
        return x / (l2 + 1e-8), l2

    else:
        l2 = np.linalg.norm(x, order, axis)
        l2 = np.expand_dims(l2, axis)
        l2[l2 == 0] = 1
        return x / l2,


def cross(x, y, axis=0):
    T = torch if isinstance(x, torch.Tensor) else np
    return T.cross(x, y, axis)


def cat(x, axis=1):
    if isinstance(x[0], torch.Tensor):
        return torch.cat(x, dim=axis)
    return np.concatenate(x, axis=axis)
    
def look_at_rotation(camera_position, at=None, up=None, inverse=False, cv=False):
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.
    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.
    Input:
        camera_position: 3
        at: 1 x 3 or N x 3  (0, 0, 0) in default
        up: 1 x 3 or N x 3  (0, 1, 0) in default
    """

    if at is None:
        at = torch.zeros_like(camera_position)
    else:
        at = torch.tensor(at).type_as(camera_position)
    if up is None:
        up = torch.zeros_like(camera_position)
        up[2] = -1
    else:
        up = torch.tensor(up).type_as(camera_position)

    z_axis = normalize(at - camera_position)[0]
    x_axis = normalize(cross(up, z_axis))[0]
    y_axis = normalize(cross(z_axis, x_axis))[0]

    R = cat([x_axis[:, None], y_axis[:, None], z_axis[:, None]], axis=1)
    return R 

def gen_path(pos_gen, at=(0, 0, 0), up=(0, -1, 0), frames=180):
    c2ws = []
    for t in range(frames):
        c2w = torch.eye(4)
        cam_pos = torch.tensor(pos_gen(t * (360.0 / frames) / 180 * np.pi))
        cam_rot = look_at_rotation(cam_pos, at=at, up=up, inverse=False, cv=True)
        c2w[:3, 3], c2w[:3, :3] = cam_pos, cam_rot
        c2ws.append(c2w)
    return torch.stack(c2ws)
