import os
import shutil

root = "/gpfs/data/ssrinath/projects/stnerf/mixvoxels/hand_v2_no_bg"
image_dir = os.path.join(root, 'image', 'undist')
target_dir = os.path.join(root, 'frames_1')
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
for i, f in enumerate(sorted(os.listdir(image_dir))):
    print(f)
    cam, frame = f.split('_')
    target = os.path.join(target_dir, cam)
    if not os.path.exists(target):
        os.mkdir(target)
    shutil.copy(os.path.join(image_dir, f), os.path.join(target, frame))
