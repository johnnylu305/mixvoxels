import os
import imageio
input_dir = '/gpfs/data/ssrinath/projects/stnerf/mixvoxels/log/hand_v2_no_bg/imgs_path_all'
mp4_list = []
num_files = len(os.listdir(input_dir))
frames = []
for i in range(num_files):
    if os.path.isfile(os.path.join(input_dir, "cam_{}_comp_video.mp4".format(i))):
        mp4_list.append(os.path.join(input_dir, 'cam_{}_comp_video.mp4'.format(i)))

for f in mp4_list:
    vid = imageio.get_reader(f, 'ffmpeg').get_data(0)
    frames.append(vid)

writer = imageio.get_writer(os.path.join(input_dir, "concatenated_1_frame.mp4"), fps=30)

for frame in frames:
    writer.append_data(frame)
writer.close()