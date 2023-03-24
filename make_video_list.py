import os

input_dir = '/gpfs/data/ssrinath/projects/stnerf/mixvoxels/log/hand_v2_no_bg/imgs_path_all'
mp4_list = []
num_files = len(os.listdir(input_dir))
for i in range(num_files):
    if os.path.isfile(os.path.join(input_dir, "cam_{}_comp_video.mp4".format(i))):
        mp4_list.append('cam_{}_comp_video.mp4'.format(i))
with open(os.path.join(input_dir, 'inputs.txt'), 'w') as o:
    for f in mp4_list:
        o.write('file \'' + f + '\'\n')
    o.close()


#ffmpeg -f concat -i inputs.txt -c copy concatenated.mp4

