import os
from os.path import isfile, join

import cv2
import hydra
import numpy as np
from tqdm import tqdm


@hydra.main(config_path="../../configs", config_name="test")
def main(cfg):
    print('-'*30, 'Converting Images to a Video', '-'*30)
    img_path= os.path.join(hydra.utils.get_original_cwd(), cfg.MODEL_PATH, f'result/{cfg.TEST_ID}/demo_frames/')
    video_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.MODEL_PATH, f'result/{cfg.TEST_ID}/demo_video/')
    if not os.path.exists(video_dir):
        os.makedirs(video_dir, exist_ok=True)
    out_path = video_dir + f'{cfg.TEST_ID}.mp4'
    fps = 10 #15 # 30 #0.5
    frame_array = []
    files = [f for f in os.listdir(img_path) if isfile(join(img_path, f))]
    #for sorting the file names properly
    files.sort(key = lambda x: x[5:-4])
    files.sort()

    print('Loading images...')
    for i in tqdm(range(len(files)), total=len(files)):
        filename=img_path + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        
        #inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    print('Generating a video...')
    for i in tqdm(range(len(frame_array)), total=len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
    print(f'Video Saving to: {out_path}')


if __name__ == '__main__':
    main()
