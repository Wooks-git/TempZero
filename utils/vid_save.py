import numpy as np
import skvideo.io
from utils.psnr import psnr
import os

def save_video(video, save_path, f_name,duration, fps, psnr):
    os.makedirs(save_path,exist_ok=True)

    writer = skvideo.io.FFmpegWriter(save_path+f'/{f_name}', inputdict={'-r':f'{fps}'},outputdict={
                '-vcodec': 'libx264',  # use the h.264 codec
                '-crf': '0',  # set the constant rate factor to 0, which is lossless
                '-preset': 'ultrafast',  # the slower the better compression, in princple, try
                '-r':f'{fps}'
                # other options see https://trac.ffmpeg.org/wiki/Encode/H.264
            })

    for _ in range(duration):
        
        for frame in video:
            
            frame = frame * 255
            frame = frame.astype(np.uint8)
            frame = frame.transpose(1, 2, 0)

            writer.writeFrame(frame)

    writer.close()