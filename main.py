import logging
import argparse
from glob import glob
from tqdm import tqdm
import sys,os
import math
from time import time
import json

from inception_i3d.pytorch_i3d import InceptionI3d
from original_attacks.fgsm import FGSM
from ours_attacks.temp_sign import temp_sign_comparison

import torch
import torch.nn as nn
import torchvision
from torchvision import io

import numpy as np
from numpy import random

from utils.args_parser import tempzero_argparser
from utils.vid_save import save_video
from utils.make_folder import get_folder

def main():  
    ####CFG to parser####
    args = tempzero_argparser()
    args_save = vars(args) #args to dict
    epsilon = args.epsilon
    train_folder = args.train_folder
    targeted_attack = args.targeted_attack
    our_method = args.our_method
    logger_name = args.logger_name
    c_value = args.c_value
    save_folder = args.save_foler
    save_path = get_folder(save_folder)
    
    os.makedirs(save_path+'/base_attack')
    os.makedirs(save_path+'/ours_attack')
    os.makedirs(save_path+'/concate_videos')
    
    #####################
    
    CE_loss = nn.CrossEntropyLoss().to(torch.device('cuda'))
    
    ####log setting####
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # log 출력 형식
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # log 출력
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # log를 파일에 출력
    file_handler = logging.FileHandler(f'{save_path}/{logger_name}.log')
    
    with open(f'{save_path}/args.json', 'w') as f:
        json.dump(args_save, f, indent=4)
        
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    #####################
    
    file_list = glob(f'{train_folder}\\*/*.mp4',recursive=True)
    tmp_label_list = sorted(glob(f'{train_folder}\\*'))
    label_list = []
    
    for label in tmp_label_list:
        label_list.append(label.split('\\')[-1])
    
    num_of_vid = 0
    
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load('inception_i3d/models/rgb_imagenet.pt'))
    i3d.cuda()
    i3d.eval()
    
    fgsm_success_rate = 0
    ours_success_rate = 0
    
    for source in tqdm(file_list):
        num_of_vid += 1
        f_name = source.split('\\')[-1]
        c_name = source.split('\\')[-2]
        
        input_fps = int(round(torchvision.io.read_video(source)[2]['video_fps']))

        true_label = label_list.index(source.split('\\')[-2])
        
        ## shape : [T,H,W,C]
        vid = torchvision.io.read_video(source)[0]
        vid = vid.permute(0, 3, 1,2) #tchw
        # vid = torch.nn.functional.interpolate(vid, size=(224, 224), mode="nearest")

        vid = vid / 255
        
        if vid.shape[0]%2!=0:
            last_frmae = vid[-1:,:,:,:]
            vid = torch.cat((vid,last_frmae),dim=0).numpy()
            vid = torch.tensor(vid, dtype=torch.float, device='cuda')

        else:
            vid = vid.cuda()
            
        vid = vid.unsqueeze(0)

        vid = vid.permute(0, 2, 1, 3,4)
        vid = vid[:,:,:256,:,:]
        
        vid_label = torch.zeros((1,400), dtype=torch.float)
        
        if targeted_attack==False:
            while(True):
                random_target = random.randint(400)
                if random_target != true_label:
                    target_label = random_target
                    break
                
            vid_label[0, target_label] = 1
            vid_label= vid_label.cuda()
            
        else:
            target_label = true_label
            vid_label[0, true_label] = 1
            vid_label= vid_label.cuda()
            
        logger.info(f'######{num_of_vid}#######')
        logger.info(f'video_path : {source} fps : {input_fps} ')
        logger.info(f'input label : {true_label} target_label : {target_label}')
        
        vid.requires_grad = True
        
        output = i3d(vid)
        output = output.mean(2)
        
        loss = CE_loss(output, vid_label)
        i3d.zero_grad()
        loss.backward()

        data_grad = vid.grad.data

        ####FGSM####
        fgsm_attack = FGSM(vid, epsilon, data_grad, targeted_attack)
        base_vid = fgsm_attack.forward()
        
        fgsm_out = i3d(base_vid)
        fgsm_out = fgsm_out.mean(2)
        fgsm_out = torch.argmax(fgsm_out)
        ############
        
        if our_method=='temp_zero' or 'tempzero' or 'temp_sign' or 'sing_comparison':
            temp_sign_attack = temp_sign_comparison(vid, epsilon, data_grad, c_value, targeted_attack)
            ours_vid, epsilon_c = temp_sign_attack.forward()
            ours_out = i3d(ours_vid)
            ours_out = ours_out.mean(2)
            ours_out = torch.argmax(ours_out)

        if targeted_attack:
            if fgsm_out==target_label:
                fgsm_success_rate +=1
            if ours_out==target_label:
                ours_success_rate +=1
                
        else:
            if fgsm_out!=true_label:
                fgsm_success_rate +=1
            if ours_out!=true_label:
                ours_success_rate +=1
        
        logger.info(f'FGSM_predict_label : {fgsm_out} \t {our_method}_predict_label : {ours_out}')
        logger.info(f'fgsm success rate : {fgsm_success_rate/num_of_vid} cw success rate : {ours_success_rate/num_of_vid}')
        
        base_vid = base_vid.squeeze(0)
        base_vid = base_vid.permute(1, 0, 2, 3)
        
        ours_vid = ours_vid.squeeze(0)
        ours_vid = ours_vid.permute(1, 0, 2, 3)
        
        concate_vid = torch.cat((base_vid, ours_vid),dim=3)
        base_vid = base_vid.cpu().detach().numpy()
        ours_vid = ours_vid.cpu().detach().numpy()
        concate_vid = concate_vid.cpu().detach().numpy()
        
        i3d.zero_grad()
        
        save_video(base_vid, save_path+f'/base_attack/{c_name}', f_name, 7, input_fps, 0) 
        save_video(ours_vid, save_path+f'/ours_attack/{c_name}', f_name, 7, input_fps, 0)
        save_video(concate_vid, save_path+f'/concate_videos/{c_name}', f_name, 7, input_fps, 0)
        
if __name__=='__main__':
    main()