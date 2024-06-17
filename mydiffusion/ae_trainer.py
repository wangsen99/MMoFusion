# Copyright (c) HuaWei, Inc. and its affiliates.
# liu.haiyang@huawei.com
# Train script for audio2pose

import os
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
import pprint
from diffusion import logger

from utils import other_tools

from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from train_ae import BaseTrainer


class CustomTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.g_name = args.g_name
        self.pose_length = args.pose_length
        self.grad_norm = args.grad_norm
        self.loss_meters = {
            'rec_val': other_tools.AverageMeter('rec_val'),
            'vel_val': other_tools.AverageMeter('vel_val'),
            'kl_val': other_tools.AverageMeter('kl_val'),
            'all': other_tools.AverageMeter('all'),
            'rec_l1': other_tools.AverageMeter('rec_l1'), 
            'vel_l1': other_tools.AverageMeter('vel_l1'),
            'kl_loss': other_tools.AverageMeter('kl_loss'),
        }
        self.best_epochs = {
            'rec_val': [np.inf, 0],
            'vel_val': [np.inf, 0],
            'kl_val': [np.inf, 0],
                           }
        self.rec_loss = torch.nn.L1Loss(reduction='none')
        self.vel_loss = torch.nn.MSELoss(reduction='none')
        self.variational_encoding = args.variational_encoding
        self.rec_weight = args.rec_weight
        self.vel_weight = args.vel_weight
        self.steps = 0
    
    def train(self, args):
        self.model.train()
        its_len = args.max_num_steps
        t_start = time.time()
        
        for its, batch in enumerate(self.train_loader):
            self.steps = its+1
            wavlm, pose_seq, style, emotion = batch
            tar_pose = pose_seq
            # print(pose_seq.shape)
            t_data = time.time() - t_start
            tar_pose = tar_pose.cuda()
            t_data = time.time() - t_start 

            self.opt.zero_grad()
            poses_feat, pose_mu, pose_logvar, recon_data = \
                self.model(None, tar_pose, variational_encoding=self.variational_encoding)
            
            recon_loss = self.rec_loss(recon_data, tar_pose) # 128*34*123
            recon_loss = torch.mean(recon_loss, dim=(1, 2)) # 128
            self.loss_meters['rec_l1'].update(torch.sum(recon_loss).item()*self.rec_weight)
            recon_loss = torch.sum(recon_loss*self.rec_weight)
            # rec vel loss
            if self.vel_weight > 0:  # use pose diff
                target_diff = tar_pose[:, 1:] - tar_pose[:, :-1]
                recon_diff = recon_data[:, 1:] - recon_data[:, :-1]
                vel_rec_loss = torch.mean(self.vel_loss(recon_diff, target_diff), dim=(1, 2))
                self.loss_meters['vel_l1'].update(torch.sum(vel_rec_loss).item()*self.vel_weight)
                recon_loss += (torch.sum(vel_rec_loss)*self.vel_weight)
            # KLD
            if self.variational_encoding:
                KLD = -0.5 * torch.sum(1 + pose_logvar - pose_mu.pow(2) - pose_logvar.exp())
                if self.steps < 10:
                    KLD_weight = 0
                else:
                    KLD_weight = min(1.0, (self.steps - 10) * 0.05) * 0.01
                loss = recon_loss + KLD_weight * KLD
                self.loss_meters['kl_loss'].update(KLD_weight * KLD.item())
            else:
                loss = recon_loss
            self.loss_meters['all'].update(loss.item())
            if self.grad_norm != 0 and "LSTM" in self.g_name: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
#             logger.warning(total_norm)
            loss.backward()
            self.opt.step()
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            # --------------------------- recording ---------------------------------- #
            if self.steps % self.log_period == 0:
                self.recording(self.steps, its, its_len, self.loss_meters, lr_g, 0, t_data, t_train, mem_cost)   
            #if its == 1:break
            # self.opt_s.step(epoch)
            if (self.steps) % args.save_period == 0:
                if self.rank == 0:
                # trainer.test(epoch)
                    other_tools.save_checkpoints(os.path.join(self.checkpoint_path, f"last_{self.steps}.bin"), self.model, opt=None, epoch=None, lrs=None)
            if self.steps > its_len:
                break
                    
    def val(self, epoch):
        self.model.eval()
        with torch.no_grad():
            its_len = len(self.val_loader)
            for its, dict_data in enumerate(self.val_loader):
                tar_pose = dict_data["pose"]
                tar_pose = tar_pose.cuda()
                
                poses_feat, pose_mu, pose_logvar, recon_data = \
                self.model(None, tar_pose, variational_encoding=self.variational_encoding)
                if self.vel_weight > 0:  # use pose diff
                    target_diff = tar_pose[:, 1:] - tar_pose[:, :-1]
                    recon_diff = recon_data[:, 1:] - recon_data[:, :-1]
                    vel_rec_loss = torch.mean(self.vel_loss(recon_diff, target_diff), dim=(0, 1, 2))
                    self.loss_meters['vel_val'].update(vel_rec_loss.item())
                #print(recon_data.shape, tar_pose.shape)    
                recon_loss = F.l1_loss(recon_data, tar_pose, reduction='none')
                recon_loss = torch.mean(recon_loss, dim=(0, 1, 2))
                self.loss_meters['rec_val'].update(recon_loss.item())
            self.val_recording(epoch, self.loss_meters)