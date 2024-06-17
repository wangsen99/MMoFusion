# Copyright (c) HuaWei, Inc. and its affiliates.
# liu.haiyang@huawei.com
# Train script for audio2pose

import os
import signal
import time
import csv
import sys
[sys.path.append(i) for i in ['.', '..', '../model', '../train']]
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from data_loader.h5_data_loader import AllSpeechGestureDataset, RandomSampler, SequentialSampler
import torch.multiprocessing as mp
import numpy as np
import time
import pprint
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader
from diffusion import logger
from configs.parse_args import parse_args
from utils import other_tools
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func


class BaseTrainer(object):
    def __init__(self, args):
        self.rank = dist.get_rank()
        self.checkpoint_path = args.root_path+args.out_root_path + "custom/" + args.name + "/" #wandb.run.dir #args.root_path+args.out_root_path+"/"+args.name
        self.batch_size = args.batch_size
        self.gpus = len(args.gpu)
        
        self.best_epochs = {
            'fid_val': [np.inf, 0],
            'rec_val': [np.inf, 0],
                           }
        self.loss_meters = {
            'fid_val': other_tools.AverageMeter('fid_val'),
            'rec_val': other_tools.AverageMeter('rec_val'),
            'all': other_tools.AverageMeter('all'),
            'rec': other_tools.AverageMeter('rec'), 
            'gen': other_tools.AverageMeter('gen'),
            'dis': other_tools.AverageMeter('dis'), 
        } 
      
        self.log_period = args.log_period
        print("Loading dataset into memory ...")
        self.train_data = AllSpeechGestureDataset(args.h5file, motion_dim=args.motion_dim, style_dim=args.style_dim,
                                       sequence_length=args.n_poses, npy_root="../process", 
                                       version=args.version, dataset=args.dataset)  

        self.train_loader = DataLoader(self.train_data, num_workers=args.num_workers,
                              sampler=RandomSampler(0, len(self.train_data)),
                              batch_size=args.batch_size,
                              pin_memory=True,
                              drop_last=False)
        
        self.train_length = args.max_num_steps
        logger.info(f"Init train dataloader success")
        
        model_module = __import__(f"model.{args.model}", fromlist=["something"])
        
        # self.model = torch.nn.DataParallel(getattr(model_module, args.g_name)(args), args.gpu).cuda()
        self.model = getattr(model_module, args.g_name)(args).cuda()

        if self.rank == 0:
            logger.info(self.model)
            logger.info(f"init {args.g_name} success")
        
        self.opt = create_optimizer(args, self.model)
        # self.opt_s = create_scheduler(args, self.opt)
       
    def recording(self, steps, its, its_len, loss_meters, lr_g, lr_d, t_data, t_train, mem_cost):
        if self.rank == 0:
            pstr = "[%03d/%03d]  "%(steps, its_len)
            # step=epoch*self.train_length+its
            for name, loss_meter in self.loss_meters.items():
                if "val" not in name:
                    if loss_meter.count > 0:
                        pstr += "{}: {:.3f}\t".format(loss_meter.name, loss_meter.avg)
                        loss_meter.reset()
            pstr += "glr: {:.1e}\t".format(lr_g)
            pstr += "dlr: {:.1e}\t".format(lr_d)
            pstr += "dtime: %04d\t"%(t_data*1000)        
            pstr += "ntime: %04d\t"%(t_train*1000)
            pstr += "mem: {:.2f} ".format(mem_cost*self.gpus)
            logger.info(pstr)
     
    def val_recording(self, epoch, metrics):
        if self.rank == 0: 
            pstr_curr = "Curr info >>>>  "
            pstr_best = "Best info >>>>  "

            for name, metric in metrics.items():
                if "val" in name:
                    if metric.count > 0:
                        pstr_curr += "{}: {:.3f}     \t".format(metric.name, metric.avg)
                        logger.info({metric.name: metric.avg}, step=epoch*self.train_length)
                        if metric.avg < self.best_epochs[metric.name][0]:
                            self.best_epochs[metric.name][0] = metric.avg
                            self.best_epochs[metric.name][1] = epoch
                            other_tools.save_checkpoints(os.path.join(self.checkpoint_path, f"{metric.name}.bin"), self.model, opt=None, epoch=None, lrs=None)        
                        metric.reset()
            for k, v in self.best_epochs.items():
                pstr_best += "{}: {:.3f}({:03d})\t".format(k, v[0], v[1])
            logger.info(pstr_curr)
            logger.info(pstr_best)  

def main_worker(rank, world_size, args):
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
    logger.configure(args)
    other_tools.set_random_seed(args)
    other_tools.print_exp_info(args)
      
    # return one intance of trainer
    trainer = __import__(f"{args.trainer}_trainer", fromlist=["something"]).CustomTrainer(args) if args.trainer != "base" else BaseTrainer(args) 
     
    logger.info("Training from starch ...")          

    trainer.train(args) 
            
    
            
if __name__ == "__main__":
    os.environ["MASTER_ADDR"]='localhost'
    os.environ["MASTER_PORT"]='2222'
    args = parse_args()
    device_name = 'cuda:' + args.gpu
    mydevice = torch.device(device_name)
    torch.cuda.set_device(int(args.gpu))
    args.no_cuda = args.gpu

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    # pprint(config)

    config = EasyDict(config)

    time_local = time.localtime()
    name_expend = "%02d%02d_%02d%02d%02d_"%(time_local[1], time_local[2],time_local[3], time_local[4], time_local[5])
    config.name = name_expend + config.name

    config.h5file = '../process/' + config.dataset + '_' + config.version + '_train.h5'

    main_worker(0, 1, config)