import pdb
import logging
logging.getLogger().setLevel(logging.INFO)
from torch.utils.data import DataLoader
from data_loader.h5_data_loader import RandomSampler, WholeSpeechGestureDataset, UpperSpeechGestureDataset
import torch
import yaml
from pprint import pprint
from easydict import EasyDict
from configs.parse_args import parse_args
import os
import sys
[sys.path.append(i) for i in ['.', '..', '../model', '../train']]
from utils.model_util import create_gaussian_diffusion
from utils import other_tools
from training_loop import TrainLoop
from diffusion import logger
import time

def create_model_and_diffusion(args):
    
    if args.model_name == 'mmofusion':
        from model.mmofusion  import MMoFusion
        print('use mmofusion')

    model = MMoFusion(args, modeltype='', njoints=args.njoints, nfeats=1, cond_mode=args.cond_mode, audio_feat=args.audio_feat,
                arch='trans_enc', latent_dim=args.latent_dim, n_seed=args.n_seed, cond_mask_prob=args.cond_mask_prob, device=device_name,
                style_dim=args.speaker_num, emo_dim=args.emo_dim, num_layers=args.num_layers, num_heads=args.num_heads, source_audio_dim=args.audio_feature_dim, 
                audio_feat_dim_latent=args.audio_feat_dim_latent)

    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def main(args):

    logger.configure(args)
    other_tools.set_random_seed(args)
    # Get data, data loaders and collate function ready
    print("Loading dataset into memory ...")
    if args.body=='whole':
        trn_dataset = WholeSpeechGestureDataset(args.h5file, motion_dim=args.motion_dim, style_dim=args.speaker_num,
                                        sequence_length=args.n_poses, npy_root="../process", 
                                        version=args.version, dataset=args.dataset)
    elif args.body== 'upper':
        trn_dataset = UpperSpeechGestureDataset(args.h5file, motion_dim=args.motion_dim, style_dim=args.speaker_num,
                                sequence_length=args.n_poses, npy_root="../process", 
                                version=args.version, dataset=args.dataset)

    train_loader = DataLoader(trn_dataset, num_workers=args.num_workers,
                              sampler=RandomSampler(0, len(trn_dataset)),
                              batch_size=args.batch_size,
                              pin_memory=True,
                              drop_last=False)

    model, diffusion = create_model_and_diffusion(args)
    model.to(mydevice)
    TrainLoop(args, model, diffusion, mydevice, data=train_loader).run_loop()


if __name__ == '__main__':
    '''
    python train.py --config=./configs/mmofusion.yml --gpu 0
    '''

    args = parse_args()
    device_name = 'cuda:' + args.gpu
    mydevice = torch.device(device_name)
    torch.cuda.set_device(int(args.gpu))
    args.no_cuda = args.gpu

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v

    config = EasyDict(config)
    config.model_name = config.name
    print(config.model_name)

    time_local = time.localtime()
    name_expend = "%02d%02d_%02d%02d%02d_"%(time_local[1], time_local[2],time_local[3], time_local[4], time_local[5])
    config.name = name_expend + config.name + '_' + config.body

    main(config)
