import pdb
import sys
[sys.path.append(i) for i in ['.', '..', '../process', '../model']]
from model.mdm  import MDM
from utils.model_util import create_gaussian_diffusion, load_model_wo_clip
from utils import metric, other_tools, data_tools
import subprocess
import os
from datetime import datetime
import copy
import librosa
import numpy as np
import yaml
from pprint import pprint
import torch
import torch.nn.functional as F
from easydict import EasyDict
import math
from process_BEAT_bvh import wav2wavlm, pose2bvh, pose2bvh_bugfix, pose2euler
from process_TWH_bvh import pose2bvh as pose2bvh_twh
from process_TWH_bvh import wavlm_init, load_metadata
from torch.utils.data import DataLoader
from data_loader.h5_data_loader import SpeechGestureDataset, RandomSampler, SequentialSampler
import argparse

def create_model_and_diffusion(args):
    model = MDM(args, modeltype='', njoints=args.njoints, nfeats=1, cond_mode=config.cond_mode, audio_feat=args.audio_feat,
                arch='trans_enc', latent_dim=args.latent_dim, n_seed=args.n_seed, cond_mask_prob=args.cond_mask_prob, device=device_name,
                style_dim=args.speaker_num, source_audio_dim=args.audio_feature_dim,
                audio_feat_dim_latent=args.audio_feat_dim_latent)
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion

def main(args, save_dir, model_path, tst_path=None, max_len=0, skip_timesteps=0, tst_prefix=None, dataset='BEAT', 
         wav_path=None, txt_path=None, wavlm_path=None, word2vector_path=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # sample
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args)
    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(mydevice)
    model.eval()
    sample_fn = diffusion.p_sample_loop  # predict x_start

    if args.e_name is not None:
        eval_model_module = __import__(f"model.{args.eval_model}", fromlist=["something"])
        eval_model = getattr(eval_model_module, args.e_name)(args)
        other_tools.load_checkpoints(eval_model, args.root_path+args.e_path, args.e_name)  
        eval_model.to(mydevice)
        eval_model.eval()
        
            
    tst_audio_dir = os.path.join(tst_path, 'audio_' + dataset)
    tst_wav_dir = os.path.join(tst_path, 'wav_' + dataset)
    tst_text_dir = os.path.join(tst_path, 'text_' + dataset)
    tst_gesture_dir = os.path.join(tst_path, 'gesture_' + dataset)
    tst_sem_dir = os.path.join(tst_path, 'sem_' + dataset)
    alignmenter = metric.alignment(0.3, 2)
    if args.body =='upper':
        pose_joints = 62
    else:
        pose_joints = 76
    srgr_calculator = metric.SRGR(4, pose_joints)
    l1_calculator = metric.L1div()
    t_start = 10
    t_end = 500
    align = 0 
    pose_fps = 30
    for its, file in enumerate(os.listdir(tst_gesture_dir)):
        filename = file.split(".")[0]
        print(f"Processing: {filename}")
        if dataset == 'BEAT':
            # speaker_id = int(filename.split('_')[0]) - 1
            speaker_id = 9
            speaker = np.zeros([args.speaker_num])
            speaker[speaker_id] = 1
            
        audio_path = os.path.join(tst_audio_dir, filename + '.npy')
        audio = np.load(audio_path)
        wav_path = os.path.join(tst_wav_dir, filename + '.npy')
        wav = np.load(wav_path)
        text_path = os.path.join(tst_text_dir, filename + '.npy')
        text = np.load(text_path)
        sem_path = os.path.join(tst_sem_dir, filename + '.npy')
        sem = np.load(sem_path)
        textaudio = np.concatenate((audio, text), axis=-1)
        textaudio = torch.FloatTensor(textaudio)
        textaudio = textaudio.to(mydevice)

        gesture_path = os.path.join(tst_gesture_dir, filename + '.npy')
        gesture = np.load(gesture_path)

        seed=123456
        torch.manual_seed(seed)

        style=speaker
        n_frames = textaudio.shape[0]

        if n_frames >=6000:
            n_frames = 6000
            textaudio = textaudio[:n_frames]

        real_n_frames = copy.deepcopy(n_frames)     # 1830

        stride_poses = args.n_poses - args.n_seed

        if n_frames < stride_poses:
            num_subdivision = 1
            n_frames = stride_poses
        else:
            num_subdivision = math.ceil(n_frames / stride_poses)
            n_frames = num_subdivision * stride_poses
            print('real_n_frames: {}, num_subdivision: {}, stride_poses: {}, n_frames: {}, speaker_id: {}'.format(real_n_frames, num_subdivision, stride_poses, n_frames, np.where(style==np.max(style))[0][0]))

        model_kwargs_ = {'y': {}}
        model_kwargs_['y']['mask'] = (torch.zeros([1, 1, 1, args.n_poses]) < 1).to(mydevice)
        model_kwargs_['y']['style'] = torch.as_tensor([style]).float().to(mydevice)
        model_kwargs_['y']['mask_local'] = torch.ones(1, args.n_poses).bool().to(mydevice)
        
        textaudio_pad = torch.zeros([n_frames - real_n_frames, args.audio_feature_dim]).to(mydevice)
        textaudio = torch.cat((textaudio, textaudio_pad), 0)
        audio_reshape = textaudio.reshape(num_subdivision, stride_poses, args.audio_feature_dim).transpose(0, 1)

        if dataset == 'BEAT':
            data_mean_ = np.load("../process/gesture_BEAT_mean_" + args.version + ".npy")
            data_std_ = np.load("../process/gesture_BEAT_std_" + args.version + ".npy")
        elif dataset == 'TWH':
            data_mean_ = np.load("../process/gesture_TWH_mean_v0" + ".npy")
            data_std_ = np.load("../process/gesture_TWH_std_v0" + ".npy")

        data_mean = np.array(data_mean_)
        data_std = np.array(data_std_)
        # std = np.clip(data_std, a_min=0.01, a_max=None)
        
        shape_ = (1, model.njoints, model.nfeats, args.n_poses)
        out_list = []
        for i in range(0, num_subdivision):
            print(i, num_subdivision)
            model_kwargs_['y']['audio'] = audio_reshape[:, i:i + 1]
            if i == 0:
                if args.name == 'DiffuseStyleGesture':
                    pad_zeros = torch.zeros([args.n_seed, 1, args.audio_feature_dim]).to(mydevice)
                    model_kwargs_['y']['audio'] = torch.cat((pad_zeros, model_kwargs_['y']['audio']), 0).transpose(0, 1)      # attention 3
                elif args.name == 'DiffuseStyleGesture+':
                    model_kwargs_['y']['audio'] = model_kwargs_['y']['audio'].transpose(0, 1)       # attention 4

                # model_kwargs_['y']['seed'] = torch.zeros([1, args.njoints, 1, args.n_seed]).to(mydevice)

                if args.body == 'upper':
                    seed_gesture = gesture[:args.n_seed, 6*3:192*3]
                else:
                    seed_gesture = gesture[:args.n_seed]
                # seed_gesture_vel = seed_gesture[1:] - seed_gesture[:-1]
                # seed_gesture_acc = seed_gesture_vel[1:] - seed_gesture_vel[:-1]
                # seed_gesture_ = np.concatenate((seed_gesture[2:], seed_gesture_vel[1:], seed_gesture_acc), axis=1)      # (args.n_seed, args.njoints)
                seed_gesture_ = torch.from_numpy(seed_gesture).float().transpose(0, 1).unsqueeze(0).to(mydevice)
                model_kwargs_['y']['seed'] = seed_gesture_.unsqueeze(2)

            else:
                if args.name == 'DiffuseStyleGesture':
                    pad_audio = audio_reshape[-args.n_seed:, i - 1:i]
                    model_kwargs_['y']['audio'] = torch.cat((pad_audio, model_kwargs_['y']['audio']), 0).transpose(0, 1)        # attention 3
                elif args.name == 'DiffuseStyleGesture+':
                    model_kwargs_['y']['audio'] = model_kwargs_['y']['audio'].transpose(0, 1)  # attention 4

                model_kwargs_['y']['seed'] = out_list[-1][..., -args.n_seed:].to(mydevice)

            sample = sample_fn(
                model,
                shape_,
                clip_denoised=False,
                model_kwargs=model_kwargs_,
                skip_timesteps=skip_timesteps,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,  # None, torch.randn(*shape_, device=mydevice)
                const_noise=False,
            )
            # smoothing motion transition
            if len(out_list) > 0 and args.n_seed != 0:
                last_poses = out_list[-1][..., -args.n_seed:]        # # (1, model.njoints, 1, args.n_seed)
                out_list[-1] = out_list[-1][..., :-args.n_seed]  # delete last 4 frames
                # if smoothing:
                #     # Extract predictions
                #     last_poses_root_pos = last_poses[:, :12]        # (1, 3, 1, 8)
                #     next_poses_root_pos = sample[:, :12]        # (1, 3, 1, 88)
                #     root_pos = last_poses_root_pos[..., 0]      # (1, 3, 1)
                #     predict_pos = next_poses_root_pos[..., 0]
                #     delta_pos = (predict_pos - root_pos).unsqueeze(-1)      # # (1, 3, 1, 1)
                #     sample[:, :12] = sample[:, :12] - delta_pos
                for j in range(len(last_poses)):
                    n = len(last_poses)
                    prev = last_poses[..., j]
                    next = sample[..., j]
                    sample[..., j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)
            out_list.append(sample)

        out_list = [i.detach().data.cpu().numpy()[:, :args.njoints] for i in out_list]
        if len(out_list) > 1:
            out_dir_vec_1 = np.vstack(out_list[:-1]) # (num-1, 684, 1, 120)
            sampled_seq_1 = out_dir_vec_1.squeeze(2).transpose(0, 2, 1).reshape(batch_size, -1, model.njoints) # (1, 1680, 684)
            out_dir_vec_2 = np.array(out_list[-1]).squeeze(2).transpose(0, 2, 1)
            sampled_seq = np.concatenate((sampled_seq_1, out_dir_vec_2), axis=1)
        else:
            sampled_seq = np.array(out_list[-1]).squeeze(2).transpose(0, 2, 1)

        if args.body == 'upper':
            zeros_sample = np.zeros((sampled_seq[0].shape[0], 42*3))
            root_sample = np.concatenate([zeros_sample[:, :6*3], sampled_seq[0]], axis=1)
            lower_sample = np.concatenate([root_sample, zeros_sample[:, 6*3:]], axis=1)
            out_matrix = lower_sample
        else :
            out_matrix = sampled_seq[0]

        
        out_poses = np.multiply(out_matrix, data_std) + data_mean
        print(out_poses.shape, real_n_frames)

        out = torch.FloatTensor(out_poses).unsqueeze(0).to(mydevice)
        tar = torch.FloatTensor(gesture).unsqueeze(0).to(mydevice)

        if gesture.shape[0] < real_n_frames:
            real_n_frames = gesture.shape[0]
        out_poses = out_poses[:real_n_frames]
        tar_poses = gesture[:real_n_frames]
        print(tar_poses.shape)

        out_euler = pose2euler(out_poses, gt=False, fixroot=False, normalize_root=False, upper_body=False)
        tar_euler = pose2euler(tar_poses, gt=True, fixroot=False)

        if args.body == 'upper':
            test_out_euler = out_euler[:, 6:192]
            test_tar_euler = tar_euler[:, 6:192]
        else:
            test_out_euler = out_euler
            test_tar_euler = tar_euler

        motion_stride = 120
        motion_stride_poses = 150
        for i in range(0, num_subdivision):
            if i == 0:
                cat_results = out[:, i*motion_stride:i*motion_stride+motion_stride_poses, :]
                cat_targets = tar[:, i*motion_stride:i*motion_stride+motion_stride_poses, :]
                #cat_sem = in_sem[:,i*self.stride:i*self.stride+self.pose_length]
            else:
                cat_results = torch.cat([cat_results, out[:,i*motion_stride:i*motion_stride+motion_stride_poses, :]], 0)
                cat_targets = torch.cat([cat_targets, tar[:,i*motion_stride:i*motion_stride+motion_stride_poses, :]], 0)

        latent_out = eval_model(cat_results)
        latent_ori = eval_model(cat_targets)
        
        # speaker_name = filename.split("_")[0]
        speaker_name = speaker_id
        pose2bvh(save_dir, filename, out_euler, pipeline='../process/resource/data_pipe_30fps' + '_speaker' + str(speaker_name) + '.sav', gt=False)
        # pose2bvh(save_dir, filename, tar_euler, pipeline='../process/resource/data_pipe_30fps' + '_speaker' + str(speaker_name) + '.sav', gt=True)
        if its == 0:
            latent_out_all = latent_out.detach().data.cpu().numpy()
            latent_ori_all = latent_ori.detach().data.cpu().numpy()
        else:
            latent_out_all = np.concatenate([latent_out_all, latent_out.detach().data.cpu().numpy()], axis=0)
            latent_ori_all = np.concatenate([latent_ori_all, latent_ori.detach().data.cpu().numpy()], axis=0)
        

        _ = l1_calculator.run(test_out_euler)
        _ = srgr_calculator.run(test_out_euler, test_tar_euler, sem)
        onset_raw, onset_bt, onset_bt_rms = alignmenter.load_audio(wav.reshape(-1), t_start, t_end, True)
        beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = alignmenter.load_pose(test_out_euler, t_start, t_end, pose_fps, True)
        align += alignmenter.calculate_align(onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist, pose_fps)

    fid = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
    print(f"fid score: {fid}")
    diversity = data_tools.FIDCalculator.get_diversity_scores(latent_out_all, 500)
    print(f"diversity score: {diversity}")
    l1div = l1_calculator.avg()
    print(f"l1div score: {l1div}")
    srgr = srgr_calculator.avg()
    print(f"srgr score: {srgr}")
    align_avg = align/len(os.listdir(tst_gesture_dir))
    print("align score:", align_avg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DiffuseStyleGesture')
    parser.add_argument('--config', default='./configs/DiffuseStyleGesture.yml')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--tst_prefix', nargs='+')
    parser.add_argument('--no_cuda', type=list, default=['0'])
    parser.add_argument('--wav_path', type=str, default=None)
    parser.add_argument('--txt_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='sample_dir')
    parser.add_argument('--max_len', type=int, default=0)
    parser.add_argument('--skip_timesteps', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='BEAT')
    parser.add_argument('--wavlm_path', type=str, default='./WavLM/WavLM-Large.pt')
    parser.add_argument('--word2vector_path', type=str, default='./crawl-300d-2M.vec')
    
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    for k, v in vars(args).items():
        config[k] = v
    # pprint(config)
    config = EasyDict(config)

    assert config.name in ['DiffuseStyleGesture', 'DiffuseStyleGesture+']
    if config.name == 'DiffuseStyleGesture+':
        config.cond_mode = 'cross_local_attention4_style1_sample'
    elif config.name == 'DiffuseStyleGesture':
        config.cond_mode = 'cross_local_attention3_style1_sample'

    if config.dataset == 'BEAT':
        config.audio_feature_dim = 1434
        if 'v0' in config.version:
            config.motion_dim = 684
        elif 'v2' in config.version:
            config.motion_dim = 1141
    else:
        raise NotImplementedError

    device_name = 'cuda:' + args.gpu
    mydevice = torch.device('cuda:' + config.gpu)
    torch.cuda.set_device(int(config.gpu))
    args.no_cuda = args.gpu

    batch_size = 1
    print(config.model_path)
    model_root = config.model_path.split('/')[-5]
    model_spicific = config.model_path.split('/')[-1].split('.')[0]
    config.save_dir = "./" + model_root + '/' + 'sample_dir_' + model_spicific + '_' + config.name + '_' + config.body + '/'

    print('model_root', model_root, 'tst_path', config.tst_path, 'save_dir', config.save_dir)

    config.h5file = '../process/' + config.dataset + '_' + config.version + '_test' + '.h5'

    main(config, config.save_dir, config.model_path, tst_path=config.tst_path, max_len=config.max_len,
         skip_timesteps=config.skip_timesteps, tst_prefix=config.tst_prefix, dataset=config.dataset, 
         wav_path=config.wav_path, txt_path=config.txt_path, wavlm_path=config.wavlm_path, word2vector_path=config.word2vector_path)

