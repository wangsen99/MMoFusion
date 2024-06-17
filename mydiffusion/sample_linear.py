import pdb
import sys
[sys.path.append(i) for i in ['.', '..', '../process', '../model']]
from utils.model_util import create_gaussian_diffusion, load_model_wo_clip
from utils import metric, other_tools, data_tools
# import subprocess
import os
from datetime import datetime
import copy
import numpy as np
import yaml
import torch
import torch.nn.functional as F
from easydict import EasyDict
import math
from process_BEAT_bvh import wav2wavlm, pose2bvh, pose2bvh_bugfix, pose2euler
import argparse
from model.cfg_sampler import ClassifierFreeSampleModel
from model.motion_autoencoder import HalfEmbeddingNet

def create_model_and_diffusion(args):
    
    if args.model_name == 'mmofusion':
        from model.mmofusion  import MMoFusion
        print('use mmofusion')

    model = MMoFusion(args, modeltype='', njoints=args.njoints, nfeats=1, cond_mode=args.cond_mode, audio_feat=args.audio_feat,
                arch='trans_enc', latent_dim=args.latent_dim, n_seed=args.n_seed, cond_mask_prob=args.cond_mask_prob, device=args.device_name,
                style_dim=args.speaker_num, emo_dim=args.emo_dim, num_layers=args.num_layers, num_heads=args.num_heads, source_audio_dim=args.audio_feature_dim, 
                audio_feat_dim_latent=args.audio_feat_dim_latent)
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion

def main(args, save_dir, model_path, tst_path=None, max_len=0, skip_timesteps=0, tst_prefix=None, dataset='BEAT', 
         wav_path=None, mydevice=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # sample
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args)
    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler

    model.to(mydevice)
    model.eval()
    sample_fn = diffusion.p_sample_loop  # predict x_start
    if args.e_name is not None:
        eval_model_module = __import__(f"model.{args.eval_model}", fromlist=["something"])
        eval_model = getattr(eval_model_module, args.e_name)(args)
        other_tools.load_checkpoints(eval_model, args.e_path, args.e_name)  
        eval_model.to(mydevice)
        eval_model.eval()
            
    tst_audio_dir = os.path.join(tst_path, 'audio_' + dataset)
    tst_wav_dir = os.path.join(tst_path, 'wav_' + dataset)
    tst_text_dir = os.path.join(tst_path, 'text_' + dataset)
    tst_gesture_dir = os.path.join(tst_path, 'gesture_' + dataset)
    tst_sem_dir = os.path.join(tst_path, 'sem_' + dataset)
    tst_emo_dir = os.path.join(tst_path, 'emo_' + dataset)
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
    randm_num = args.randm_num
    for its, file in enumerate(os.listdir(tst_gesture_dir)):
        filename = file.split(".")[0]
        print(f"Processing: {filename}, i={its}")
        if dataset == 'BEAT':
            if 'our_style' in args.name:
                speaker_id = int(filename.split('_')[0]) - 1
                speaker = speaker_id
            else:
                speaker_id = int(filename.split('_')[0]) - 1
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
        emo_path = os.path.join(tst_emo_dir, filename + '.npy')
        emo = np.load(emo_path)
        textaudio = np.concatenate((audio, text), axis=-1)
        textaudio = torch.FloatTensor(textaudio)
        textaudio = textaudio.to(mydevice)

        gesture_path = os.path.join(tst_gesture_dir, filename + '.npy')
        gesture = np.load(gesture_path)

        emotion = torch.from_numpy(emo).to(mydevice)
        
        seed=123456
        torch.manual_seed(seed)

        style=speaker

        n_frames = textaudio.shape[0]

        if n_frames >=6000:
            n_frames = 6000
            textaudio = textaudio[:n_frames]

        real_n_frames = copy.deepcopy(n_frames)   
        # stride_poses = args.n_poses + args.n_seed
        stride_poses = args.n_poses
        stride = args.stride

        if n_frames < stride_poses:
            num_subdivision = 1
            n_frames = stride_poses
        else:
            num_subdivision = math.ceil(n_frames / stride)
            n_frames = num_subdivision * stride
            print('real_n_frames: {}, num_subdivision: {}, stride_poses: {}, stride: {}, pad_frames: {}, speaker_id: {}'.format(real_n_frames, num_subdivision, stride_poses, stride, n_frames - real_n_frames + args.n_seed, style))
        
        textaudio_pad = torch.zeros([n_frames - real_n_frames + args.n_seed, args.audio_feature_dim]).to(mydevice)
        textaudio = torch.cat((textaudio, textaudio_pad), 0)
        textaudio_shape = textaudio.shape[0]
        if emotion.shape[0] < textaudio_shape:
            emotion_pad = torch.zeros([textaudio_shape - emotion.shape[0]]).to(mydevice)
            emotion = torch.cat((emotion, emotion_pad), 0)

        if dataset == 'BEAT':
            data_mean_ = np.load("../process/gesture_BEAT_mean_" + args.version + ".npy")
            data_std_ = np.load("../process/gesture_BEAT_std_" + args.version + ".npy")
        elif dataset == 'TWH':
            data_mean_ = np.load("../process/gesture_TWH_mean_v0" + ".npy")
            data_std_ = np.load("../process/gesture_TWH_std_v0" + ".npy")

        data_mean = np.array(data_mean_)
        data_std = np.array(data_std_)
        # std = np.clip(data_std, a_min=0.01, a_max=None)
        
        shape_ = (num_subdivision, model.njoints, model.nfeats, args.n_poses)
        for i in range(0, num_subdivision):
            if i == 0:
                audio_sub = textaudio[i*stride:i*stride+stride_poses, :].unsqueeze(0) 
                emo_sub = emotion[i*stride:i*stride+stride_poses].unsqueeze(0) 
            else:

                audio_sub = torch.cat((audio_sub, textaudio[i*stride:i*stride+stride_poses, :].unsqueeze(0)), 0)
                emo_sub = torch.cat((emo_sub, emotion[i*stride:i*stride+stride_poses].unsqueeze(0)), 0)
            # model_kwargs_['y']['audio'] = model_kwargs_['y']['audio'].unsqueeze(0)     # attention 4


            
        model_kwargs_ = {'y': {}}

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs_['y']['scale'] = torch.ones(num_subdivision, device=mydevice) * args.guidance_param

        model_kwargs_['y']['mask'] = (torch.zeros([num_subdivision, 1, 1, args.n_poses]) < 1).to(mydevice)
        model_kwargs_['y']['style'] = torch.as_tensor([style]).int().repeat(num_subdivision, 1).to(mydevice)

        model_kwargs_['y']['mask_local'] = torch.ones(num_subdivision, args.n_poses).bool().to(mydevice)

        model_kwargs_['y']['audio'] = audio_sub.to(mydevice)
        model_kwargs_['y']['emotion'] = emo_sub.int().to(mydevice)

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

        # print(sample.shape) 

        sample_seq = sample[:, :args.njoints] # ([b, 648, 1, 150])

        sample_seq = sample_seq.squeeze(2).permute(0, 2, 1).reshape(num_subdivision, -1, model.njoints)

        # sample_seq_euler = matrix_to_euler_angles(sample_seq)

        # b, s, c = sample_seq.shape 
        seed = args.n_seed #  30
        # stride # 120
        pos = sample_seq[:, :, :9]
        joints = sample_seq[:, :, 9:]
        # joints = joints.reshape(num_subdivision, joints.shape[1], -1, 9)
        # joints = joints.reshape(num_subdivision, joints.shape[1], -1, 3, 3)

        b, s, c1 = joints.shape

        fade_forward = torch.ones((1, s, 1)).to(mydevice)
        fade_back = torch.ones((1, s, 1)).to(mydevice)
        fade_forward[:, stride:, :] = torch.linspace(1, 0, seed)[None, :, None].to(mydevice)
        fade_back[:, :seed, :] = torch.linspace(0, 1, seed)[None, :, None].to(mydevice)

        pos[:-1] *= fade_forward
        pos[1:] *= fade_back

        full_pos = torch.zeros((s + stride * (b - 1), 9)).to(mydevice)
        idx = 0
        for pos_slice in pos:
            full_pos[idx : idx + s] += pos_slice 
            idx += stride
        

        # stitch joint angles with linear
        fade_f = torch.ones((1, seed, 1)).to(mydevice)
        fade_b = torch.ones((1, seed, 1)).to(mydevice)
        fade_f = torch.linspace(1, 0, seed)[None, :, None].to(mydevice)
        fade_b = torch.linspace(0, 1, seed)[None, :, None].to(mydevice)

        left, right = joints[:-1, stride:], joints[1:, :seed]
        merged = fade_f * left + fade_b * right

        full_q = torch.zeros((s + stride * (b - 1), c1)).to(pos.device)
        full_q[:stride] += joints[0, :stride]
        idx = stride

        for it, q_slice in enumerate(merged):
            it += 1
            full_q[idx : idx + seed] += q_slice
            full_q[idx + seed : idx + stride] += joints[it, seed:stride]
            idx += stride

        full_q[idx : idx + seed] += joints[-1, :seed]

        # full_q = full_q.reshape(full_q.shape[0], -1)
        out_matrix = torch.cat([full_pos, full_q], 1).detach().data.cpu().numpy()

        if args.body == 'upper':
            zeros_sample = np.zeros((out_matrix.shape[0], 42*3))
            root_sample = np.concatenate([zeros_sample[:, :6*3],out_matrix], axis=1)
            lower_sample = np.concatenate([root_sample, zeros_sample[:, 6*3:]], axis=1)
            out_matrix = lower_sample
        
        out_poses = np.multiply(out_matrix, data_std) + data_mean
        print(out_poses.shape, real_n_frames)

        out = torch.FloatTensor(out_poses).unsqueeze(0).to(mydevice)
        tar = torch.FloatTensor(gesture).unsqueeze(0).to(mydevice)
        
        if gesture.shape[0] < real_n_frames:
            real_n_frames = gesture.shape[0]
        out_poses = out_poses[:real_n_frames]
        tar_poses = gesture[:real_n_frames]
        # print(tar_poses.shape)

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
        
        speaker_name = filename.split("_")[0]
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
    diversity = data_tools.FIDCalculator.get_diversity_scores(latent_out_all, randm_num)
    print(f"diversity score: {diversity}")
    l1div = l1_calculator.avg()
    print(f"l1div score: {l1div}")
    srgr = srgr_calculator.avg()
    print(f"srgr score: {srgr}")
    align_avg = align/len(os.listdir(tst_gesture_dir))
    print("align score:", align_avg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMoFusion')
    parser.add_argument('--config', default='./configs/mmofusion.yml')
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
    config.model_name = config.name
    config.cond_mode = ''

    config.device_name = 'cuda:' + args.gpu
    mydevice = torch.device('cuda:' + config.gpu)
    torch.cuda.set_device(int(config.gpu))
    args.no_cuda = args.gpu

    config.save_dir = "./outputs/" + 'sample_dir_' + config.name + '_' + config.body + '/'

    print('tst_path', config.tst_path, 'save_dir', config.save_dir, 'name', config.name)

    # config.h5file = '../process/' + config.dataset + '_' + config.version + '_test' + '.h5'a

    main(config, config.save_dir, config.model_path, tst_path=config.tst_path, max_len=config.max_len,
         skip_timesteps=config.skip_timesteps, tst_prefix=config.tst_prefix, dataset=config.dataset, 
         wav_path=config.wav_path, mydevice=mydevice)