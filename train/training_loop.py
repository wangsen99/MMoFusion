import functools
import os
import pdb

import numpy as np

import blobfile as bf
import torch
from torch.optim import AdamW

from diffusion import logger
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler

import sys
[sys.path.append(i) for i in ['../process']]



class TrainLoop:
    def __init__(self, args, model, diffusion, device, data=None):
        self.args = args
        self.data = data
        self.model = model
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps
        self.start_anneal_steps = args.start_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size     # * dist.get_world_size()
        self.num_steps = args.max_num_steps
        self.save_iters = args.save_iters
        self.n_seed = args.n_seed
        self.checkpoint_path = args.root_path+args.out_root_path + args.name + "/"
        logger.info(f'max_steps:{self.num_steps}')

        self.resume_checkpoint = args.resume_checkpoint

        self.sync_cuda = torch.cuda.is_available()
        self.device = device
        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        if args.audio_feat == "mfcc" or args.audio_feat == 'wavlm':
            self.opt = AdamW([
                {'params': self.mp_trainer.master_params, 'lr':self.lr, 'weight_decay':self.weight_decay}
            ])

        if self.resume_step:
            self._load_optimizer_state()
            
        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None

        self.use_ddp = False
        self.ddp_model = self.model
        self.mask_train = (torch.zeros([self.batch_size, 1, 1, args.n_poses]) < 1).to(self.device)
        self.mask_test = (torch.zeros([1, 1, 1, args.n_poses]) < 1).to(self.device)
        self.mask_local_train = torch.ones(self.batch_size, args.n_poses).bool().to(self.device)
        self.mask_local_test = torch.ones(1, args.n_poses).bool().to(self.device)


    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.info(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                    torch.load(resume_checkpoint, map_location=self.device)
                )

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = torch.load(
                opt_checkpoint, map_location=self.device
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):

        for its, batch in enumerate(self.data):
            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break

            cond_ = {'y':{}}
            wavlm, pose_seq, style, emotion = batch

            motion = pose_seq.permute(0, 2, 1).unsqueeze(2).to(self.device, non_blocking=True)

            cond_['y']['seed'] = motion[..., 0:self.n_seed]
            cond_['y']['style'] = style.to(self.device, non_blocking=True)
            cond_['y']['mask_local'] = self.mask_local_train
            cond_['y']['audio'] = wavlm.to(self.device, non_blocking=True)   
            cond_['y']['mask'] = self.mask_train        # [..., self.n_seed:]
            cond_['y']['emotion'] = emotion.to(self.device, non_blocking=True)

            self.run_step(motion, cond_)
            if self.step % self.log_interval == 0:
                for k,v in logger.get_current().name2val.items():
                    if k == 'loss':
                        logger.info('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))


            if self.step % self.save_iters == 0:
                self.save()

                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
            if self.step + self.resume_step > self.num_steps: 
                break
            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break



    def run_step(self, batch, cond):
        self.forward_backward(batch, cond) 
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                dataset='beat'
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        elif (self.step + self.resume_step) > self.start_anneal_steps:
            frac_done = (self.step + self.resume_step) / (self.lr_anneal_steps - self.start_anneal_steps)
            lr = self.lr * (1 - frac_done)
            for param_group in self.opt.param_groups:
                param_group["lr"] = lr
        else:
            return

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)


    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"


    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.checkpoint_path, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)
        logger.log(f"saving opt...")
        with bf.BlobFile(
            bf.join(self.checkpoint_path, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)