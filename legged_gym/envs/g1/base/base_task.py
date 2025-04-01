import sys
import numpy as np
import torch
import time
import genesis as gs

# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, sim_device, headless):
        
        self.render_fps = 50
        self.last_frame_time = 0

        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            assert sim_device in ["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"], f"Invalid sim device {sim_device}"
            self.device = torch.device(sim_device)
        self.headless = headless

        self.num_envs = 1 if cfg.env.num_envs == 0 else cfg.env.num_envs
        self.num_build_envs = self.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float32)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.int32)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float32)
        else: 
            self.privileged_obs_buf = None

        self.extras = {}
        
        self.create_sim()

    def get_observations(self):
        return self.obs_buf
    
    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        raise NotImplementedError