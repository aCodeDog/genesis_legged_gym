import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
import numpy as np
import os

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.g1.base.legged_robot import LeggedRobot
from legged_gym.utils.math import wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.gs_utils import *
from .g1_config import G1Cfg

class G1(LeggedRobot):
    cfg:G1Cfg
    def _reset_dofs(self, envs_idx):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        
        dof_pos = torch.zeros((len(envs_idx), self.num_actions), dtype=gs.tc_float, device=self.device)
        dof_pos[:, [0, 3, 6, 9]] = self.default_dof_pos[[0, 3, 6, 9]] + gs_rand_float(-0.1, 0.1, (len(envs_idx), 4), self.device)
        dof_pos[:, [1, 4, 7, 10]] = self.default_dof_pos[[0, 1, 4, 7]] + gs_rand_float(-0.1, 0.1, (len(envs_idx), 4), self.device)
        dof_pos[:, [2, 5, 8, 11]] = self.default_dof_pos[[0, 2, 5, 8]] + gs_rand_float(-0.1, 0.1, (len(envs_idx), 4), self.device)
        self.dof_pos[envs_idx] = dof_pos
        
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.robot.zero_all_dofs_velocity(envs_idx)
    def  _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase  
        
    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask
    
    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.link_contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.2) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.link_contact_forces[:, self.feet_indices, 2] > 5.

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.links_pos[:, self.feet_indices, 2] - 0.05
        delta_z = feet_z - self.last_feet_z
        self.feet_heights += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_heights - self.cfg.rewards.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_heights *= ~contact
        return rew_pos
    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.link_contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1.0, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_arm_pos_diff(self):
        # Penalize dof positions too close to the limit
        diff = (self.dof_pos[:,-14:] - self.default_dof_pos[-14:])
        return torch.sum(torch.square(diff), dim=1)

    def _reward_arm_pos_diff_exp(self):
        # Penalize dof positions too close to the limit
        diff = (self.dof_pos[:,-12:] - self.default_dof_pos[-12:])
        
        error = torch.sum(torch.square(diff), dim=1)
        return torch.exp(-error/self.cfg.rewards.tracking_sigma)

    def _reward_waist_pos_diff(self):
        # Penalize dof positions too close to the limit
        diff = (self.dof_pos[:,[2,5,8]] - self.default_dof_pos[[2,5,8]])
        return torch.sum(torch.square(diff), dim=1)
 
    def _reward_waist_pos_diff_exp(self):
        # Penalize dof positions too close to the limit
        diff = (self.dof_pos[:,[2,5,8]] - self.default_dof_pos[[2,5,8]])
        
        error = torch.sum(torch.square(diff), dim=1)
        return torch.exp(-error/self.cfg.rewards.tracking_sigma)
    
    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = torch.cat((self.last_base_lin_vel[:,:] - self.base_lin_vel[:, :],self.last_base_ang_vel[:,:] - self.base_ang_vel[:, :]), dim=1)
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.links_pos[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_ankle_pos_diff(self):
        # Penalize dof positions too close to the limit
        diff = (self.dof_pos[:,[13,14,17,18]] - self.default_dof_pos[[13,14,17,18]])
        return torch.sum(torch.square(diff), dim=1)
 
    def _reward_ankle_pos_diff_exp(self):
        # Penalize dof positions too close to the limit
        diff = (self.dof_pos[:,[13,14,17,18]] - self.default_dof_pos[[13,14,17,18]])
        
        error = torch.sum(torch.square(diff), dim=1)
        return torch.exp(-error/self.cfg.rewards.tracking_sigma)
    
    def _reward_hip_pos_diff(self):
        # Penalize dof positions too close to the limit
        diff = (self.dof_pos[:,[3,4,6,7]] - self.default_dof_pos[[3,4,6,7]])
        return torch.sum(torch.square(diff), dim=1)
    
    def _reward_hip_pos_diff_exp(self):
        # Penalize dof positions too close to the limit
        diff = (self.dof_pos[:,[3,4,6,7]] - self.default_dof_pos[[3,4,6,7]])
        error = torch.sum(torch.square(diff), dim=1)
        return torch.exp(-error/self.cfg.rewards.tracking_sigma)