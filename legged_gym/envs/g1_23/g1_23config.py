# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1_23RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 1000

        n_scan = 132
        n_priv = 3+3 +3
        n_priv_latent = 4 + 1 + 23 +23 
        n_proprio = 3 + 2 + 3 + 2 + 23*3 + 5 +23 +2 +2
        history_len = 10

        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv #n_scan + n_proprio + n_priv #187 + 47 + 5 + 12 
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 23
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        obs_type = "og"


        
        
        
        history_encoding = True
        reorder_dofs = True
        
        
        # action_delay_range = [0, 5]

        # additional visual inputs 

        # action_delay_range = [0, 5]

        # additional visual inputs 
        include_foot_contacts = True
        
        randomize_start_pos = False
        randomize_start_vel = False
        randomize_start_yaw = False
        rand_yaw_range = 1.2
        randomize_start_y = False
        rand_y_range = 0.5
        randomize_start_pitch = False
        rand_pitch_range = 1.6

        contact_buf_len = 100

        next_goal_threshold = 0.2
        reach_goal_delay = 0.1
        num_future_goal_obs = 2

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.7] # x,y,z [m]
        rot = [0.0,  0., 0.0, 1] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_roll_joint' : 0,  
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_pitch_joint' : -0.4,         
           'left_knee_joint' : 0.8,       
           'left_ankle_pitch_joint': -0.3,
           'left_ankle_roll_joint': 0.,    
           'right_hip_roll_joint' : 0, 
           'right_hip_yaw_joint' : 0., 
           'right_hip_pitch_joint' : -0.4,                                       
           'right_knee_joint' : 0.8,                                             
           'right_ankle_pitch_joint': -0.3,
           'right_ankle_roll_joint': 0.,
           'left_elbow_joint': 0.8,
           'right_elbow_joint': 0.8,
           
           
           'left_shoulder_pitch_joint': 0.7,
           'left_shoulder_roll_joint': 0.,
           'left_shoulder_yaw_joint': 0.,
           'left_wrist_roll_joint': 0.,
           
            'right_shoulder_pitch_joint': 0.7,
            'right_shoulder_roll_joint': 0.,
            'right_shoulder_yaw_joint': 0.,
            'right_wrist_roll_joint': 0.,
           
           'waist_yaw_joint': 0.,
        }

    class init_state_slope( LeggedRobotCfg.init_state ):
        pos = [2, 0.0, 0.74] # x,y,z [m]
        rot = [0.0,  0., 0.0, 1] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_roll_joint' : 0,  
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_pitch_joint' : -0.4,         
           'left_knee_joint' : 0.8,       
           'left_ankle_pitch_joint': -0.3,
           'left_ankle_roll_joint': 0.,    
           'right_hip_roll_joint' : 0, 
           'right_hip_yaw_joint' : 0., 
           'right_hip_pitch_joint' : -0.4,                                       
           'right_knee_joint' : 0.8,                                             
           'right_ankle_pitch_joint': -0.3,
           'right_ankle_roll_joint': 0.,
           'left_elbow_joint': 0.8,
           'right_elbow_joint': 0.8,
           
           
           'left_shoulder_pitch_joint': 0.7,
           'left_shoulder_roll_joint': 0.,
           'left_shoulder_yaw_joint': 0.,
           'left_wrist_roll_joint': 0.,
           
            'right_shoulder_pitch_joint': 0.7,
            'right_shoulder_roll_joint': 0.,
            'right_shoulder_yaw_joint': 0.,
            'right_wrist_roll_joint': 0.,
           
           'waist_yaw_joint': 0.,
        }
        
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip_yaw': 150,
                     'hip_roll': 150,
                     'hip_pitch': 200,
                     'knee': 300,
                     'ankle': 20,
                     'elbow': 20,
                     'shoulder':50,
                     'wrist':10,
                     'waist': 10,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 5,
                     'hip_roll': 5,
                     'hip_pitch': 5,
                     'knee': 5,
                     'ankle': 2,
                     'elbow': 2,
                     'shoulder':2,
                     'wrist':1,
                     'waist': 10,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        hf2mesh_method = "grid"  # grid or fast
        max_error = 0.1 # for fast
        max_error_camera = 2

        y_range = [-0.4, 0.4]
        
        edge_width_thresh = 0.05
        horizontal_scale = 0.05 # [m] influence computation time by a lot
        horizontal_scale_camera = 0.1
        vertical_scale = 0.005 # [m]
        border_size = 0 # [m]
        height = [0.02, 0.06]
        simplify_grid = False
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = True

        all_vertical = False
        no_flat = True
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]
        measure_horizontal_noise = 0.0

        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 18.
        terrain_width = 4
        num_rows= 10 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 40 # number of terrain cols (types)
        
        terrain_dict = {"smooth slope": 0., 
                        "rough slope up": 0.0,
                        "rough slope down": 0.0,
                        "rough stairs up": 0., 
                        "rough stairs down": 0., 
                        "discrete": 0., 
                        "stepping stones": 0.0,
                        "gaps": 0., 
                        "smooth flat": 0.5,
                        "pit": 0.0,
                        "wall": 0.0,
                        "platform": 0.,
                        "large stairs up": 0.,
                        "large stairs down": 0.,
                        "parkour": 0.,
                        "parkour_hurdle": 0.,
                        "parkour_flat": 1,
                        "parkour_step": 0.,
                        "parkour_gap": 0.,
                        "demo": 0.0,}
        terrain_proportions = list(terrain_dict.values())
        
        # trimesh only:
        slope_treshold = 1.5# slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = True

        num_goals = 8

    class asset( LeggedRobotCfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1_new.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_23dof/urdf/g1_23dof.urdf'
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        
        terminate_after_contacts_on = ["pelvis","torso"]#, "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.728
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.2 # tracking reward = exp(-error^2/sigma)
        soft_dof_vel_limit = 1
        soft_torque_limit = 0.4
        max_contact_force = 300. # forces above this value are penalized
        cycle_time =0.64
        target_feet_height = 0.06 
        target_joint_pos_scale = 0.17 

        class scales( LeggedRobotCfg.rewards.scales ):
            # torques = -0.0002
            # dof_pos_limits = -10.0
            tracking_goal_vel =2
            tracking_yaw = 0.5
            # regularization rewards
            lin_vel_z = -0.03
            ang_vel_xy = -0.03
            orientation = -1.
            dof_acc = -5e-7
            collision = -10.
            action_rate = -0.01
            delta_torques = -1.0e-7
            torques = -0.00001
            hip_pos = -0.05
            dof_error = -0.02
            feet_stumble = -0.02
            feet_edge = -0.05
            feet_air_time=0.05
            base_height = -0.1
            action_rate_l2 = -0.01
            no_fly = 0.05
            joint_pos = 0.05
            feet_clearance = 0.05



class G1_23RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_g1'
        resume = False
        load_run = "/home/zifanw/rl_robot/extreme-parkour/legged_gym/logs/parkour_new/000-02-g1p"  # -1 = last run
        #checkpoint = 3600 # -1 = last saved model
        # resume_path = "/home/zifanw/rl_robot/extreme-parkour/legged_gym/logs/parkour_new/000-02-g1" # updated from load_run and chkpt

    class depth_encoder:
        if_depth = G1_23RoughCfg.depth.use_camera
        depth_shape = G1_23RoughCfg.depth.resized
        buffer_len = G1_23RoughCfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.e-3
        num_steps_per_env = G1_23RoughCfg.depth.update_interval * 24

    class estimator:
        train_with_estimated_states = True
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = G1_23RoughCfg.env.n_priv
        num_prop = G1_23RoughCfg.env.n_proprio
        num_scan = G1_23RoughCfg.env.n_scan

  
