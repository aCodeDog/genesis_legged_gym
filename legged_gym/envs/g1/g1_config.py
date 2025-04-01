from legged_gym.envs.g1.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1Cfg( LeggedRobotCfg ):
    
    class env( LeggedRobotCfg.env ):
        num_envs = 4000
        num_observations = 48+17*3
        num_actions = 29
    
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = "plane" # none, plane, heightfield
        friction = 1.0
        restitution = 0.
        
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.74] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
        "left_hip_pitch_joint": -0.20,
        "right_hip_pitch_joint": -0.20,
        "waist_yaw_joint": 0.0,
        "left_hip_roll_joint": 0.0,
        "right_hip_roll_joint": 0.0,
        "waist_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "waist_pitch_joint": 0.0,
        "left_knee_joint": 0.42,
        "right_knee_joint": 0.42,
        "left_shoulder_pitch_joint": 0.35,
        "right_shoulder_pitch_joint": 0.35,
        "left_ankle_pitch_joint": -0.23,
        "right_ankle_pitch_joint": -0.23,
        "left_shoulder_roll_joint": 0.28,
        "right_shoulder_roll_joint": -0.28,
        "left_ankle_roll_joint": 0.0,
        "right_ankle_roll_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.87,
        "right_elbow_joint": 0.87,
        "left_wrist_roll_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,

        }
        # initial state randomization
        yaw_angle_range = [0., 3.14] # min max [rad]

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        # control_type = 'P'
        stiffness = {
        "left_hip_pitch_joint": 200.0,
        "right_hip_pitch_joint": 200.0,
        "waist_yaw_joint": 200.0,
        "left_hip_roll_joint": 150.0,
        "right_hip_roll_joint": 150.0,
        "waist_roll_joint": 200.0,
        "left_hip_yaw_joint": 150.0,
        "right_hip_yaw_joint": 150.0,
        "waist_pitch_joint": 200.0,
        "left_knee_joint": 200.0,
        "right_knee_joint": 200.0,
        "left_shoulder_pitch_joint": 40.0,
        "right_shoulder_pitch_joint": 40.0,
        "left_ankle_pitch_joint": 20.0,
        "right_ankle_pitch_joint": 20.0,
        "left_shoulder_roll_joint": 40.0,
        "right_shoulder_roll_joint": 40.0,
        "left_ankle_roll_joint": 20.0,
        "right_ankle_roll_joint": 20.0,
        "left_shoulder_yaw_joint": 40.0,
        "right_shoulder_yaw_joint": 40.0,
        "left_elbow_joint": 40.0,
        "right_elbow_joint": 40.0,
        "left_wrist_roll_joint": 40.0,
        "right_wrist_roll_joint": 40.0,
        "left_wrist_pitch_joint": 40.0,
        "right_wrist_pitch_joint": 40.0,
        "left_wrist_yaw_joint": 40.0,
        "right_wrist_yaw_joint": 40.0,
        }

        damping = {
        "left_hip_pitch_joint": 5.0,
        "right_hip_pitch_joint": 5.0,
        "waist_yaw_joint": 5.0,
        "left_hip_roll_joint": 5.0,
        "right_hip_roll_joint": 5.0,
        "waist_roll_joint": 5.0,
        "left_hip_yaw_joint": 5.0,
        "right_hip_yaw_joint": 5.0,
        "waist_pitch_joint": 5.0,
        "left_knee_joint": 5.0,
        "right_knee_joint": 5.0,
        "left_shoulder_pitch_joint": 10.0,
        "right_shoulder_pitch_joint": 10.0,
        "left_ankle_pitch_joint": 2.0,
        "right_ankle_pitch_joint": 2.0,
        "left_shoulder_roll_joint": 10.0,
        "right_shoulder_roll_joint": 10.0,
        "left_ankle_roll_joint": 2.0,
        "right_ankle_roll_joint": 2.0,
        "left_shoulder_yaw_joint": 10.0,
        "right_shoulder_yaw_joint": 10.0,
        "left_elbow_joint": 10.0,
        "right_elbow_joint": 10.0,
        "left_wrist_roll_joint": 10.0,
        "right_wrist_roll_joint": 10.0,
        "left_wrist_pitch_joint": 10.0,
        "right_wrist_pitch_joint": 10.0,
        "left_wrist_yaw_joint": 10.0,
        "right_wrist_yaw_joint": 10.0,
        } # [N*m*s/rad]
        action_scale = 0.5 # action scale: target angle = actionScale * action + defaultAngle
        dt =  0.02  # control frequency 50Hz
        decimation = 4 # decimation: Number of control action updates @ sim DT per policy DT

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_29/g1_29dof.urdf'
        dof_names = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint',
                    'right_hip_roll_joint', 'waist_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint',
                    'waist_pitch_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint',
                    'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
                    'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint',
                    'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
                    'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint',
                    'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint']
    
        foot_name = ["ankle_roll"]
        penalize_contacts_on = ["hip", "knee","wrist","elbow"]
        terminate_after_contacts_on = ["torso","pelvis"]
        # links that are not merged because of fixed joints
        links_to_keep = ['']
        self_collisions = False
        
    class normalization(LeggedRobotCfg.normalization):
        class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 4.
        clip_actions = 5.

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.728
        cycle_time = 0.64 
        max_contact_force = 700
        target_feet_height = 0.06
        target_joint_pos_scale = 0.17
        only_positive_rewards = False
        min_dist = 0.2
        max_dist = 0.5
        class scales( LeggedRobotCfg.rewards.scales ):
            # limitation
            dof_pos_limits = -10.0
            collision = -0.0
            # command tracking
            tracking_lin_vel = 3.
            tracking_ang_vel = 0.5
            # smooth
            lin_vel_z = -1
            base_height = -1.0
            ang_vel_xy = -1
            orientation = -50
            dof_vel = -2.5e-7
            dof_acc = -5e-6
            action_rate = -0.01
            torques = 0.
            # gait
            feet_air_time = 1.0
            # dof_close_to_default = -0.1
            
            feet_contact_number = 1.
            feet_clearance = 1.
            
            arm_pos_diff = -5
            arm_pos_diff_exp = 0.5
            
            waist_pos_diff = -5
            waist_pos_diff_exp = 0.5
            
            base_acc=0.2
            action_smoothness = -0.002
            
            ankle_pos_diff = -2
            ankle_pos_diff_exp = 0.2
            
            feet_distance = 0.2
            
            hip_pos_diff = -10
            hip_pos_diff_exp = 0.5
    
    class commands( LeggedRobotCfg.commands ):
        curriculum = True
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [0, 1] # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0, 0]    # min max [rad/s]
            heading = [-3.14, 3.14]
    
    class domain_rand:
        randomize_friction = True
        friction_range = [0.8, 1.2]
        randomize_base_mass = False
        added_mass_range = [-0.5, 0.5]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.
        simulate_action_latency = False # 1 step delay
        randomize_com_displacement = False
        com_displacement_range = [-0.01, 0.01]
    
    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [-1, 0, 6]       # [m]
        lookat = [0., 0, 0.]  # [m]
        num_rendered_envs = 10  # number of environments to be rendered
        add_camera = False

class G1CfgPPO( LeggedRobotCfgPPO ):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [128, 128, 128]
        critic_hidden_dims = [128, 128, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'g1'
        save_interval = 100
        resume = False
        load_run = -1
        checkpoint = -1
        max_iterations = 5000
        policy_class_name = "ActorCriticRecurrent"