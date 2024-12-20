# Genesis Environments for Legged Robots #
This repository provides the Genesis environment used to legged robots.  

---


### Installation ###
1. Create a new python virtual env with python >= 3.9
2. Install genesis

   - `pip install genesis-world`

3. Install rsl_rl (PPO implementation)
   - Clone https://github.com/leggedrobotics/rsl_rl
   -  `cd rsl_rl && git checkout v1.0.2 && pip install -e .` 
4. Install this genesis veersion of legged_gym
    - Clone this repository
   - `cd legged_gym && pip install -e .`

### CODE STRUCTURE ###
 same as legged gym. compared with the original version,you only need to add dof_names,max_joint_velocity,links_to_keep in the config flie.

### Usage ###
1. Train:  
  ```python legged_gym/scripts/train.py --task=anymal_c_flat```
    -  To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
    -  To run headless (no rendering) add `--headless`.
    - **Important**: To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.
    - The trained policy is saved in `issacgym_anymal/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
    -  The following command line arguments override the values set in the config files:
     - --task TASK: Task name.
     - --resume:   Resume training from a checkpoint
     - --experiment_name EXPERIMENT_NAME: Name of the experiment to run or load.
     - --run_name RUN_NAME:  Name of the run.
     - --load_run LOAD_RUN:   Name of the run to load when resume=True. If -1: will load the last run.
     - --checkpoint CHECKPOINT:  Saved model checkpoint number. If -1: will load the last checkpoint.
     - --num_envs NUM_ENVS:  Number of environments to create.
     - --seed SEED:  Random seed.
     - --max_iterations MAX_ITERATIONS:  Maximum number of training iterations.
2. Play a trained policy:  
```python legged_gym/scripts/play.py --task=anymal_c_flat```
    - By default, the loaded policy is the last model of the last run of the experiment folder.
    - Other runs/model iteration can be selected by setting `load_run` and `checkpoint` in the train config.

### Adding a new environment ###
The base environment `legged_robot` implements a rough terrain locomotion task. The corresponding cfg does not specify a robot asset (URDF/ MJCF) and has no reward scales. 

1. Add a new folder to `envs/` with `'<your_env>_config.py`, which inherit from an existing environment cfgs  
2. If adding a new robot:
    - Add the corresponding assets to `resources/`.
    - In `cfg` set the asset path, define body names, default_joint_positions and PD gains. Specify the desired `train_cfg` and the name of the environment (python class).
    - In `train_cfg` set `experiment_name` and `run_name`
3. (If needed) implement your environment in <your_env>.py, inherit from an existing environment, overwrite the desired functions and/or add your reward functions.
4. Register your env in `isaacgym_anymal/envs/__init__.py`.
5. Modify/Tune other parameters in your `cfg`, `cfg_train` as needed. To remove a reward set its scale to zero. Do not modify parameters of other envs!



# Acknowledgements
This code builds upon following open-source code-bases. Please visit the URLs to see the respective LICENSES:

1. https://github.com/leggedrobotics/legged_gym/tree/master
2. https://github.com/aCodeDog/awesome-loco-manipulation
3. https://github.com/aCodeDog/legged-robots-manipulation