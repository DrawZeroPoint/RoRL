import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.torch.vae.conv_vae import imsize48_default_architecture

__C = edict()
cfg = __C

# General options
__C.algorithm = 'Skew-Fit-SAC'
__C.double_algo = False
__C.online_vae_exploration = False
__C.render = False

__C.ALGORITHM = edict()
__C.ALGORITHM.batch_size = 1024
__C.ALGORITHM.num_epochs = 170
__C.ALGORITHM.num_eval_steps_per_epoch = 500
__C.ALGORITHM.num_expl_steps_per_train_loop = 500
__C.ALGORITHM.num_trains_per_train_loop = 1000
__C.ALGORITHM.min_num_steps_before_training = 10000
__C.ALGORITHM.oracle_data = False

# Environment options
__C.ENV = edict()
__C.ENV.id = 'UR5eVisualReachEnv-v0'
__C.ENV.cls = None
__C.ENV.kwargs = None
__C.ENV.init_camera = None
__C.ENV.imsize = 48

__C.GENERATE_VAE_DATASET = edict()
# How many training samples to create, one for one image
__C.GENERATE_VAE_DATASET.N = 2000
# ratio percentage of generated images will be used as train set
__C.GENERATE_VAE_DATASET.ratio = .9
__C.GENERATE_VAE_DATASET.use_cached = True
# The npy file containing all data for training vae
# If none is given, cached dataset in tmp will be used
# as long as file exist and use_cached=True
__C.GENERATE_VAE_DATASET.dataset_path = osp.abspath(
    '../../dataset/UR5eVisualReachEnv-v0_N2000__size48_random_oracle_split_0.npy'
)
__C.GENERATE_VAE_DATASET.show = False
__C.GENERATE_VAE_DATASET.oracle_dataset = False
# How many steps taken before obtaining the observation image to dataset
__C.GENERATE_VAE_DATASET.n_random_steps = 100
__C.GENERATE_VAE_DATASET.non_presampled_goal_img_is_garbage = True
__C.GENERATE_VAE_DATASET.random_and_oracle_policy_data_split = 0
__C.GENERATE_VAE_DATASET.random_and_oracle_policy_data = False
__C.GENERATE_VAE_DATASET.random_rollout_data = False
__C.GENERATE_VAE_DATASET.oracle_dataset_using_set_to_goal = False
__C.GENERATE_VAE_DATASET.uniform_dataset_generator = None
__C.GENERATE_VAE_DATASET.generate_uniform_dataset_kwargs = None

__C.POLICY = edict()
# Pre trained policy model file path
__C.POLICY.model_path = None
__C.POLICY.hidden_sizes = [400, 300]

__C.PRIORITY_FUNCTION = edict()
__C.PRIORITY_FUNCTION.sampling_method = 'importance_sampling'
__C.PRIORITY_FUNCTION.decoder_distribution = 'gaussian_identity_variance'
__C.PRIORITY_FUNCTION.num_latents_to_sample = 10

__C.Q_FUNCTION = edict()
__C.Q_FUNCTION.hidden_sizes = [400, 300]

__C.REPLAY_BUFFER = edict()
__C.REPLAY_BUFFER.start_skew_epoch = 10
__C.REPLAY_BUFFER.max_size = int(100000)
__C.REPLAY_BUFFER.fraction_goals_rollout_goals = 0.2
__C.REPLAY_BUFFER.fraction_goals_env_goals = 0.5
__C.REPLAY_BUFFER.exploration_rewards_type = 'None'
__C.REPLAY_BUFFER.vae_priority_type = 'vae_prob'
__C.REPLAY_BUFFER.power = -0.5
# rlkit.envs.vae_wrapper.VAEWrappedEnv.sample_goals
__C.REPLAY_BUFFER.relabeling_goal_sampling_mode = 'custom_goal_sampler'

__C.REWARD = edict()
__C.REWARD.type = 'latent_distance'

# Skew-fit options
__C.SKEW_FIT = edict()
__C.SKEW_FIT.custom_goal_sampler = 'replay_buffer'
__C.SKEW_FIT.max_path_length = 100
__C.SKEW_FIT.exploration_goal_sampling_mode = 'custom_goal_sampler'
__C.SKEW_FIT.evaluation_goal_sampling_mode = 'presampled'
__C.SKEW_FIT.training_mode = 'train'
__C.SKEW_FIT.testing_mode = 'test'
__C.SKEW_FIT.observation_key = 'latent_observation'
__C.SKEW_FIT.desired_goal_key = 'latent_desired_goal'
__C.SKEW_FIT.save_video = True
__C.SKEW_FIT.save_video_period = 50
__C.SKEW_FIT.presample_goals = True
__C.SKEW_FIT.presampled_goals_path = '/home/dzp/RoRL/goals/roworld_ur5e_reach_goals.npy'
__C.SKEW_FIT.presample_image_goals_only = False

__C.TWIN_SAC_TRAINER = edict()
__C.TWIN_SAC_TRAINER.reward_scale = 1
__C.TWIN_SAC_TRAINER.discount = 0.99
__C.TWIN_SAC_TRAINER.soft_target_tau = 1e-3
__C.TWIN_SAC_TRAINER.target_update_period = 1
__C.TWIN_SAC_TRAINER.use_automatic_entropy_tuning = True

__C.VAE = edict()
# Make sure this equal to ENV.imsize
__C.VAE.imsize = 48
__C.VAE.decoder_distribution = 'gaussian_identity_variance'
__C.VAE.input_channels = 3
__C.VAE.representation_size = 16
__C.VAE.decoder_activation = 'gaussian'
__C.VAE.architecture = imsize48_default_architecture

__C.VAE_TRAINER = edict()
__C.VAE_TRAINER.beta = 20
__C.VAE_TRAINER.lr = 1e-3
# Define path to trained vae model if it has been trained
# If it is None, it will be filled with actual model during runtime
__C.VAE_TRAINER.vae_path = osp.abspath('../../models/vae/roworld-ur5e-reach_2020_08_18_20_25_13.pkl')
__C.VAE_TRAINER.vae_training_schedule = vae_schedules.custom_schedule
__C.VAE_TRAINER.num_epochs = 3000
# Period for saving intermediate result images
__C.VAE_TRAINER.save_period = 200
# FIXME Set this to true will cause error!
__C.VAE_TRAINER.parallel_train = False
# If save vae data, the generated data will be put into train_data and test_data
__C.VAE_TRAINER.save_vae_data = True
# These data will be generated in runtime
__C.VAE_TRAINER.train_data = None
__C.VAE_TRAINER.test_data = None


def get_output_dir(imdb, weights_filename):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if weights_filename is None:
        weights_filename = 'default'
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def get_output_tb_dir(imdb, weights_filename):
    """Return the directory where tensorflow summaries are placed.
  If the directory does not exist, it is created.
  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'tensorboard', __C.EXP_DIR, imdb.name))
    if weights_filename is None:
        weights_filename = 'default'
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
                type(value), type(d[subkey]))
        d[subkey] = value
