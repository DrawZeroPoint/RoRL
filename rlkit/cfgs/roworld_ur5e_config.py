import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.torch.vae.conv_vae import imsize48_default_architecture

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

# General options
__C.algorithm = 'Skew-Fit-SAC'
__C.double_algo = False
__C.online_vae_exploration = False
__C.img_size = 48

__C.ALGORITHM = edict()
__C.ALGORITHM.batch_size = 1024
__C.ALGORITHM.num_epochs = 170
__C.ALGORITHM.num_eval_steps_per_epoch = 500
__C.ALGORITHM.num_expl_steps_per_train_loop = 500
__C.ALGORITHM.num_trains_per_train_loop = 1000
__C.ALGORITHM.min_num_steps_before_training = 10000
__C.ALGORITHM.vae_training_schedule = vae_schedules.custom_schedule
__C.ALGORITHM.oracle_data = False
__C.ALGORITHM.vae_save_period = 50
__C.ALGORITHM.parallel_vae_train = False

# Environment options
__C.ENV.id = 'UR5eReachEnv-v0'
__C.ENV.init_camera = None

__C.GENERATE_VAE_DATASET = edict()
__C.GENERATE_VAE_DATASET.N = 2
__C.GENERATE_VAE_DATASET.test_p = .9
__C.GENERATE_VAE_DATASET.use_cached = True
__C.GENERATE_VAE_DATASET.show = False
__C.GENERATE_VAE_DATASET.oracle_dataset = False
__C.GENERATE_VAE_DATASET.n_random_steps = 1
__C.GENERATE_VAE_DATASET.non_presampled_goal_img_is_garbage = True

__C.POLICY = edict()
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
__C.REPLAY_BUFFER.relabeling_goal_sampling_mode = 'custom_goal_sampler'

__C.REWARD = edict()
__C.REWARD.type = 'latent_distance'

# Skew-fit options
__C.SKEW_FIT = edict()
__C.SKEW_FIT.save_video = True
__C.SKEW_FIT.save_video_period = 50
__C.SKEW_FIT.custom_goal_sampler = 'replay_buffer'
__C.SKEW_FIT.max_path_length = 100
__C.SKEW_FIT.exploration_goal_sampling_mode = 'custom_goal_sampler'
__C.SKEW_FIT.evaluation_goal_sampling_mode = 'presampled'
__C.SKEW_FIT.training_mode = 'train'
__C.SKEW_FIT.testing_mode = 'test'
__C.SKEW_FIT.observation_key = 'latent_observation'
__C.SKEW_FIT.desired_goal_key = 'latent_desired_goal'
__C.SKEW_FIT.presample_goals = True
__C.SKEW_FIT.presampled_goals_path = osp.join(
    osp.dirname(mwmj.__file__),
    "goals",
    "door_goals.npy",
)

__C.TWIN_SAC_TRAINER = edict()
__C.TWIN_SAC_TRAINER.reward_scale = 1
__C.TWIN_SAC_TRAINER.discount = 0.99
__C.TWIN_SAC_TRAINER.soft_target_tau = 1e-3
__C.TWIN_SAC_TRAINER.target_update_period = 1
__C.TWIN_SAC_TRAINER.use_automatic_entropy_tuning = True

__C.VAE = edict()
__C.VAE.decoder_distribution = 'gaussian_identity_variance'
__C.VAE.input_channels = 3
__C.VAE.architecture = imsize48_default_architecture

__C.VAE_TRAINER = edict()
__C.VAE_TRAINER.representation_size = 16
__C.VAE_TRAINER.beta = 20
__C.VAE_TRAINER.num_epochs = 0
__C.VAE_TRAINER.dump_skew_debug_plots = False
__C.VAE_TRAINER.decoder_activation = 'gaussian'
__C.VAE_TRAINER.save_period = 1
__C.VAE_TRAINER.lr = 1e-3


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
