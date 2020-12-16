import time
from roworld.core.image_env import ImageEnv
from rorlkit.core import logger
from rorlkit.envs.vae_wrapper import temporary_mode

import cv2
import torch
import numpy as np
import os.path as osp

from rorlkit.samplers.data_collector.vae_env import (
    VAEWrappedEnvPathCollector,
)
from rorlkit.torch.her.her import HERTrainer
from rorlkit.torch.sac.policies import MakeDeterministic
from rorlkit.torch.sac.sac import SACTrainer
from rorlkit.torch.skewfit.online_vae_algorithm import OnlineVaeAlgorithm
from rorlkit.util.io import load_local_or_remote_file
from rorlkit.util.video import dump_video


def vae_experiment(cfgs):
    train_vae_and_update_variant(cfgs)


def train_vae_and_update_variant(variant):
    from rorlkit.core import logger
    skewfit_variant = variant['skewfit_variant']
    train_vae_variant = variant['train_vae_variant']
    if skewfit_variant.get('vae_path', None) is None:
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'vae_progress.csv', relative_to_snapshot_dir=True
        )
        vae, vae_train_data, vae_test_data = train_vae(train_vae_variant, return_data=True)
        if skewfit_variant.get('save_vae_data', False):
            skewfit_variant['vae_train_data'] = vae_train_data
            skewfit_variant['vae_test_data'] = vae_test_data
        logger.save_extra_data(vae, 'vae.pkl', mode='pickle')
        logger.remove_tabular_output(
            'vae_progress.csv',
            relative_to_snapshot_dir=True,
        )
        logger.add_tabular_output(
            'progress.csv',
            relative_to_snapshot_dir=True,
        )
        skewfit_variant['vae_path'] = vae  # just pass the VAE directly
    else:
        if skewfit_variant.get('save_vae_data', False):
            vae_train_data, vae_test_data, info = generate_vae_dataset(
                train_vae_variant['generate_vae_dataset_kwargs']
            )
            skewfit_variant['vae_train_data'] = vae_train_data
            skewfit_variant['vae_test_data'] = vae_test_data


def train_vae(variant, return_data=False):
    from rlkit.util.ml_util import PiecewiseLinearSchedule
    from rorlkit.torch.vae.conv_vae import (
        ConvVAE,
    )
    import rorlkit.torch.vae.conv_vae as conv_vae
    from rorlkit.torch.vae.vae_trainer import ConvVAETrainer
    from rorlkit.core import logger
    import rlkit.torch.pytorch_util as ptu
    from rlkit.pythonplusplus import identity
    import torch

    beta = variant["beta"]
    representation_size = variant["representation_size"]
    train_data, test_data = prepare_vae_dataset(variant)
    # train_data, test_data, info = generate_vae_dataset(cfgs)
    # logger.save_extra_data(info)
    logger.get_snapshot_dir()

    if 'beta_schedule_kwargs' in variant:
        beta_schedule = PiecewiseLinearSchedule(
            **variant['beta_schedule_kwargs'])
    else:
        beta_schedule = None
    if variant.get('decoder_activation', None) == 'sigmoid':
        decoder_activation = torch.nn.Sigmoid()
    else:
        decoder_activation = identity
    architecture = variant['vae_kwargs'].get('architecture', None)
    if not architecture and variant.get('imsize') == 84:
        architecture = conv_vae.imsize84_default_architecture
    elif not architecture and variant.get('imsize') == 48:
        architecture = conv_vae.imsize48_default_architecture
    variant['vae_kwargs']['architecture'] = architecture
    variant['vae_kwargs']['imsize'] = variant.get('imsize')

    m = ConvVAE(
        representation_size,
        decoder_output_activation=decoder_activation,
        **variant['vae_kwargs']
    )
    m.to(ptu.device)
    t = ConvVAETrainer(train_data, test_data, m, beta=beta,
                       beta_schedule=beta_schedule, **variant['algo_kwargs'])
    save_period = variant['save_period']
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)
        t.train_epoch(epoch)
        t.test_epoch(
            epoch,
            save_reconstruction=should_save_imgs,
            save_vae=False,
        )
        if should_save_imgs:
            t.dump_samples(epoch)
        t.update_train_weights()
    logger.save_extra_data(m, 'vae.pkl', mode='pickle')
    if return_data:
        return m, train_data, test_data
    return m


def prepare_vae_dataset(variant, flatted_data=False):
    """Pre-process the dataset to RoRL standard input format (b, c*w*h).
    The raw images generated by OpenCV is HWC rgb ndarray with shape (b, h, w, c), but
    the standard format needs the order to be CWH.
    :param variant: dict
    :param flatted_data: if true, the raw_images is generated by mujoco, otherwise by OpenCV
    """
    ratio = variant.get('test_p', 0.9)
    imsize = variant.get('imsize')
    assert imsize is not None
    data_path = variant.get('data_path')
    proc_images = []
    raw_images = np.load(data_path)
    if not flatted_data:
        for image in raw_images:
            image = cv2.resize(image, (imsize, imsize))
            cv2.imshow('samples', image)
            cv2.waitKey(1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # The images are in HWC order
            proc_images.append(image)
        dataset = np.asarray(proc_images)
        # swap order and reshape
        dataset = torch.from_numpy(dataset).permute(0, 3, 2, 1).flatten(start_dim=1).numpy()
    else:
        dataset = raw_images

    n = int(len(dataset) * ratio)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    return train_dataset, test_dataset


def get_exploration_strategy(variant, env):
    from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
    from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
    from rlkit.exploration_strategies.ou_strategy import OUStrategy

    exploration_type = variant['exploration_type']
    exploration_noise = variant.get('exploration_noise', 0.1)
    if exploration_type == 'ou':
        es = OUStrategy(
            action_space=env.action_space,
            max_sigma=exploration_noise,
            min_sigma=exploration_noise,  # Constant sigma
        )
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            max_sigma=exploration_noise,
            min_sigma=exploration_noise,  # Constant sigma
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            prob_random_action=exploration_noise,
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    return es


def skewfit_preprocess_variant(cfgs):
    if cfgs.SKEW_FIT.get("do_state_exp", False):
        cfgs['observation_key'] = 'state_observation'
        cfgs['desired_goal_key'] = 'state_desired_goal'
        cfgs['achieved_goal_key'] = 'state_achieved_goal'


def skewfit_experiment(cfgs):
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.online_vae_replay_buffer import \
        OnlineVaeRelabelingBuffer
    from rlkit.torch.networks import FlattenMlp
    from rlkit.torch.sac.policies import TanhGaussianPolicy
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer

    skewfit_preprocess_variant(cfgs)
    env = get_envs(cfgs)

    # TODO
    uniform_dataset_fn = cfgs.GENERATE_VAE_DATASET.get('uniform_dataset_generator', None)
    if uniform_dataset_fn:
        uniform_dataset = uniform_dataset_fn(
            **cfgs.GENERATE_VAE_DATASET.generate_uniform_dataset_kwargs
        )
    else:
        uniform_dataset = None

    observation_key = cfgs.SKEW_FIT.get('observation_key', 'latent_observation')
    desired_goal_key = cfgs.SKEW_FIT.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
        env.observation_space.spaces[observation_key].low.size
        + env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size
    hidden_sizes = cfgs.Q_FUNCTION.get('hidden_sizes', [400, 300])
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=cfgs.POLICY.get('hidden_sizes', [400, 300]),
    )

    vae = env.vae

    replay_buffer = OnlineVaeRelabelingBuffer(
        vae=env.vae,
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        priority_function_kwargs=cfgs.PRIORITY_FUNCTION,
        **cfgs.REPLAY_BUFFER
    )
    vae_trainer = ConvVAETrainer(
        cfgs.VAE_TRAINER.train_data,
        cfgs.VAE_TRAINER.test_data,
        env.vae,
        beta=cfgs.VAE_TRAINER.beta,
        lr=cfgs.VAE_TRAINER.lr,
    )

    # assert 'vae_training_schedule' not in cfgs, "Just put it in algo_kwargs"
    max_path_length = cfgs.SKEW_FIT.max_path_length
    trainer = SACTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **cfgs.TWIN_SAC_TRAINER
    )
    trainer = HERTrainer(trainer)
    eval_path_collector = VAEWrappedEnvPathCollector(
        cfgs.SKEW_FIT.evaluation_goal_sampling_mode,
        env,
        MakeDeterministic(policy),
        decode_goals=True,  # TODO check this
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    expl_path_collector = VAEWrappedEnvPathCollector(
        cfgs.SKEW_FIT.exploration_goal_sampling_mode,
        env,
        policy,
        decode_goals=True,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )

    algorithm = OnlineVaeAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        vae=vae,
        vae_trainer=vae_trainer,
        uniform_dataset=uniform_dataset,  # TODO used in test vae
        max_path_length=max_path_length,
        parallel_vae_train=cfgs.VAE_TRAINER.parallel_train,
        **cfgs.ALGORITHM
    )

    if cfgs.SKEW_FIT.custom_goal_sampler == 'replay_buffer':
        env.custom_goal_sampler = replay_buffer.sample_buffer_goals

    algorithm.to(ptu.device)
    vae.to(ptu.device)
    algorithm.train()


def get_envs(cfgs):
    from roworld.core.image_env import ImageEnv
    from rlkit.envs.vae_wrapper import VAEWrappedEnv
    from rlkit.util.io import load_local_or_remote_file

    render = cfgs.get('render', False)
    reward_params = cfgs.get("reward_params", dict())
    do_state_exp = cfgs.get("do_state_exp", False)  # TODO

    vae_path = cfgs.VAE_TRAINER.get("vae_path", None)
    init_camera = cfgs.ENV.get("init_camera", None)

    presample_goals = cfgs.SKEW_FIT.get('presample_goals', False)
    presample_image_goals_only = cfgs.SKEW_FIT.get('presample_image_goals_only', False)
    presampled_goals_path = cfgs.SKEW_FIT.get('presampled_goals_path', None)

    vae = load_local_or_remote_file(vae_path) if type(vae_path) is str else vae_path
    if cfgs.ENV.id:
        import gym
        import roworld
        roworld.register_all_envs()
        env = gym.make(cfgs.ENV.id)
    else:
        env = cfgs.ENV.cls(**cfgs.ENV.kwargs)

    if not do_state_exp:
        if isinstance(env, ImageEnv):
            image_env = env
        else:
            image_env = ImageEnv(
                env,
                cfgs.ENV.imsize,
                init_camera=init_camera,
                transpose=True,
                normalize=True,
            )
        if presample_goals:
            """
            This will fail for online-parallel as presampled_goals will not be
            serialized. Also don't use this for online-vae.
            """
            if presampled_goals_path is None:
                image_env.non_presampled_goal_img_is_garbage = True
                vae_env = VAEWrappedEnv(
                    image_env,
                    vae,
                    imsize=image_env.imsize,
                    decode_goals=render,
                    render_goals=render,
                    render_rollouts=render,
                    reward_params=reward_params,
                    **cfgs.get('vae_wrapped_env_kwargs', {})
                )
                presampled_goals = cfgs['generate_goal_dataset_fctn'](
                    env=vae_env,
                    env_id=cfgs.get('env_id', None),
                    **cfgs['goal_generation_kwargs']
                )
                del vae_env
            else:
                presampled_goals = load_local_or_remote_file(
                    presampled_goals_path
                ).item()
            del image_env
            image_env = ImageEnv(
                env,
                cfgs.ENV.get('imsize'),
                init_camera=init_camera,
                transpose=True,
                normalize=True,
                presampled_goals=presampled_goals,
            )
            vae_env = VAEWrappedEnv(
                image_env,
                vae,
                imsize=image_env.imsize,
                decode_goals=render,
                render_goals=render,
                render_rollouts=render,
                reward_params=reward_params,
                presampled_goals=presampled_goals,
                sample_from_true_prior=True,
            )
            print("Pre sampling all goals only")
        else:
            vae_env = VAEWrappedEnv(
                image_env,
                vae,
                imsize=image_env.imsize,
                decode_goals=render,
                render_goals=render,
                render_rollouts=render,
                reward_params=reward_params,
                goal_sampling_mode='vae_prior',
                sample_from_true_prior=True,
            )
            if presample_image_goals_only:
                presampled_goals = cfgs['generate_goal_dataset_fctn'](
                    image_env=vae_env.wrapped_env,
                    **cfgs['goal_generation_kwargs']
                )
                image_env.set_presampled_goals(presampled_goals)
                print("Pre sampling image goals only")
            else:
                print("Not using presampled goals")

        env = vae_env
    return env


def get_video_save_func(rollout_function, env, policy, variant):
    logdir = logger.get_snapshot_dir()
    save_period = variant.get('save_video_period', 50)
    do_state_exp = variant.get("do_state_exp", False)
    dump_video_kwargs = variant.get("dump_video_kwargs", dict())
    if do_state_exp:
        imsize = variant.get('imsize')
        dump_video_kwargs['imsize'] = imsize
        image_env = ImageEnv(
            env,
            imsize,
            init_camera=variant.get('init_camera', None),
            transpose=True,
            normalize=True,
        )

        def save_video(algo, epoch):
            if epoch % save_period == 0 or epoch == algo.num_epochs:
                filename = osp.join(logdir, 'video_{epoch}_env.mp4'.format(epoch=epoch))
                dump_video(image_env, policy, filename, rollout_function,
                           **dump_video_kwargs)
    else:
        image_env = env
        dump_video_kwargs['imsize'] = env.imsize

        def save_video(algo, epoch):
            if epoch % save_period == 0 or epoch == algo.num_epochs:
                filename = osp.join(logdir, 'video_{epoch}_env.mp4'.format(epoch=epoch))
                temporary_mode(
                    image_env,
                    mode='video_env',
                    func=dump_video,
                    args=(image_env, policy, filename, rollout_function),
                    kwargs=dump_video_kwargs
                )
                filename = osp.join(logdir, 'video_{epoch}_vae.mp4'.format(epoch=epoch))
                temporary_mode(
                    image_env,
                    mode='video_vae',
                    func=dump_video,
                    args=(image_env, policy, filename, rollout_function),
                    kwargs=dump_video_kwargs
                )
    return save_video