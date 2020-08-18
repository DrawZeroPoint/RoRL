import time
from roworld.core.image_env import ImageEnv
from rlkit.core import logger
from rlkit.envs.vae_wrapper import temporary_mode

import cv2
import numpy as np
import os.path as osp

from rlkit.samplers.data_collector.vae_env import (
    VAEWrappedEnvPathCollector,
)
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.skewfit.online_vae_algorithm import OnlineVaeAlgorithm
from rlkit.util.io import load_local_or_remote_file
from rlkit.util.video import dump_video


def roworld_full_experiment(cfgs):
    train_vae_and_update_config(cfgs)
    skewfit_experiment(cfgs)


def train_vae_and_update_config(cfgs):
    from rlkit.core import logger
    if cfgs.VAE_TRAINER.get('vae_path', None) is None:
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'vae_progress.csv', relative_to_snapshot_dir=True
        )
        vae, vae_train_data, vae_test_data = train_vae(cfgs, return_data=True)
        cfgs.VAE_TRAINER.vae_path = vae  # just pass the VAE directly
        if cfgs.VAE_TRAINER.save_vae_data:
            cfgs.VAE_TRAINER.train_data = vae_train_data
            cfgs.VAE_TRAINER.test_data = vae_test_data

        logger.remove_tabular_output(
            'vae_progress.csv', relative_to_snapshot_dir=True,
        )
        logger.add_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True,
        )
    else:
        vae_train_data, vae_test_data, _ = generate_vae_dataset(cfgs)
        if cfgs.VAE_TRAINER.save_vae_data:
            cfgs.VAE_TRAINER.train_data = vae_train_data
            cfgs.VAE_TRAINER.test_data = vae_test_data


def train_vae(cfgs, return_data=False):
    from rlkit.util.ml_util import PiecewiseLinearSchedule
    from rlkit.torch.vae.conv_vae import (
        ConvVAE,
    )
    import rlkit.torch.vae.conv_vae as conv_vae
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer
    from rlkit.core import logger
    import rlkit.torch.pytorch_util as ptu
    from rlkit.pythonplusplus import identity
    import torch

    train_data, test_data, info = generate_vae_dataset(cfgs)
    logger.save_extra_data(info)
    logger.get_snapshot_dir()

    # FIXME default gaussian
    if cfgs.VAE.get('decoder_activation', None) == 'sigmoid':
        decoder_activation = torch.nn.Sigmoid()
    else:
        decoder_activation = identity

    architecture = cfgs.VAE.get('architecture', None)
    if not architecture and cfgs.ENV.get('img_size') == 84:
        architecture = conv_vae.imsize84_default_architecture
    elif not architecture and cfgs.ENV.get('img_size') == 48:
        architecture = conv_vae.imsize48_default_architecture

    vae_model = ConvVAE(
        representation_size=cfgs.VAE.representation_size,
        architecture=architecture,
        decoder_output_activation=decoder_activation,
        input_channels=cfgs.VAE.input_channels,
        decoder_distribution=cfgs.VAE.decoder_distribution,
        imsize=cfgs.VAE.img_size,
    )
    vae_model.to(ptu.device)

    # FIXME the function of beta_schedule?
    if 'beta_schedule_kwargs' in cfgs.VAE_TRAINER:
        beta_schedule = PiecewiseLinearSchedule(
            **cfgs.VAE_TRAINER.beta_schedule_kwargs)
    else:
        beta_schedule = None

    t = ConvVAETrainer(train_data, test_data, vae_model,
                       lr=cfgs.VAE_TRAINER.lr,
                       beta=cfgs.VAE_TRAINER.beta,
                       beta_schedule=beta_schedule)

    save_period = cfgs.VAE_TRAINER.save_period
    for epoch in range(cfgs.VAE_TRAINER.num_epochs):
        should_save_imgs = (epoch % save_period == 0)
        t.train_epoch(epoch)
        t.test_epoch(
            epoch,
            save_reconstruction=should_save_imgs,
            # save_vae=False,
        )
        if should_save_imgs:
            t.dump_samples(epoch)
        t.update_train_weights()
    logger.save_extra_data(vae_model, 'vae.pkl', mode='pickle')
    if return_data:
        return vae_model, train_data, test_data
    return vae_model


def generate_vae_dataset(cfgs):
    env_id = cfgs.ENV.id
    img_size = cfgs.ENV.imsize
    init_camera = cfgs.ENV.init_camera

    N = cfgs.GENERATE_VAE_DATASET.N
    use_cached = cfgs.GENERATE_VAE_DATASET.use_cached
    n_random_steps = cfgs.GENERATE_VAE_DATASET.n_random_steps
    dataset_path = cfgs.GENERATE_VAE_DATASET.dataset_path  # FIXME
    non_presampled_goal_img_is_garbage = cfgs.GENERATE_VAE_DATASET.non_presampled_goal_img_is_garbage
    random_and_oracle_policy_data_split = cfgs.GENERATE_VAE_DATASET.random_and_oracle_policy_data_split
    random_and_oracle_policy_data = cfgs.GENERATE_VAE_DATASET.random_and_oracle_policy_data
    random_rollout_data = cfgs.GENERATE_VAE_DATASET.random_rollout_data
    oracle_dataset_using_set_to_goal = cfgs.GENERATE_VAE_DATASET.oracle_dataset_using_set_to_goal

    num_channels = cfgs.VAE.input_channels
    policy_file = cfgs.POLICY.model_path

    from roworld.core.image_env import ImageEnv, unormalize_image
    import rlkit.torch.pytorch_util as ptu

    info = {}
    if dataset_path is not None:
        dataset = load_local_or_remote_file(dataset_path)
        N = dataset.shape[0]
    else:
        filename = "/tmp/{}_N{}_{}_size{}_random_oracle_split_{}.npy".format(
            env_id,
            str(N),
            init_camera.__name__ if init_camera else '',
            img_size,
            random_and_oracle_policy_data_split,
        )
        if use_cached and osp.isfile(filename):
            dataset = np.load(filename)
            print("loaded data from saved file", filename)
        else:
            now = time.time()

            assert env_id is not None
            import gym
            import roworld
            roworld.register_all_envs()
            env = gym.make(env_id)

            if not isinstance(env, ImageEnv):
                env = ImageEnv(
                    env,
                    img_size,
                    init_camera=init_camera,
                    transpose=True,
                    normalize=True,
                    non_presampled_goal_img_is_garbage=non_presampled_goal_img_is_garbage,
                )
            else:
                env.imsize = img_size
                env.non_presampled_goal_img_is_garbage = non_presampled_goal_img_is_garbage

            env.reset()
            info['env'] = env
            if random_and_oracle_policy_data:
                policy_file = load_local_or_remote_file(policy_file)
                policy = policy_file['policy']
                policy.to(ptu.device)
            if random_rollout_data:
                from rlkit.exploration_strategies.ou_strategy import OUStrategy
                policy = OUStrategy(env.action_space)

            dataset = np.zeros((N, img_size * img_size * num_channels), dtype=np.uint8)
            obs = env.reset()
            for i in range(N):
                if random_and_oracle_policy_data:
                    num_random_steps = int(N * random_and_oracle_policy_data_split)
                    if i < num_random_steps:
                        # Randomly obtain observation
                        env.reset()
                        for _ in range(n_random_steps):
                            obs = env.step(env.action_space.sample())[0]
                    else:
                        # Obtain observation with policy
                        obs = env.reset()
                        policy.reset()
                        for _ in range(n_random_steps):
                            policy_obs = np.hstack((
                                obs['state_observation'],
                                obs['state_desired_goal'],
                            ))
                            action, _ = policy.get_action(policy_obs)
                            obs, _, _, _ = env.step(action)
                elif oracle_dataset_using_set_to_goal:
                    goal = env.sample_goal()
                    env.set_to_goal(goal)
                    obs = env._get_obs()
                elif random_rollout_data:
                    if i % n_random_steps == 0:
                        g = dict(
                            state_desired_goal=env.sample_goal_for_rollout())
                        env.set_to_goal(g)
                        policy.reset()
                        # env.reset()
                    u = policy.get_action_from_raw_action(
                        env.action_space.sample())
                    obs = env.step(u)[0]
                else:
                    env.reset()
                    # The output obs will be the last observation after stepping n_random_steps
                    for _ in range(n_random_steps):
                        obs = env.step(env.action_space.sample())[0]

                img = obs['image_observation']
                dataset[i, :] = unormalize_image(img)

                if cfgs.GENERATE_VAE_DATASET.show:
                    img = img.reshape(3, img_size, img_size).transpose()
                    img = img[::-1, :, ::-1]
                    cv2.imshow('img', img)
                    cv2.waitKey(1)
                    # radius = input('waiting...')
            print("Done making training data", filename, time.time() - now)
            np.save(filename, dataset)

    n = int(N * cfgs.GENERATE_VAE_DATASET.ratio)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    return train_dataset, test_dataset, info


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
