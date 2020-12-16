import os.path as osp
import multiworld.envs.mujoco as mwmj
from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v0

from rorlkit.launchers.launcher_util import run_experiment
import rorlkit.torch.vae.vae_schedules as vae_schedules
from rorlkit.launchers.vae_experiments import vae_experiment
from rorlkit.torch.vae.conv_vae import imsize48_default_architecture


if __name__ == "__main__":
    variant = dict(
        algorithm='Skew-Fit-SAC',
        double_algo=False,
        online_vae_exploration=False,
        imsize=48,
        env_id='SawyerDoorHookResetFreeEnv-v1',
        init_camera=sawyer_door_env_camera_v0,
        skewfit_variant=dict(
            save_video=True,
            custom_goal_sampler='replay_buffer',
            online_vae_trainer_kwargs=dict(
                beta=20,
                lr=1e-3,
            ),
            save_video_period=50,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            twin_sac_trainer_kwargs=dict(
                reward_scale=1,
                discount=0.99,
                soft_target_tau=1e-3,
                target_update_period=1,
                use_automatic_entropy_tuning=True,
            ),
            max_path_length=100,
            algo_kwargs=dict(
                batch_size=1024,
                num_epochs=170,
                num_eval_steps_per_epoch=500,
                num_expl_steps_per_train_loop=500,
                num_trains_per_train_loop=1000,
                min_num_steps_before_training=10000,
                vae_training_schedule=vae_schedules.custom_schedule,
                oracle_data=False,
                vae_save_period=50,
                parallel_vae_train=False,
            ),
            replay_buffer_kwargs=dict(
                start_skew_epoch=10,
                max_size=int(100000),
                fraction_goals_rollout_goals=0.2,
                fraction_goals_env_goals=0.5,
                exploration_rewards_type='None',
                vae_priority_type='vae_prob',
                priority_function_kwargs=dict(
                    sampling_method='importance_sampling',
                    decoder_distribution='gaussian_identity_variance',
                    num_latents_to_sample=10,
                ),
                power=-0.5,
                relabeling_goal_sampling_mode='custom_goal_sampler',
            ),
            exploration_goal_sampling_mode='custom_goal_sampler',
            evaluation_goal_sampling_mode='presampled',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance',
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            presampled_goals_path=osp.join(
                osp.dirname(mwmj.__file__),
                "goals",
                "door_goals.npy",
            ),
            presample_goals=True,
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            ),
        ),
        train_vae_variant=dict(
            data_path='/home/dzp/samples_img.npy',
            imsize=48,
            representation_size=16,
            beta=20,
            num_epochs=1000,
            dump_skew_debug_plots=False,
            decoder_activation='gaussian',
            generate_vae_dataset_kwargs=dict(
                N=2000,
                test_p=.9,
                use_cached=True,
                show=False,
                oracle_dataset=False,
                n_random_steps=1,
                non_presampled_goal_img_is_garbage=True,
            ),
            vae_kwargs=dict(
                decoder_distribution='gaussian_identity_variance',
                input_channels=3,
                architecture=imsize48_default_architecture,
            ),
            algo_kwargs=dict(
                lr=1e-3,
            ),
            save_period=200,
        ),
    )

    mode = 'local'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    run_experiment(
        vae_experiment,
        exp_prefix=exp_prefix,
        mode=mode,
        variant=variant,
        use_gpu=True,
    )
