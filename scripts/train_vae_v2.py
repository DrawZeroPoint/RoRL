from rorlkit.launchers.launcher_util import run_experiment
from rorlkit.launchers.vae_experiments import vae_experiment
from rorlkit.torch.vae.conv_vae import imsize48_default_architecture


if __name__ == "__main__":
    variant = dict(
        imsize=48,
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
