"""
This script aims to implement Skew-Fit for reaching task with UR5e + 2F85,
under Webots environment
"""
from rorlkit.cfgs.roworld_ur5e_config import cfg
from rorlkit.launchers.launcher_util import run_experiment
from rorlkit.launchers.roworld_experiments import roworld_full_experiment


if __name__ == "__main__":
    n_seeds = 1
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )
    for _ in range(n_seeds):
        run_experiment(
            roworld_full_experiment,
            exp_prefix=exp_prefix,
            mode='local',
            variant=cfg,
            use_gpu=True,
        )

