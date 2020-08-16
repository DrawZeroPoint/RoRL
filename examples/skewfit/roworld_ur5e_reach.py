"""
This script aims to implement Skew-Fit for reaching task with UR5e + 2F85,
under Webots environment
"""
import rlkit.util.hyperparameter as hyp

from rlkit.cfgs.roworld_ur5e_config import cfg
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.skewfit_experiments import skewfit_full_experiment


if __name__ == "__main__":
    sweeper = hyp.DeterministicHyperparameterSweeper({}, cfg)

    n_seeds = 1
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )
    for exp_id, cfg in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                skewfit_full_experiment,
                exp_prefix=exp_prefix,
                mode='local',
                variant=cfg,
                use_gpu=True,
            )

