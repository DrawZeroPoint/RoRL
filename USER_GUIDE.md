# Before running the examples

1. Install MuJoCo (latest version), mujoco-py (2.0.2.5, latest version cause a [issue](https://github.com/openai/mujoco-py/issues/607))
2. Install missing dependencies, like gtimer, opencv-python via pip;

# Run the examples

### Tested examples

[her_sac_gym_fetch_reach](examples/her/her_sac_gym_fetch_reach.py)

# Visualize the results

You may need to add these paths to your bashrc

```shell
export PYTHONPATH=$PYTHONPATH:/home/$USER/RoRL:/home/$USER/multiworld
```

Run the policy in terminal:

```shell
python scripts/run_goal_conditioned_policy.py data/her-sac-fetch-experiment/her-sac-fetch-experiment_2021_06_17_12_31_44_0000--s-0/params.pkl --gpu
```

