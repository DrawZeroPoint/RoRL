# HER

## FetchPush-v1

[train_script](examples/her/her_sac_gym_fetch_push.py)

```
args = dict(
    algorithm='HER-SAC',
    version='normal',
    algo_kwargs=dict(
        batch_size=128,
        num_epochs=2000,
        num_eval_steps_per_epoch=5000,
        num_expl_steps_per_train_loop=4000,
        num_trains_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=200,
    ),
    sac_trainer_kwargs=dict(
        discount=0.99,
        soft_target_tau=5e-3,
        target_update_period=1,
        policy_lr=3E-4,
        qf_lr=3E-4,
        reward_scale=1,
        use_automatic_entropy_tuning=True,
    ),
    replay_buffer_kwargs=dict(
        max_size=int(1E6),
        fraction_goals_rollout_goals=0.2,  # equal to k = 4 in HER paper
        fraction_goals_env_goals=0,
    ),
    qf_kwargs=dict(
        hidden_sizes=[400, 300],
    ),
    policy_kwargs=dict(
        hidden_sizes=[400, 300],
    ),
)
```
