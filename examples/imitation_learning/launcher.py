from experiment_launcher import Launcher
from experiment_launcher.utils import is_local


if __name__ == '__main__':
    LOCAL = is_local()
    TEST = False
    USE_CUDA = False

    N_SEEDS = 3

    launcher = Launcher(exp_name='loco_mujoco_evalution',
                        exp_file='experiment',
                        n_seeds=N_SEEDS,
                        n_cores=3,
                        memory_per_core=1500,
                        days=2,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        use_timestamp=True,
                        )

    default_params = dict(n_epochs=1,
                          n_steps_per_epoch=100000,
                          n_epochs_save=25,
                          n_eval_episodes=10,
                          n_steps_per_fit=1000,
                          use_cuda=USE_CUDA)

    env_ids = ["Atlas.walk", "Atlas.carry",
               "Talos.walk", "Talos.carry",
               "UnitreeH1.walk", "UnitreeH1.run", "UnitreeH1.carry",
               "HumanoidTorque.walk", "HumanoidTorque.run",
               "HumanoidMuscle.walk", "HumanoidMuscle.run",
               "UnitreeA1.simple", "UnitreeA1.hard"]

    for env_id in env_ids:
        launcher.add_experiment(env_id__=env_id, **default_params)

    launcher.run(LOCAL, TEST)