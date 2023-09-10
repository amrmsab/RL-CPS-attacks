import os
import matplotlib.pyplot as plt
import numpy as np
from supplyChainEnv import RLEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import TD3, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback
import json
from stable_baselines3.common.env_util import make_vec_env
# import torch


class NumpyEncoder(json.JSONEncoder):
    """Custom function for json encoding for numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def moving_average(values, window):
    """
    Smooth values by doing a moving average over window

    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the training results (rewards per episode)

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=2)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()


class CollectSuccessfulAttackData(BaseCallback):
    """
    Stop the training once a maximum number of episodes are played.

    For multiple callback_env presumes that, the desired behavior is that the agent trains on each callback_env for ``max_episodes``
    and in total for ``max_episodes * n_envs`` episodes.

    :param max_episodes: Maximum number of episodes to stop training.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about when training ended by
        reaching ``max_episodes``
    """

    def __init__(self, eval_env, log_dir, reward_threshold: float = 0., verbose: int = 0):
        super().__init__(verbose=verbose)
        self.callback_env = eval_env
        self.active_envs = list(range(len(self.callback_env.envs)))
        self.simulation_time = 0
        self.verbose = verbose
        self.n_episodes = 0
        self.attack_data = []
        self.datum = None
        self.log_dir = log_dir
        self.reward_threshold = reward_threshold

    def _init_callback(self) -> None:
        pass

    def _on_step(self) -> bool:
        # Check that the `dones` local variable is defined

        assert "dones" in self.locals, \
            "`dones` variable is not defined, please check your code next to `callback.on_step()`"

        for e in self.active_envs:
            if self.locals["dones"][e]:
                self.n_episodes += 1

                if self.verbose > 0:
                    print(
                        f"""    Done flag in environment {e}/{self.callback_env.num_envs}.
    The last episode had reward {self.callback_env.envs[e].get_episode_rewards()[-1]} \
and length {self.callback_env.envs[e].get_episode_lengths()[-1]}.
    Number of episodes is now {self.n_episodes}."""
                    )

        return True


if __name__ == "__main__":
    continue_training_model = None  # set to None to start training from scratch, else include str location of model to train over.
    n_envs = 4  # number of environments to train in parallel.
    learning_rate = 2.5e-3  # RL agents learning rate hyperparameter
    time_steps = 8e4  # Simulation timesteps to train RL agent for.
    save_model_name = None  # set to None to not save, else include str name to save model as.

    # Create log dir
    log_dir = os.getcwd() + '/log/'
    os.makedirs(log_dir, exist_ok=True)

    # Create environment
    env = make_vec_env(RLEnv,
                       env_kwargs={'sampling_time': 0.2, 'episode_length': 20,
                                   'case_file_location': 'kundur/kundur_full.xlsx',
                                   'attack_points': {1: ['wref0']},
                                   'observation_points': {'v': [], 'omega': [0], 'domega': [0]},
                                   'freq_bounds_hz': [57.5, 61.5],
                                   'voltage_bounds_pu': [-0.144, 0.026], },
                       n_envs=n_envs, monitor_dir=log_dir)

    # Callback to save successful attacks  
    callback_ep = CollectSuccessfulAttackData(env, log_dir, reward_threshold=1e3, verbose=1)

    # Train Model ################################################################
    if continue_training_model is not None:
        model = PPO.load(continue_training_model)
        model.env = env
        model.learning_rate = learning_rate
        model.device = 'cuda'
    else:
        model = PPO(policy='MlpPolicy', env=env, learning_rate=learning_rate,
                    device='cuda', verbose=0, seed=1)

    print(f'Model training on device: {model.device}')

    model.learn(total_timesteps=int(time_steps), callback=callback_ep)
    ################################################################################

    # Plot training reward results
    results_plotter.plot_results([log_dir], int(time_steps), results_plotter.X_TIMESTEPS, "RL rewards over episodes")
    plot_results(log_dir)

    # Simulate learnt agent ########################################################
    obs = env.envs[0].env.reset()
    cumm_reward = 0
    # q_value = []
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        # q_value.append(float(model.critic.forward(torch.tensor(obs.reshape((1, -1))),
        #                                           torch.tensor(np.array(action).reshape((1, -1))))[1][0][0]))
        obs, reward, done, _ = env.envs[0].env.step(action)
        cumm_reward += reward
        print(i, action, done, cumm_reward)
        if done:
            break

    env.envs[0].env.render()
    ################################################################################

    # Save model
    if save_model_name is not None:
        model.save(save_model_name)
