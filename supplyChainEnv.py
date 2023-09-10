from typing import List

import andes
from andes.utils import get_case
import gym
from gym import spaces
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class RLEnv(gym.Env):
    """Custom Environment that follows gym interface.

    The environment simulates a power system that is targeted by a cyberattack.
    It allows attacks comprising the frequency and/or voltage setpoints of generators in
    the power system.
    """

    def __init__(self, sampling_time: float = 0.05,
                 episode_length: float = 10.,
                 case_file_location='ieee14/ieee14_full.xlsx',
                 attack_points: dict = None,
                 observation_points: dict = None,
                 freq_bounds_hz=None,
                 voltage_bounds_pu=None,
                 ):
        """
        Initialize the environment.

        :param sampling_time: float. The time in seconds between attack actions.
        :param episode_length: float. The maximum time length of the episode (attack).
        :param case_file_location: str. The Andes file location of the power system testbed
        :param attack_points: dict. Dictionary detailing the attack points
            written as {gen_idx: [<'wref0'>, <'vref0'>]
            Include 'wref0' in the List if the attack comprises the frequency setpoint of generator gen_idx
            Include 'vref0' in the List if the attack comprises the voltage setpoint of generator gen_idx
        :param observation_points: dict. Dictionary detailing the power system states observable by the attacker.
            written as {'v': [Bus # ,#, ..], 'omega': [Generator #, #, ..], 'domega': [Generator #, #, ..]}
            The List for 'v' includes the buses at which the agent can monitor the voltage.
            The List for 'omega' includes the generators at which the agent can monitor the frequency.
            The List for 'domega' includes the generators at which the agent can monitor the rate-of-change of frequency.
        :param freq_bounds_hz: List[float] The bounds on the RL agent's actions compromising the frequency setpoint.
        :param voltage_bounds_pu: List[float] The bounds on the RL agent's actions compromising the frequency setpoint.

        Return None
        """
        super(RLEnv, self).__init__()

        self.case_location = get_case(case_file_location)  # get benchmark system from Andes
        andes.config_logger(stream_level=50, stream=False)  # configure Andes to reduce logging

        self.attack_points = (attack_points or {'GENROU_1': ['wref0', 'vref0']})
        self.observation_points = observation_points or {'v': [0], 'omega': [0], 'domega': [0]}

        self.voltage_bounds_pu = np.array(voltage_bounds_pu or [-0.05, 0.05])
        self.freq_bounds_hz = np.array(freq_bounds_hz or [59, 61])

        self.num_observations = \
            len(self.observation_points.get('v', [])) + \
            len(self.observation_points.get('omega', [])) + \
            len(self.observation_points.get('domega', []))

        self.num_actions = sum([len(self.attack_points[gen]) for gen in self.attack_points])

        self.actions = None  # array to store actions
        self.domega_pu = None  # array to save domega

        self.prev_omega = None
        self.prev_observation = None

        self.simulation_time = 0  # reset simulation time to 0 seconds
        self.episode_length = episode_length  # maximum episode length (seconds)
        self.ts = sampling_time  # time between agent actions (seconds)

        # Define action and observation space #############################################
        # Observations
        self.observation_space = spaces.Box(low=-0.3, high=0.3,
                                            shape=(self.num_observations,),
                                            dtype=np.float32)
        # Actions
        self.action_space = spaces.Box(low=-1, high=1,
                                       shape=(self.num_actions,),
                                       dtype=np.float32)
        ###################################################################################

        self.ss = None  # parameter to store simulation results
        self.reset()  # initiates system

        # Extract the governor and exciter IDs from the Andes testbed ####################
        self.ss.TDS.config.tf = 1e-3
        self.ss.TDS.run()

        self.governor_map = dict(zip(self.ss.TGOV1.syn.v, self.ss.TGOV1.idx.v))
        self.exciter_map = dict(zip(self.ss.ESST3A.syn.v, self.ss.ESST3A.idx.v))
        self.exciter_map.update(dict(zip(self.ss.EXDC2.syn.v, self.ss.EXDC2.idx.v)))
        self.exciter_map.update(dict(zip(self.ss.EXST1.syn.v, self.ss.EXST1.idx.v)))

        self.V0 = dict(zip(self.ss.ESST3A.syn.v, self.ss.ESST3A.vref0.v))
        self.V0.update(dict(zip(self.ss.EXDC2.syn.v, self.ss.EXDC2.vref0.v)))
        self.V0.update(dict(zip(self.ss.EXST1.syn.v, self.ss.EXST1.vref0.v)))
        ##################################################################################

        self.figure, self.axis = None, None

        return

    def step(self, action: List[float], suppress_done=True):
        """
        Inject an attack into the power system and simulate the power system for ts seconds.

        :param action: List[float]. Attack values in the order attack_points was initialized. Attack values should be
        in the range [-1,1] and the function will interpolate them to be within the allowed bounds.
        :param suppress_done: Bool. Suppresses ending the episode if the attack destabilizes the power system.

        :return: (List[float] - observations as outlined in initialization,
                    float - reward to the RL agent,
                    Bool - True if the episode reaches maximum length or if attack destabilizes the grid,
                    {})
        """

        # Modify the setpoints of the generators per the attack/actions ################################
        curr_actions = []  # array to log actions
        i = 0
        for gen_idx in self.attack_points:
            for point in self.attack_points[gen_idx]:
                if point == 'wref0':
                    a = np.interp(action[i], [-1, 1], self.freq_bounds_hz / 60)
                    try:
                        self.ss.TGOV1.alter(point, self.governor_map[gen_idx], a)
                    except:
                        self.ss.TGOV1.alter(point, list(self.governor_map.values())[gen_idx], a)
                elif point == 'vref0':
                    a = np.interp(action[i], [-1, 1],
                                  self.V0[gen_idx] + self.voltage_bounds_pu)

                    self.ss.EXDC2.alter(point, self.exciter_map[gen_idx], a)

                    # if self.exciter_map[gen_idx][:-2] == 'EXDC2':
                    #     self.ss.EXDC2.alter(point, self.exciter_map[gen_idx], a)
                    # elif self.exciter_map[gen_idx][:-2] == 'ESST3A':
                    #     self.ss.ESST3A.alter(point, self.exciter_map[gen_idx], a)
                    # elif self.exciter_map[gen_idx][:-2] == 'EXST1':
                    #     self.ss.EXST1.alter(point, self.exciter_map[gen_idx], a)
                    # else:
                    #     raise Exception(f"Cannot find the {self.exciter_map[gen_idx][:-2]} exciter for {gen_idx}.")

                else:
                    raise Exception(f"Attack point {point} is undefined.")
                curr_actions.append(a)
                i += 1
        ######################################################################################################

        self.actions.append(curr_actions)  # log actions for future monitoring

        # run simulation #########################################
        self.simulation_time += self.ts
        self.ss.TDS.config.tf = self.simulation_time
        run_status = self.ss.TDS.run()

        if not run_status:
            raise Exception('Attack compromised simulation')
        ###########################################################

        # collect observations ####################################
        observation = np.append(
            self.ss.Bus.v.v[self.observation_points.get('v', [])] - 1,
            self.ss.GENROU.omega.v[self.observation_points.get('omega', [])] - 1)

        domega_pu = (self.ss.GENROU.omega.v - self.prev_omega) / self.ts
        self.domega_pu.append(domega_pu)

        self.prev_omega = self.ss.GENROU.omega.v.copy()

        self.prev_observation = observation

        observation = np.append(
            observation,
            domega_pu[self.observation_points.get('domega', [])]
        )
        ###########################################################

        # Compute done flag ##########################################################
        # True if (1) of/uf, (2) rocof, (3) voltage trip, or (4) simulation end
        done = \
            np.any(np.logical_and(57.4 / 60 < self.ss.GENROU.omega.v,
                                  self.ss.GENROU.omega.v < 61.7 / 60), where=False) or \
            np.any(np.abs(domega_pu) >= 1 / 60) or \
            np.any(np.logical_and(0.7 < self.ss.Bus.v.v, self.ss.Bus.v.v < 1.3), where=False)
        done = (done and not suppress_done) or self.simulation_time >= self.episode_length
        ########################################################################################

        # compute reward ################################################
        # explanation of reward function in conference paper.
        reward = \
            1 * np.sum((domega_pu / (1 / 60)) ** 2) \
            + 5 * done
        ##################################################################

        info = {}

        return (np.reshape(np.array(observation, dtype=np.float32), (self.num_observations,)),
                reward, bool(done), info)

    def reset(self):
        """
        Reset the environment to the initial state.
        
        :return: List[float] - observations as outlined in initialization.
        """
        # uncomment to render after every episode
        # try:
        #     self.render()
        # except:
        #     pass

        andes.main.remove_output()  # clean any output files

        self.ss = andes.run(self.case_location, verbosity=50)  # loads power system and runs power flow
        self.ss.TDS.config.no_tqdm = 1 # reduces Andes logging

        self.prev_omega = np.ones(len(self.ss.GENROU), dtype=np.float32)  # array to store last omega obs
        self.actions = []  # array to store actions
        self.domega_pu = []  # array to save domega

        self.simulation_time = 0  # reset simulation time to 0 seconds

        for i in self.ss.Toggle.as_dict()['idx']:  # disable any toggles
            self.ss.Toggle.alter('u', i, 0)

        return np.zeros((self.num_observations,), dtype=np.float32)

    def render(self, renew=False):
        """
        Render plots of the power system state to inspect impact of attacks on system.
        
        :param renew: Bool. Whether to generate new Matplotlib figure to plot the figures.
        :return: None
        """
        self.ss.TDS.load_plotter()

        if self.figure is None or renew:
            self.figure, self.axis = plt.subplots(2, 2)
        else:
            for x in range(2):
                for y in range(2):
                    self.axis[x, y].clear()

        x, y = 0, 0
        self.axis[x, y].set_ylabel("(a)\nAttack (pu)")
        self.axis[x, y].plot(self.ts * np.array(range(int(round(self.simulation_time, 2) // self.ts) + 1)),
                             self.actions)
        L = []
        for k in self.attack_points:
            for v in self.attack_points[k]:
                L.append(f'${v[0]}_{{{v[1:-1]}}}$ @ G{k}')
        self.axis[x, y].legend(L, loc='upper left')

        x, y = 1, 0
        self.axis[x, y].set_ylabel("(b)\nVoltage (pu)")
        self.axis[x, y].set_xlabel("Time (s)")
        self.axis[x, y].plot(self.ss.TDS.plotter.get_values(0),
                             self.ss.TDS.plotter.get_values(self.ss.TDS.plotter.find('v Bus')[0]))

        x, y = 0, 1
        self.axis[x, y].set_ylabel("(c)\nFrequency (Hz)")
        self.axis[x, y].plot(self.ss.TDS.plotter.get_values(0),
                             60 * self.ss.TDS.plotter.get_values(self.ss.TDS.plotter.find('omega')[0]))

        x, y = 1, 1
        self.axis[x, y].set_xlabel("Time (s)")
        self.axis[x, y].set_ylabel("(d)\nRate-of-change of\nFrequency (Hz/s)")
        self.axis[x, y].plot(self.ts * np.array(range(int(round(self.simulation_time, 2) // self.ts) + 1)),
                             60 * np.array(self.domega_pu))

        self.figure.tight_layout()

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

        return None

    def close(self):
        andes.main.remove_output()
        return None

# TEST: uncomment to run
# ss = RLEnv(sampling_time=0.1,
#              episode_length=20,
#              attack_points={'GENROU_1': ['wref0'], 'GENROU_2': ['wref0'], 'GENROU_3': ['wref0', 'vref0']},
#              freq_bounds_hz=[59, 61],
#              voltage_bounds_pu=[-0.03, 0.03])

# cum_reward = 0
# for i in range(200):
#     _, reward, done, info = ss.step([(i % 50) / 50, (i % 20) / 20, (i % 30) / 30, (i % 25) / 25],
#                                     suppress_done=True)
#     cum_reward += reward
#     print(cum_reward)
#     if done:
#         print(info)
#         break
#
# ss.render()
