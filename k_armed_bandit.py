from abc import ABC, abstractmethod
import functools
import itertools
import multiprocessing as mp
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Reward(ABC):

    def __init__(self, num_actions: int):
        self._num_actions = num_actions
        self._action_to_reward = None
        self._build()

    @abstractmethod
    def _build(self):
        pass

    def expected_optimal_reward(self):
        return max(self.expected_reward(action) for action in range(self._num_actions))

    @abstractmethod
    def expected_reward(self, action: int):
        pass

    def __call__(self, action: int):
        reward = self._action_to_reward[action]()
        return reward


class NormalReward(Reward):

    def _build(self):
        means = np.random.normal(size=self._num_actions)
        scales = np.random.uniform(0, 1, size=self._num_actions)

        self._action_to_reward = [functools.partial(np.random.normal, loc=mean, scale=scale)
                                  for i, (mean, scale) in enumerate(zip(means, scales))]

    def expected_reward(self, action: int):
        return self._action_to_reward[action].keywords['loc']


class Agent:

    def __init__(self, num_actions: int, reward_function: callable, initial_value_estimate: float = 0):
        num_actions = int(num_actions)

        self._num_actions = num_actions
        self._reward_function = reward_function
        self._n = [0] * self._num_actions

        if isinstance(initial_value_estimate, float) or isinstance(initial_value_estimate, int):
            self._value_estimate = [float(initial_value_estimate)] * self._num_actions
        elif isinstance(initial_value_estimate, list):
            assert len(initial_value_estimate) == num_actions, 'The number of initial values must be equal to the ' \
                                                               'number of actions.'
            self._value_estimate = [float(ive) for ive in initial_value_estimate]

    def _update_value_estimate(self, action: int, reward: float):
        self._n[action] += 1
        step_size = 1 / self._n[action]
        value_estimate = self._value_estimate[action]
        self._value_estimate[action] = value_estimate + step_size * (reward - value_estimate)


class EpsilonGreedyAgent(Agent):
    r"""
    Implement an epsilon greedy agent.

    :ivar _num_actions int: The number of actions to take, which is the same as the number of arms in a k armed bandit
    problem.
    :ivar _epsilon float: It defines the way an agent is to tradeoff action exploration and exploitation. THe closer
    it is to zero, the more the agent exploits. Conversely, the closer it is to one, the more the agent explores.
    :ivar _reward_function callable: A callable which accepts an action and returns its respective reward value.
    :ivar _n list[int]: The number of times the agent has taken each action.
    :ivar _value_estimate list[float]: The current value estimate for each action expected reward.

    Some facts to consider:
    - The value estimate of an actions is computed as the average reward received throughout past times the actions was
    taken. Specifically, the action value estimate $$Q_{t+1} = Q_t + \frac1t \left(R_t - Q_t)\right) $$ where $$R_t$$
    is the reward received at time $$t$$.
    - At every time step, the agent randomly chooses between exploration and exploitation based on epsilon.  If its
    choice is to explore, then it uniformly choose one of the actions. Otherwise, it chooses the action having the
    current maximum value estimate.
    - After choosing an action and receiving the reward for it, the agent updates the action value estimate accordingly.

    """

    def __init__(self, *, epsilon: float, **kwargs):
        super().__init__(**kwargs)
        epsilon = float(epsilon)
        self._epsilon = epsilon

    def act(self):

        action = self._get_random_action() if self._explore() else self._get_max_action()
        reward = self._reward_function(action)
        self._update_value_estimate(action, reward)

        return action, reward

    def _get_random_action(self) -> int:
        return random.choice(range(self._num_actions))

    def _get_max_action(self) -> int:

        action_and_value_estimate_pairs = [(action, value_estimate)
                                           for action, value_estimate in
                                           sorted(enumerate(self._value_estimate), key=lambda x: x[-1], reverse=True)]

        max_value_estimate = action_and_value_estimate_pairs[0][1]
        actions = [action for action, value_estimate in action_and_value_estimate_pairs
                   if value_estimate == max_value_estimate]

        action = random.choice(actions)

        return action

    def _explore(self) -> bool:
        explore = random.choices([True, False], weights=[self._epsilon, 1 - self._epsilon])[0]
        return explore


class ConstantStepSizeEpsilonGreedyAgent(EpsilonGreedyAgent):
    """
    Implement an epsilon-greedy agent with constant step size.

    :ivar _step_size: float = The step size used to update the value estimates.

    Some facts to consider:
    - Adopting a constant step size makes the agent to give more importance to recently received rewards. In fact,
    the importance of a reward value obtained in time $$t$$ exponentially decays along time.

    """

    def __init__(self, *, step_size: float, **kwargs):
        super().__init__(**kwargs)

        assert 0 < step_size <= 1, "step_size must be a value between zero (exclusively) and one (inclusively)."

        self._step_size = step_size

    def _update_value_estimate(self, action: int, reward: float):
        self._n[action] += 1
        value_estimate = self._value_estimate[action]
        self._value_estimate[action] = value_estimate + self._step_size * (reward - value_estimate)


class VaryingStepSizeEpsilonGreedyAgent(EpsilonGreedyAgent):
    """
    Implement an epsilon-greedy agent with a custom and varying step size.

    :ivar _step_size_maker = A callable which accepts an action and a time instant and returns a step size value.

    Some facts to consider:
    - One adopts a varying step-size to customizing the weighting of past rewards in the update of value estimates.
    """

    def __init__(self, *, step_size_maker: callable, **kwargs):
        super().__init__(**kwargs)
        self._step_size_maker = step_size_maker

    def _update_value_estimate(self, action: int, reward: float):
        self._n[action] += 1
        value_estimate = self._value_estimate[action]
        step_size = self._step_size_maker(action, self._n[action])
        assert 0 < step_size <= 1, f'{self._step_size_maker} return a step size value = {step_size} out of the ' \
                                   f'allowed range (0, 1].'
        self._value_estimate[action] = value_estimate + step_size * (reward - value_estimate)


class UpperConfidenceBoundAgent(Agent):
    """
    Implement an epsilon greedy agent which takes into account the uncertainty underlying action value estimates.
    Specifically, when choosing an action, the agent choose not the one with the maximum current (exploitation) value
    estimate or a random one (exploration); instead it balances its choice based on both the maximum current value
    estimate and the uncertainty associated with a given action. Note that as time goes, the uncertainty of the value
    estimate decreases and the agent tends to give more importance to past value estimates.
    """

    def __init__(self, *,  c, **kwargs):
        super().__init__(**kwargs)
        assert c > 0, 'The exploration parameter must be greater than zero.'
        self._c = c

    def act(self):

        action = self._get_max_action()
        reward = self._reward_function(action)
        self._update_value_estimate(action, reward)

        return action, reward

    def _get_max_action(self) -> int:

        t = sum(self._n)

        # Actions not yet chosen are considered maximal.
        zero_actions = [(action, value_estimate)
                        for action, value_estimate in enumerate(self._value_estimate) if self._n[action] == 0]

        if len(zero_actions) != 0:
            # When performing argmax the indices is restricted to the zero actions. But we need to return
            # the actual index.
            zero_action_id_to_action_id = {z_id: a_id for z_id, (a_id, _) in enumerate(zero_actions)}

            max_actions = arg_max(zero_actions, key=lambda x: x[-1])
            max_action = random.choice(max_actions)
            max_action = zero_action_id_to_action_id[max_action]

        else:
            action_value_estimates = [value_estimate + self._c * np.sqrt(np.log(t) / self._n[action])
                                      for action, value_estimate in enumerate(self._value_estimate)]

            max_actions = arg_max(action_value_estimates)
            max_action = random.choice(max_actions)

        return max_action


def arg_max(sequence, key=None):

    def _get_value(v):
        return v if key is None else key(v)

    max_value = None
    positions = set()

    for i, value in enumerate(sequence):
        if max_value is None:
            max_value = _get_value(value)
            positions.add(i)
        else:
            if _get_value(value) == max_value:
                positions.add(i)
            elif _get_value(value) > max_value:
                positions = {i}

    return list(positions)


def run(epochs, num_actions, epsilon, reward_function):
    agent = UpperConfidenceBoundAgent(num_actions=num_actions, c=1, reward_function=reward_function)
    rewards = [agent.act()[-1] for _ in range(epochs)]
    return rewards


def main():
    num_actions = 10
    epochs = 1000
    num_runs = 2000
    num_epsilons = 1
    epsilons = np.linspace(0, 1, num_epsilons)
    reward_function = NormalReward(num_actions)
    rewards = {'epsilon': np.ones((0,)), 'epoch': np.ones((0,)), 'reward': np.ones((0,))}

    with mp.Pool(processes=8) as pool:
        for epsilon in epsilons:
            iterable = zip(itertools.repeat(epochs, num_runs), itertools.repeat(num_actions, num_runs),
                           itertools.repeat(epsilon, num_runs), itertools.repeat(reward_function, num_runs))

            current_rewards = pool.starmap(run, iterable)
            current_rewards = np.array(current_rewards)
            current_rewards = current_rewards.ravel()

            rewards['epsilon'] = np.concatenate([rewards['epsilon'], np.full((epochs*num_runs,), epsilon)], axis=0)
            rewards['epoch'] = np.concatenate([rewards['epoch'], np.tile(range(epochs), num_runs)], axis=0)
            rewards['reward'] = np.concatenate([rewards['reward'], current_rewards], axis=0)

    rewards = pd.DataFrame.from_dict(rewards)
    rewards['optimal'] = reward_function.expected_optimal_reward()
    palette = sns.color_palette("hls", num_epsilons)
    sns.lineplot(x='epoch', y='reward', hue='epsilon', data=rewards, legend='full', palette=palette)
    plt.show()


if __name__ == '__main__':
    main()
