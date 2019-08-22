class TrainInformation:
    def __init__(self):
        self._average = 0.0
        self._best_reward = -float('inf')
        self._best_average = -float('inf')
        self._rewards = []
        self._average_range = 100
        self._index = 0

    @property
    def best_reward(self):
        return self._best_reward

    @property
    def best_average(self):
        return self._best_average

    @property
    def average(self):
        avg_range = self._average_range * -1
        return sum(self._rewards[avg_range:]) / len(self._rewards[avg_range:])

    @property
    def index(self):
        return self._index

    def _update_best_reward(self, episode_reward):
        if episode_reward > self.best_reward:
            self._best_reward = episode_reward

    def _update_best_average(self):
        if self.average > self.best_average:
            self._best_average = self.average
            return True
        return False

    def update_rewards(self, episode_reward):
        self._rewards.append(episode_reward)
        self._update_best_reward(episode_reward)
        return self._update_best_average()

    def update_index(self):
        self._index += 1
