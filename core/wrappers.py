import cv2
import numpy as np
from collections import deque
from core.helpers import is_atari
from gym import make, ObservationWrapper, wrappers, Wrapper
from gym.spaces import Box


class ClassicControl(Wrapper):
    def __init__(self, env, atari):
        super(ClassicControl, self).__init__(env)
        self._atari = atari

    def reset(self):
        if not self._atari:
            self.env.reset()
            return self.env.render(mode='rgb_array')
        else:
            return self.env.reset()


class FrameDownsample(ObservationWrapper):
    def __init__(self, env):
        super(FrameDownsample, self).__init__(env)
        self.observation_space = Box(low=0,
                                     high=255,
                                     shape=(84, 84, 1),
                                     dtype=np.uint8)
        self._width = 84
        self._height = 84

    def observation(self, observation):
        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame,
                           (self._width, self._height),
                           interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class MaxAndSkipEnv(Wrapper):
    def __init__(self, env, atari, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip
        self._atari = atari

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if not self._atari:
                obs = self.env.render(mode='rgb_array')
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class FireResetEnv(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env)
        if len(env.unwrapped.get_action_meanings()) < 3:
            raise ValueError('Expected an action space of at least 3!')

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class FrameBuffer(ObservationWrapper):
    def __init__(self, env, num_steps, dtype=np.float32):
        super(FrameBuffer, self).__init__(env)
        obs_space = env.observation_space
        self._dtype = dtype
        self.observation_space = Box(obs_space.low.repeat(num_steps, axis=0),
                                     obs_space.high.repeat(num_steps, axis=0),
                                     dtype=self._dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low,
                                    dtype=self._dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class ImageToPyTorch(ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(low=0.0,
                                     high=1.0,
                                     shape=(obs_shape[::-1]),
                                     dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class NormalizeFloats(ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


def wrap_environment(environment, monitor=False):
    env = make(environment)
    atari = is_atari(environment)
    env = ClassicControl(env, atari)
    env = MaxAndSkipEnv(env, atari)
    try:
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
    except AttributeError:
        # Some environments, such as the classic control environments, don't
        # have a get_action_meanings method. Since these environments don't
        # contain a 'FIRE' action, this wrapper is irrelevant and can be safely
        # ignored if the attribute doesn't exist.
        pass
    env = FrameDownsample(env)
    env = ImageToPyTorch(env)
    env = FrameBuffer(env, 4)
    env = NormalizeFloats(env)
    if monitor:
        env = wrappers.Monitor(env, 'videos', force=True)
    return env
