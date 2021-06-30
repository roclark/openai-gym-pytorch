import atari_py as ap
import math
import numpy as np
import re
import torch
from .model import CNNDQN
from torch import FloatTensor, LongTensor
from torch.autograd import Variable


class Range:
    def __init__(self, start, end):
        self._start = start
        self._end = end

    def __eq__(self, input_num):
        return self._start <= input_num <= self._end


def compute_td_loss(model, target_net, batch, gamma, device):
    state, action, reward, next_state, done = batch

    state = Variable(FloatTensor(np.float32(state))).to(device)
    next_state = Variable(FloatTensor(np.float32(next_state))).to(device)
    action = Variable(LongTensor(action)).to(device)
    reward = Variable(FloatTensor(reward)).to(device)
    done = Variable(FloatTensor(done)).to(device)

    q_values = model(state)
    next_q_values = target_net(next_state)

    q_value = q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - Variable(expected_q_value.data).to(device)).pow(2).mean()
    loss.backward()


def update_epsilon(episode, args):
    eps_final = args.epsilon_final
    eps_start = args.epsilon_start
    decay = args.epsilon_decay
    epsilon = eps_final + (eps_start - eps_final) * \
        math.exp(-1 * ((episode + 1) / decay))
    return epsilon


def set_device(force_cpu):
    device = torch.device('cpu')
    if not force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
    return device


def load_model(checkpoint, model, target_model, device):
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    target_model.load_state_dict(model.state_dict())
    return model, target_model


def initialize_models(env, device, checkpoint):
    model = CNNDQN(env.observation_space.shape,
                   env.action_space.n).to(device)
    target_model = CNNDQN(env.observation_space.shape,
                          env.action_space.n).to(device)
    if checkpoint:
        model, target_model = load_model(checkpoint, model, target_model,
                                         device)
    return model, target_model


def camel_to_snake_case(string):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def is_atari(environment):
    for field in ['ramDeterministic', 'ramNoFrameSkip', 'NoFrameskip',
                  'Deterministic', 'ram']:
        environment = environment.replace(field, '')
    environment = re.sub(r'-v\d+', '', environment)
    environment = camel_to_snake_case(environment)
    if environment in ap.list_games():
        return True
    else:
        return False
