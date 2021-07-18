from argparse import ArgumentParser

from core.agents import models
from core.constants import (BATCH_SIZE,
                            ENVIRONMENT,
                            EPSILON_START,
                            EPSILON_FINAL,
                            EPSILON_DECAY,
                            GAMMA,
                            INITIAL_LEARNING,
                            LEARNING_RATE,
                            MEMORY_CAPACITY,
                            MODEL,
                            NUM_EPISODES,
                            TARGET_UPDATE_FREQUENCY)
from core.helpers import Range


def parse_args():
    parser = ArgumentParser(description='')
    parser.add_argument('--batch-size', type=int, help='Specify the batch '
                        'size to use when updating the replay buffer. '
                        'Default: %s' % BATCH_SIZE, default=BATCH_SIZE)
    parser.add_argument('--buffer-capacity', type=int, help='The capacity to '
                        'use in the experience replay buffer. Default: %s'
                        % MEMORY_CAPACITY, default=MEMORY_CAPACITY)
    parser.add_argument('--checkpoint', type=str, help='Specify a .dat file '
                        'to be used as a checkpoint to initialize weights for '
                        'a new training run. Defaults to no checkpoint.')
    parser.add_argument('--environment', type=str, help='The OpenAI gym '
                        'environment to use. Default: %s' % ENVIRONMENT,
                        default=ENVIRONMENT)
    parser.add_argument('--epsilon-start', type=float, help='The initial '
                        'value for epsilon to be used in the epsilon-greedy '
                        'algorithm. Default: %s' % EPSILON_START,
                        choices=[Range(0.0, 1.0)], default=EPSILON_START,
                        metavar='EPSILON_START')
    parser.add_argument('--epsilon-final', type=float, help='The final value '
                        'for epislon to be used in the epsilon-greedy '
                        'algorithm. Default: %s' % EPSILON_FINAL,
                        choices=[Range(0.0, 1.0)], default=EPSILON_FINAL,
                        metavar='EPSILON_FINAL')
    parser.add_argument('--epsilon-decay', type=int, help='The decay factor '
                        'for epsilon in the epsilon-greedy algorithm. '
                        'Default: %s' % EPSILON_DECAY, default=EPSILON_DECAY)
    parser.add_argument('--force-cpu', action='store_true', help='By default, '
                        'the program will run on the first supported GPU '
                        'identified by the system, if applicable. If a '
                        'supported GPU is installed, but all computations are '
                        'desired to run on the CPU only, specify this '
                        'parameter to ignore the GPUs. All actions will run '
                        'on the CPU if no supported GPUs are found. Default: '
                        'False')
    parser.add_argument('--gamma', type=float, help='Specify the discount '
                        'factor, gamma, to use in the Q-table formula. '
                        'Default: %s' % GAMMA, choices=[Range(0.0, 1.0)],
                        default=GAMMA, metavar='GAMMA')
    parser.add_argument('--initial-learning', type=int, help='The number of '
                        'iterations to explore prior to updating the model '
                        'and begin the learning process. Default: %s'
                        % INITIAL_LEARNING, default=INITIAL_LEARNING)
    parser.add_argument('--model', type=str, help='The model to use for '
                        f'training. Default: {MODEL}', default=MODEL,
                        choices=models.keys())
    parser.add_argument('--learning-rate', type=float, help='The learning '
                        'rate to use for the optimizer. Default: %s'
                        % LEARNING_RATE, default=LEARNING_RATE)
    parser.add_argument('--num-episodes', type=int, help='The number of '
                        'episodes to run in the given environment. Default: '
                        '%s' % NUM_EPISODES, default=NUM_EPISODES)
    parser.add_argument('--render', action='store_true', help='Specify to '
                        'render a visualization in another window of the '
                        'learning process. Note that a Desktop Environment is '
                        'required for visualization. Rendering scenes will '
                        'lower the learning speed. Default: False')
    parser.add_argument('--target-update-frequency', type=int, help='Specify '
                        'the number of iterations to run prior to updating '
                        'target network with the primary network\'s weights. '
                        'Default: %s' % TARGET_UPDATE_FREQUENCY,
                        default=TARGET_UPDATE_FREQUENCY)
    return parser.parse_args()
