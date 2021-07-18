from .a3c import ActorCritic
from .dqn import CNNDQN


models = {
    'a3c': ActorCritic,
    'dqn': CNNDQN
}
