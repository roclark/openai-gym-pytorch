from argparse import ArgumentParser
from core.helpers import (initialize_models,
                          set_device)
from core.wrappers import wrap_environment


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Specify the trained '
                        'model to test.')
    parser.add_argument('--environment', type=str, help='Specify the '
                        'environment to test against.',
                        default='PongNoFrameskip-v4')
    parser.add_argument('--force-cpu', action='store_true', help='Force '
                        'computation to be done on the CPU. This may result '
                        'in longer processing time.')
    return parser.parse_args()


def main():
    args = parse_args()
    env = wrap_environment(args.environment, monitor=True)
    device = set_device(args.force_cpu)
    model, target_model = initialize_models(env, device, args.checkpoint)

    done = False
    state = env.reset()
    episode_reward = 0.0

    while not done:
        action = model.act(state, device)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = next_state

    print(f'Episode Reward: {round(episode_reward, 3)}')
    env.close()


if __name__ == '__main__':
    main()
