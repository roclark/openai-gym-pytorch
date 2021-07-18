from core.agents import models
from core.argparser import parse_args
from core.configs import configs
from core.helpers import (compute_td_loss,
                          initialize_model,
                          initialize_model_with_target,
                          is_atari,
                          set_device,
                          update_epsilon)
from core.optimizers import GlobalAdam
from core.replay_buffer import ReplayBuffer
from core.train_information import TrainInformation
from core.wrappers import wrap_environment

import torch
import torch.multiprocessing as _mp
from torch.distributions import Categorical
from torch.optim import Adam
from torch.nn.functional import log_softmax, softmax


def update_graph(model, target_model, optimizer, replay_buffer, args, device,
                 info):
    if len(replay_buffer) > args.initial_learning:
        if not info.index % args.target_update_frequency:
            target_model.load_state_dict(model.state_dict())
        optimizer.zero_grad()
        batch = replay_buffer.sample(args.batch_size)
        compute_td_loss(model, target_model, batch, args.gamma, device)
        optimizer.step()


def complete_episode(model, environment, info, episode_reward, episode,
                     epsilon):
    new_best = info.update_rewards(episode_reward)
    if new_best:
        print('New best average reward of %s! Saving model'
              % round(info.best_average, 3))
        torch.save(model.state_dict(), '%s.dat' % environment)
    print('Episode %s - Reward: %s, Best: %s, Average: %s '
          'Epsilon: %s' % (episode, episode_reward, info.best_reward,
                           round(info.average, 3), round(epsilon, 4)))


def train_loop(env, args, model, state, hx, cx, device):
    log_policies, values, rewards, entropies = [], [], [], []

    while True:
        logits, value, hx, cx = model(state.unsqueeze(0), hx, cx)
        policy = softmax(logits, dim=1)
        log_policy = log_softmax(logits, dim=1)
        entropy = -(policy * log_policy).sum(1, keepdim=True)

        multinomial = Categorical(policy)
        action = multinomial.sample().item()

        next_state, reward, done, _ = env.step(action)
        state = torch.from_numpy(next_state).to(device)

        values.append(value)
        log_policies.append(log_policy[0, action])
        rewards.append(reward)
        entropies.append(entropy)

        if done:
            state = torch.from_numpy(env.reset()).to(device)
            return model, state, values, log_policies, rewards, entropies


def setup_process(rank, args, model_ref, device):
    torch.manual_seed(123 + rank)
    env = wrap_environment(args.environment)
    model = initialize_model(env, device, args.checkpoint, model_ref)
    state = torch.from_numpy(env.reset()).to(device)
    model.train()
    return env, model, state


def update_network(optimizer, total_loss, model, global_model):
    optimizer.zero_grad()
    total_loss.backward()

    for local_param, global_param in zip(model.parameters(), global_model.parameters()):
        if global_param.grad is not None:
            break
        global_param._grad = local_param.grad

    optimizer.step()
    return optimizer, model, global_model


def calculate_loss(args, loss_values, device):
    R = torch.zeros((1, 1), dtype=torch.float).to(device)
    gae = torch.zeros((1, 1), dtype=torch.float).to(device)
    actor_loss, critic_loss, entropy_loss = 0, 0, 0
    next_value = R

    for value, log_policy, reward, entropy in loss_values[::-1]:
        gae = gae * args.gamma * args.tau
        gae = gae + reward + args.gamma * next_value.detach() - value.detach()
        next_value = value
        actor_loss = actor_loss + log_policy * gae
        R = R * args.gamma + reward
        critic_loss = critic_loss + (R - value) ** 2 / 2
        entropy_loss = entropy_loss + entropy

    total_loss = -actor_loss + critic_loss - args.beta * entropy_loss
    return total_loss


def train(rank, global_model, optimizer, model_ref, device, args):
    env, model, state = setup_process(rank, args, model_ref, device)
    info = TrainInformation()

    for episode in range(args.num_episodes):
        model.load_state_dict(global_model.state_dict())
        hx = torch.zeros((1, 512), dtype=torch.float).to(device)
        cx = torch.zeros((1, 512), dtype=torch.float).to(device)

        train_outputs = train_loop(env, args, model, state, hx, cx, device)
        model, state, values, log_policies, rewards, entropies = train_outputs

        loss_values = list(zip(values, log_policies, rewards, entropies))
        total_loss = calculate_loss(args, loss_values, device)
        optimizer, model, global_model = update_network(optimizer, total_loss,
                                                        model, global_model)


def complete_episode(environment, info, episode_reward, episode, model, args):
    best = info.best_reward
    new_best = info.update_rewards(episode_reward)
    save_model = False
    if episode_reward > best:
        print(f'New high score of {round(episode_reward, 3)}! Saving model')
        save_model = True
    print(f'Episode {episode} - Reward: {round(episode_reward, 3)}, '
          f'Best: {round(info.best_reward, 3)}, '
          f'Average: {round(info.average, 3)}')


def test_loop(env, model, global_model, state, done, args, info,
              episode_reward, hx, cx):
    if done:
        model.load_state_dict(global_model.state_dict())
    with torch.no_grad():
        if done:
            hx = torch.zeros((1, 512), dtype=torch.float)
            cx = torch.zeros((1, 512), dtype=torch.float)
        else:
            hx = hx.detach()
            cx = cx.detach()
    logit, value, hx, cx = model(state.unsqueeze(0), hx, cx)
    policy = softmax(logit, dim=1)
    action = torch.argmax(policy).item()
    next_state, reward, done, _ = env.step(action)
    if args.render:
        env.render()
    episode_reward += reward
    if done:
        info.update_index()
        complete_episode(args.environment, info, episode_reward, info.index,
                         model, args)
        episode_reward = 0.0
        next_state = env.reset()
    state = torch.from_numpy(next_state)
    return model, hx, cx, state, done, info, episode_reward


def test(env, global_model, model_ref, device, args):
    torch.manual_seed(123 + vars(args).get('num_processes', 1))
    info = TrainInformation()
    env = wrap_environment(args.environment)
    model = initialize_model(env, device, args.checkpoint, model_ref)
    model.eval()

    state = torch.from_numpy(env.reset())
    done = True
    episode_reward = 0.0
    hx, cx = None, None

    while True:
        loop_outputs = test_loop(env, model, global_model, state, done, args,
                                 info, episode_reward, hx, cx)
        model, hx, cx, state, done, info, episode_reward = loop_outputs


def merge_settings(args, settings):
    for setting, value in settings.items():
        setattr(args, setting, value)
    return args


def main():
    args = parse_args()
    model_ref = models[args.model]
    settings = configs[args.model]
    args = merge_settings(args, settings)
    torch.manual_seed(123)
    mp = _mp.get_context('spawn')
    env = wrap_environment(args.environment)
    device = set_device(args.force_cpu)
    if vars(args).get('target_model', False):
        model, target = initialize_model_with_target(env, device,
                                                     args.checkpoint,
                                                     model_ref)
    else:
        model = initialize_model(env, device, args.checkpoint, model_ref)

    if vars(args).get('optimizer', False):
        optimizer = GlobalAdam(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = Adam(model.parameters(), lr=args.learning_rate)

    processes = []

    for rank in range(vars(args).get('num_processes', 1)):
        process = mp.Process(target=train, args=(rank, model, optimizer, model_ref, device, args))
        process.start()
        processes.append(process)
    process = mp.Process(target=test, args=(env, model, model_ref, device, args))
    process.start()
    processes.append(process)

    for process in processes:
        process.join()

    env.close()


if __name__ == '__main__':
    main()
