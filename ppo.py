import os
import torch
import torch.multiprocessing as _mp
import torch.nn.functional as F

from core.argparser import parse_args
#from core.wrappers import wrap_environment
from torch.distributions import Categorical
from wrappers import wrap_environment
from model import PPO


def test(args, global_model):
    done = True
    episode_reward = 0.0
    torch.manual_seed(123)

    env = wrap_environment(args.environment)
    model = PPO(env.observation_space.shape, env.action_space.n)
    model.eval()
    state = torch.from_numpy(env.reset())

    while True:
        if done:
            model.load_state_dict(global_model.state_dict())
        logits, value = model(state)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, info = env.step(action)
        episode_reward += reward

        if done:
            state = env.reset()
            print(f'Reward: {episode_reward}')
            episode_reward = 0.0
        state = torch.from_numpy(state)


def main():
    args = parse_args()
    args.tau = 1.0
    args.epsilon = 0.2
    args.beta = 0.01
    args.batch_size = 1
    args.num_processes = 16
    args.local_steps = 512
    mp = _mp.get_context('spawn')
    env = wrap_environment(args.environment)
    model = PPO(env.observation_space.shape, env.action_space.n)
    model.share_memory()
    process = mp.Process(target=test, args=(args, model))
    process.start()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for episode in range(args.num_episodes):
        episode_reward = 0.0
        state = torch.from_numpy(env.reset())
        current_state = state
        states, actions, dones, log_policies, rewards, values = [], [], [], [], [], []

        for _ in range(args.local_steps):
            states.append(current_state)
            logits, value = model(state)
            #values.append(value.squeeze())
            values.append(value)
            policy = F.softmax(logits, dim=1)
            action_policy = Categorical(policy)
            action = action_policy.sample()
            actions.append(action)
            log_policy = action_policy.log_prob(action)
            log_policies.append(log_policy)

            state, reward, done, info = env.step(action)
            state = torch.from_numpy(state)
            rewards.append(reward)
            dones.append(done)
            current_state = state

        _, next_value = model(state)
        next_value = next_value.squeeze()
        actions = torch.cat(actions)
        values = torch.cat(values).detach()
        log_policies = torch.cat(log_policies).detach()
        states = torch.cat(states)
        gae = 0
        R = []

        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = gae * args.gamma * args.tau
            gae = gae + reward + args.gamma * next_value.detach() * (1 - done) - value.detach()
            next_value = value
            R.append(gae + value)
        R = R[::-1]
        R = torch.cat(R).detach()
        advantages = R - values

        for epoch in range(10):
            index = torch.randperm(args.local_steps)

            for batch in range(args.batch_size):
                batch_indices = index[
                    int(batch * args.local_steps):
                    int((batch + 1) * args.local_steps)]
                logits, value = model(states[batch_indices])
                new_policy = F.softmax(logits, dim=1)
                new_action_policy = Categorical(new_policy)
                new_log_policy = new_action_policy.log_prob(actions[batch_indices])
                ratio = torch.exp(new_log_policy - log_policies[batch_indices])
                actor_loss = -torch.mean(torch.min(ratio * advantages[batch_indices],
                                         torch.clamp(ratio, 1.0 - args.epsilon, 1.0 + args.epsilon) *
                                         advantages[batch_indices]))
                critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze())
                entropy_loss = torch.mean(new_action_policy.entropy())
                total_loss = actor_loss + critic_loss - args.beta * entropy_loss
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    main()
