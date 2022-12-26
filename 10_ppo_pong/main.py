from typing import Sequence

import torch
import envpool
import tqdm
import numpy

device = torch.device("cuda:0")


class ConvPong(torch.nn.Module):
    def __init__(self, input_shape: Sequence[int], num_actions: int):
        super().__init__()
        self.input_shape = input_shape
        self.feature = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )
        self.feature_size = self._feature_size()
        self.flatten_feature = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.feature_size, out_features=128),
            torch.nn.ReLU()
        )

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(in_features=128, out_features=num_actions)
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(in_features=128, out_features=1)
        )

    def forward(self, x):
        x /= 255.0
        shape = x.shape
        if len(shape) == 5:  # batch_size, seq_len, c, h, w
            x.resize_(shape[0] * shape[1], shape[2], shape[3], shape[4])

        feat = self.feature(x)
        feat = self.flatten_feature(feat)
        probs = self.actor(feat)
        probs = torch.nn.functional.softmax(probs, dim=-1)
        value = self.critic(feat)
        if len(shape) == 5:
            probs = torch.reshape(probs, (shape[0], shape[1], probs.shape[-1]))
            value = torch.reshape(value, (shape[0], shape[1], 1))

        return torch.distributions.Categorical(probs), value

    def act(self, x) -> int:
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32, device=device)
            x /= 255.0
            x.unsqueeze(0)
            x = self.feature(x)
            x = self.flatten_feature(x)
            probs = self.actor(x)
            return torch.argmax(probs, 1).item()

    def _feature_size(self):
        with torch.no_grad():
            return self.feature(torch.zeros(1, *self.input_shape)).shape[1]


def whiten(values, shift_mean=True):
    """Whiten values."""
    mean, var = torch.mean(values), torch.var(values)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def compute_return(final_value: torch.Tensor, values, rewards, masks, gamma=0.99, tau=0.95) -> torch.Tensor:
    values = torch.cat([values, final_value.unsqueeze(0)]).squeeze(-1)
    n = len(rewards)
    gae = torch.zeros(dtype=torch.float32, size=(len(rewards[0]),), device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    masks = torch.tensor(masks, dtype=torch.float32, device=device)
    returns = []
    for step in reversed(range(n)):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.append(gae + values[step])
    returns = torch.stack(returns[::-1])
    return returns


def do_ppo_update(model, optimizer,
                  ppo_batch_size, stats: torch.Tensor, actions, log_probs, returns, advantages, clip=0.2):
    with torch.no_grad():
        stats = torch.permute(stats, [1, 0, 2, 3, 4])
        actions = torch.permute(actions, [1, 0])
        log_probs = torch.permute(log_probs, [1, 0])
        returns = torch.permute(returns, [1, 0])
        advantages = torch.permute(advantages, [1, 0])

    ids = numpy.arange(stats.shape[0])
    numpy.random.shuffle(ids)

    for _ in range(3):
        for start in range(0, stats.shape[0], ppo_batch_size):
            end = start + ppo_batch_size
            to_train = ids[start: end]
            state = stats[to_train, :]
            action = actions[to_train, :]
            old_log_prob = log_probs[to_train, :]
            advantage = advantages[to_train, :]
            return_ = returns[to_train, :]

            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)
            ratio = (new_log_probs - old_log_prob).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage
            actor_loss = - torch.min(surr2, surr1).mean()
            critic_loss = (return_ - value.squeeze(-1)) ** 2
            critic_loss = critic_loss.mean()
            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def test_env(model):
    env = envpool.make_gym("Pong-v5", num_envs=1, episodic_life=True, reward_clip=True, stack_num=4)
    total_reward = 0
    state, _ = env.reset()
    while True:
        action = model.act(state)
        state, reward, terminated, truncated, info = env.step(numpy.array([action]))
        total_reward += reward
        if terminated or truncated:
            env.close()
            return total_reward


def main():
    env = envpool.make_gym("Pong-v5", num_envs=64, episodic_life=True, reward_clip=True, stack_num=4)
    model = ConvPong(input_shape=env.observation_space.shape, num_actions=env.action_space.n)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    ppo_mini_batch_size = 2
    num_steps = 512
    state, _ = env.reset()

    for batch_id in tqdm.tqdm(range(5000), desc="batch"):
        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []
        entropy = 0
        with torch.no_grad():
            model.eval()
            for _ in range(num_steps):
                states.append(state)
                state = torch.tensor(state, dtype=torch.float32).to(device)
                dist, value = model(state)
                action = dist.sample()
                actions.append(action.cpu().numpy())
                state, reward, terminated, truncated, info = env.step(actions[-1])
                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()
                log_probs.append(log_prob.cpu().numpy())
                values.append(value.cpu().numpy())
                rewards.append(reward)
                masks.append([not term and not trunc for term, trunc in zip(terminated, truncated)])
            states = numpy.stack(states)
            states = torch.tensor(states, dtype=torch.float32, device=device)
            actions = torch.tensor(numpy.stack(actions), dtype=torch.long, device=device)
            log_probs = torch.tensor(numpy.stack(log_probs), dtype=torch.float32, device=device)
            values = torch.tensor(numpy.stack(values), dtype=torch.float32, device=device)

            rewards = numpy.stack(rewards)
            print("training reward", rewards.sum())
            _, next_value = model(torch.tensor(state, dtype=torch.float32).to(device))
            returns = compute_return(next_value, values, rewards, masks)
            values = values.squeeze(-1)
            advantages = returns - values
            advantages = whiten(advantages)

        model.train()
        do_ppo_update(model, optimizer, ppo_mini_batch_size,
                      states, actions, log_probs, returns, advantages)

        if (batch_id + 1) % 10 == 0:
            with torch.no_grad():
                total_reward = test_env(model)
                print(f"batch {batch_id + 1}, reward {total_reward[0]} ")


if __name__ == '__main__':
    main()
