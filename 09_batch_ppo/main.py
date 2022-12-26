import contextlib
import time
from typing import Tuple

import envpool
import gymnasium
import numpy
import torch
import tqdm
from gymnasium.utils.save_video import save_video

device = torch.device("cuda:0")


class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs: int, num_actions: int):
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_actions)
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        value = self.critic(x)
        probs = self.actor(x)
        probs = torch.nn.functional.softmax(probs, dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist, value

    def act(self, x) -> int:
        with torch.no_grad():
            probs = self.actor(torch.tensor(x, dtype=torch.float32, device=device))
            action = torch.argmax(probs, 1).item()
            return action


def compute_return(final_value: torch.Tensor, values, rewards, masks, gamma=0.99, tau=0.95) -> torch.Tensor:
    values = values + [final_value]
    values = torch.stack(values).squeeze(-1)
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


WITH_TIMER = False


@contextlib.contextmanager
def timeit(label: str):
    begin = time.time()
    yield
    if WITH_TIMER:
        print(f"time {label} {time.time() - begin}")


def do_ppo_update(model, optimizer,
                  ppo_batch_size, stats: torch.Tensor, actions, log_probs, returns, advantages, clip=0.2):
    with torch.no_grad():
        stats = torch.permute(stats, [1, 0, 2])
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


def whiten(values, shift_mean=True):
    """Whiten values."""
    mean, var = torch.mean(values), torch.var(values)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def test_env(model: ActorCritic):
    env = envpool.make_gym("CartPole-v1", num_envs=1)
    episode_id = 0
    total_reward = 0
    state, _ = env.reset()
    while True:
        action = model.act(state)
        state, reward, terminated, truncated, info = env.step(numpy.array([action]))
        total_reward += reward
        if terminated or truncated:
            if episode_id == 10:
                env.close()
                return total_reward
            episode_id += 1
            state, _ = env.reset()


def main():
    env = envpool.make_gym("CartPole-v1", num_envs=1024)
    model = ActorCritic(num_inputs=env.observation_space.shape[0], num_actions=env.action_space.n)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    ppo_mini_batch_size = 512
    num_steps = 1000
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
            total_reward = 0
            with timeit(f"prepare num_steps {num_steps} data"):
                model.eval()
                for _ in range(num_steps):
                    state = torch.tensor(state).to(device)
                    states.append(state)
                    dist, value = model(state)
                    action = dist.sample()
                    actions.append(action)
                    state, reward, terminated, truncated, info = env.step(action.cpu().numpy())
                    total_reward += reward.sum()
                    log_prob = dist.log_prob(action)
                    entropy += dist.entropy().mean()
                    log_probs.append(log_prob)
                    values.append(value)
                    rewards.append(reward)
                    masks.append([not term and not trunc for term, trunc in zip(terminated, truncated)])

            rewards = numpy.stack(rewards)
            _, next_value = model(torch.tensor(state).to(device))
            returns = compute_return(next_value, values, rewards, masks)
            values = torch.stack(values).squeeze(-1)
            advantages = returns - values
            advantages = whiten(advantages)
            log_probs = torch.stack(log_probs)
            actions = torch.stack(actions)
            states = torch.stack(states)

        with timeit(f"ppo update"):
            model.train()
            do_ppo_update(model, optimizer, ppo_mini_batch_size,
                          states, actions, log_probs, returns, advantages)

        if (batch_id + 1) % 10 == 0:
            with torch.no_grad():
                total_reward = test_env(model)
                print(f"batch {batch_id + 1}, total reward {total_reward[0]} / 10 episodes")


if __name__ == '__main__':
    main()
