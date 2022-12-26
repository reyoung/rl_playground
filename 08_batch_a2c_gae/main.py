import time

import gymnasium
from typing import Callable, List, Dict, Tuple, SupportsFloat

import numpy
import torch
import tqdm
import envpool


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
        probs = torch.nn.functional.softmax(self.actor(x), dim=len(x.shape) - 1)
        dist = torch.distributions.Categorical(probs)
        return dist, value

    def act(self, x) -> int:
        with torch.no_grad():
            dist, _ = self.forward(torch.tensor(x, dtype=torch.float32, device="cuda:0").unsqueeze(0))
            action = dist.sample().squeeze(0).item()
            return action


def compute_return(final_value: torch.Tensor, values: List[torch.Tensor],
                   rewards: List[List[SupportsFloat]], masks: List[List[bool]], gamma=0.99, tau=0.95) -> torch.Tensor:
    values = values + [final_value]
    values = torch.stack(values).squeeze(-1)
    n = len(rewards)
    gae = torch.zeros(dtype=torch.float32, size=(len(rewards[0]),), device="cuda:0")
    rewards = torch.tensor(rewards, dtype=torch.float32, device="cuda:0")
    masks = torch.tensor(masks, dtype=torch.float32, device="cuda:0")
    returns = []
    for step in reversed(range(n)):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.append(gae + values[step])

    returns = list(reversed(returns))
    returns = torch.stack(returns)
    return returns


def test_env(model: ActorCritic):
    env = gymnasium.make('CartPole-v1', render_mode="rgb_array")
    episode_id = 0
    total_reward = 0
    state, _ = env.reset()
    while True:
        action = model.act(state)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            if episode_id == 10:
                env.close()
                return total_reward
            episode_id += 1
            state, _ = env.reset()


def main():
    env = envpool.make_gym("CartPole-v1", num_envs=64)

    model = ActorCritic(num_inputs=env.observation_space.shape[0], num_actions=env.action_space.n)
    model.to("cuda:0")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    for batch_id in tqdm.tqdm(range(5000), desc="batch"):
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0

        state, _ = env.reset()
        for _ in range(1000):
            state = torch.tensor(state, dtype=torch.float32, device="cuda:0")
            dist, value = model(state)
            action = dist.sample()
            state, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            masks.append([not term and not trunc for term, trunc in zip(terminated, truncated)])
        rewards = numpy.stack(rewards)
        with torch.no_grad():
            _, value = model(torch.tensor(state, dtype=torch.float32, device="cuda:0"))
            returns = compute_return(value, values, rewards, masks)

        values = torch.stack(values).squeeze(-1)
        log_probs = torch.stack(log_probs)
        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (batch_id + 1) % 100 == 0:
            with torch.no_grad():
                total_reward = test_env(model)
                print(f"batch {batch_id + 1}, total reward {total_reward} / 10 episodes")


if __name__ == '__main__':
    main()
