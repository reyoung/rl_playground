from typing import Tuple, SupportsFloat, Any, List

import gymnasium
import numpy
import torch.nn
from gymnasium import Env
from gymnasium.core import WrapperActType, ActType, ObsType
from gymnasium.utils.save_video import save_video
import tqdm


class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs: int, num_actions: int):
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_actions),
            torch.nn.Softmax()
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        value = self.critic(x)
        probs = self.actor(x)
        dist = torch.distributions.Categorical(probs)
        return dist, value

    def act(self, x) -> int:
        with torch.no_grad():
            dist, _ = self.forward(torch.tensor(x, dtype=torch.float32, device="cuda:0").unsqueeze(0))
            action = dist.sample().squeeze(0).item()
            return action


def compute_return(final_value: float, values: List[float],
                   rewards: List[SupportsFloat], masks: List[bool], gamma=0.99, tau=0.95) -> torch.Tensor:
    values = values + [final_value]
    returns = []
    gae = 0
    for step in reversed(range(len(rewards))):
        delta = float(rewards[step]) + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.append(gae + values[step])
    returns = list(reversed(returns))
    return torch.tensor(returns, dtype=torch.float32, device='cuda:0')


def test_env(env, model: ActorCritic) -> float:
    episode_id = 0
    total_reward = 0
    state, _ = env.reset()
    while True:
        action = model.act(state)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            if episode_id == 10:
                return total_reward
            episode_id += 1
            state, _ = env.reset()


def main():
    env = gymnasium.make('CartPole-v1', render_mode="rgb_array")
    env = gymnasium.wrappers.AutoResetWrapper(env)
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
            state = torch.tensor(state, dtype=torch.float32, device="cuda:0").unsqueeze(0)
            dist, value = model(state)
            action = dist.sample().squeeze(0)
            state, reward, terminated, truncated, info = env.step(action.item())
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            done = terminated or truncated
            masks.append(not done)
        values = torch.cat(values)

        with torch.no_grad():
            _, value = model(torch.tensor(state, dtype=torch.float32, device="cuda:0").unsqueeze(0))
            returns = compute_return(value.item(), values.flatten().cpu().tolist(), rewards, masks)

        log_probs = torch.cat(log_probs)
        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (batch_id + 1) % 100 == 0:
            with torch.no_grad():
                total_reward = test_env(env, model)

            print(f"batch {batch_id + 1}, total reward {total_reward} / 10 episodes")

    # let's record some videos for trained model
    state, _ = env.reset()
    episode = 0
    frames = []
    while True:
        frames.append(env.render())
        action = model.act(state)
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            frames.append(env.render())
            state, _ = env.reset()
            save_video(frames,
                       "videos",
                       episode_trigger=lambda *args: True,
                       episode_index=episode,
                       fps=env.metadata['render_fps'])
            episode += 1
            if episode == 10:
                break


if __name__ == '__main__':
    main()
