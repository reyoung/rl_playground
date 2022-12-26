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
            torch.nn.Linear(128, num_actions),
            torch.nn.Softmax(dim=0)
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

def compute_return(final_value: torch.Tensor, values, rewards, masks, gamma=0.99, tau=0.95) -> torch.Tensor:
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


WITH_TIMER = False


@contextlib.contextmanager
def timeit(label: str):
    begin = time.time()
    yield
    if WITH_TIMER:
        print(f"time {label} {time.time() - begin}")


def do_ppo_update(model, optimizer,
                  ppo_epochs, ppo_n_batch, ppo_batch_size, stats, actions, log_probs, returns, advantages, clip=0.2):
    n = stats.shape[0]
    ids = numpy.arange(n)
    numpy.random.shuffle(ids)

    for _ in range(ppo_epochs):
        for start in range(0, ppo_n_batch, ppo_batch_size):
            end = start + ppo_batch_size
            to_train = ids[start: end]

            state = stats[to_train, :]
            action = actions[to_train, :]
            old_log_prob = log_probs[to_train, :]
            return_ = returns[to_train, :]
            advantage = advantages[to_train, :]

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
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    ppo_epochs = 4
    ppo_mini_batch_size = 5
    ppo_n_batch = 50
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
        with timeit(f"prepare num_steps {num_steps} data"):
            for _ in range(num_steps):
                state = torch.tensor(state).to(device)
                states.append(state)
                dist, value = model(state)
                action = dist.sample()
                actions.append(action)
                state, reward, terminated, truncated, info = env.step(action.cpu().numpy())
                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                masks.append([not term and not trunc for term, trunc in zip(terminated, truncated)])

        rewards = numpy.stack(rewards)
        with torch.no_grad():
            _, next_value = model(torch.tensor(state).to(device))
            returns = compute_return(next_value, values, rewards, masks)
            values = torch.stack(values).squeeze(-1)
            advantages = returns - values
            log_probs = torch.stack(log_probs)
            actions = torch.stack(actions)
            states = torch.stack(states)

        with timeit(f"ppo update {ppo_n_batch} batches"):
            do_ppo_update(model, optimizer, ppo_epochs, ppo_n_batch, ppo_mini_batch_size,
                          states, actions, log_probs, returns, advantages)

        if (batch_id + 1) % 10 == 0:
            with torch.no_grad():
                total_reward = test_env(model)
                print(f"batch {batch_id + 1}, total reward {total_reward} / 10 episodes")


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
