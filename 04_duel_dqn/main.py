import dataclasses
import gymnasium
import torch
from typing import Optional, List
import random
import math
from collections import deque
import numpy
from gymnasium.utils.save_video import save_video


@dataclasses.dataclass()
class Record:
    state: gymnasium.core.ObsType
    next_state: gymnasium.core.ObsType
    action: gymnasium.core.ActType
    reward: gymnasium.core.SupportsFloat
    done: bool


@dataclasses.dataclass()
class RecordBatch:
    state: torch.Tensor
    next_state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, rng: Optional[random.Random] = None):
        # 超过maxlen 会被删除
        self.buffer = deque(maxlen=capacity)
        if rng is None:
            rng = random.Random()
        self.rng = rng

    def append(self, record: Record):
        self.buffer.append(record)

    def sample(self, batch_size: int) -> RecordBatch:
        batch = self.rng.sample(self.buffer, k=batch_size)
        states = []
        next_states = []
        dones = []
        actions = []
        rewards = []

        for record in batch:
            states.append(record.state)
            next_states.append(record.next_state)
            dones.append(record.done)
            actions.append(record.action)
            rewards.append(record.reward)

        return RecordBatch(
            state=torch.tensor(states, dtype=torch.float32, device="cuda:0"),
            next_state=torch.tensor(next_states, dtype=torch.float32, device="cuda:0"),
            action=torch.tensor(actions, dtype=torch.long, device="cuda:0"),
            reward=torch.tensor(rewards, dtype=torch.float32, device='cuda:0'),
            done=torch.tensor(dones, dtype=torch.long, device="cuda:0")
        )

    def __len__(self) -> int:
        return len(self.buffer)


def exploration(frame: int) -> float:
    # 越早的frame增加探索的概率
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 1000
    return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame / epsilon_decay)


class DuelingDQN(torch.nn.Module):
    def __init__(self, num_inputs: int, num_actions: int):
        super().__init__()
        self.base = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 128),
            torch.nn.ReLU()
        )
        self.advantage = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_actions)
        )
        self.value = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )
        self.rng = random.Random()
        self.num_actions = num_actions

    def forward(self, state: torch.Tensor):
        x = self.base(state)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

    def act(self, state, explore_rate: float = 0) -> int:
        if self.rng.random() > explore_rate:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32, device='cuda:0').unsqueeze(0)
                q_value = self.forward(state)
                result: torch.Tensor = q_value[0].argmax()
                return result.item()
        else:
            return random.randrange(self.num_actions)


def do_learn(replay: ReplayBuffer, cur_model: DuelingDQN, trg_model: DuelingDQN,
             optimizer: torch.optim.Optimizer) -> torch.Tensor:
    use_one_model = False  # true only for testing. Test for disabling two model trick
    gamma = 0.99
    batch_size = 128
    batch = replay.sample(batch_size=batch_size)

    # Q values 表示当前状态不同Action的期望收益
    q_value = cur_model(batch.state)
    # 找到 action 的 q_value
    q_value = q_value.gather(1, batch.action.unsqueeze(1)).squeeze(1)

    if use_one_model:
        next_q_value = cur_model(batch.next_state)
        next_q_value = next_q_value.max(1)[0]
        expected_q_value = batch.reward + gamma * next_q_value * (1 - batch.done)
    else:
        with torch.no_grad():
            # next q values 表示下一个状态不同Action之间的期望收益
            next_q_value = trg_model(batch.next_state)
            # find the max policy's Q value
            next_q_value = next_q_value.max(1)[0]

            # expected q value = current step reward + next_step q value excluding done.
            expected_q_value = batch.reward + gamma * next_q_value * (1 - batch.done)
            expected_q_value = expected_q_value.detach()

    loss = (q_value - expected_q_value).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def main():
    env = gymnasium.make('CartPole-v1', render_mode="rgb_array")
    current_model = DuelingDQN(env.observation_space.shape[0], env.action_space.n)
    current_model.to("cuda:0")
    target_model = DuelingDQN(env.observation_space.shape[0], env.action_space.n)
    target_model.to("cuda:0")

    def sync_model():
        target_model.load_state_dict(current_model.state_dict())

    sync_model()

    optimizer = torch.optim.Adam(current_model.parameters())
    replay = ReplayBuffer(1000)
    replay_warmup = 200

    state, _ = env.reset()
    episode_reward = 0
    losses = []
    total_frames_to_train = 50000
    episode_id = 0
    for frame in range(total_frames_to_train):
        epsilon = exploration(frame)
        action = current_model.act(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay.append(Record(next_state=next_state, state=state, reward=reward, done=done, action=action))
        state = next_state
        episode_reward += reward
        if done:
            state, _ = env.reset()
            if len(replay) > replay_warmup:
                print("episode_reward ", episode_reward, ", loss ", numpy.array(losses).mean())

            losses.clear()
            episode_reward = 0
            episode_id += 1

        if len(replay) > replay_warmup:
            losses.append(do_learn(replay, current_model, target_model, optimizer).item())
            if frame % 200 == 0:
                sync_model()

    # let's record some videos for trained model

    state, _ = env.reset()
    episode = 0
    frames = []
    while True:
        frames.append(env.render())
        action = current_model.act(state)
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


main()
