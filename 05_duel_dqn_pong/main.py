import threading

import gymnasium
import torch
from typing import Sequence, Optional, Any
import random
import dataclasses
import math
import numpy
from collections import deque
import tqdm
from threading import Thread
import queue
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
    not_done: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, batch_size: int, rng: Optional[random.Random] = None):
        # 超过maxlen 会被删除
        self.buffer = deque(maxlen=capacity)
        if rng is None:
            rng = random.Random()
        self.rng = rng
        self.queue_size = 5
        self.num_threads = 20
        self.samples = queue.Queue(maxsize=self.queue_size)
        self.batch_size = batch_size
        self.thread_mtx = threading.Lock()
        self.sample_thread = []
        self.thread_exited = threading.Event()

    def append(self, record: Record):
        with self.thread_mtx:
            self.buffer.append((torch.tensor(record.state, dtype=torch.uint8),
                                torch.tensor(record.next_state, dtype=torch.uint8),
                                torch.tensor(record.done, dtype=torch.bool),
                                torch.tensor(record.reward, dtype=torch.float32),
                                torch.tensor(record.action, dtype=torch.uint8)))

    def sample(self) -> RecordBatch:
        if len(self.sample_thread) == 0:
            self.start_sample_thread()

        batch = self.samples.get()
        return batch

    def start_sample_thread(self):
        for _ in range(self.num_threads):
            self.sample_thread.append(Thread(target=self._thread_main))
        for th in self.sample_thread:
            th.start()

    def _thread_main(self):
        copy_stream = torch.cuda.Stream()

        while not self.thread_exited.isSet():
            with self.thread_mtx:
                items = zip(*self.rng.sample(self.buffer, k=self.batch_size))

            state, next_state, done, reward, action = (torch.stack(item) for item in items)
            state = state.to(torch.float32)
            next_state = next_state.to(torch.float32)
            done = ~done
            done = done.to(torch.long)
            action = action.to(torch.long)

            with torch.cuda.stream(copy_stream):
                dev = "cuda:0"
                batch = RecordBatch(
                    state=state.to(dev),
                    next_state=next_state.to(dev),
                    action=action.to(dev),
                    reward=reward.to(dev),
                    not_done=done.to(dev),
                )
            event: torch.cuda.Event = copy_stream.record_event()
            event.synchronize()
            while True:
                try:
                    self.samples.put(batch, timeout=1)
                    break
                except queue.Full:
                    if self.thread_exited.isSet():
                        break

    def stop_samples(self):
        self.thread_exited.set()
        for th in self.sample_thread:
            th.join()

        self.sample_thread.clear()
        self.thread_exited = threading.Event()

    def __len__(self) -> int:
        return len(self.buffer)


class DuelingConvDQN(torch.nn.Module):
    def __init__(self, input_shape: Sequence[int], num_actions: int):
        super().__init__()
        self.input_shape = input_shape
        self.feature = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )
        self.feature_size = self._feature_size()

        self.advantage = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.feature_size, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=num_actions)
        )

        self.value = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.feature_size, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=1)
        )
        self.rng = random.Random()
        self.num_actions = num_actions

    def forward(self, state: torch.Tensor):
        x = self.feature(state)
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

    def _feature_size(self) -> int:
        with torch.no_grad():
            return self.feature(torch.zeros(1, *self.input_shape)).shape[1]


def exploration(frame: int) -> float:
    # 越早的frame增加探索的概率
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 30000
    return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame / epsilon_decay)


def do_learn(replay: ReplayBuffer, cur_model: DuelingConvDQN, trg_model: DuelingConvDQN,
             optimizer: torch.optim.Optimizer) -> torch.Tensor:
    use_one_model = False  # true only for testing. Test for disabling two model trick
    gamma = 0.99

    batch = replay.sample()

    # Q values 表示当前状态不同Action的期望收益
    q_value = cur_model(batch.state)
    # 找到 action 的 q_value
    q_value = q_value.gather(1, batch.action.unsqueeze(1)).squeeze(1)

    if use_one_model:
        next_q_value = cur_model(batch.next_state)
        next_q_value = next_q_value.max(1)[0]
        expected_q_value = batch.reward + gamma * next_q_value * batch.not_done
    else:
        with torch.no_grad():
            # next q values 表示下一个状态不同Action之间的期望收益
            next_q_value = trg_model(batch.next_state)
            # find the max policy's Q value
            next_q_value = next_q_value.max(1)[0]

            # expected q value = current step reward + next_step q value excluding done.
            expected_q_value = batch.reward + gamma * next_q_value * batch.not_done
            expected_q_value = expected_q_value.detach()

    loss = (q_value - expected_q_value).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


class ImageToPyTorch(gymnasium.ObservationWrapper):
    """
    Image shape to num_channels x weight x height
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(low=0.0, high=1.0,
                                                      shape=(1, old_shape[0], old_shape[1]))

    def observation(self, observation):
        return numpy.expand_dims(observation, 0)


class ClipReward(gymnasium.RewardWrapper):
    def __init__(self, env, min_reward, max_reward):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)

    def reward(self, reward):
        return numpy.clip(reward, self.min_reward, self.max_reward)


class EarlyStop(gymnasium.Wrapper):
    def __init__(self, env, min_reward, max_reward):
        super().__init__(env)
        self._total_reward = 0
        self._min = min_reward
        self._max = max_reward
        self._total_positive = 0
        self._total_negative = 0

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        self._total_reward += reward
        if reward < 0:
            self._total_negative += 1
        if reward > 0:
            self._total_positive += 1
        if self._total_reward < self._min or self._total_reward > self._max:
            terminated = True

        return state, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._total_reward = 0
        print(self._total_negative, " : ", self._total_positive)
        self._total_positive = 0
        self._total_negative = 0
        return obs, info


def main():
    env = gymnasium.make("PongNoFrameskip-v4", render_mode="rgb_array", full_action_space=False)
    env = gymnasium.wrappers.AtariPreprocessing(env=env, terminal_on_life_loss=True, grayscale_obs=True,
                                                noop_max=0)
    env = ClipReward(env, -1, 1)
    env = EarlyStop(env, -4, 10)
    # env = gymnasium.wrappers.TransformReward(env, lambda r: r * 10000 + 1)
    env = ImageToPyTorch(env)

    current_model = DuelingConvDQN(env.observation_space.shape, env.action_space.n)
    current_model.to("cuda:0")
    target_model = DuelingConvDQN(env.observation_space.shape, env.action_space.n)
    target_model.to("cuda:0")

    def sync_model():
        target_model.load_state_dict(current_model.state_dict())

    sync_model()

    optimizer = torch.optim.Adam(current_model.parameters(), lr=1e-4)
    replay = ReplayBuffer(100000, batch_size=256)
    replay_warmup = 10000

    state, _ = env.reset()
    episode_reward = 0
    losses = []
    total_frames_to_train = 2000000
    episode_id = 0
    for frame in tqdm.tqdm(range(total_frames_to_train), desc="training", unit="frames"):
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
            if frame % 100 == 0:
                sync_model()

    replay.stop_samples()

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
                       fps=24)
            episode += 1
            if episode == 10:
                break


main()
