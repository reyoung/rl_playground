import dataclasses
import time
from typing import Tuple, SupportsFloat, Any, List, Dict, Callable, Optional
import threading
import gymnasium
import numpy
import torch.nn
from gymnasium import Env
from gymnasium.core import WrapperActType, ActType, ObsType
from gymnasium.utils.save_video import save_video
import tqdm
import queue


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


class ActionSpaceTransform(gymnasium.ActionWrapper):
    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__(env)
        self.action_space = gymnasium.spaces.Box(shape=(2,), dtype=numpy.float32, low=-1, high=1)

    def action(self, action: WrapperActType) -> ActType:
        return numpy.argmax(action)


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


@dataclasses.dataclass
class UpdateRequest:
    gradients: Dict[str, torch.Tensor]
    on_params_updated: Callable[[Dict[str, torch.Tensor]], None]


class OptimizeThread:
    def __init__(self, model: ActorCritic, start=True):
        self.model = model
        self.model.to("cpu")
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.thread: Optional[threading.Thread] = None
        self.exited: Optional[threading.Event] = None
        self.job_queue = queue.Queue(maxsize=2)
        if start:
            self.start()

    def start(self):
        if self.thread is not None:
            raise ValueError("already started")
        self.thread = threading.Thread(target=self.thread_main)
        self.exited = threading.Event()
        self.thread.start()

    def join(self):
        if self.thread is None:
            raise ValueError("not started")
        self.exited.set()
        self.thread.join()
        self.exited = None
        self.thread = None

    def post(self, req: UpdateRequest):
        self.job_queue.put(req)

    def thread_main(self):
        while not self.exited.isSet():
            try:
                req: UpdateRequest = self.job_queue.get(timeout=1)
            except queue.Empty:
                continue

            self.optimizer.zero_grad()

            for param_name, param in self.model.named_parameters():
                param.grad = req.gradients[param_name]

            self.optimizer.step()

            param_dict = {param_name: param.data for param_name, param in self.model.named_parameters()}
            req.on_params_updated(param_dict)


def a2c_thread_impl(model, optimize_thread: OptimizeThread, n_batches: int, n_local_steps: int,
                    log_enabled: Callable[[int], bool], stream: torch.cuda.Stream, enable_tqdm: bool):
    env = gymnasium.make('CartPole-v1')
    env = gymnasium.wrappers.AutoResetWrapper(env)

    model.to("cuda:0")

    batch_range = range(n_batches)
    if enable_tqdm:
        batch_range = tqdm.tqdm(batch_range, desc="batch")

    for batch_id in batch_range:
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0

        state, _ = env.reset()

        for _ in range(n_local_steps):
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

        # zero gradient
        for p in model.parameters():
            p.grad = None

        loss.backward()
        grad_dict = {param_name: param.grad.cpu() for param_name, param in model.named_parameters()}

        value_updated = threading.Event()

        def update_value(val_dict):
            for param_name, param in model.named_parameters():
                param.data.copy_(val_dict[param_name], non_blocking=True)

            stream.synchronize()
            value_updated.set()

        optimize_thread.post(UpdateRequest(gradients=grad_dict, on_params_updated=update_value))
        value_updated.wait()

        if log_enabled(batch_id + 1):
            with torch.no_grad():
                total_reward = test_env(env, model)

            print(f"batch {batch_id + 1}, total reward {total_reward} / 10 episodes")


def a2c_thread_main(model: ActorCritic, optimize_thread: OptimizeThread, n_batches: int, n_local_steps: int,
                    log_enabled: Callable[[int], bool], enable_tqdm: bool):
    stream = torch.cuda.Stream(device="cuda:0")
    with torch.cuda.stream(stream):
        a2c_thread_impl(model, optimize_thread, n_batches, n_local_steps, log_enabled, stream, enable_tqdm)


def main():
    env = gymnasium.make('CartPole-v1', render_mode="rgb_array")
    env = gymnasium.wrappers.AutoResetWrapper(env)

    model = ActorCritic(num_inputs=env.observation_space.shape[0], num_actions=env.action_space.n)
    optimize_thread = OptimizeThread(model=model, start=False)
    env.close()

    num_a2c_thread = 4
    threads = []
    for thread_id in range(num_a2c_thread):
        tid = thread_id
        th_model = ActorCritic(num_inputs=env.observation_space.shape[0], num_actions=env.action_space.n)
        th_model.load_state_dict(model.state_dict())
        thread = threading.Thread(target=a2c_thread_main, args=(th_model, optimize_thread,
                                                                5000, 1000,
                                                                lambda bid: bid % (100 * num_a2c_thread) == bid * 100,
                                                                tid == 0
                                                                ))
        thread.start()
        threads.append(thread)

    optimize_thread.start()

    for th in threads:
        th.join()


if __name__ == '__main__':
    main()
