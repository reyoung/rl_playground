import gymnasium as gym
from gymnasium.utils.save_video import save_video

env = gym.make("CartPole-v1", render_mode="rgb_array")

env.reset()
iter_cnt = 0
frames = []
while True:
    frames.append(env.render())
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if terminated:
        frames.append(env.render())
        if iter_cnt % 10 == 0:
            save_video(frames,
                       "videos",
                       episode_trigger=lambda *args: True,
                       episode_index=iter_cnt,
                       fps=50)

        iter_cnt += 1
        if iter_cnt == 1000:
            break
        observation, info = env.reset()
        frames.clear()

env.close()
