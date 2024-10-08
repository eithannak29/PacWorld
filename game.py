import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make('ALE/MsPacman-v5', render_mode='human')

obs, info = env.reset()

done = False

while not done:
    env.render()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
env.close()
