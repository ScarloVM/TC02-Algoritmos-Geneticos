import gymnasium as gym

def main():
    env = gym.make("MountainCar-v0", render_mode="human", max_episode_steps=1000)
    obs, info = env.reset(seed=0)

    terminated = truncated = False
    while True:
        # Acción aleatoria (reemplaza por tu política/AG cuando lo tengas)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

if __name__ == "__main__":
    main()
