import numpy as np, gymnasium as gym

ACTIONS, FEATS = 3, 6

def featurize(obs):
    x, v = obs
    return np.array([x, v, 1.0, x*x, v*v, x*v], dtype=np.float32)

def act(obs, theta):
    W = theta.reshape(ACTIONS, FEATS)
    f = featurize(obs)
    return int((W @ f).argmax())

def main():
    theta = np.load("models/best_theta.npz")["theta"]
    env = gym.make("MountainCar-v0", render_mode="human", max_episode_steps=500)
    obs, info = env.reset(seed=42)

    terminated = truncated = False
    total = 0.0
    while True:
        a = act(obs, theta)
        obs, r, terminated, truncated, info = env.step(a)
        total += r
        if terminated or truncated:
            print(f"Fin episodio | reward total: {total:.1f}")
            total = 0.0
            obs, info = env.reset()

if __name__ == "__main__":
    main()
