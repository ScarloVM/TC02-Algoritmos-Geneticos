import gymnasium as gym
import numpy as np
import pygame

ACTION_LEFT, ACTION_NONE, ACTION_RIGHT = 0, 1, 2

def main():
    env = gym.make("MountainCar-v0", render_mode="rgb_array", max_episode_steps=1000)
    obs, info = env.reset(seed=0)

    pygame.init()
    frame = env.render()                      # H x W x 3
    H, W = frame.shape[:2]
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("MountainCar - Pygame UI")
    clock = pygame.time.Clock()

    running = True
    action = ACTION_NONE
    terminated = truncated = False

    while running:
        # --- input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = ACTION_LEFT
        elif keys[pygame.K_RIGHT]:
            action = ACTION_RIGHT
        else:
            action = ACTION_NONE

        # --- paso del entorno ---
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

        # --- render ---
        frame = env.render()
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(60)  # ~60 FPS

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()
