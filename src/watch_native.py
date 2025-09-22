import gymnasium as gym
import numpy as np
import pygame
import sys
import os

# Constantes
ACTIONS = 3
FEATS = 6

def featurize(obs):
    x, v = obs
    return np.array([x, v, 1.0, x*x, v*v, x*v], dtype=np.float32)

def act(obs, theta):
    W = theta.reshape(ACTIONS, FEATS)
    f = featurize(obs)
    return int((W @ f).argmax())

class TripleGameRenderer:
    def __init__(self):
        pygame.init()
        self.width = 1200 
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("MountainCar")
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (200, 0, 0)
        self.BLUE = (0, 0, 200)
        self.GREEN = (0, 180, 0)
        self.GRAY = (128, 128, 128)
        
        # Configuraciones de cada juego
        self.game_width = self.width // 3
        self.game_height = self.height - 100
        
        self.car_colors = [self.RED, self.BLUE, self.GREEN]
        self.config_names = ["Configuración 1", "Configuración 2", "Configuración 3"]
        
    #Dibuja la montaña y el carro 
    def draw_mountain(self, surface, offset_x, x_pos):
        section_width = self.game_width - 40
        section_height = self.game_height - 80
        
        #Dibuja la montaña
        points = []
        for i in range(section_width + 1):
            x = -1.2 + (i / section_width) * 1.8
            #Fórmula de la montaña
            y = np.sin(3 * x) * 0.45 + 0.55
            screen_y = 80 + int((1 - y) * section_height)
            points.append((offset_x + 20 + i, screen_y))
        
        #curva de la montaña
        if len(points) > 1:
            pygame.draw.lines(surface, self.GRAY, False, points, 4)
        
        #Calcular posicion del carro
        car_screen_x = offset_x + 20 + int((x_pos + 1.2) / 1.8 * section_width)
        car_y = np.sin(3 * x_pos) * 0.45 + 0.55
        car_screen_y = 80 + int((1 - car_y) * section_height)
        
        #Dibujar la meta
        goal_x = offset_x + 20 + int((0.5 + 1.2) / 1.8 * section_width)
        goal_y = 80 + int((1 - (np.sin(3 * 0.5) * 0.45 + 0.55)) * section_height)
        self.draw_flag(surface, goal_x, goal_y)
        
        return car_screen_x, car_screen_y
    

    # dibuja la bandera
    def draw_flag(self, surface, x, y):
        # Asta
        pole_height = 35
        pygame.draw.line(surface, self.BLACK, (x, y), (x, y - pole_height), 3)
        flag_x = x + 2
        flag_y = y - pole_height
        square_size = 3
        
        #para los cuadros
        for row in range(4):
            for col in range(6):
                color = self.BLACK if (row + col) % 2 == 0 else self.WHITE
                rect_x = flag_x + col * square_size
                rect_y = flag_y + row * square_size
                pygame.draw.rect(surface, color, (rect_x, rect_y, square_size, square_size))
        pygame.draw.rect(surface, self.BLACK, (flag_x, flag_y, 18, 12), 2)
        pygame.draw.rect(surface, (139, 69, 19), (x - 2, y, 4, 3))
    
    #Dibuja el carro
    def draw_car(self, surface, x, y, color):
        # Cuerpo
        car_width = 28
        car_height = 16
        pygame.draw.rect(surface, color, (int(x - car_width//2), int(y - car_height//2), car_width, car_height))
        pygame.draw.rect(surface, self.BLACK, (int(x - car_width//2), int(y - car_height//2), car_width, car_height), 2)
        
        # Ruedas
        wheel_radius = 7
        pygame.draw.circle(surface, self.BLACK, (int(x - 7), int(y + 5)), wheel_radius)
        pygame.draw.circle(surface, self.BLACK, (int(x + 7), int(y + 5)), wheel_radius)
    

    #Dibuja un frame con los 3 juegos
    def draw_frame(self, states, step_counts, episode_counts, dones, total_rewards, restart_time_left=0):
        self.screen.fill(self.WHITE)
        
        # Dibujar separadores verticales
        for i in range(1, 3):
            x = i * self.game_width
            pygame.draw.line(self.screen, self.BLACK, (x, 0), (x, self.height), 2)
        
        #Dibujar cada juego
        for i in range(3):
            offset_x = i * self.game_width
            x_pos, v_pos = states[i]
            
            #Dibujar montaña y obtener posición del carro
            car_x, car_y = self.draw_mountain(self.screen, offset_x, x_pos)
            
            #Dibujar el carro
            self.draw_car(self.screen, car_x, car_y, self.car_colors[i])
            
            #Dibujar labels
            config_text = self.font.render(self.config_names[i], True, self.car_colors[i])
            self.screen.blit(config_text, (offset_x + 20, 10))
            
            #Estado del episodio
            if dones[i]:
                if states[i][0] >= 0.5:
                    status_text = self.small_font.render("Éxito", True, self.GREEN)
                else:
                    status_text = self.small_font.render("Tiempo agotado", True, self.RED)
                self.screen.blit(status_text, (offset_x + 20, 45))
            
            #Información adicional
            pos_text = self.small_font.render(f"Pos: {x_pos:.3f}", True, self.BLACK)
            vel_text = self.small_font.render(f"Vel: {v_pos:.3f}", True, self.BLACK)
            step_text = self.small_font.render(f"Steps: {step_counts[i]}", True, self.BLACK)
            episode_text = self.small_font.render(f"Episode: {episode_counts[i]}", True, self.BLACK)
            reward_text = self.small_font.render(f"Reward: {total_rewards[i]:.1f}", True, self.BLACK)
            
            self.screen.blit(pos_text, (offset_x + 20, self.height - 110))
            self.screen.blit(vel_text, (offset_x + 20, self.height - 90))
            self.screen.blit(step_text, (offset_x + 20, self.height - 70))
            self.screen.blit(episode_text, (offset_x + 20, self.height - 50))
            self.screen.blit(reward_text, (offset_x + 20, self.height - 30))

        
        pygame.display.flip()

#Carga el theta de una configuración específica
def load_theta(config_num):

    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, "models", f"best_theta_config{config_num}.npz")

    if not os.path.exists(model_path):
        parent_dir = os.path.dirname(current_dir)
        model_path = os.path.join(parent_dir, "models", f"best_theta_config{config_num}.npz")

    if not os.path.exists(model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        model_path = os.path.join(project_dir, "models", f"best_theta_config{config_num}.npz")
    
    if not os.path.exists(model_path):
        print(f"Archivo {model_path} no encontrado")
        models_dir = os.path.dirname(model_path)
        if os.path.exists(models_dir):
            print(f"Archivos disponibles en {models_dir}: {os.listdir(models_dir)}")
        return None
    
    try:
        data = np.load(model_path)
        print(f"Configuración {config_num} cargada exitosamente")
        return data['theta']
    except Exception as e:
        print(f"Error cargando {model_path}: {e}")
        return None

def main():
    print("=== MountainCar ===")
    
    # Cargar los 3 modelos
    thetas = []
    for i in range(1, 4):
        theta = load_theta(i)
        if theta is None:
            print(f"Error cargando configuración {i}")
            print("Asegúrate de que los archivos best_theta_config1.npz, best_theta_config2.npz y best_theta_config3.npz existan en la carpeta models/")
            return
        thetas.append(theta)
    
    print("\nModelos cargados exitosamente")
    print("\nIniciando simulación...\n")
    
    # Crear 3 entornos
    envs = [gym.make("MountainCar-v0", render_mode=None) for _ in range(3)]

    renderer = TripleGameRenderer()
    
    # Variables de estado
    observations = [env.reset()[0] for env in envs]
    step_counts = [0, 0, 0]
    episode_counts = [1, 1, 1]
    dones = [False, False, False]
    total_rewards = [0.0, 0.0, 0.0]
    
    restart_timer = 0.0
    restart_delay = 3.0 
    
    clock = pygame.time.Clock()
    running = True
    
    try:
        while running:
            dt = clock.tick(60) / 1000.0  # Delta time en segundos
            
            # Manejar eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            #timer de reinicio
            if all(dones):
                restart_timer += dt
                if restart_timer >= restart_delay:
                    # Reiniciar automáticamente
                    observations = [env.reset()[0] for env in envs]
                    step_counts = [0, 0, 0]
                    episode_counts = [c + 1 for c in episode_counts]
                    dones = [False, False, False]
                    total_rewards = [0.0, 0.0, 0.0]
                    restart_timer = 0.0
            else:
                restart_timer = 0.0
            
            # Ejecutar un paso en cada juego (solo si no está en modo reinicio)
            if restart_timer == 0.0:
                for i in range(3):
                    if not dones[i]:
                        action = act(observations[i], thetas[i])
                        
                        # Ejecutar acción
                        obs, reward, terminated, truncated, info = envs[i].step(action)
                        observations[i] = obs
                        step_counts[i] += 1
                        total_rewards[i] += reward
                        
                        # Verificar si terminó
                        if terminated or truncated:
                            dones[i] = True
                            if terminated and obs[0] >= 0.5:
                                print(f"Config {i+1} - Fin episodio | reward total: {total_rewards[i]:.1f} | Meta alcanzada en {step_counts[i]} pasos")
                            else:
                                print(f"Config {i+1} - Fin episodio | reward total: {total_rewards[i]:.1f} | Tiempo agotado después de {step_counts[i]} pasos")
            
            # Renderizar
            restart_time_left = restart_delay - restart_timer if all(dones) else 0
            renderer.draw_frame(observations, step_counts, episode_counts, dones, total_rewards, restart_time_left)
    
    finally:
        # Limpiar
        for env in envs:
            env.close()
        pygame.quit()

if __name__ == "__main__":
    main()