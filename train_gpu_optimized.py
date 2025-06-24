from agent_optimized import OptimizedAgent
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
import ale_py
import torch

# Configuración optimizada para GPU RTX 4090 (24GB VRAM)
episodes = 10000
max_episode_steps = 10000

# Verificar si CUDA está disponible y optimizar en consecuencia
if torch.cuda.is_available():
    print(f"CUDA disponible. GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Configuración agresiva para RTX 4090
    hidden_layer = 1024  # Red mucho más grande
    batch_size = 1024    # Batch grande para aprovechar paralelización
    learning_rate = 0.0003  # LR ligeramente más alto para batch grande
    
    # Usar mixed precision para mejor rendimiento
    torch.backends.cudnn.benchmark = True
    
else:
    print("CUDA no disponible, usando CPU")
    # Configuración para CPU
    hidden_layer = 128
    batch_size = 64
    learning_rate = 0.0001

step_repeat = 4
gamma = 0.99
epsilon = 1
min_epsilon = 0.1
epsilon_decay = 0.995

env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
env = ResizeObservation(env, (64, 64))
env = GrayscaleObservation(env, keep_dim=True)

agent = OptimizedAgent(env, hidden_layer=hidden_layer,
              learning_rate=learning_rate, step_repeat=step_repeat,
              gamma=gamma)

summary_writer_suffix = f'dqn_gpu_opt_lr={learning_rate}_hl={hidden_layer}_bs={batch_size}'

print(f"Iniciando entrenamiento con:")
print(f"- Hidden layer size: {hidden_layer}")
print(f"- Batch size: {batch_size}")
print(f"- Device: {agent.device}")

agent.train(episodes=episodes,
            max_episode_steps=max_episode_steps,
            summary_writer_suffix=summary_writer_suffix,
            batch_size=batch_size,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon)
