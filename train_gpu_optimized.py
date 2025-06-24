from agent_optimized import OptimizedAgent
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
import ale_py
import torch

# Configuración ÓPTIMA para RTX 4090 basada en benchmarks
episodes = 10000
max_episode_steps = 10000

# Verificar si CUDA está disponible y usar configuración óptima
if torch.cuda.is_available():
    print(f"CUDA disponible. GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # CONFIGURACIÓN ÓPTIMA basada en benchmarks reales
    hidden_layer = 512     # Sweet spot para RTX 4090
    batch_size = 128       # Eficiente sin overhead
    learning_rate = 0.0002 # Ligeramente más alto para red más grande
    
    # Usar mixed precision para mejor rendimiento
    torch.backends.cudnn.benchmark = True
    print("Usando configuración ÓPTIMA para RTX 4090")
    
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
