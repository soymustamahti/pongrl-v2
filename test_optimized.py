from agent_optimized import OptimizedAgent
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
import ale_py
import torch

# Configuración para test
hidden_layer = 1024 if torch.cuda.is_available() else 128
learning_rate = 0.0001
step_repeat = 4
gamma = 0.99

print(f"Configurando test con:")
print(f"- CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"- GPU: {torch.cuda.get_device_name(0)}")
print(f"- Hidden layer size: {hidden_layer}")

env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
env = ResizeObservation(env, (64, 64))
env = GrayscaleObservation(env, keep_dim=True)

agent = OptimizedAgent(env, hidden_layer=hidden_layer,
              learning_rate=learning_rate, step_repeat=step_repeat,
              gamma=gamma)

agent.test()
