import time
import torch
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
import ale_py
from agent import Agent
from agent_optimized import OptimizedAgent

def benchmark_agent(agent_class, name, episodes=5):
    """Benchmark de un agente por algunos episodios"""
    print(f"\n=== Benchmarking {name} ===")
    
    # Configuración
    if "Optimized" in name and torch.cuda.is_available():
        hidden_layer = 1024
        batch_size = 512
    else:
        hidden_layer = 128
        batch_size = 64
    
    learning_rate = 0.0001
    step_repeat = 4
    gamma = 0.99
    
    # Crear ambiente
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env = ResizeObservation(env, (64, 64))
    env = GrayscaleObservation(env, keep_dim=True)
    
    # Crear agente
    agent = agent_class(env, hidden_layer=hidden_layer,
                       learning_rate=learning_rate, step_repeat=step_repeat,
                       gamma=gamma)
    
    total_time = 0
    total_steps = 0
    episode_times = []
    
    print(f"- Device: {agent.device}")
    print(f"- Hidden layer: {hidden_layer}")
    print(f"- Batch size: {batch_size}")
    
    for episode in range(episodes):
        episode_start = time.time()
        
        done = False
        obs, info = env.reset()
        obs = agent.process_observation(obs)
        episode_steps = 0
        
        while not done and episode_steps < 1000:  # Límite para benchmark
            if hasattr(agent, 'get_action'):
                action = agent.get_action(obs, epsilon=0.1)
            else:
                # Agente original
                import random
                if random.random() < 0.1:
                    action = env.action_space.sample()
                else:
                    q_values = agent.model.forward(obs.unsqueeze(0).to(agent.device))[0]
                    action = torch.argmax(q_values, dim=-1).item()
            
            reward = 0
            for i in range(step_repeat):
                next_obs, reward_temp, done, truncated, info = env.step(action=action)
                reward += reward_temp
                if done:
                    break
            
            obs = agent.process_observation(next_obs)
            episode_steps += 1
            
            # Simular entrenamiento si hay suficientes muestras
            if hasattr(agent, 'memory') and agent.memory.can_sample(batch_size):
                if hasattr(agent, 'train_step'):
                    agent.train_step(batch_size)
                else:
                    # Simular entrenamiento para agente original
                    observations, actions, rewards, next_observations, dones = agent.memory.sample_buffer(batch_size)
                    # Simplificado para benchmark
                    pass
        
        episode_time = time.time() - episode_start
        episode_times.append(episode_time)
        total_time += episode_time
        total_steps += episode_steps
        
        print(f"  Episode {episode}: {episode_time:.2f}s, {episode_steps} steps")
    
    avg_time = total_time / episodes
    avg_steps = total_steps / episodes
    steps_per_second = total_steps / total_time
    
    print(f"\n{name} Results:")
    print(f"- Average time per episode: {avg_time:.2f}s")
    print(f"- Average steps per episode: {avg_steps:.1f}")
    print(f"- Steps per second: {steps_per_second:.1f}")
    print(f"- Total time: {total_time:.2f}s")
    
    return {
        'avg_time': avg_time,
        'avg_steps': avg_steps,
        'steps_per_second': steps_per_second,
        'total_time': total_time
    }

def main():
    print("=== GPU vs CPU Benchmark ===")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Benchmark agente original
    original_results = benchmark_agent(Agent, "Original Agent", episodes=3)
    
    # Benchmark agente optimizado
    optimized_results = benchmark_agent(OptimizedAgent, "Optimized Agent", episodes=3)
    
    # Comparación
    print("\n=== COMPARISON ===")
    speedup = original_results['avg_time'] / optimized_results['avg_time']
    throughput_improvement = optimized_results['steps_per_second'] / original_results['steps_per_second']
    
    print(f"Speedup (tiempo por episodio): {speedup:.2f}x")
    print(f"Throughput improvement: {throughput_improvement:.2f}x")
    
    if speedup > 1:
        print(f"✅ Optimized agent is {speedup:.2f}x FASTER")
    else:
        print(f"❌ Optimized agent is {1/speedup:.2f}x SLOWER")

if __name__ == "__main__":
    main()
