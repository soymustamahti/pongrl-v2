import time
import torch
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
import ale_py
from agent import Agent
from agent_optimized import OptimizedAgent
import numpy as np

def quick_training_benchmark(agent, batch_size, steps=200):
    """Benchmark rápido y preciso"""
    print(f"Testing with batch_size={batch_size}, steps={steps}")
    
    # Llenar memoria rápidamente
    obs, info = agent.env.reset()
    obs = agent.process_observation(obs)
    
    # Llenar memoria inicial
    for i in range(batch_size * 3):
        action = agent.env.action_space.sample()
        next_obs, reward, done, truncated, info = agent.env.step(action)
        next_obs = agent.process_observation(next_obs)
        
        obs_cpu = obs.cpu() if obs.is_cuda else obs
        next_obs_cpu = next_obs.cpu() if next_obs.is_cuda else next_obs
        
        agent.memory.store_transition(obs_cpu, action, reward, next_obs_cpu, done)
        obs = next_obs
        if done:
            obs, info = agent.env.reset()
            obs = agent.process_observation(obs)
    
    # Benchmark de entrenamiento
    times = []
    losses = []
    
    # Warmup
    for _ in range(10):
        if hasattr(agent, 'train_step'):
            agent.train_step(batch_size)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Medición real
    for step in range(steps):
        start_time = time.time()
        
        if hasattr(agent, 'train_step'):
            loss = agent.train_step(batch_size)
            if loss is not None:
                losses.append(loss)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        step_time = time.time() - start_time
        times.append(step_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    steps_per_sec = 1.0 / avg_time
    avg_loss = np.mean(losses) if losses else 0
    
    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'steps_per_sec': steps_per_sec,
        'avg_loss': avg_loss
    }

def find_optimal_config():
    """Encontrar la configuración óptima para RTX 4090"""
    print("=== Finding Optimal Configuration for RTX 4090 ===")
    
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env = ResizeObservation(env, (64, 64))
    env = GrayscaleObservation(env, keep_dim=True)
    
    configs = [
        # (hidden_size, batch_size, mixed_precision, name)
        (128, 64, False, "Small/CPU-like"),
        (128, 128, False, "Small/Medium Batch"),
        (128, 256, False, "Small/Large Batch"),
        (256, 128, False, "Medium/Medium"),
        (256, 256, False, "Medium/Large"),
        (512, 128, False, "Large/Medium"),
        (512, 256, False, "Large/Large"),
        (256, 256, True, "Medium/Large + MP"),
        (512, 256, True, "Large/Large + MP"),
    ]
    
    results = []
    
    print("Config                    | Time/Step(ms) | Steps/Sec | Memory(GB)")
    print("-" * 70)
    
    for hidden_size, batch_size, use_mp, name in configs:
        try:
            # Crear agente con configuración específica
            agent = OptimizedAgent(env, 
                                 hidden_layer=hidden_size,
                                 learning_rate=0.0001,
                                 step_repeat=4, 
                                 gamma=0.99,
                                 use_mixed_precision=use_mp)
            
            # Benchmark rápido
            result = quick_training_benchmark(agent, batch_size, steps=100)
            
            # Memoria GPU
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.max_memory_allocated() / 1e9
                torch.cuda.reset_peak_memory_stats()
            else:
                gpu_memory = 0
            
            time_ms = result['avg_time'] * 1000
            steps_sec = result['steps_per_sec']
            
            print(f"{name:<25} | {time_ms:>10.2f} | {steps_sec:>8.1f} | {gpu_memory:>8.2f}")
            
            results.append({
                'config': name,
                'hidden_size': hidden_size,
                'batch_size': batch_size,
                'mixed_precision': use_mp,
                'time_ms': time_ms,
                'steps_per_sec': steps_sec,
                'memory_gb': gpu_memory
            })
            
            # Limpiar
            del agent
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{name:<25} | {'OOM':>10} | {'OOM':>8} | {'OOM':>8}")
                torch.cuda.empty_cache()
            else:
                raise e
    
    # Encontrar la mejor configuración
    if results:
        best_config = max(results, key=lambda x: x['steps_per_sec'])
        print(f"\n🏆 MEJOR CONFIGURACIÓN:")
        print(f"   {best_config['config']}")
        print(f"   {best_config['steps_per_sec']:.1f} steps/sec")
        print(f"   {best_config['time_ms']:.2f}ms/step")
        print(f"   {best_config['memory_gb']:.2f}GB memoria")
    
    return results

def compare_with_original():
    """Comparar con el agente original usando configuración óptima"""
    print("\n=== Comparison with Original Agent ===")
    
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env = ResizeObservation(env, (64, 64))
    env = GrayscaleObservation(env, keep_dim=True)
    
    # Agente original
    print("Testing Original Agent...")
    original_agent = Agent(env, hidden_layer=128, learning_rate=0.0001, 
                          step_repeat=4, gamma=0.99)
    original_result = quick_training_benchmark(original_agent, batch_size=64, steps=200)
    
    # Agente optimizado con configuración inteligente
    print("Testing Optimized Agent (smart config)...")
    optimized_agent = OptimizedAgent(env, hidden_layer=256, learning_rate=0.0001,
                                   step_repeat=4, gamma=0.99, use_mixed_precision=True)
    optimized_result = quick_training_benchmark(optimized_agent, batch_size=256, steps=200)
    
    # Comparación
    print(f"\nResultados:")
    print(f"Original:  {original_result['steps_per_sec']:.1f} steps/sec ({original_result['avg_time']*1000:.2f}ms/step)")
    print(f"Optimized: {optimized_result['steps_per_sec']:.1f} steps/sec ({optimized_result['avg_time']*1000:.2f}ms/step)")
    
    speedup = optimized_result['steps_per_sec'] / original_result['steps_per_sec']
    print(f"Speedup: {speedup:.2f}x")
    
    if speedup > 1.2:
        print(f"✅ Optimización exitosa: {speedup:.2f}x más rápido")
    else:
        print(f"⚠️ Optimización marginal: {speedup:.2f}x")

def main():
    print("=== RTX 4090 Optimization Analysis ===")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")
    
    # Encontrar configuración óptima
    find_optimal_config()
    
    # Comparar con original
    compare_with_original()

if __name__ == "__main__":
    main()
