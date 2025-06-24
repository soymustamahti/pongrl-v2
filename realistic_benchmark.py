import time
import torch
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
import ale_py
from agent import Agent
from agent_optimized import OptimizedAgent
import numpy as np

def realistic_training_benchmark():
    """Benchmark que simula entrenamiento completo realista"""
    print("=== Realistic Training Pipeline Benchmark ===")
    
    # Configuraciones a comparar
    configs = [
        {
            'name': 'Original Agent',
            'agent_class': Agent,
            'hidden_layer': 128,
            'batch_size': 64,
            'use_mixed_precision': False
        },
        {
            'name': 'Optimized Agent (Best Config)',
            'agent_class': OptimizedAgent,
            'hidden_layer': 256,
            'batch_size': 256,
            'use_mixed_precision': True
        },
        {
            'name': 'Optimized Agent (Conservative)',
            'agent_class': OptimizedAgent,
            'hidden_layer': 512,
            'batch_size': 128,
            'use_mixed_precision': True
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"\n=== Testing {config['name']} ===")
        
        # Crear ambiente
        env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
        env = ResizeObservation(env, (64, 64))
        env = GrayscaleObservation(env, keep_dim=True)
        
        # Crear agente
        if config['agent_class'] == OptimizedAgent:
            agent = config['agent_class'](
                env, 
                hidden_layer=config['hidden_layer'],
                learning_rate=0.0001,
                step_repeat=4,
                gamma=0.99,
                use_mixed_precision=config['use_mixed_precision']
            )
        else:
            agent = config['agent_class'](
                env,
                hidden_layer=config['hidden_layer'],
                learning_rate=0.0001,
                step_repeat=4,
                gamma=0.99
            )
        
        print(f"Config: {config['hidden_layer']} hidden, {config['batch_size']} batch, MP: {config['use_mixed_precision']}")
        
        # Simular episodio completo con entrenamiento
        result = simulate_training_episode(agent, config['batch_size'], steps_per_episode=500)
        result['config'] = config['name']
        results.append(result)
        
        # Limpiar memoria
        del agent, env
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Comparar resultados
    print(f"\n{'='*80}")
    print("RESULTADOS COMPLETOS")
    print(f"{'='*80}")
    
    print(f"{'Config':<30} | {'Total Time':<10} | {'Steps/Sec':<10} | {'Train Steps':<12} | {'GPU Util':<8}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['config']:<30} | {result['total_time']:<10.2f} | {result['overall_steps_per_sec']:<10.1f} | {result['training_steps_per_sec']:<12.1f} | {result['gpu_memory']:<8.2f}")
    
    # Calcular speedups
    baseline = next(r for r in results if 'Original' in r['config'])
    
    print(f"\n{'Config':<30} | {'Overall Speedup':<15} | {'Training Speedup':<16}")
    print("-" * 65)
    
    for result in results:
        overall_speedup = result['overall_steps_per_sec'] / baseline['overall_steps_per_sec']
        training_speedup = result['training_steps_per_sec'] / baseline['training_steps_per_sec']
        
        print(f"{result['config']:<30} | {overall_speedup:<15.2f}x | {training_speedup:<16.2f}x")
    
    return results

def simulate_training_episode(agent, batch_size, steps_per_episode=500):
    """Simula un episodio completo de entrenamiento"""
    
    # Llenar memoria inicial
    obs, info = agent.env.reset()
    obs = agent.process_observation(obs)
    
    # Llenar buffer
    for i in range(batch_size * 5):
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
    
    print(f"Buffer llenado con {agent.memory.mem_ctr} experiencias")
    
    # Simular episodio con timing detallado
    environment_times = []
    training_times = []
    total_training_steps = 0
    
    total_start = time.time()
    
    for step in range(steps_per_episode):
        # Tiempo de ambiente (selección de acción + step)
        env_start = time.time()
        
        if hasattr(agent, 'get_action'):
            action = agent.get_action(obs, epsilon=0.1)
        else:
            # Agente original
            import random
            if random.random() < 0.1:
                action = agent.env.action_space.sample()
            else:
                q_values = agent.model.forward(obs.unsqueeze(0).to(agent.device))[0]
                action = torch.argmax(q_values, dim=-1).item()
        
        next_obs, reward, done, truncated, info = agent.env.step(action)
        next_obs = agent.process_observation(next_obs)
        
        obs_cpu = obs.cpu() if obs.is_cuda else obs
        next_obs_cpu = next_obs.cpu() if next_obs.is_cuda else next_obs
        
        agent.memory.store_transition(obs_cpu, action, reward, next_obs_cpu, done)
        obs = next_obs
        
        env_time = time.time() - env_start
        environment_times.append(env_time)
        
        # Entrenamiento (cada 4 steps)
        if step % 4 == 0 and agent.memory.can_sample(batch_size):
            train_start = time.time()
            
            if hasattr(agent, 'train_step'):
                agent.train_step(batch_size)
            else:
                # Simular entrenamiento original
                observations, actions, rewards, next_observations, dones = agent.memory.sample_buffer(batch_size)
                dones = dones.unsqueeze(1).float()
                
                q_values = agent.model(observations)
                actions = actions.unsqueeze(1).long()
                qsa_batch = q_values.gather(1, actions)
                
                next_actions = torch.argmax(agent.model(next_observations), dim=1, keepdim=True)
                next_q_values = agent.target_model(next_observations).gather(1, next_actions)
                target_b = rewards.unsqueeze(1) + (1 - dones) * agent.gamma * next_q_values
                
                loss = torch.nn.functional.mse_loss(qsa_batch, target_b.detach())
                
                agent.model.zero_grad()
                loss.backward()
                agent.optimizer.step()
            
            train_time = time.time() - train_start
            training_times.append(train_time)
            total_training_steps += 1
        
        if done:
            obs, info = agent.env.reset()
            obs = agent.process_observation(obs)
    
    total_time = time.time() - total_start
    
    # Calcular métricas
    avg_env_time = np.mean(environment_times)
    avg_train_time = np.mean(training_times) if training_times else 0
    
    overall_steps_per_sec = steps_per_episode / total_time
    training_steps_per_sec = total_training_steps / sum(training_times) if training_times else 0
    
    # Memoria GPU
    gpu_memory = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    
    print(f"Resultados:")
    print(f"- Tiempo total: {total_time:.2f}s")
    print(f"- Tiempo promedio ambiente: {avg_env_time*1000:.2f}ms")
    print(f"- Tiempo promedio entrenamiento: {avg_train_time*1000:.2f}ms")
    print(f"- Steps totales de entrenamiento: {total_training_steps}")
    print(f"- Steps de entrenamiento/sec: {training_steps_per_sec:.1f}")
    print(f"- Memoria GPU: {gpu_memory:.2f}GB")
    
    return {
        'total_time': total_time,
        'avg_env_time': avg_env_time,
        'avg_train_time': avg_train_time,
        'overall_steps_per_sec': overall_steps_per_sec,
        'training_steps_per_sec': training_steps_per_sec,
        'total_training_steps': total_training_steps,
        'gpu_memory': gpu_memory
    }

def main():
    print("=== Complete Pipeline Benchmark ===")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB" if torch.cuda.is_available() else "")
    
    realistic_training_benchmark()

if __name__ == "__main__":
    main()
