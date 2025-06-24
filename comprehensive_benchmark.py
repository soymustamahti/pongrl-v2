import time
import torch
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
import ale_py
from agent import Agent
from agent_optimized import OptimizedAgent
import numpy as np

def training_benchmark(agent_class, name, training_steps=1000):
    """Benchmark realista de entrenamiento con muchos pasos"""
    print(f"\n=== Training Benchmark: {name} ===")
    
    # Configuración
    if "Optimized" in name and torch.cuda.is_available():
        hidden_layer = 1024
        batch_size = 1024  # Batch grande para GPU
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
    
    print(f"- Device: {agent.device}")
    print(f"- Hidden layer: {hidden_layer}")
    print(f"- Batch size: {batch_size}")
    print(f"- Target training steps: {training_steps}")
    
    # Llenar memoria inicial
    print("Llenando memoria inicial...")
    obs, info = env.reset()
    obs = agent.process_observation(obs)
    
    for i in range(batch_size * 10):  # Llenar suficiente memoria
        if hasattr(agent, 'get_action'):
            action = agent.get_action(obs, epsilon=1.0)  # Random al inicio
        else:
            action = env.action_space.sample()
        
        next_obs, reward, done, truncated, info = env.step(action)
        next_obs = agent.process_observation(next_obs)
        
        # Convertir a CPU para almacenamiento
        obs_cpu = obs.cpu() if obs.is_cuda else obs
        next_obs_cpu = next_obs.cpu() if next_obs.is_cuda else next_obs
        
        agent.memory.store_transition(obs_cpu, action, reward, next_obs_cpu, done)
        
        obs = next_obs
        if done:
            obs, info = env.reset()
            obs = agent.process_observation(obs)
    
    print(f"Memoria llenada con {agent.memory.mem_ctr} experiencias")
    
    # Benchmark de entrenamiento
    training_times = []
    losses = []
    
    print("Iniciando benchmark de entrenamiento...")
    start_time = time.time()
    
    for step in range(training_steps):
        step_start = time.time()
        
        if hasattr(agent, 'train_step'):
            # Agente optimizado
            loss = agent.train_step(batch_size)
            if loss is not None:
                losses.append(loss)
        else:
            # Agente original - simular entrenamiento
            if agent.memory.can_sample(batch_size):
                observations, actions, rewards, next_observations, dones = agent.memory.sample_buffer(batch_size)
                dones = dones.unsqueeze(1).float()
                
                # Forward pass
                q_values = agent.model(observations)
                actions = actions.unsqueeze(1).long()
                qsa_batch = q_values.gather(1, actions)
                
                next_actions = torch.argmax(agent.model(next_observations), dim=1, keepdim=True)
                next_q_values = agent.target_model(next_observations).gather(1, next_actions)
                target_b = rewards.unsqueeze(1) + (1 - dones) * agent.gamma * next_q_values
                
                loss = torch.nn.functional.mse_loss(qsa_batch, target_b.detach())
                losses.append(loss.item())
                
                # Backward pass
                agent.model.zero_grad()
                loss.backward()
                agent.optimizer.step()
        
        step_time = time.time() - step_start
        training_times.append(step_time)
        
        if step % 100 == 0:
            avg_time = np.mean(training_times[-100:])
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"  Step {step}: {avg_time*1000:.2f}ms/step, Loss: {avg_loss:.4f}")
    
    total_time = time.time() - start_time
    avg_step_time = np.mean(training_times)
    steps_per_second = 1.0 / avg_step_time
    avg_loss = np.mean(losses) if losses else 0
    
    print(f"\n{name} Training Results:")
    print(f"- Total time: {total_time:.2f}s")
    print(f"- Average time per step: {avg_step_time*1000:.2f}ms")
    print(f"- Steps per second: {steps_per_second:.1f}")
    print(f"- Average loss: {avg_loss:.4f}")
    print(f"- Total steps completed: {training_steps}")
    
    return {
        'total_time': total_time,
        'avg_step_time': avg_step_time,
        'steps_per_second': steps_per_second,
        'avg_loss': avg_loss
    }

def gpu_utilization_test():
    """Test específico para mostrar utilización de GPU"""
    print("\n=== GPU Utilization Test ===")
    
    if not torch.cuda.is_available():
        print("CUDA no disponible")
        return
    
    # Configuración grande para saturar GPU
    batch_sizes = [64, 256, 1024, 2048]
    hidden_sizes = [128, 512, 1024]
    
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env = ResizeObservation(env, (64, 64))
    env = GrayscaleObservation(env, keep_dim=True)
    
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)
    
    print("Batch Size | Hidden Size | Time/Step (ms) | GPU Memory (GB)")
    print("-" * 60)
    
    for hidden_size in hidden_sizes:
        for batch_size in batch_sizes:
            try:
                # Crear modelo
                from model import Model
                model = Model(action_dim=env.action_space.n, 
                            hidden_dim=hidden_size, 
                            observation_shape=obs.shape).cuda()
                
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                
                # Crear batch de datos
                batch_obs = obs.unsqueeze(0).repeat(batch_size, 1, 1, 1).cuda()
                target = torch.randn(batch_size, env.action_space.n).cuda()
                
                # Warmup
                for _ in range(10):
                    with torch.amp.autocast('cuda'):
                        output = model(batch_obs)
                        loss = torch.nn.functional.mse_loss(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                torch.cuda.synchronize()
                
                # Benchmark
                times = []
                for _ in range(50):
                    start = time.time()
                    with torch.amp.autocast('cuda'):
                        output = model(batch_obs)
                        loss = torch.nn.functional.mse_loss(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    torch.cuda.synchronize()
                    times.append(time.time() - start)
                
                avg_time = np.mean(times) * 1000  # ms
                gpu_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
                
                print(f"{batch_size:>10} | {hidden_size:>11} | {avg_time:>13.2f} | {gpu_memory:>13.2f}")
                
                # Limpiar memoria
                del model, optimizer, batch_obs, target
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"{batch_size:>10} | {hidden_size:>11} | {'OOM':>13} | {'OOM':>13}")
                    torch.cuda.empty_cache()
                else:
                    raise e

def main():
    print("=== Comprehensive GPU Benchmark ===")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Benchmark de entrenamiento realista
    print("\n" + "="*50)
    print("TRAINING BENCHMARK (1000 steps)")
    print("="*50)
    
    original_results = training_benchmark(Agent, "Original Agent", training_steps=1000)
    optimized_results = training_benchmark(OptimizedAgent, "Optimized Agent", training_steps=1000)
    
    # Comparación detallada
    print("\n" + "="*50)
    print("DETAILED COMPARISON")
    print("="*50)
    
    speedup = original_results['avg_step_time'] / optimized_results['avg_step_time']
    throughput_improvement = optimized_results['steps_per_second'] / original_results['steps_per_second']
    
    print(f"Speedup (tiempo por paso): {speedup:.2f}x")
    print(f"Throughput improvement: {throughput_improvement:.2f}x")
    print(f"Time reduction: {(1 - 1/speedup)*100:.1f}%")
    
    if speedup > 1.5:
        print(f"🚀 Optimized agent is {speedup:.2f}x FASTER!")
    elif speedup > 1.1:
        print(f"✅ Optimized agent is {speedup:.2f}x faster")
    else:
        print(f"⚠️  Optimization gain is minimal: {speedup:.2f}x")
    
    # Test de utilización de GPU
    if torch.cuda.is_available():
        gpu_utilization_test()

if __name__ == "__main__":
    main()
