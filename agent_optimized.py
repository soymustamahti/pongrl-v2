from buffer import ReplayBuffer
from model import Model, soft_update
import torch
import torch.optim as optim
import torch.nn.functional as F
import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import random
import os
import cv2
import numpy as np


class OptimizedAgent():
    """
    Versión optimizada del agente para GPUs potentes.
    Minimiza transferencias CPU-GPU y usa batching eficiente.
    """

    def __init__(self, env, hidden_layer, learning_rate, step_repeat, gamma, use_mixed_precision=True):
        
        self.env = env
        self.step_repeat = step_repeat
        self.gamma = gamma
        self.use_mixed_precision = use_mixed_precision

        obs, info = self.env.reset()
        obs = self.process_observation(obs)

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f'Loaded model on device {self.device}')

        # Buffer más grande para GPU
        buffer_size = 1000000 if self.device.startswith('cuda') else 500000
        self.memory = ReplayBuffer(max_size=buffer_size, input_shape=obs.shape, device=self.device)

        self.model = Model(action_dim=env.action_space.n, hidden_dim=hidden_layer, observation_shape=obs.shape).to(self.device)
        self.target_model = Model(action_dim=env.action_space.n, hidden_dim=hidden_layer, observation_shape=obs.shape).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Gradient scaler para mixed precision
        if self.use_mixed_precision and self.device.startswith('cuda'):
            self.scaler = torch.cuda.amp.GradScaler()
            print("Usando Mixed Precision Training")
        else:
            self.scaler = None

        self.learning_rate = learning_rate

        # Cache para reducir transferencias
        self._obs_cache = None

    def process_observation(self, obs):
        """Procesar observación y mantenerla en GPU si es posible"""
        obs = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)
        if self.device.startswith('cuda'):
            obs = obs.to(self.device, non_blocking=True)
        return obs

    def get_action(self, obs, epsilon):
        """Obtener acción manteniendo tensores en GPU"""
        if random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                if not obs.is_cuda and self.device.startswith('cuda'):
                    obs = obs.to(self.device, non_blocking=True)
                
                if self.use_mixed_precision and self.scaler:
                    with torch.cuda.amp.autocast():
                        q_values = self.model(obs.unsqueeze(0))[0]
                else:
                    q_values = self.model(obs.unsqueeze(0))[0]
                
                action = torch.argmax(q_values, dim=-1).item()
            return action

    def train_step(self, batch_size):
        """Paso de entrenamiento optimizado"""
        if not self.memory.can_sample(batch_size):
            return None

        observations, actions, rewards, next_observations, dones = self.memory.sample_buffer(batch_size)
        
        # Asegurar que todo está en GPU
        if self.device.startswith('cuda'):
            observations = observations.to(self.device, non_blocking=True)
            next_observations = next_observations.to(self.device, non_blocking=True)
            actions = actions.to(self.device, non_blocking=True)
            rewards = rewards.to(self.device, non_blocking=True)
            dones = dones.to(self.device, non_blocking=True)

        dones = dones.unsqueeze(1).float()
        actions = actions.unsqueeze(1).long()

        if self.use_mixed_precision and self.scaler:
            with torch.cuda.amp.autocast():
                # Forward pass con mixed precision
                q_values = self.model(observations)
                qsa_batch = q_values.gather(1, actions)

                with torch.no_grad():
                    next_actions = torch.argmax(self.model(next_observations), dim=1, keepdim=True)
                    next_q_values = self.target_model(next_observations).gather(1, next_actions)
                    target_b = rewards.unsqueeze(1) + (1 - dones) * self.gamma * next_q_values

                loss = F.mse_loss(qsa_batch, target_b.detach())

            # Backward pass con gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Forward pass normal
            q_values = self.model(observations)
            qsa_batch = q_values.gather(1, actions)

            with torch.no_grad():
                next_actions = torch.argmax(self.model(next_observations), dim=1, keepdim=True)
                next_q_values = self.target_model(next_observations).gather(1, next_actions)
                target_b = rewards.unsqueeze(1) + (1 - dones) * self.gamma * next_q_values

            loss = F.mse_loss(qsa_batch, target_b.detach())

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        return loss.item()

    def train(self, episodes, max_episode_steps, summary_writer_suffix, batch_size, epsilon, epsilon_decay, min_epsilon):

        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{summary_writer_suffix}'
        writer = SummaryWriter(summary_writer_name)

        if not os.path.exists('models'):
            os.makedirs('models')

        total_steps = 0
        update_target_steps = 0

        print(f"Iniciando entrenamiento optimizado:")
        print(f"- Device: {self.device}")
        print(f"- Mixed Precision: {self.use_mixed_precision and self.scaler is not None}")
        print(f"- Buffer size: {self.memory.mem_size}")
        print(f"- Batch size: {batch_size}")

        for episode in range(episodes):
            done = False
            episode_reward = 0
            obs, info = self.env.reset()
            obs = self.process_observation(obs)
            episode_steps = 0
            episode_start_time = time.time()
            losses = []

            while not done and episode_steps < max_episode_steps:
                action = self.get_action(obs, epsilon)
                
                reward = 0
                for i in range(self.step_repeat):
                    next_obs, reward_temp, done, truncated, info = self.env.step(action=action)
                    reward += reward_temp
                    if done:
                        break
                
                next_obs = self.process_observation(next_obs)
                
                # Convertir a CPU solo para almacenar en memoria
                obs_cpu = obs.cpu() if obs.is_cuda else obs
                next_obs_cpu = next_obs.cpu() if next_obs.is_cuda else next_obs
                
                self.memory.store_transition(obs_cpu, action, reward, next_obs_cpu, done)
                obs = next_obs

                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                # Entrenar más frecuentemente
                if self.memory.can_sample(batch_size) and total_steps % 4 == 0:
                    loss = self.train_step(batch_size)
                    if loss is not None:
                        losses.append(loss)
                        writer.add_scalar("Loss/model", loss, total_steps)

                    # Actualizar target network más frecuentemente para redes grandes
                    update_target_steps += 1
                    if update_target_steps % 8 == 0:
                        soft_update(self.target_model, self.model, tau=0.01)

            # Guardar modelo
            self.model.save_the_model()

            # Logging
            avg_loss = np.mean(losses) if losses else 0
            writer.add_scalar('Score', episode_reward, episode)
            writer.add_scalar('Epsilon', epsilon, episode)
            writer.add_scalar('Avg_Loss', avg_loss, episode)
            writer.add_scalar('Episode_Steps', episode_steps, episode)

            if epsilon > min_epsilon:
                epsilon *= epsilon_decay
            
            episode_time = time.time() - episode_start_time

            print(f"Episode {episode}: Score={episode_reward:.1f}, Time={episode_time:.2f}s, Steps={episode_steps}, Avg_Loss={avg_loss:.4f}, Epsilon={epsilon:.3f}")

            # Limpiar cache de GPU ocasionalmente
            if episode % 100 == 0 and self.device.startswith('cuda'):
                torch.cuda.empty_cache()

    def test(self):
        """Método de test optimizado"""
        self.model.load_the_model()
        self.model.eval()

        obs, info = self.env.reset()
        done = False
        obs = self.process_observation(obs)
        episode_reward = 0

        print("Iniciando test con modelo entrenado...")

        with torch.no_grad():
            while not done:
                action = self.get_action(obs, epsilon=0.05)  # Poca exploración
                
                reward = 0
                for i in range(self.step_repeat):
                    next_obs, reward_temp, done, truncated, info = self.env.step(action=action)
                    reward += reward_temp

                    # Visualización
                    frame = self.env.env.env.render() 
                    resized_frame = cv2.resize(frame, (500, 400))
                    resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Pong AI - Optimized", resized_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    time.sleep(0.05)
                    if done:
                        break
                
                obs = self.process_observation(next_obs)
                episode_reward += reward

        print(f"Test completado. Score final: {episode_reward}")
        cv2.destroyAllWindows()
