from agent_optimized import OptimizedAgent
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
import ale_py
import torch

class SmartOptimizedAgent(OptimizedAgent):
    """
    Agente optimizado que auto-configura parámetros basado en hardware
    """
    
    def __init__(self, env, learning_rate=0.0001, step_repeat=4, gamma=0.99):
        # Auto-configurar basado en GPU disponible
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"Auto-configurando para {gpu_name} ({gpu_memory:.1f}GB)")
            
            if "4090" in gpu_name or "4080" in gpu_name or gpu_memory > 20:
                # GPU de alta gama
                hidden_layer = 512  # Más conservador
                use_mixed_precision = True
                print("Configuración: GPU High-End")
            elif "3080" in gpu_name or "3090" in gpu_name or gpu_memory > 10:
                # GPU media-alta
                hidden_layer = 256
                use_mixed_precision = True
                print("Configuración: GPU Mid-High")
            else:
                # GPU básica
                hidden_layer = 128
                use_mixed_precision = False
                print("Configuración: GPU Basic")
        else:
            # CPU
            hidden_layer = 128
            use_mixed_precision = False
            print("Configuración: CPU")
        
        # Llamar al constructor padre
        super().__init__(env, hidden_layer, learning_rate, step_repeat, gamma, use_mixed_precision)
    
    def get_optimal_batch_size(self):
        """Determinar batch size óptimo basado en configuración"""
        if not torch.cuda.is_available():
            return 64
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        hidden_size = self.model.fc1.in_features  # Aproximación
        
        if gpu_memory > 20 and hasattr(self, 'use_mixed_precision') and self.use_mixed_precision:
            return 256  # Más conservador para RTX 4090
        elif gpu_memory > 10:
            return 128
        else:
            return 64

def smart_benchmark():
    """Benchmark con configuración inteligente"""
    print("=== Smart Benchmark ===")
    
    # Crear ambiente
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env = ResizeObservation(env, (64, 64))
    env = GrayscaleObservation(env, keep_dim=True)
    
    # Agente inteligente
    smart_agent = SmartOptimizedAgent(env)
    optimal_batch = smart_agent.get_optimal_batch_size()
    
    print(f"Configuración automática:")
    print(f"- Hidden layer: {smart_agent.model.fc1.out_features}")
    print(f"- Batch size óptimo: {optimal_batch}")
    print(f"- Mixed precision: {smart_agent.use_mixed_precision}")
    print(f"- Device: {smart_agent.device}")
    
    return smart_agent, optimal_batch

if __name__ == "__main__":
    smart_benchmark()
