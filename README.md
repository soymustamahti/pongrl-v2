# Pong Reinforcement Learning with Deep Q-Network (DQN)

This project implements a Deep Q-Network (DQN) agent to play the classic Atari Pong game using reinforcement learning. The agent learns to play Pong by observing game states and taking actions to maximize its score.

## Overview

The project consists of a complete reinforcement learning pipeline that trains an AI agent to play Pong using Double DQN with experience replay. The agent uses a convolutional neural network to process game frames and learns optimal actions through trial and error.

## Project Structure

```
├── agent.py                    # Main DQN agent implementation
├── agent_optimized.py          # GPU-optimized agent with mixed precision
├── model.py                    # Neural network architecture
├── buffer.py                   # Experience replay buffer
├── train.py                    # Training script (CPU/small GPU)
├── train_gpu_optimized.py      # Training script optimized for powerful GPUs
├── test.py                     # Testing/evaluation script
├── test_optimized.py           # Testing script for optimized agent
├── benchmark.py                # Performance comparison tool
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Components Explained

### 1. `model.py` - Neural Network Architecture

- **CNN Layers**: Three convolutional layers to extract features from game frames
  - Conv1: 1→8 channels, 4x4 kernel, stride 2
  - Conv2: 8→16 channels, 4x4 kernel, stride 2
  - Conv3: 16→32 channels, 3x3 kernel, stride 2
- **Fully Connected Layers**: Three dense layers (256 hidden units each) for decision making
- **Output Layer**: Produces Q-values for each possible action
- **Weight Initialization**: Kaiming normal for Conv2D, Xavier normal for Linear layers
- **Soft Update Function**: For target network updates with parameter τ=0.005

### 2. `buffer.py` - Experience Replay Buffer

- **Memory Storage**: Stores up to 500,000 experience tuples (state, action, reward, next_state, done)
- **Efficient Sampling**: Random batch sampling for training
- **Memory Management**: Circular buffer that overwrites old experiences
- **Device Support**: Automatic GPU/CPU tensor handling

### 3. `agent.py` - DQN Agent Implementation

- **Double DQN**: Uses separate online and target networks to reduce overestimation bias
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation
- **Experience Replay**: Learns from stored past experiences
- **Target Network**: Periodically updated for stable learning
- **Preprocessing**: Converts RGB frames to grayscale and normalizes pixel values
- **TensorBoard Logging**: Tracks training metrics (loss, score, epsilon)

### 4. Training Configuration

- **Environment**: Atari Pong with 64x64 grayscale observation
- **Episodes**: 10,000 training episodes
- **Frame Skipping**: 4 frames per action (step_repeat=4)
- **Learning Rate**: 0.0001 with Adam optimizer
- **Discount Factor**: γ=0.99
- **Batch Size**: 64 experiences per training step
- **Epsilon Decay**: Starts at 1.0, decays by 0.995 per episode, minimum 0.1

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   The project requires:

   - `gymnasium[atari]` - Atari game environments
   - `ale-py` - Arcade Learning Environment
   - `torch`, `torchvision`, `torchaudio` - PyTorch for neural networks
   - `tensorboard` - Training visualization
   - `opencv-python` - Image processing
   - `pygame` - Game rendering support
   - `numpy` - Numerical operations

3. **CUDA Support** (Optional):
   - The requirements include CUDA 11.8 support for GPU acceleration
   - The agent automatically detects and uses GPU if available

## Usage

### Training the Agent

To train a new agent from scratch:

```bash
python train.py
```

**Training Process**:

- The agent starts with random actions (epsilon=1.0)
- Gradually reduces exploration as it learns (epsilon decay)
- Saves model weights to `models/latest.pt` after each episode
- Logs training metrics to TensorBoard in the `runs/` directory

**Training Parameters**:

- 10,000 episodes maximum
- 10,000 steps per episode maximum
- Experience replay starts after 320 experiences (5 × batch_size)
- Target network updates every 4 steps
- Model saves after every episode

### GPU-Optimized Training (Recommended for RTX 4090+)

For high-end GPUs with 16GB+ VRAM:

```bash
python train_gpu_optimized.py
```

**Optimized Features**:

- Automatically detects GPU and configures optimal settings
- Uses 1024 hidden units and 1024 batch size for RTX 4090
- Mixed precision training for 2x speedup
- Larger experience buffer (1M vs 500K)
- Advanced gradient clipping and regularization

**Expected Performance**:

- RTX 4090: ~2-3x faster than CPU
- RTX 3080/4080: ~1.5-2x faster than CPU
- Older GPUs: May be slower due to overhead

### Testing the Agent

To test a trained agent:

```bash
python test.py
```

**Testing Features**:

- Loads trained weights from `models/latest.pt`
- Displays game window using OpenCV
- Uses minimal exploration (5% random actions)
- Shows real-time gameplay at 20 FPS
- Press 'q' to quit the game window

### Using VS Code Task

You can also run the training using the predefined VS Code task:

- Press `Ctrl+Shift+P` (Linux) and type "Tasks: Run Task"
- Select "Run Python script" to execute `train.py`

## Monitoring Training

### TensorBoard Visualization

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir=runs
```

**Available Metrics**:

- **Score**: Episode rewards over time
- **Loss/model**: Training loss values
- **Epsilon**: Exploration rate decay

### Console Output

The training script provides real-time feedback:

- Episode completion with scores
- Episode duration and step count
- Model saving confirmations

## Algorithm Details

### Double DQN

- **Main Network**: Updated every step with gradient descent
- **Target Network**: Soft-updated every 4 steps (τ=0.005)
- **Action Selection**: Main network selects actions, target network evaluates them
- **Loss Function**: Mean Squared Error between predicted and target Q-values

### Experience Replay

- **Buffer Size**: 500,000 transitions
- **Sampling**: Random batch sampling breaks temporal correlations
- **Training Trigger**: Starts when buffer has 5× batch_size experiences

### Preprocessing

- **Frame Stacking**: Single grayscale frame input
- **Normalization**: Pixel values divided by 255
- **Resizing**: 64×64 resolution for efficient processing

## File Structure After Training

```
├── models/
│   └── latest.pt      # Trained model weights
├── runs/
│   └── [timestamp]/   # TensorBoard logs
│       ├── events.out.tfevents.*
│       └── ...
└── [project files]
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:

   - Reduce batch_size in `train.py`
   - Use CPU by setting `device='cpu'` in `agent.py`

2. **Slow Training**:

   - Ensure CUDA is properly installed for GPU acceleration
   - Reduce image resolution or network size

3. **Poor Performance**:

   - Increase training episodes
   - Adjust learning rate or network architecture
   - Check epsilon decay schedule

4. **Import Errors**:
   - Verify all dependencies are installed
   - Check Python version compatibility (recommended: Python 3.8+)

### Performance Expectations

- **Initial Performance**: Random gameplay (score around -21 to -18)
- **Learning Progress**: Gradual improvement over thousands of episodes
- **Convergence**: Agent should start winning games after 5,000-10,000 episodes
- **Training Time**: Several hours to days depending on hardware

## Customization

### Hyperparameter Tuning

Modify parameters in `train.py`:

- `learning_rate`: Controls learning speed
- `hidden_layer`: Network capacity
- `gamma`: Future reward importance
- `epsilon_decay`: Exploration reduction rate
- `batch_size`: Training batch size

### Network Architecture

Modify `model.py` to experiment with:

- Different CNN architectures
- Layer sizes and activation functions
- Regularization techniques

### Environment Settings

Change game settings in `train.py` and `test.py`:

- Different Atari games
- Frame resolution
- Action repeat frequency

## GPU Optimization Features

### New Optimized Components

#### `agent_optimized.py` - GPU-Optimized Agent

- **Mixed Precision Training**: Uses FP16 for faster training on modern GPUs
- **Minimized CPU-GPU Transfers**: Keeps tensors on GPU when possible
- **Larger Buffer**: 1M experiences for GPU setups vs 500K for CPU
- **AdamW Optimizer**: Better regularization with weight decay
- **Gradient Clipping**: Prevents exploding gradients in large networks
- **Efficient Batching**: Optimized for larger batch sizes (512-1024)

#### `train_gpu_optimized.py` - High-Performance Training

- **Auto-Detection**: Automatically configures for GPU vs CPU
- **Large Networks**: 1024 hidden units for RTX 4090 class GPUs
- **Big Batches**: 1024 batch size for maximum GPU utilization
- **Advanced Features**: Mixed precision, gradient clipping, memory management

#### `benchmark.py` - Performance Testing

- **Side-by-side Comparison**: Tests both original and optimized agents
- **Detailed Metrics**: Steps/second, memory usage, speedup measurements
- **Hardware Detection**: Automatic configuration based on available hardware

This implementation provides a solid foundation for understanding and experimenting with deep reinforcement learning in the classic Pong environment.
