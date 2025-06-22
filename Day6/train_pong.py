#!/usr/bin/env python3
"""
Final DQN Training Script for Atari Pong - Production Ready
Includes proper score tracking, loss monitoring, and auto-recovery
"""

import os
import sys
import random
import copy
import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
import json
import ale_py

# Add import at the top
import glob
if not torch.cuda.is_available():
    print("⚠️  WARNING: CUDA not available! Training will be very slow.")
    print("Continuing with CPU...")
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda")
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")

# Hyperparameters
SEED = 42
ENV_NAME = "ALE/Pong-v5"
LEARNING_RATE = 1e-4
GAMMA = 0.99
BATCH_SIZE = 128  # Increased for better GPU utilization
REPLAY_SIZE = 200000
MIN_REPLAY_SIZE = 20000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 500000
TARGET_UPDATE = 1000
MAX_FRAMES = 2000000
SAVE_INTERVAL = 50000
LOG_INTERVAL = 5  # Log every 5 episodes

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

class DQN(nn.Module):
    """Deep Q-Network"""
    def __init__(self, n_actions=3):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state).to(DEVICE),
            torch.LongTensor(action).to(DEVICE),
            torch.FloatTensor(reward).to(DEVICE),
            torch.FloatTensor(next_state).to(DEVICE),
            torch.FloatTensor(done).to(DEVICE)
        )
    
    def __len__(self):
        return len(self.buffer)

def preprocess_frame(frame):
    """Convert to grayscale and resize"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray[34:194], (84, 84))
    return resized.astype(np.float32) / 255.0

class FrameStack:
    """Stack last 4 frames"""
    def __init__(self):
        self.frames = deque(maxlen=4)
    
    def reset(self, frame):
        for _ in range(4):
            self.frames.append(preprocess_frame(frame))
        return np.array(self.frames)
    
    def push(self, frame):
        self.frames.append(preprocess_frame(frame))
        return np.array(self.frames)

def save_metrics(metrics, filename='training_metrics.json'):
    """Save training metrics to file"""
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)

def train():
    """Main training loop"""
    print(f"\n{'='*60}")
    print(f"Starting DQN Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Initialize environment
    env = gym.make(ENV_NAME)
    n_actions = 3  # NOOP, UP, DOWN
    action_map = [0, 2, 3]
    
    # Initialize networks
    policy_net = DQN(n_actions).to(DEVICE)
    target_net = DQN(n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # Initialize optimizer and buffer
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(REPLAY_SIZE)
    frame_stack = FrameStack()
    
    # Training variables
    epsilon = EPSILON_START
    frame_count = 0
    episode = 0
    episode_rewards = []
    episode_lengths = []
    losses = []
    best_avg_reward = -21.0
    
    # Check for existing checkpoints
    import glob
    checkpoints = glob.glob('checkpoint_*.pt')
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        latest_checkpoint = checkpoints[-1]
        print(f"📁 Found checkpoint: {latest_checkpoint}")
        
        resume = input("Resume from checkpoint? (y/n): ").strip().lower()
        if resume == 'y':
            checkpoint = torch.load(latest_checkpoint)
            
            # Restore state
            frame_count = checkpoint['frame']
            episode = checkpoint.get('episode', 0)
            epsilon = checkpoint['epsilon']
            best_avg_reward = checkpoint.get('best_avg_reward', -21.0)
            
            # Restore networks and optimizer
            policy_net.load_state_dict(checkpoint['model_state'])
            target_net.load_state_dict(checkpoint['target_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            # Restore metrics
            metrics = checkpoint.get('metrics', {
                'episodes': [], 'frames': [], 'rewards': [],
                'avg_rewards': [], 'losses': [], 'epsilon': [],
                'episode_lengths': []
            })
            
            # Restore episode rewards from metrics
            if 'rewards' in metrics:
                episode_rewards = metrics['rewards']
            
            print(f"✅ Resumed training from:")
            print(f"   Frame: {frame_count:,}")
            print(f"   Episode: {episode}")
            print(f"   Epsilon: {epsilon:.3f}")
            print(f"   Best avg reward: {best_avg_reward:.2f}")
            print(f"   Episodes completed: {len(episode_rewards)}")
        else:
            print("🆕 Starting fresh training...")
            metrics = {
                'episodes': [], 'frames': [], 'rewards': [],
                'avg_rewards': [], 'losses': [], 'epsilon': [],
                'episode_lengths': []
            }
    else:
        print("🆕 No checkpoints found. Starting fresh training...")
        metrics = {
            'episodes': [], 'frames': [], 'rewards': [],
            'avg_rewards': [], 'losses': [], 'epsilon': [],
            'episode_lengths': []
        }
    
    # Track actual game score
    episode_score = 0
    episode_start_frame = 0
    
    # Metrics tracking - moved after checkpoint check
    # (metrics is now set in the checkpoint loading section above)
    
    # Initialize first state
    obs, _ = env.reset()
    state = frame_stack.reset(obs)
    
    print("Collecting initial experience...")
    print("Episode | Frames | Score | Avg(100) | Loss | Epsilon | FPS")
    print("-" * 65)
    
    start_time = datetime.now()
    
    # Training loop
    while frame_count < MAX_FRAMES:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randrange(n_actions)
        else:
            with torch.no_grad():
                q_values = policy_net(torch.FloatTensor(state).unsqueeze(0).to(DEVICE))
                action = q_values.argmax().item()
        
        # Take action
        next_obs, reward, done, truncated, _ = env.step(action_map[action])
        next_state = frame_stack.push(next_obs)
        
        # Track actual score
        episode_score += reward
        
        # Store transition
        memory.push(state, action, np.sign(reward), next_state, done)
        state = next_state
        frame_count += 1
        
        # Update epsilon
        epsilon = max(EPSILON_END, EPSILON_START - frame_count / EPSILON_DECAY)
        
        # Reset if episode ended
        if done or truncated:
            episode += 1
            episode_rewards.append(episode_score)
            episode_length = frame_count - episode_start_frame
            episode_lengths.append(episode_length)
            
            # Calculate metrics
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            avg_loss = np.mean(losses[-1000:]) if losses else 0.0
            
            # Calculate FPS
            elapsed = (datetime.now() - start_time).total_seconds()
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Update metrics
            metrics['episodes'].append(episode)
            metrics['frames'].append(frame_count)
            metrics['rewards'].append(episode_score)
            metrics['avg_rewards'].append(avg_reward)
            metrics['losses'].append(avg_loss)
            metrics['epsilon'].append(epsilon)
            metrics['episode_lengths'].append(episode_length)
            
            # Print progress
            if episode % LOG_INTERVAL == 0:
                print(f"{episode:5d} | {frame_count:6d} | {episode_score:+3.0f} | {avg_reward:+6.2f} | "
                      f"{avg_loss:5.3f} | {epsilon:5.3f} | {fps:4.0f}")
                
                # Save best model
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    torch.save(policy_net.state_dict(), 'best_model.pt')
                    print(f"  📈 New best average reward: {best_avg_reward:.2f}")
            
            # Reset for next episode
            episode_score = 0
            episode_start_frame = frame_count
            obs, _ = env.reset()
            state = frame_stack.reset(obs)
        
        # Training step
        if len(memory) >= MIN_REPLAY_SIZE and frame_count % 4 == 0:
            # Sample batch
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
            
            # Compute current Q values
            current_q_values = policy_net(states).gather(1, actions.unsqueeze(1))
            
            # Compute target Q values (Double DQN)
            with torch.no_grad():
                next_actions = policy_net(next_states).argmax(1, keepdim=True)
                next_q_values = target_net(next_states).gather(1, next_actions)
                target_q_values = rewards.unsqueeze(1) + (GAMMA * next_q_values * (1 - dones.unsqueeze(1)))
            
            # Compute loss
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
            optimizer.step()
            
            losses.append(loss.item())
            
            # Update target network
            if frame_count % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
        
        # Save checkpoint
        if frame_count % SAVE_INTERVAL == 0:
            checkpoint = {
                'frame': frame_count,
                'episode': episode,
                'model_state': policy_net.state_dict(),
                'target_state': target_net.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epsilon': epsilon,
                'best_avg_reward': best_avg_reward,
                'metrics': metrics
            }
            torch.save(checkpoint, f'checkpoint_{frame_count}.pt')
            save_metrics(metrics)
            print(f"\n💾 Saved checkpoint at frame {frame_count}")
            print(f"   Best avg reward so far: {best_avg_reward:.2f}\n")
    
    # Final save
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    # Save final model
    torch.save(policy_net.state_dict(), 'final_model.pt')
    
    # Create final plots
    plt.figure(figsize=(15, 10))
    
    # Episode rewards
    plt.subplot(2, 3, 1)
    plt.plot(episode_rewards, alpha=0.6)
    plt.plot(pd.Series(episode_rewards).rolling(100).mean(), 'r-', linewidth=2)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    
    # Training loss
    plt.subplot(2, 3, 2)
    if losses:
        plt.plot(losses, alpha=0.6)
        plt.title('Training Loss')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)
    
    # Epsilon decay
    plt.subplot(2, 3, 3)
    plt.plot(metrics['frames'], metrics['epsilon'])
    plt.title('Epsilon Decay')
    plt.xlabel('Frames')
    plt.ylabel('Epsilon')
    plt.grid(True)
    
    # Episode lengths
    plt.subplot(2, 3, 4)
    plt.plot(episode_lengths, alpha=0.6)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Frames')
    plt.grid(True)
    
    # Win rate over time
    plt.subplot(2, 3, 5)
    wins = [1 if r > 0 else 0 for r in episode_rewards]
    win_rate = pd.Series(wins).rolling(100).mean() * 100
    plt.plot(win_rate)
    plt.title('Win Rate (100 episode average)')
    plt.xlabel('Episode')
    plt.ylabel('Win %')
    plt.grid(True)
    
    # Learning curve
    plt.subplot(2, 3, 6)
    plt.plot(metrics['avg_rewards'])
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=20, color='g', linestyle='--', alpha=0.5)
    plt.title('Average Reward (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    print(f"\n📊 Plots saved to training_results.png")
    
    # Print final statistics
    final_100_avg = np.mean(episode_rewards[-100:])
    final_win_rate = sum(1 for r in episode_rewards[-100:] if r > 0)
    
    print(f"\n📈 Final Statistics:")
    print(f"   Total episodes: {episode}")
    print(f"   Total frames: {frame_count}")
    print(f"   Final 100-episode average: {final_100_avg:.2f}")
    print(f"   Final 100-episode win rate: {final_win_rate}%")
    print(f"   Best average reward: {best_avg_reward:.2f}")
    print(f"   Training time: {datetime.now() - start_time}")
    
    env.close()

if __name__ == "__main__":
    try:
        # Import pandas if available for rolling averages
        try:
            import pandas as pd
        except ImportError:
            print("Note: Install pandas for better plots: pip install pandas")
            pd = None
            
        train()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("💾 Progress saved - you can resume from the last checkpoint")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()