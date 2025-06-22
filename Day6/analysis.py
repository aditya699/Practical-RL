#!/usr/bin/env python3
"""
Quick fix to complete the video recording
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import cv2
from collections import deque
import imageio
import os
import ale_py

class DQN(nn.Module):
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

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray[34:194], (84, 84))
    return resized.astype(np.float32) / 255.0

def record_simple_gameplay(model_path='final_model.pt', num_episodes=3):
    """Simple gameplay recording without dimension issues"""
    print(f"\n🎮 Recording gameplay from: {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = DQN().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval()
    
    env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
    action_map = [0, 2, 3]
    
    all_frames = []
    scores = []
    
    # Fixed dimensions for all frames
    HEIGHT, WIDTH = 224, 160  # Divisible by 16 for ffmpeg
    
    for ep in range(num_episodes):
        print(f"Recording episode {ep+1}/{num_episodes}...")
        
        obs, _ = env.reset()
        frames_buffer = deque(maxlen=4)
        
        for _ in range(4):
            frames_buffer.append(preprocess_frame(obs))
        
        done = False
        total_reward = 0
        
        while not done:
            state = np.array(frames_buffer)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = action_map[q_values.argmax().item()]
            
            obs, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            frames_buffer.append(preprocess_frame(obs))
            
            # Get frame and resize to fixed dimensions
            frame = env.render()
            frame_resized = cv2.resize(frame, (WIDTH, HEIGHT))
            
            # Add text overlay
            cv2.putText(frame_resized, f'Score: {total_reward:+.0f}', (5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            all_frames.append(frame_resized)
            total_reward += reward
            
            # Limit length
            if len(all_frames) > 10000:  # Safety limit
                done = True
        
        scores.append(total_reward)
        print(f"  Episode {ep+1} score: {total_reward:+.0f}")
        
        # Add transition frame (same dimensions)
        transition = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        cv2.putText(transition, f'Episode {ep+1} Complete', (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(transition, f'Score: {total_reward:+.0f}', (40, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for _ in range(30):  # 1 second pause
            all_frames.append(transition)
    
    env.close()
    
    # Save video
    video_name = f"pong_{model_path.replace('.pt', '')}_gameplay.mp4"
    print(f"Saving {len(all_frames)} frames to {video_name}...")
    imageio.mimsave(video_name, all_frames, fps=30)
    print(f"✅ Video saved: {video_name}")
    print(f"📊 Scores: {scores}, Average: {np.mean(scores):.1f}")

# Record for both models
for model in ['best_model.pt', 'final_model.pt']:
    if os.path.exists(model):
        record_simple_gameplay(model, num_episodes=2)

print("\n✅ All videos recorded successfully!")
print("\nYou now have:")
print("- training_analysis_*.png (comprehensive plots)")
print("- pong_best_model_gameplay.mp4")
print("- pong_final_model_gameplay.mp4")