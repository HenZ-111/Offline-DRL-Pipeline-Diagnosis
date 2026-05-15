import os
import torch
import torch.optim as optim
import numpy as np
from env import PipelineEnv
from dqn import DQN
from replay_buffer import ReplayBuffer
import csv
import pandas as pd

def evaluate_policy(env, model, device):
    model.eval()
    total_reward = 0
    total_steps = 0

    for i in range(len(env.files)):
        state = env.reset(randset=False, i_file=i)
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_state, reward, done = env.step(action)

            total_reward += reward
            total_steps += 1

            state = next_state

    file_path, label = env.files[0]
    signal = pd.read_excel(file_path, header=None).values.squeeze()
    state_number = (len(signal) - env.window_size) / env.step_size + 1
    avg_reward = total_reward / total_steps * state_number
    return avg_reward

WINDOW_SIZE = 128
ACTION_DIM = 3
LR = 1e-3
GAMMA = 0.99
BATCH_SIZE = 128
MEMORY_SIZE = 10000
EPISODES = 50
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
env = PipelineEnv("TRAIN_DATA", WINDOW_SIZE)
env_eval = PipelineEnv("TEST_DATA", WINDOW_SIZE)
q_net = DQN(WINDOW_SIZE, ACTION_DIM).to(device)
target_q_net = DQN(WINDOW_SIZE, ACTION_DIM).to(device)
target_q_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_SIZE)

epsilon = EPSILON_START
start_episode = 0
model_path = "dqn_models/dqn_pipeline.pth"
if os.path.exists(model_path):
    ckpt = torch.load(model_path, map_location=device)
    q_net.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    epsilon = ckpt["epsilon"]
    start_episode = ckpt["episode"] + 1
    memory = ckpt["replay_buffer"]
    print(f"Resume training from episode {start_episode}")
else:
    print("No checkpoint found, training from scratch")

log_path = "dqn_train_log.csv"
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "train_reward", "eval_reward", "epsilon"])

episode = 0
for episode in range(start_episode, start_episode + EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    if episode % TARGET_UPDATE_FREQ == 0:
        target_q_net.load_state_dict(q_net.state_dict())
        print(f"Target network updated at episode {episode}")

    while not done:
        if np.random.rand() < epsilon:
            action = np.random.randint(ACTION_DIM)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                q_values = q_net(state_tensor)
                action = q_values.argmax().item()
        next_state, reward, done = env.step(action)
        memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if len(memory) > BATCH_SIZE:
            s, a, r, ns, d = memory.sample(BATCH_SIZE)
            s = s.to(device)
            a = a.to(device)
            r = r.to(device)
            ns = ns.to(device)
            d = d.to(device)

            q = q_net(s).gather(1, a.unsqueeze(1)).squeeze()
            q_next = target_q_net(ns).max(1)[0]
            q_target = r + GAMMA * q_next * (1 - d)

            loss = (q - q_target.detach()).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    print(f"Episode {episode}, Reward {total_reward}, Epsilon {epsilon}")
    eval_reward = evaluate_policy(env_eval, q_net, device)

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            episode,
            total_reward,
            eval_reward,
            epsilon
        ])

os.makedirs("dqn_models", exist_ok=True)
torch.save({
    "model": q_net.state_dict(),
    "optimizer": optimizer.state_dict(),
    "episode": episode,
    "epsilon": epsilon,
    "replay_buffer": memory
}, model_path)
print("Model saved to dqn_pipeline.pth")
