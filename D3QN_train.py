# D3QN_train.py
import os
import torch
import torch.optim as optim
import numpy as np
from env import PipelineEnv
from Dueling_dqn import DuelingDQN
from priortized_replay_buffer import PrioritizedReplayBuffer
import csv
import pandas as pd

# 用于每一轮后计算一次off_reward
def evaluate_policy(env, model, device):
    model.eval()

    total_reward = 0
    total_steps = 0

    for file_path, label in env.files:

        signal = pd.read_excel(file_path, header=None).values.squeeze()

        ptr = 0
        done = False

        while not done:

            state = signal[ptr:ptr + env.window_size]

            state_tensor = torch.FloatTensor(state).to(device)

            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            # reward计算
            if action == label:
                reward = 1
            else:
                if label == 0:
                    reward = -1
                elif label == 1:
                    reward = -2
                else:
                    reward = -3

            total_reward += reward
            total_steps += 1

            ptr += env.step_size
            done = ptr + env.window_size >= len(signal)

    model.train()

    avg_reward = total_reward / total_steps

    return avg_reward

# 超参数，和dqn_train一模一样，具体解释看dqn_train
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

#这一部分是涉及PER机制的超参数
ALPHA = 0.6   # PER
BETA_START = 0.4
BETA_INC = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

env = PipelineEnv("TRAIN_DATA", WINDOW_SIZE)

# 网络
q_net = DuelingDQN(WINDOW_SIZE, ACTION_DIM).to(device)
target_q_net = DuelingDQN(WINDOW_SIZE, ACTION_DIM).to(device)
target_q_net.load_state_dict(q_net.state_dict())
target_q_net.eval()

optimizer = optim.Adam(q_net.parameters(), lr=LR)
memory = PrioritizedReplayBuffer(MEMORY_SIZE, alpha=ALPHA)

epsilon = EPSILON_START
beta = BETA_START
start_episode = 0

model_path = "d3qn_models/d3qn_per.pth"

# 恢复训练
if os.path.exists(model_path):
    ckpt = torch.load(model_path, map_location=device)
    q_net.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    epsilon = ckpt["epsilon"]
    beta = ckpt["beta"]
    start_episode = ckpt["episode"] + 1
    print(f"Resume from episode {start_episode}")
else:
    print("Training from scratch")

# 编写训练日志的准备
log_path = "d3qn_train_log.csv"
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "train_reward", "eval_reward", "epsilon"])

for episode in range(start_episode, start_episode + EPISODES):
    state = env.reset()
    done = False
    total_reward = 0

    if episode % TARGET_UPDATE_FREQ == 0:
        target_q_net.load_state_dict(q_net.state_dict())

    while not done:
        # ε-greedy
        if np.random.rand() < epsilon:
            action = np.random.randint(ACTION_DIM)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = q_net(state_tensor)
            action = q_values.argmax(dim=1).item()

        next_state, reward, done = env.step(action)
        memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(memory) > BATCH_SIZE:
            s, a, r, ns, d, w, idxs = memory.sample(BATCH_SIZE, beta)

            s, a, r, ns, d, w = (
                s.to(device), a.to(device), r.to(device),
                ns.to(device), d.to(device), w.to(device)
            )

            # Double DQN
            q = q_net(s).gather(1, a.unsqueeze(1)).squeeze()
            next_actions = q_net(ns).argmax(dim=1)
            q_next = target_q_net(ns).gather(1, next_actions.unsqueeze(1)).squeeze()

            q_target = r + GAMMA * q_next * (1 - d)

            td_error = q - q_target.detach()
            loss = (w * td_error.pow(2)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新 priority
            new_prios = td_error.abs().detach().cpu().numpy() + 1e-6
            memory.update_priorities(idxs, new_prios)

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    beta = min(1.0, beta + BETA_INC)

    print(f"Episode {episode} | Reward {total_reward:.2f} | Epsilon {epsilon:.3f}")
    eval_reward = evaluate_policy(env, q_net, device)
    # 写日志
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            episode,
            total_reward,
            eval_reward,
            epsilon
        ])

# 保存
os.makedirs("d3qn_models", exist_ok=True)
torch.save({
    "model": q_net.state_dict(),
    "optimizer": optimizer.state_dict(),
    "episode": episode,
    "epsilon": epsilon,
    "beta": beta
}, model_path)

print("D3QN + PER model saved.")
