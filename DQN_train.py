# train.py
import os
import torch
import torch.optim as optim
import numpy as np
from env import PipelineEnv
from dqn import DQN
from replay_buffer import ReplayBuffer
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


# 超参数
WINDOW_SIZE = 128  # 代表state的一个时间窗口，即128个点
ACTION_DIM = 3  # 动作数量
LR = 1e-3  #
GAMMA = 0.99  #
BATCH_SIZE = 128  # 代表1episode要使用128个经验来训练网络
MEMORY_SIZE = 10000  # 代表经验池的上限
EPISODES = 50  # 一次训练经历50个episode
EPSILON_START = 1.0  # 开始的探索率，探索率越大说明越趋向于explore而不是exploit
EPSILON_END = 0.05  # 训练到最后锁死的探索率
EPSILON_DECAY = 0.995  #
TARGET_UPDATE_FREQ = 10  # 目标网络更新频率

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

env = PipelineEnv("TRAIN_DATA", WINDOW_SIZE)

# 初始化（永远先初始化）
q_net = DQN(WINDOW_SIZE, ACTION_DIM).to(device)
target_q_net = DQN(WINDOW_SIZE, ACTION_DIM).to(device)  # 目标网络
target_q_net.load_state_dict(q_net.state_dict())  # 初始化目标网络的权重与q_net相同
optimizer = optim.Adam(q_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_SIZE)

epsilon = EPSILON_START
start_episode = 0

model_path = "dqn_models/dqn_pipeline.pth"

# 统一入口：如果 checkpoint 存在，就恢复
if os.path.exists(model_path):
    ckpt = torch.load(model_path, map_location=device)
    q_net.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    epsilon = ckpt["epsilon"]
    start_episode = ckpt["episode"] + 1
    print(f"Resume training from episode {start_episode}")
else:
    print("No checkpoint found, training from scratch")

# 编写训练日志的准备
log_path = "dqn_train_log.csv"
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "train_reward", "eval_reward", "epsilon"])

for episode in range(start_episode, start_episode + EPISODES):
    state = env.reset()
    done = False
    total_reward = 0

    # 每隔一定步数，更新目标网络
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
            # 使用目标网络计算下一个状态的最大 Q 值
            q_next = target_q_net(ns).max(1)[0]
            q_target = r + GAMMA * q_next * (1 - d)

            loss = (q - q_target.detach()).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    print(f"Episode {episode}, Reward {total_reward}, Epsilon {epsilon}")
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

# 训练结束后保存模型
os.makedirs("dqn_models", exist_ok=True)
torch.save({
    "model": q_net.state_dict(),
    "optimizer": optimizer.state_dict(),
    "episode": episode,
    "epsilon": epsilon
}, model_path)
print("Model saved to dqn_pipeline.pth")
