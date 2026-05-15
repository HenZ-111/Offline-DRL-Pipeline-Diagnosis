import os
import torch
import torch.optim as optim
import numpy as np
from env import PipelineEnv
from Dueling_dqn import DuelingDQN
from priortized_replay_buffer import PrioritizedReplayBuffer
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
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = q_net(state_tensor)
            action = q_values.argmax(dim=1).item()

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

ALPHA = 0.2
BETA_START = 0.7
BETA_INC = 1e-2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
env = PipelineEnv("TRAIN_DATA", WINDOW_SIZE)
env_eval = PipelineEnv("TEST_DATA", WINDOW_SIZE)
q_net = DuelingDQN(WINDOW_SIZE, ACTION_DIM).to(device)
target_q_net = DuelingDQN(WINDOW_SIZE, ACTION_DIM).to(device)
target_q_net.load_state_dict(q_net.state_dict())
target_q_net.eval()
optimizer = optim.Adam(q_net.parameters(), lr=LR)
memory = PrioritizedReplayBuffer(MEMORY_SIZE, alpha=ALPHA)
epsilon = EPSILON_START
beta = BETA_START
start_episode = 0

model_path = "d3qn_models/d3qn_pipeline.pth"
if os.path.exists(model_path):
    ckpt = torch.load(model_path, map_location=device)
    q_net.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    epsilon = ckpt["epsilon"]
    beta = ckpt["beta"]
    start_episode = ckpt["episode"] + 1
    memory = ckpt["replay_buffer"]
    print(f"Resume from episode {start_episode}")
else:
    print("Training from scratch")

log_path = "d3qn_train_log.csv"
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "train_reward", "eval_reward", "epsilon"])

# td-error
small_td_errors = []
big_td_errors = []
no_td_errors = []

episode = 0
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
        memory.push(state, action, reward, next_state, done, env.label)
        state = next_state
        total_reward += reward

        if len(memory) > BATCH_SIZE:
            s, a, r, ns, d, w, idxs, labels = memory.sample(BATCH_SIZE, beta)
            s, a, r, ns, d, w = (s.to(device), a.to(device), r.to(device), ns.to(device), d.to(device), w.to(device))

            q = q_net(s).gather(1, a.unsqueeze(1)).squeeze()
            next_actions = q_net(ns).argmax(dim=1)
            q_next = target_q_net(ns).gather(1, next_actions.unsqueeze(1)).squeeze()
            q_target = r + GAMMA * q_next * (1 - d)

            # td-error
            td_error = q - q_target.detach()
            td_error_np = td_error.detach().cpu().numpy()
            small_td = []
            big_td = []
            no_td = []
            for i, label in enumerate(labels):
                if label == 1:
                    small_td.append(abs(td_error_np[i]))
                elif label == 2:
                    big_td.append(abs(td_error_np[i]))
                elif label == 0:
                    no_td.append(abs(td_error_np[i]))
            if len(small_td) > 0:
                small_td_errors.append(np.mean(small_td))
            if len(big_td) > 0:
                big_td_errors.append(np.mean(big_td))
            if len(no_td) > 0:
                no_td_errors.append(np.mean(no_td))

            loss = (w * td_error.pow(2)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            new_prios = td_error.abs().detach().cpu().numpy() + 1e-6
            memory.update_priorities(idxs, new_prios)

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    beta = min(1.0, beta + BETA_INC)

    print(f"Episode {episode} | Reward {total_reward:.2f} | Epsilon {epsilon:.3f}")
    eval_reward = evaluate_policy(env_eval, q_net, device)

    avg_small = np.mean(small_td_errors) if small_td_errors else 0
    avg_big = np.mean(big_td_errors) if big_td_errors else 0
    avg_no = np.mean(no_td_errors) if no_td_errors else 0
    with open("td_error_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([episode, avg_no, avg_small, avg_big])
    small_td_errors.clear()
    big_td_errors.clear()
    no_td_errors.clear()

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([episode, total_reward, eval_reward, epsilon])

os.makedirs("d3qn_models", exist_ok=True)
torch.save({
    "model": q_net.state_dict(),
    "optimizer": optimizer.state_dict(),
    "episode": episode,
    "epsilon": epsilon,
    "beta": beta,
    "replay_buffer": memory
}, model_path)
print("D3QN model saved.")
