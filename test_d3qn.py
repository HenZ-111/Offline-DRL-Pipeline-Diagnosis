import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
from env import PipelineEnv
from Dueling_dqn import DuelingDQN


# =====================
# 参数
# =====================
DATA_ROOT = "TEST_DATA"
WINDOW_SIZE = 128
ACTION_DIM = 3
MODEL_PATH = "d3qn_models/d3qn_15_0131_21.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULT_DIR = "test_d3qn_results"
os.makedirs(RESULT_DIR, exist_ok=True)

# =====================
# 加载模型
# =====================

model = DuelingDQN(WINDOW_SIZE, ACTION_DIM).to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

print("D3QN model loaded.")


# =====================
# env 只用于读取文件
# =====================

env = PipelineEnv(DATA_ROOT, WINDOW_SIZE)

label_names = ["noLeak", "smallLeak", "bigLeak"]

# =====================
# 测试开始
# =====================
y_true = []
y_pred = []
file_names = []
confidence_list = []


# =====================
# 全数据集测试
# =====================

with torch.no_grad():

    for file_path, label in env.files:

        signal = pd.read_excel(
            file_path,
            header=None
        ).values.squeeze()

        ptr = 0
        step = env.step_size
        q_sum = np.zeros(ACTION_DIM)
        count = 0

        # 滑动窗口
        while ptr + WINDOW_SIZE < len(signal):

            state = signal[
                ptr:ptr + WINDOW_SIZE
            ]
            state_tensor = torch.FloatTensor(
                state
            ).unsqueeze(0).to(DEVICE)
            q = model(state_tensor).cpu().numpy().squeeze()
            q_sum += q
            count += 1
            ptr += step

        if count == 0:
            continue

        avg_q = q_sum / count
        pred = int(np.argmax(avg_q))
        confidence = float(np.max(avg_q))
        y_true.append(label)
        y_pred.append(pred)
        confidence_list.append(confidence)
        file_names.append(
            os.path.basename(file_path)
        )


# =====================
# Overall指标
# =====================

acc = accuracy_score(y_true, y_pred)
print("\nOverall Accuracy:", acc)
print(
classification_report(
    y_true,
    y_pred,
    target_names=label_names
)
)

# =====================
# CSV1 每样本预测
# =====================

df_pred = pd.DataFrame({
    "file": file_names,
    "true_label": y_true,
    "pred_label": y_pred,
    "confidence": confidence_list
})

df_pred.to_csv(
os.path.join(
RESULT_DIR,
"d3qn_predictions.csv"
),
index=False
)

print("Prediction CSV saved.")


# =====================
# CSV2 分类指标
# =====================

report = classification_report(
y_true,
y_pred,
target_names=label_names,
output_dict=True
)

df_report = pd.DataFrame(
report
).transpose()

df_report.to_csv(
os.path.join(
RESULT_DIR,
"d3qn_classification_report.csv"
)
)

print("Report CSV saved.")


# =====================
# 混淆矩阵
# =====================

cm = confusion_matrix(
y_true,
y_pred
)

plt.figure(figsize=(6,5))
plt.imshow(cm)
plt.title("D3QN Confusion Matrix")
plt.colorbar()
ticks = np.arange(len(label_names))
plt.xticks(ticks,label_names)
plt.yticks(ticks,label_names)
for i in range(len(label_names)):
    for j in range(len(label_names)):
        plt.text(
            j,
            i,
            cm[i,j],
            ha="center",
            va="center"
        )

plt.ylabel("True")
plt.xlabel("Pred")
plt.tight_layout()
plt.savefig(
os.path.join(
RESULT_DIR,
"d3qn_confusion_matrix.png"
),
dpi=300
)
plt.show()
print("Confusion matrix saved.")