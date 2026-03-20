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

from env import PipelineEnv, LABEL_MAP
from dqn import DQN

# =====================
# 基本参数
# =====================
DATA_ROOT = "TEST_DATA"
WINDOW_SIZE = 128
ACTION_DIM = 3
MODEL_PATH = "dqn_models/dqn_pipeline_15_0201_17.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULT_DIR = "test_dqn_results"
os.makedirs(RESULT_DIR, exist_ok=True)

# =====================
# 加载模型
# =====================
model = DQN(WINDOW_SIZE, ACTION_DIM).to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

print("Model loaded successfully.")

# =====================
# 加载环境（只用来取数据）
# =====================
env = PipelineEnv(DATA_ROOT, WINDOW_SIZE)

# =====================
# 测试开始
# =====================
y_true = []
y_pred = []
file_names = []

with torch.no_grad():
    for file_path, label in env.files:
        # 读取完整信号
        signal = pd.read_excel(file_path, header=None).values.squeeze()

        q_sum = np.zeros(ACTION_DIM)
        count = 0
        ptr = 0
        step_size = env.step_size

        while ptr + WINDOW_SIZE < len(signal):
            state = signal[ptr:ptr + WINDOW_SIZE]
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

            q_values = model(state_tensor).cpu().numpy().squeeze()
            q_sum += q_values
            count += 1

            ptr += step_size

        # 使用 Q 值平均作为最终决策
        avg_q = q_sum / count
        pred_label = int(np.argmax(avg_q))

        y_true.append(label)
        y_pred.append(pred_label)
        file_names.append(os.path.basename(file_path))

# =====================
# 统计指标
# =====================
acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

label_names = ["noLeak", "smallLeak", "bigLeak"]

report = classification_report(
    y_true,
    y_pred,
    target_names=label_names,
    output_dict=True
)

print("Overall Accuracy:", acc)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=label_names))

# =====================
# 保存 CSV（论文画表）
# =====================
# 1️⃣ 每个样本预测结果
df_pred = pd.DataFrame({
    "file": file_names,
    "true_label": y_true,
    "pred_label": y_pred
})
df_pred.to_csv(os.path.join(RESULT_DIR, "dqn_test_predictions.csv"), index=False)

# 2️⃣ 分类指标
df_report = pd.DataFrame(report).transpose()
df_report.to_csv(os.path.join(RESULT_DIR, "dqn_classification_report.csv"))

print("CSV results saved.")

# =====================
# 绘制混淆矩阵
# =====================
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix (DQN)")
plt.colorbar()

tick_marks = np.arange(len(label_names))
plt.xticks(tick_marks, label_names)
plt.yticks(tick_marks, label_names)

for i in range(len(label_names)):
    for j in range(len(label_names)):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center")

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "dqn_confusion_matrix.png"), dpi=300)
plt.show()

print("Confusion matrix saved.")
