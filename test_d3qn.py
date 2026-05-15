import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    ConfusionMatrixDisplay
)
from env import PipelineEnv
from Dueling_dqn import DuelingDQN


DATA_ROOT = "TEST_DATA"
WINDOW_SIZE = 128
ACTION_DIM = 3
MODEL_PATH = "d3qn_models/d3qn_pipeline.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_DIR = "test_d3qn_results"
os.makedirs(RESULT_DIR, exist_ok=True)

model = DuelingDQN(WINDOW_SIZE, ACTION_DIM).to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
print("D3QN model loaded.")
model.eval()

env = PipelineEnv(DATA_ROOT, WINDOW_SIZE)

y_true = []
y_pred = []
file_names = []
confidence_list = []

with torch.no_grad():
    for i in range(len(env.files)):
        state = env.reset(randset=False, i_file=i)
        file_path, label = env.files[i]
        done = False
        q_sum = np.zeros(ACTION_DIM)
        count = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q = model(state_tensor).cpu().numpy().squeeze()
            action = model(state_tensor).argmax().item()

            q_sum += q
            count += 1

            next_state, _, done = env.step(action)
            state = next_state

        avg_q = q_sum / count
        pred = int(np.argmax(avg_q))

        confidence = float(np.max(avg_q))
        y_true.append(label)
        y_pred.append(pred)
        confidence_list.append(confidence)
        file_names.append(os.path.basename(file_path))

label_names = ["noLeak", "smallLeak", "bigLeak"]

acc = accuracy_score(y_true, y_pred)
print("\nOverall Accuracy:", acc)
print(classification_report(y_true, y_pred, target_names=label_names))
df_pred = pd.DataFrame({"file": file_names, "true_label": y_true, "pred_label": y_pred, "confidence": confidence_list})
df_pred.to_csv(os.path.join(RESULT_DIR, "d3qn_predictions.csv"), index=False)
print("Prediction CSV saved.")

report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv(os.path.join(RESULT_DIR, "d3qn_classification_report.csv"))
print("Report CSV saved.")

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
fig, ax = plt.subplots()
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title('(f)Confusion Matrix - D3QN', fontsize=16, y=-0.24, fontweight=500)
ax.set_xlabel('Predicted label', fontsize=12)
ax.set_ylabel('True label', fontsize=12)
ax.tick_params(axis='both', labelsize=11)
for text in disp.text_.ravel():
    text.set_fontsize(13)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix_D3QN.png"), dpi=300)
plt.show()
print("Confusion matrix saved.")
