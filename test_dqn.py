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
from dqn import DQN

DATA_ROOT = "TEST_DATA"
WINDOW_SIZE = 128
ACTION_DIM = 3
MODEL_PATH = "dqn_models/dqn_pipeline.pth"
RESULT_DIR = "test_dqn_results"
os.makedirs(RESULT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(WINDOW_SIZE, ACTION_DIM).to(device)
ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["model"])
print("Model loaded successfully.")
model.eval()

env = PipelineEnv(DATA_ROOT, WINDOW_SIZE)

y_true = []
y_pred = []
file_names = []

with torch.no_grad():
    for i in range(len(env.files)):
        state = env.reset(randset=False, i_file=i)
        file_path, label = env.files[i]
        done = False
        q_sum = np.zeros(ACTION_DIM)
        count = 0

        while not done:
            state_tensor = torch.FloatTensor(state).to(device)
            q_values = model(state_tensor).cpu().numpy().squeeze()
            action = model(state_tensor).argmax().item()

            q_sum += q_values
            count += 1

            next_state, _, done = env.step(action)
            state = next_state

        avg_q = q_sum / count
        pred_label = int(np.argmax(avg_q))

        y_true.append(label)
        y_pred.append(pred_label)
        file_names.append(os.path.basename(file_path))

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

df_pred = pd.DataFrame({
    "file": file_names,
    "true_label": y_true,
    "pred_label": y_pred
})
df_pred.to_csv(os.path.join(RESULT_DIR, "dqn_test_predictions.csv"), index=False)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv(os.path.join(RESULT_DIR, "dqn_classification_report.csv"))
print("CSV results saved.")

labels = ["noLeak", "smallLeak", "bigLeak"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots()
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title('(d)Confusion Matrix - DQN', fontsize=16, y=-0.24, fontweight=500)
ax.set_xlabel('Predicted label', fontsize=12)
ax.set_ylabel('True label', fontsize=12)
ax.tick_params(axis='both', labelsize=11)
for text in disp.text_.ravel():
    text.set_fontsize(13)
plt.tight_layout()
plt.savefig("confusion_matrix_DQN.png", dpi=300)
plt.show()
print("Confusion matrix saved.")
