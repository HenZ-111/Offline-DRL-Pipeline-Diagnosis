import os
import numpy as np
import pandas as pd

LABEL_MAP = {
    "noLeak": 0,
    "smallLeak": 1,
    "bigLeak": 2
}

class PipelineEnv:
    def __init__(self, data_root, window_size=128):
        self.data_root = data_root
        self.window_size = window_size
        self.files = []
        self._load_files()
        self.step_size = 4

    def _load_files(self):
        for label_name, label in LABEL_MAP.items():
            folder = os.path.join(self.data_root, label_name)
            for file in os.listdir(folder):
                if file.endswith(".xlsx"):
                    self.files.append((os.path.join(folder, file), label))

    def reset(self, randset=True, i_file=0):
        if randset is True:
            self.current_file, self.label = self.files[np.random.randint(len(self.files))]
        else:
            self.current_file, self.label = self.files[i_file]
        signal = pd.read_excel(self.current_file, header=None).values.squeeze()
        self.signal = signal
        self.ptr = 0
        return self.signal[self.ptr:self.ptr + self.window_size]

    def step(self, action):
        if action == self.label:
            reward = 1
        else:
            if self.label == 0:
                reward = -1
            elif self.label == 1:
                reward = -2
            else:
                reward = -3

        self.ptr += self.step_size
        done = self.ptr + self.window_size >= len(self.signal)

        next_state = self.signal[self.ptr:self.ptr + self.window_size] if not done else None
        return next_state, reward, done
