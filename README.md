# SeqDRL-Detection
This project proposes a **deep reinforcement learning framework** grounded in **sequential decision-making**, leveraging **DQN** and **D3QN** models for training and inference on one-dimensional time-series data.   
Experiments on a simulated pipeline leak dataset show that the framework achieves strong classification performance in **low-data scenarios** and is readily extendable to other **anomaly detection** applications.  

本项目提出了一种基于**序贯决策**的**深度强化学习框架**，并在此框架上使用**DQN模型**、**D3QN模型**于一维时序数据上完成训练、推理。  
经过本人在学校提供的模拟实验中获取的管道泄漏数据集上进行的实验，  
可以证明该框架在**小样本条件**下有较强的分类性能，并可推广应用于其他**异常检测**任务。
***

![License](https://img.shields.io/badge/license-MIT-lightgrey)
---

## 目录
- [方法](#Methodology-)
- [准备](#Preparation-)
- [环境配置](#Installation-)
- [使用方法](#Usage-)
- [项目结构](#Project-Structure-)
- [注意事项](#Notes-)
- [引用](#Citation-)
- [许可证](#License-)
- [联系方式](#Contact-)

---

## Methodology 🧠




## Preparation 💻




## Installation 🛠️





## Usage 🎯





## Project Structure 📁
### DQN模型  
├─ dqn.py           # Defines the DQN network structure  
├─ env.py           # RL environment for pipeline data  
├─ replay_buffer.py # Experience replay buffer  
├─ DQN_train.py     # Training script for DQN  
├─ test_dqn.py      # Testing script for DQN  
 - #### dqn.py
本文件定义了DQN模型中使用到的网络结构。  
 - #### env.py
强化学习中的环境（Environment）：  
1. LABEL_MAP定义数据标签和数值编码的映射关系。因此如果要使用自己的数据集需要进行修改。  
    
2. 本项目实验时使用的环境PipelineEnv。  
     - 初始化时可以输入时间窗口以控制状态的长度。在本模型的框架中，状态是时序数据中连续的一段离散点值。  
     - 类别中的初始化部分，可以自定义时间窗口滑动的步长，如果需要更改则需要在此处调整step_size的数值。  
     - 本环境仅对xlsx文件进行了数据读取的兼容，如果需要使用其他格式，请修改_load_files。  
     - step方法完成对奖励函数的定义，本项目在实验时采用了差异化的奖励函数以权衡预测错误的不同代价。可以自适应调整此部分，以实现自己想要的训练效果。  
 - #### replay_buffer.py  
    - 初始化时可以输入所希望的经验池容量。
    - 从push的输入可以窥见一条经验具体是什么，done定义了“游戏”是否结束，当done为0时，说明游戏未结束。  
    - sample用来采样，可以输入需要的经验数量，即批大小（batch_size），len方法可以输出此时经验池有多少经验。  
 - #### DQN_train.py
1. evaluate_policy方法使用测试集对每回合训练完的模型进行一个评估，从而绘制评估曲线。
     - 需要输入环境、模型，device用于适配cpu或者gpu，通过修改输入的环境可以更改测试的数据集。
     - 输出是数据集上的平均return，表示平均一个样本（一局游戏）获得的总rewards。
2. 超参数的设置：
     - WINDOW_SIZE -------- 时间窗口，样本上读取的状态长度
     - ACTION_DIM --------- 动作空间维度，在此实验中是3，因为只有三个动作，用标量0、1、2表示。
     - LR ----------------- 学习率
     - GAMMA -------------- 折扣因子，用以权衡后续选择的影响
     - BATCH_SIZE --------- 单回合抽取经验的数量
     - MEMORY_SIZE -------- 经验池容量、
     - EPISODES ----------- 运行一次本代码，经历的回合数
     - EPSILON_START ------ 初始探索率
     - EPSILON_END -------- 最低探索率，保证Agent有更加包容、开放的视角
     - EPSILON_DECAY ------ 探索率衰减因子，可以使得训练后期趋于“贪婪”，使训练更加稳定
     - TARGET_UPDATE_FREQ
3. 
 - #### test_dqn.py


### D3QN模型



## Notes 👀





## Citation 💞




## License 📋


## Contact 📬
邮箱: h626242105@hotmail.com

