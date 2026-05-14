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
- [方法](##🧠Methodology/方法)
- [功能特点](#功能特点)
- [安装与运行](#安装与运行)
- [使用示例](#使用示例)
- [项目结构](#项目结构)
- [技术细节](#技术细节)
- [贡献](#贡献)
- [许可证](#许可证)
- [联系方式](#联系方式)

---

## 🧠Methodology/方法




## 🔍Preparation/准备




## ⚙️Installation/环境配置





## 🚀Usage/使用方法





## 📁Project Structure/项目结构  
### DQN模型  
####  dqn.py
本文件定义了DQN模型中使用到的网络结构。  
#### env.py
强化学习中的环境（Environment）：  
1. LABEL_MAP定义数据标签和数值编码的映射关系。因此如果要使用自己的数据集需要进行修改。  
    
2. 本项目实验时使用的环境PipelineEnv。  
    1. 初始化时可以输入时间窗口以控制状态的长度。在本模型的框架中，状态是时序数据中连续的一段离散点值。  
    2. 类别中的初始化部分，可以自定义时间窗口滑动的步长，如果需要更改则需要在此处调整step_size的数值。  
    3. 本环境仅对xlsx文件进行了数据读取的兼容，如果需要使用其他格式，请修改_load_files。  
    4. step方法完成对奖励函数的定义，本项目在实验时采用了差异化的奖励函数以权衡预测错误的不同代价。可以自适应调整此部分，以实现自己想要的训练效果。  
####  replay_buffer.py
深度强化学习框架中使用到的经验回放池：  
1. 初始化时可以输入所希望的经验池容量。  
2. 从push的输入可以窥见一条经验具体是什么，done定义了“游戏”是否结束，当done为0时，说明游戏未结束。  
3. sample用来采样，可以输入需要的经验数量，即批大小（batch_size），len方法可以输出此时经验池有多少经验。  
####  DQN_train.py


#### test_dqn.py


### D3QN模型



## ⚠️Notes/注意事项





## 📚Citation/引用




## 📄License/许可证




