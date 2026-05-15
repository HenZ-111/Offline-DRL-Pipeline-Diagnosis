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
 - ### dqn.py
本文件定义了DQN模型中使用到的网络结构。  
 - ### env.py
强化学习中的环境（Environment）：  
1. LABEL_MAP定义数据标签和数值编码的映射关系。因此如果要使用自己的数据集需要进行修改。  
    
2. 本项目实验时使用的环境PipelineEnv。  
     - 初始化时可以输入时间窗口以控制状态的长度。在本模型的框架中，状态是时序数据中连续的一段离散点值。  
     - 类别中的初始化部分，可以自定义时间窗口滑动的步长，如果需要更改则需要在此处调整step_size的数值。  
     - 本环境仅对xlsx文件进行了数据读取的兼容，如果需要使用其他格式，请修改_load_files。  
     - step方法完成对奖励函数的定义，本项目在实验时采用了差异化的奖励函数以权衡预测错误的不同代价。可以自适应调整此部分，以实现自己想要的训练效果。  
 - ### replay_buffer.py  
    - 初始化时可以输入所希望的经验池容量，超过容量时自动删除最老的经验。
    - 从push的输入可以窥见一条经验具体是什么，done定义了“游戏”是否结束，当done为0时，说明游戏未结束。  
    - sample用来采样，可以输入需要的经验数量，即批大小（batch_size），len方法可以输出此时经验池有多少经验。  
 - ### DQN_train.py
1. evaluate_policy方法使用测试集对每回合训练完的模型进行一个评估，从而绘制评估曲线。

     - 需要输入环境、模型和device，device用于适配cpu或者gpu，通过修改输入的环境可以更改测试的数据集。
     - 输出是数据集上的平均return，表示平均一个样本（一局游戏）获得的总rewards，本项目命名为eval reward。
2. 超参数的设置：

     - WINDOW_SIZE --------- 时间窗口，样本上读取的状态长度
     - ACTION_DIM ---------- 动作空间维度，在此实验中是3，因为只有三个动作，用标量0、1、2表示。
     - LR ------------------ 学习率
     - GAMMA --------------- 折扣因子，用以权衡后续选择的影响
     - BATCH_SIZE ---------- 单回合抽取经验的数量
     - MEMORY_SIZE --------- 经验池容量、
     - EPISODES ------------ 运行一次本代码，经历的回合数
     - EPSILON_START ------- 初始探索率
     - EPSILON_END --------- 最低探索率，保证Agent有更加包容、开放的视角
     - EPSILON_DECAY ------- 探索率衰减因子，可以使得训练后期趋于“贪婪”，使训练更加稳定
     - TARGET_UPDATE_FREQ -- 更新目标网络的频率，单位是回合数
3. 具体的训练流程：

     - 初始化部分主要由三部分组成:模型初始化（在env的初始化更改使用的数据集路径）、试加载上次训练结果（model_path和代码底部保存模型相互呼应）、准备训练日志。 
     - 训练阶段优先与环境交互形成经验放入经验池，经验池达到一回合训练所需经验数量则开始训练。训练一回合以后计算eval reward，并写训练日志。
     - 训练结束后将训练的模型保存至当前文件夹下的dqn_models文件夹内，如果没有则会自动创建。
 - ### test_dqn.py
     - 初始化时设置了需要的超参数（要与训练时一致）、测试时使用的路径，之后进行模型加载和环境初始化。
     - 具体的测试过程比较简单，懂了怎么训练的，看看代码就懂了是如何测试的。
     - 测试结束后先输出一段文字报告，之后会保存三个结果文件到初始化定义的结果文件夹中。
         - dqn_test_predictions.csv保存了数据集每个样本的预测结果，并且附上真实结果进行对比。
         - dqn_classification_report.csv保存的是预测结果的总体报告。
         - confusion_matrix_DQN.png如其名保存的是测试结果混淆矩阵。

***

### D3QN模型
├─ dqn.py           # Defines the DQN network structure  
├─ env.py           # RL environment for pipeline data  
├─ replay_buffer.py # Experience replay buffer  
├─ DQN_train.py     # Training script for DQN  
├─ test_dqn.py      # Testing script for DQN  
 - ### dqn.py
本文件定义了DQN模型中使用到的网络结构。  
 - ### env.py
强化学习中的环境（Environment）：  
1. LABEL_MAP定义数据标签和数值编码的映射关系。因此如果要使用自己的数据集需要进行修改。  
    
2. 本项目实验时使用的环境PipelineEnv。  
     - 初始化时可以输入时间窗口以控制状态的长度。在本模型的框架中，状态是时序数据中连续的一段离散点值。  
     - 类别中的初始化部分，可以自定义时间窗口滑动的步长，如果需要更改则需要在此处调整step_size的数值。  
     - 本环境仅对xlsx文件进行了数据读取的兼容，如果需要使用其他格式，请修改_load_files。  
     - step方法完成对奖励函数的定义，本项目在实验时采用了差异化的奖励函数以权衡预测错误的不同代价。可以自适应调整此部分，以实现自己想要的训练效果。  
 - ### replay_buffer.py  
    - 初始化时可以输入所希望的经验池容量，超过容量时自动删除最老的经验。
    - 从push的输入可以窥见一条经验具体是什么，done定义了“游戏”是否结束，当done为0时，说明游戏未结束。  
    - sample用来采样，可以输入需要的经验数量，即批大小（batch_size），len方法可以输出此时经验池有多少经验。
 - ### DQN_train.py
1. evaluate_policy方法使用测试集对每回合训练完的模型进行一个评估，从而绘制评估曲线。

     - 需要输入环境、模型和device，device用于适配cpu或者gpu，通过修改输入的环境可以更改测试的数据集。
     - 输出是数据集上的平均return，表示平均一个样本（一局游戏）获得的总rewards，本项目命名为eval reward。
2. 超参数的设置：

     - WINDOW_SIZE --------- 时间窗口，样本上读取的状态长度
     - ACTION_DIM ---------- 动作空间维度，在此实验中是3，因为只有三个动作，用标量0、1、2表示。
     - LR ------------------ 学习率
     - GAMMA --------------- 折扣因子，用以权衡后续选择的影响
     - BATCH_SIZE ---------- 单回合抽取经验的数量
     - MEMORY_SIZE --------- 经验池容量、
     - EPISODES ------------ 运行一次本代码，经历的回合数
     - EPSILON_START ------- 初始探索率
     - EPSILON_END --------- 最低探索率，保证Agent有更加包容、开放的视角
     - EPSILON_DECAY ------- 探索率衰减因子，可以使得训练后期趋于“贪婪”，使训练更加稳定
     - TARGET_UPDATE_FREQ -- 更新目标网络的频率，单位是回合数
3. 具体的训练流程：

     - 初始化部分主要由三部分组成:模型初始化（在env的初始化更改使用的数据集路径）、试加载上次训练结果（model_path和代码底部保存模型相互呼应）、准备训练日志。 
     - 训练阶段优先与环境交互形成经验放入经验池，经验池达到一回合训练所需经验数量则开始训练。训练一回合以后计算eval reward，并写训练日志。
     - 训练结束后将训练的模型保存至当前文件夹下的dqn_models文件夹内，如果没有则会自动创建。
 - ### test_dqn.py
     - 初始化时设置了需要的超参数（要与训练时一致）、测试时使用的路径，之后进行模型加载和环境初始化。
     - 具体的测试过程比较简单，懂了怎么训练的，看看代码就懂了是如何测试的。
     - 测试结束后先输出一段文字报告，之后会保存三个结果文件到初始化定义的结果文件夹中。
         - dqn_test_predictions.csv保存了数据集每个样本的预测结果，并且附上真实结果进行对比。
         - dqn_classification_report.csv保存的是预测结果的总体报告。
         - confusion_matrix_DQN.png如其名保存的是测试结果混淆矩阵。


## Notes 👀





## Citation 💞




## License 📋


## Contact 📬
邮箱: h626242105@hotmail.com

