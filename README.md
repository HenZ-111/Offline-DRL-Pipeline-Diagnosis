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
- [联系方式](#Contact-)

---

## Methodology 🧠




## Preparation 💻




## Installation 🛠️





## Usage 🎯





## Project Structure 📁
### DQN模型  
├─ dqn.py ------------ # Defines the DQN network structure  
├─ env.py ------------ # RL environment for pipeline data  
├─ replay_buffer.py -- # Experience replay buffer  
├─ DQN_train.py ------ # Training script for DQN  
├─ test_dqn.py ------- # Testing script for DQN  
 - ### dqn.py
&nbsp;&nbsp;本文件定义了DQN模型中使用到的网络结构。  
 - ### env.py
&nbsp;&nbsp;强化学习中的环境（Environment）：

&nbsp;&nbsp;1. LABEL_MAP定义数据标签和数值编码的映射关系。因此如果要使用自己的数据集需要进行修改。
    
&nbsp;&nbsp;2. 本项目实验时使用的环境PipelineEnv。  
      - 初始化时可以输入时间窗口以控制状态的长度。在本模型的框架中，状态是时序数据中连续的一段离散点值。  
      - 类别中的初始化部分，可以自定义时间窗口滑动的步长，如果需要更改则需要在此处调整step_size的数值。  
      - 本环境仅对xlsx文件进行了数据读取的兼容，如果需要使用其他格式，请修改_load_files。  
      - step方法完成对奖励函数的定义，本项目在实验时采用了差异化的奖励函数以权衡预测错误的不同代价。可以自适应调整此部分，以实现自己想要的训练效果。  
 - ### replay_buffer.py  
    - 初始化时可以输入所希望的经验池容量，超过容量时自动删除最老的经验。
    - 从push的输入可以窥见一条经验具体是什么，done定义了“游戏”是否结束，当done为0时，说明游戏未结束。  
    - sample用来采样，可以输入需要的经验数量，即批大小（batch_size），len方法可以输出此时经验池有多少经验。  
 - ### DQN_train.py
&nbsp;&nbsp;1. evaluate_policy方法使用测试集对每回合训练完的模型进行一个评估，从而绘制评估曲线。

     - 需要输入环境、模型和device，device用于适配cpu或者gpu，通过修改输入的环境可以更改测试的数据集。
     - 输出是数据集上的平均return，表示平均一个样本（一局游戏）获得的总rewards，本项目命名为eval reward。
&nbsp;&nbsp;2. 超参数的设置：

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
&nbsp;&nbsp;3. 具体的训练流程：

     - 初始化部分主要由三部分组成:模型初始化（在env的初始化更改使用的数据集路径）、试加载上次训练结果（model_path和代码底部保存模型相互呼应）、准备训练日志。 
     - 训练阶段优先与环境交互形成经验放入经验池，经验池达到一回合训练所需经验数量则开始训练。一局游戏之后计算eval reward，并写训练日志。
     - 训练结束后将训练的模型保存至当前文件夹下的dqn_models文件夹内，如果没有则会自动创建；训练日志最终保存在当前文件下的d3qn_train_log.csv。
 - ### test_dqn.py
     - 初始化时设置了需要的超参数（要与训练时一致）、测试时使用的路径，之后进行模型加载和环境初始化。
     - 具体的测试过程比较简单，懂了怎么训练的，看看代码就懂了是如何测试的。
     - 测试结束后先输出一段文字报告，之后会保存三个结果文件到初始化定义的结果文件夹中。
         - dqn_test_predictions.csv保存了数据集每个样本的预测结果，并且附上真实结果进行对比。
         - dqn_classification_report.csv保存的是预测结果的总体报告。
         - confusion_matrix_DQN.png如其名保存的是测试结果混淆矩阵。

---

### D3QN模型
├─ Dueling_dqn.py ---------------- # Defines the Dueling DQN network structure  
├─ env.py ------------------------ # RL environment for pipeline data  
├─ priortized_replay_buffer.py --- # Priortized experience replay buffer  
├─ D3QN_train.py ----------------- # Training script for D3QN  
├─ test_d3qn.py ------------------ # Testing script for D3QN  
 - ### d3qn.py
&nbsp;&nbsp;本文件定义了D3QN模型中使用到的Dueling DQN结构。  
 - ### env.py
&nbsp;&nbsp;强化学习中的环境（Environment）,与DQN使用相同的环境。    
 - ### priortized_replay_buffer.py  
    - 初始化在原先的经验回放池上新添了alpha参数，用以控制优先采样偏置，偏置强度越高，则越趋向于采样模型认为更值得学习的经验来学习。
    - 由于引入了优先级机制，此时的push变得很不一样：
        - 首先为了让经验池的拓展性更高，buffer不再使用预先定义的库，而是采用列表的方式定义。后续可以定义经验池的溢出处理方式等。
        - 经验中加入label，用于训练时记录TD-error。
        - 新放入的优先级基于当前经验池中的最大优先级赋予其优先级，使Agent趋向于学习新鲜事物。
        - 本实验经验池的溢出处理方式是覆盖最旧的经验，通过pos属性实现。
    - sample用来采样，这里引入了beta参数，可以控制重要性采样修正的强度，以纠正非均匀采样带来的分布偏差。
    - update_priorities用于动态更改经验池的优先级。
 - ### D3QN_train.py
&nbsp;&nbsp;1. evaluate_policy方法使用测试集对每回合训练完的模型进行一个评估，从而绘制评估曲线，与DQN的一模一样，可以考虑放进一个库里。

&nbsp;&nbsp;2. 第一部分超参数与DQN是一样的，第二部分超参数涉及到PER机制，前面也有所提及：

     - ALPHA --------------- 优先采样偏置强度
     - BETA_START ---------- 重要性采样修正的强度
     - BETA_INC ------------ 重要性采样修正强度的增长因子，使训练后期偏向于均衡学习

&nbsp;&nbsp;3. 具体的训练流程：

     - 初始化部分主要由三部分组成:模型初始化（在env的初始化更改使用的数据集路径）、试加载上次训练结果（model_path和代码底部保存模型相互呼应）、准备训练日志。 此外，还有td_error日志的准备工作。
     - 相比DQN，一回合结束后需要记录td-error、更新优先级；此处记录的td-errror为训练使用经验的td-error对应类别的平均值，以衡量训练过程中遇到的经验的学习价值。一局游戏后在更新探索率的同时也更新重要性采样修正强度，之后写训练日志。
     - 训练结束后将训练的模型保存至当前文件夹下的d3qn_models文件夹内，如果没有则会自动创建，同时保存下列两个文件在当前文件夹下：
         - d3qn_train_log.csv，这与DQN类似。
         - td_error_log.csv，记录训练过程的td_error。
 - ### test_d3qn.py
     - 初始化时设置了需要的超参数（要与训练时一致）、测试时使用的路径，之后进行模型加载和环境初始化。
     - 具体的测试过程比较简单，懂了怎么训练的，看看代码就懂了是如何测试的。
     - 测试结束后先输出一段文字报告，之后会保存三个结果文件到初始化定义的结果文件夹中。
         - dqn_test_predictions.csv保存了数据集每个样本的预测结果，并且附上真实结果进行对比。
         - dqn_classification_report.csv保存的是预测结果的总体报告。
         - confusion_matrix_DQN.png如其名保存的是测试结果混淆矩阵。


## Notes 👀
&nbsp;&nbsp;本人实验使用的管道泄漏声波数据集是分布平均的。  
&nbsp;&nbsp;td-error在D3QN的训练过程的相关代码并不是很兼容，需要根据自己的数据集进行相应的修改，主要是跟类别相关。  




## Citation 💞





## Contact 📬
邮箱: h626242105@hotmail.com

