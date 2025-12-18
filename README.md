![logo](./docs/_static/logo2.0.png)
---

![PyPI - Python Version](https://img.shields.io/badge/pyhton-3.10-blue)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
![GitHub repo size](https://img.shields.io/github/repo-size/THUwangcy/ReChorus)
[![arXiv](https://img.shields.io/badge/arXiv-ReChorus-%23B21B1B)](https://arxiv.org/abs/2405.18058)

# ReChorus - CoGCL 模型复现版本

本项目基于 [ReChorus 2.0](https://github.com/THUwangcy/ReChorus) 框架，成功复现了 **CoGCL (Contrastive Graph Collaborative Learning)** 模型。

> 原始论文: [Enhancing Graph Contrastive Learning with Reliable and Informative Augmentation for Recommendation](https://dl.acm.org/doi/10.1145/3690624.3709214)

## 项目简介

CoGCL 是一种基于图对比学习的推荐算法，旨在解决用户行为数据的稀疏性问题。本项目将 CoGCL 从 RecBole 框架完整迁移到了 ReChorus 框架中，并实现了以下核心特性：

*   **向量量化**: 引入端到端的残差量化机制，学习具有强协同信号的离散编码。
*   **智能图增强**: 基于学习到的离散编码，生成可靠的“虚拟邻居”，而非随机扰动。
*   **多视图对比学习**: 结合结构视图（图增强）和语义视图（语义相关性采样）进行联合优化。


## 实验结果

我们在 **Grocery_and_Gourmet_Food** 和 **MovieLens-1M**两个数据集上进行了对比实验。


### 性能对比 

**Grocery_and_Gourmet_Food 数据集**

| 模型        | HR@5       | NDCG@5     | HR@10      | NDCG@10    | HR@20      | NDCG@20    |
| ----------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| BPRMF       | 0.3191     | 0.2195     | 0.4222     | 0.2529     | 0.5304     | 0.2801     |
| LightGCN    | 0.3710     | 0.2566     | 0.4925     | 0.2961     | 0.6132     | 0.3266     |
| CoGCL| **0.4000** | **0.2799** | **0.5158** | **0.3175** | **0.6281** | **0.3459** |

**MovieLens-1M 数据集**

| 模型        | HR@5       | NDCG@5     | HR@10      | NDCG@10    | HR@20      | NDCG@20    |
| ----------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| **BPRMF**   | **0.3764** | 0.2515     | **0.5516** | 0.3081     | 0.7146     | 0.3422     |
| LightGCN    | 0.3702     | 0.2477     | 0.5356     | 0.3008     | 0.6887     | 0.3409     |
| CoGCL | 0.3761     | **0.2580** | 0.5495     | **0.3138** | **0.7400** | **0.3621** |


## 快速开始

### 1. 环境准备
确保安装 PyTorch 和 PyTorch Geometric。
```bash
pip install -r requirements.txt
# 额外依赖
pip install torch-geometric
```

### 2. 数据准备
使用本项目提供的脚本预处理 MovieLens-1M 数据：
```bash
python data/MovieLens-1M/prepare_movielens.py
```

### 3. 实验运行指南

#### 3.1 超参数搜索
本项目使用 `Optuna` 框架进行自动化超参数搜索。

**Grocery_and_Gourmet_Food**:
```bash
python run_optuna.py \
    --model_name CoGCL \
    --dataset Grocery_and_Gourmet_Food \
    --n_trials 100 \
    --epoch 100 \
    --gpu 0 
```

**MovieLens-1M**:
```bash
python run_optuna.py \
    --model_name CoGCL \
    --dataset MovieLens_1M \
    --n_trials 100 \
    --epoch 100 \
    --gpu 0 
```

#### 3.2 复现最佳结果
基于我们的搜索结果，使用以下命令可直接复现报告中的性能。

**Grocery_and_Gourmet_Food**:
```bash
python src/main.py --model_name CoGCL --dataset Grocery_and_Gourmet_Food \
      --vq_loss_weight 0.008 \
      --cl_weight 0.03 \
      --sim_cl_weight 0.5 \
      --graph_replace_p 0.42 \
      --graph_add_p 0.44 \
      --drop_p 0.05 \
      --n_layers 1 \
      --lr 0.0033 \
      --l2 4e-6 \
      --cl_tau 0.47
```

**MovieLens-1M**:
```bash
python src/main.py --model_name CoGCL --dataset MovieLens_1M \
      --vq_loss_weight 0.00179 \
      --cl_weight 0.0130 \
      --sim_cl_weight 0.877 \
      --graph_replace_p 0.0735 \
      --graph_add_p 0.378 \
      --drop_p 0.455 \
      --n_layers 4 \
      --lr 0.00101 \
      --l2 1.14e-07 \
      --cl_tau 0.270
```

#### 3.3 消融实验
可以通过将相关权重或概率设为 0 来进行消融实验。例如，移除**相似性对比学习 (SimCL)**:
```bash
python src/main.py --model_name CoGCL ... --sim_cl_weight 0
```
移除**图增强**:
```bash
python src/main.py --model_name CoGCL ... --graph_replace_p 0 --graph_add_p 0
```


## 原始 ReChorus 引用

ReChorus 是一个模块化、任务灵活的推荐算法库。

```
@inproceedings{li2024rechorus2,
  title={ReChorus2. 0: A Modular and Task-Flexible Recommendation Library},
  author={Li, Jiayu and Li, Hanyu and He, Zhiyu and Ma, Weizhi and Sun, Peijie and Zhang, Min and Ma, Shaoping},
  booktitle={Proceedings of the 18th ACM Conference on Recommender Systems},
  pages={454--464},
  year={2024}
}
```

更多关于 ReChorus 的文档请参考 [Wiki](https://github.com/THUwangcy/ReChorus/tree/master/docs)。
