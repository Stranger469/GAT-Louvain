# Community Detection with Graph Attention Network

GAT与Louvain进行用户关系选择性增强的社区发现算法。

User relation reinforcement Community detection. The relation between users are calculated by GAT. We use the attentions from the output layer of GAT as relation between users, and reinforce it by weighted delta function.

# Environment

- python 3.8
- numpy 1.19
- pandas 1.1.5
- tensorflow 2.4.0
- networkx 2.5
