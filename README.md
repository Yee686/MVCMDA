# ```DELMDA``` Deep Graph Embedding Learning For MiRNA-Drug Association Prediction
# ```DELMDA``` 基于图深度学习的MiRNA-Drug关联预测研究
## Author: Yi ZhengYe
## Data: 2023-03 ~ 2023-05
## Modified: 2023-04-22

### Requirement

- python                    3.8.10
- torch-geometric           1.7.0
- torch-sparse              0.6.9
- torch-cluster             1.5.9
- torch-scatter             2.0.6
- pytorch                   1.8.1           py3.8_cuda10.2_cudnn7.6.5_0
- numpy                     1.21.2          
- scipy                     1.6.3
- networkx                  2.5.1
- tensorflow                2.2.0
- networkx                  2.8
- [GraphEmbedding](https://github.com/shenweichen/GraphEmbedding)

### File structure

- ./datasets/ :oringinal data
- ./Prediction/ :raw prediction data
  - ./Prediction/Compare/ :compare among GNNs
  - ./Prediction/Nimc/ :others
- ./NIMCcode/ :code of GNNs model and training
  - ./NIMCcode/main_cv_compare.py :code of cv all GNNs model
  - ./NIMCcode/main_cv.py : code of cv single GNNs model
  - ./NIMCcode/main_parallel_cv.py : code of cv in multi GPUs
  - ./NIMCcode/model_gat.py : code of GAT model
  - ./NIMCcode/model_gcn.py : code of GCN model
  - ./NIMCcode/model_gin.py : code of GiN model
  - ./NIMCcode/model_SAGE.py : code of GraphSAGE model
  - ./NIMCcode/model_attengcn.py : code of attention GCN model
- ./embedding_cv_element.ipynb :code of struc learning, element-wise 
- ./embedding_cv_row_col.ipynb :code of struc learning, row-col-wise
- ./embedding.ipynb :code of struc learning demo
- ./new_dataset.pt  :preprocessed data(2023.4.22)
- ./datacheck.ipynb :check dataset
- ./metric.ipynb :visualization of metric & result

### Device

- GPU: Nvidia GTX1080ti * 2
- CPU: Intel(R) Xeon(R) Gold 6146 CPU @ 3.20GHz
- RAM: 128G

### Experiment Result

| Model | AUC | AUPR |
| :---: | :---: | :---: |
| GraphSAGE + WMSE | 0.8015299048068565 | 0.027891972920350793 |
| GraphSAGE + CONSTRACTIVE | 0.7976784627831176 | 0.027981549010521998 |
| GraphSAGE + CONSTRACTIVE | 0.7942976959763675 | 0.02794336895198995 |
