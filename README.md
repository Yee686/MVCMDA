# MVCMDA
## Author: Yi ZhengYe
## Data: 2023-03 ~ 2023-05
## Modified: 2023-05-21

### TODO
- [X] multi-view
- [ ] neural architechure search
- [X] constractive learning

### Requirement

- python                    3.8.16
- torch-cluster             1.6.1+pt20cu117          
- torch-geometric           2.3.0                    
- torch-scatter             2.1.1+pt20cu117          
- torch-sparse              0.6.17+pt20cu117          
- torch-spline-conv         1.2.2+pt20cu117          
- pytorch                   2.0.0
- numpy                     1.21.2          
- scipy                     1.9.1
- tensorflow                2.2.0
- networkx                  2.8
- rdkit                     2022.9.5
- [GraphEmbedding](https://github.com/shenweichen/GraphEmbedding)
- [Mol2vec](https://github.com/samoturk/mol2vec)
- gensim                    4.1.2
- biopython                 1.79
- tqdm                      4.65.0

### File structure
```
-- ./datasets/ :oringinal data
-- ./Prediction/ :raw prediction data
  -- ./Prediction/Compare/ :compare among GNNs
  -- ./Prediction/Nimc/ :others
-- ./NIMCcode/ :code of GNNs model and training
  -- ./NIMCcode/main_cv_compare.py :code of cv all GNNs model
  -- ./NIMCcode/main_cv.py : code of cv single GNNs model
  -- ./NIMCcode/main_parallel_cv.py : code of cv in multi GPUs
  -- ./NIMCcode/model_gat.py : code of GAT model
  -- ./NIMCcode/model_gcn.py : code of GCN model
  -- ./NIMCcode/model_gin.py : code of GiN model
  -- ./NIMCcode/model_SAGE.py : code of GraphSAGE model
  -- ./NIMCcode/model_attengcn.py : code of attention GCN model
-- ./embedding_cv_element.ipynb :code of struc learning, element--wise 
-- ./embedding_cv_row_col.ipynb :code of struc learning, row--col--wise
-- ./embedding.ipynb :code of struc learning demo
-- ./new_dataset.pt  :preprocessed data(2023.4.22)
-- ./datacheck.ipynb :check dataset
-- ./metric.ipynb :visualization of metric & result
```

### Device

- GPU: Nvidia GTX1080ti * 2
- CPU: Intel(R) Xeon(R) Gold 6146 CPU @ 3.20GHz
- RAM: 128G