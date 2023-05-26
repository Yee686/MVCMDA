# MVCMDA
# 用于miRNA-药物关联预测的多通道多层次通道注意力图表示学习模型

### Contribution
1. construct miRNA-Drug association network
2. construct miRNA and Drug multi-view properties graph
   - miRNA: sequence and function
   - Drug: sequence and structure
3. design association network update strategy based neighborhood info
4. design multi-view multi-channel attention graph neural network

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
-- ./datasets/ : oringinal data
  -- ./datasets/multiview_dataset_updatamd.pt : multi-view dataset after update associate network
-- ./MatrixFactorization/ : code of MF model
  -- ./MatrixFactorization/NMF.py : code of MF model
  -- ./MatrixFactorization/NMFwithNIF.py : code of MF using neighborhood info model

-- ./MultiView/ : multi-view network construction
  -- ./MultiView/Mol2Vec.ipynb : get drug mol2vec embedding
  -- ./MultiView/rna.ipynb: get miRNA seq simlarity
  -- ./MultiView/multiview_dataset.ipynb : get multi-view dataset in pt format 

-- ./NIMCcode/ :code of GNNs model and training
  -- ./NIMCcode/main_cv_multiview.py : train code of multiview GNNs model
  -- ./NIMCcode/model_attenGNN_multiview.py : model code of multiview GNNs model

-- ./Embedding/ : graph struc embedding method
  -- ./embedding_cv_element.ipynb :code of struc learning, element--wise 
  -- ./embedding_cv_row_col.ipynb :code of struc learning, row--col--wise

-- ./Tools/ : some tools
  -- ./embedding.ipynb :code of struc learning demo
  -- ./datacheck.ipynb :check dataset
  -- ./metric.ipynb :visualization of metric & result
  -- ./metric_union.ipynb :visualization of all prediction
```

### Device
#### Device 1
- GPU: Nvidia GTX1080ti * 2
- CPU: Intel(R) Xeon(R) Gold 6146 CPU @ 3.20GHz
- RAM: 128G

#### Device 2
- GPU: NVIDIA Quadro RTX 5000
- CPU: 15 vCPU AMD EPYC 7543 32-Core Processor
- RAM: 30GB


### Info
- Author: Yi ZhengYe From HZAU
- Date: 2023-03 ~ 2023-06
- Modified: 2023-05-26