# HARP: Transferable Neural WAN TE for Changing Topologies
[HARP](https://dl.acm.org/doi/10.1145/3651890.3672237) is a transferable neural network for WAN Traffic Engineering that is designed to handle changing topologies. It was published at ACM SIGCOMM 2024.

If you use this code, please cite:
```
@inproceedings{HARP,
author = {AlQiam, Abd AlRhman and Yao, Yuanjun and Wang, Zhaodong and Ahuja, Satyajeet Singh and Zhang, Ying and Rao, Sanjay G. and Ribeiro, Bruno and Tawarmalani, Mohit},
title = {Transferable Neural WAN TE for Changing Topologies},
year = {2024},
isbn = {9798400706141},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3651890.3672237},
doi = {10.1145/3651890.3672237},
booktitle = {Proceedings of the ACM SIGCOMM 2024 Conference},
pages = {86–102},
numpages = {17},
keywords = {traffic engineering, wide-area networks, network optimization, machine learning},
location = {Sydney, NSW, Australia},
series = {ACM SIGCOMM '24}
}
```
Please contact `aalqiam@purdue.edu` for any questions.
### Environment Used
HARP was tested using the following setup:
- Ubuntu 22.04 machine
- Python 3.10.6
- `torch==2.1.0+cu121`
- `torch-scatter==2.1.2`
- Check the rest in requirements.txt
### Required Libraries
1. Install the required Python packages as listed in the requirements.txt. Use:
   `pip3 install -r requirements.txt`
2. Please follow this [link](https://pytorch.org/get-started/locally/) to install a version of PyTorch that fits your environment (CPU/GPU).
3. Identify and copy the link of a suitable [URL](https://data.pyg.org/whl/) depending on PyTorch and CUDA/CPU versions installed in the previous step. Then, run:
   - `pip install --no-index torch-scatter -f [URL]`
4. Follow [Gurobi Website](https://www.gurobi.com/) to install and setup Gurobi Optimizer.
      
### How to Use HARP
- In the `manifest` folder, The user should provide a `txt` file that holds the topology name and describes at every time step the **topology_file.json**,**set_of_pairs_file.pkl**,**traffic_matrix.pkl** file that will be read at that time step. For every timestep, a corresponding file of these three should exist in the `topologies`, `pairs`, and `traffic_matrices` folders inside a directory with the topology name. 
- For details on the data format, please check [Data Format](#data-format)
- To compute optimal values and cluterize your dataset, run:
   - ``python3 frameworks/gurobi_mlu.py --num_paths_per_pair 15 --opt_start_idx 0 --opt_end_idx 2000 --topo topo_name --framework gurobi``
   - Please refer to our paper to check the definition of a "cluster" in this context.

 - To train, run (for example):
   - ``python3 run_harp.py --topo topo_name --mode train --epochs 100 --batch_size 32 --lr 0.001 --num_paths_per_pair 8 --num_transformer_layers 2 --num_gnn_layers 3 --num_mlp1_hidden_layers 2 --num_mlp2_hidden_layers 2 --num_for_loops 3 --train_clusters 0 1 2 3 --train_start_indices 0 0 0 0 --train_end_indices 200 200 200 200 --val_clusters 4 5 --val_start_indices 0 0 --val_end_indices 90 90 --framework harp --pred 0 --dynamic 1``
   - ``python3 run_harp.py --topo abilene --mode train --epochs 100 --lr 0.007 --batch_size 32 --num_paths_per_pair 8 --num_transformer_layers 2 --num_gnn_layers 3 --num_mlp1_hidden_layers 1 --num_mlp2_hidden_layers 1 --num_for_loops 3  --train_clusters 0 --train_start_indices 0 --train_end_indices 12096 --val_clusters 0 --val_start_indices 12096 --val_end_indices 14112 --framework harp --pred 0 --dynamic 0``
   - ``python3 run_harp.py --topo kdl --mode train --epochs 100 --lr 0.007 --batch_size 8 --num_paths_per_pair 4 --num_transformer_layers 1 --num_gnn_layers 1 --num_mlp1_hidden_layers 1 --num_mlp2_hidden_layers 1 --num_for_loops 3  --train_clusters 0 --train_start_indices 0 --train_end_indices 170 --val_clusters 0 --val_start_indices 170 --val_end_indices 200 --framework harp --pred 0 --dynamic 0``

- To test, run (for example):
   - ``python3 run_harp.py --topo topo_name --mode test --num_paths_per_pair 15 --num_for_loops 14  --test_cluster 6 --test_start_idx 0 --test_end_idx 150 --framework harp --pred 0 --dynamic 1``
   - ``python3 run_harp.py --topo abilene --mode test --num_paths_per_pair 8 --num_for_loops 3  --test_cluster 0 --test_start_idx 14112 --test_end_idx 16128 --framework harp --pred 0 --dynamic 0``
   - ``python3 run_harp.py --topo kdl --mode test --num_paths_per_pair 8 --num_for_loops 3  --test_cluster 0 --test_start_idx 200 --test_end_idx 278 --framework harp --pred 0 --dynamic 0``
   - Note that only one cluster is allowed per testing mode run.
- For further explanation on command line arguments, see [Command Line Arguments Explanation](#command-line-arguments-explanation)

### Working with Public Datasets (Abilene and GEANT):
- Download `AbileneTM-all.tar` from this [link](https://www.cs.utexas.edu/~yzhang/research/AbileneTM/) and decompress it (twice) inside ``prepare_abilene`` folder.
   - `cd prepare_abilene`
   - `wget https://www.cs.utexas.edu/~yzhang/research/AbileneTM/AbileneTM-all.tar`
   - `tar -xvf AbileneTM-all.tar`
   - `gunzip *.gz`
   - Then, run ``python3 prepare_abilene_harp.py``
   - This example should serve as a reference on how to prepare any dataset.
- A preprocessed copy of the GEANT dataset in the format needed by HARP is available on this [link](https://app.box.com/s/shzgaxnt36org6dmu9q228kzk28numue)
   - Update: 09/30/2024: GEANT matrices were scaled down to have the same unit as capacities.
- A preprocessed copy of the KDL dataset in the format needed by HARP is available on this [link](https://drive.google.com/file/d/1dZLhpgJ1ZcjzsZsPZwbTr0t2_Fkz_8oP/view)

### Working with Predicted Matrices
- By default, HARP trains over ground truth matrices.
- Running HARP with ``--pred 1`` trains it over predicted matrices rather than ground truth matrices.
- Provide the predicted traffic matrices using a predictor of your choice, then put them inside the `traffic_matrices` directory inside a folder named `topo_name_pred`.
  - For example, for the GEANT dataset, original matrices will be under the `GEANT` directory whereas predicted matrices will be under the `GEANT_pred` directory.
  - Make sure that at every time step, the predicted matrix corresponds to the ground truth matrix at that time step.
     - For example: t100.pkl in the `GEANT` and the `GEANT_pred` folders correspond to each other.

## Reproduce Single-link Failure Experiments on Abilene and GEANT
- After training HARP model on GEANT and Abilene, run:
  - ``python3 run_failures.py --topo geant --num_paths_per_pair 8 --num_for_loops X --test_start_idx start --test_end_idx end --pred 0 --test_cluster 0``
  - ``python3 run_failures.py --topo abilene --num_paths_per_pair 8 --num_for_loops X --test_start_idx start --test_end_idx end --pred 0 --test_cluster 0``
- This will compute the optimal for each failure scenario, and then run HARP for that scenario.

### Data Format:
- **Traffic matrices**: Numpy array of shape (num_pairs, 1)
- **Pairs**: Numpy array of shape (num_pairs, 2)
- Note: the kth demand in the traffic matrix must correspond to the kth pair in the set of pairs file. This relation must be preserved for all snapshots. **We suggest sorting the hash map (pairs/keys and values/demands) before separating**.
- **Paths**: By default, HARP computes K shortest paths and automatically puts them in the correct folders and format.
   - If you wish to use your paths:
     - create a Python dictionary where the keys are the pairs and the values are a list of $K$ lists, where the inner lists are a sequence of edges.
     - For example: {(s, t): [[(s, a), (a, t)], [(s, a), (a, b), (b, t)]]}.
     - Put it inside `topologies/paths_dict` and name it: *topo_name*\_*K*\_paths_dict_cluster_*num_cluster*.pkl
        - For example: abilene_8_paths_dict_cluster_0.pkl
     - Make sure all pairs have the same number of paths (replicate if needed).

### Command Line Arguments Explanation
- framework: Determines the framework that solves the problem [harp, gurobi].
- num_heads: Number of transformer attention heads [int]. By default, it is equal to the number of GNN layers.
- num_for_loops: Determines HARP's Number of RAUs.
- dynamic: If your topology varies across snapshots, set it to `1`. If it is static, set it to `0`.
   - In our paper, the AnonNet network is dynamic.
   - GEANT, Abilene, and KDL networks are static.
   - **This CLA is useful to save GPU memory when training for a (static) topology that does not change across snapshots**.
- dtype: Determines the `dtype` of HARP and its data [float32, float16] corresponding to [torch.float32, torch.bfloat16]. The default is float32.
