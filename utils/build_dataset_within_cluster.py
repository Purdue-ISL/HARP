from utils.snapshot_utils import Read_Snapshot
from utils.cluster_utils import Cluster_Info
from torch.utils.data import Dataset
import numpy as np
import os
import pickle

class DM_Dataset_within_Cluster(Dataset):
    def __init__(self, props, cluster, start, end):
        self.props = props
        self.cluster = cluster
        self.list_tms = []
        self.list_tms_pred = []
        self.list_capacities = []
        self.list_node_features = []
        self.list_optimal_values = []
        self.parent_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.results_path = f"{self.parent_dir_path}/results/{self.props.topo}/{props.num_paths_per_pair}sp/{self.cluster}"
        filenames = np.loadtxt(f"{self.results_path}/filenames.txt", dtype="U", delimiter=",").reshape(-1, 3)
        filenames = filenames[start:end]
        
        if self.props.failure_id == None:
            file = open(f"{self.results_path}/optimal_values.txt")
            opts = np.loadtxt(file, dtype=np.float32).ravel()
            file.close()
            opts = opts[start:end]
        else:
            file = open(f"{self.results_path}/optimal_values_failure_id_{self.props.failure_id}.txt")
            opts = np.loadtxt(file, dtype=np.float32).ravel()
            file.close()
            if len(opts) == 0:
                exit(1)
                    
        for snapshot_filename, opt_value in zip(filenames, opts):
            topology_filename, pairs_filename, tm_filename = snapshot_filename
            snapshot = Read_Snapshot(self.props, topology_filename, pairs_filename, tm_filename)
            self.list_tms.append(snapshot.tm)
            self.list_tms_pred.append(snapshot.tm_pred)
            self.list_optimal_values.append(opt_value)
            self.list_capacities.append(snapshot.capacities)
            self.list_node_features.append(snapshot._node_features)
            
        cluster_info = Cluster_Info(snapshot, props, self.cluster)
        self.edge_index = cluster_info.sp.get_edge_index().to(props.device)
        self.pij = cluster_info.compute_ksp_paths(props.num_paths_per_pair, cluster_info.sp.pairs)
        self.pte = cluster_info.get_paths_to_edges_matrix(self.pij)
        self.padded_edge_ids_per_path = cluster_info.get_padded_edge_ids_per_path(self.pij, cluster_info.edges_map)
        self.num_pairs = cluster_info.num_pairs
        
        
    def __len__(self):
        return len(self.list_tms)
    
    def __getitem__(self, idx):
        if self.props.pred:
            return self.list_node_features[idx], self.list_capacities[idx], self.list_tms[idx], self.list_tms_pred[idx], self.list_optimal_values[idx]
        else:
            return self.list_node_features[idx], self.list_capacities[idx], self.list_tms[idx], self.list_tms[idx], self.list_optimal_values[idx]        
