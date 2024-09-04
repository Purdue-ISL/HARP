import sys
import os
import numpy as np
import networkx as nx
from torch.nn.utils.rnn import pad_sequence
import torch
from itertools import islice
from scipy.sparse import csr_matrix
import pickle

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
from snapshot_utils import Read_Snapshot

class Cluster_Info:
    def __init__(self, sp: Read_Snapshot, props, cluster):
        self.sp = sp
        self.edges_map = {(i, j): eid for eid, (i, j) in enumerate(self.sp.graph.edges())}
        self.props = props
        self.cluster = cluster
        self.parent_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    def node_ids_to_edge_tuple(self, node_ids):
        return [(node1, node2) for node1, node2 in zip(node_ids, node_ids[1:])]
    
    def find_ksp_per_pair(self, graph, src, dst, k, weight=None):
        return list(islice(nx.shortest_simple_paths(graph, src, dst, weight=weight), k))
        
    def compute_ksp_paths(self, k, pairs):
        """
        Computes or loads the k-shortest paths for each source-destination pair.

        Args:
            k (int): The number of shortest paths to compute for each pair.
            pairs (Iterable (np.array, list, OrderedSet... etc.)): A list of source-destination node pairs (src, dst).

        Returns:
            dict: A dictionary where the keys are (src, dst) tuples and the values are lists of 
                k paths represented as edge tuples.

        Process:
            1. Try to load the precomputed k-shortest paths from a pickle file. 
            The file is located at 'topologies/paths_dict/' and named based on the topology 
            and cluster information.
            2. If the file is not found, compute the k-shortest paths for each source-destination 
            pair in the given pairs list.
            - Use `find_ksp_per_pair` to compute the paths.
            - If a pair has fewer than k paths, replicate the first path until the number of 
                paths equals k.
            - Store the paths as edge tuples by converting node IDs using `node_ids_to_edge_tuple`.
            3. Save the computed k-shortest paths to the pickle file for future use.
            4. Return the dictionary containing the k-shortest paths.
        """


        filepath = f"{self.parent_dir_path}/topologies/paths_dict/{self.props.topo}_{k}_paths_dict_cluster_{self.cluster}.pkl"
        try:
            file = open(filepath, "rb")
            pij = pickle.load(file)
            self.num_pairs = len(pij)
            file.close()
        except:
            pij = dict()
            print(f"[Computing {k} Shortest Paths]")
            for src, dst in pairs:
                    all_paths = self.find_ksp_per_pair(self.sp.graph, src, dst, k, self.props.weight)
                    # Force all pairs of nodes to have the same number of paths (K)
                    # If a pair of nodes has less than K paths, replicate the first paths
                    # until they are equal to K
                    while len(all_paths) != k:
                        all_paths.append(all_paths[0])
                    pij[(src, dst)] = [self.node_ids_to_edge_tuple(all_paths[i]) for i in range(k)]
            self.num_pairs = len(pairs)
            file = open(filepath, "wb")
            pickle.dump(pij, file)
            file.close()
            
        return pij
        
    def get_padded_edge_ids_per_path(self, pij, edges_map) -> torch.Tensor:
        """
        Retrieves or computes the padded edge IDs for each path in pij.

        Args:
            pij (dict): Dictionary where the keys are (src, dst) pairs and values are lists of paths 
                        represented as edge tuples.
            edges_map (dict): A mapping from edge tuples to edge indices.

        Returns:
            torch.Tensor: A tensor of padded edge IDs for each path. The tensor is padded with -1 
                        to ensure all paths have the same length, and has shape 
                        (num_paths, max_path_length).

        Process:
            1. Try to load precomputed padded edge IDs from a file. The file is located at 
            'topologies/padded_edge_ids_per_path/' and named based on the topology and cluster 
            information.
            2. If the file is not found, compute the edge IDs for each path:
            - For each (src, dst) pair in pij, retrieve the list of paths.
            - For each path, convert its edges to indices using the edges_map.
            - Append the edge indices to a list.
            3. Pad the edge index lists so that all paths have the same length, using -1 as the 
            padding value.
            4. Convert the padded edge indices to a tensor of int64 type and save it to the file.
            5. Return the padded edge IDs tensor.
        """


        filepath = f"{self.parent_dir_path}/topologies/padded_edge_ids_per_path/{self.props.topo}_{self.props.num_paths_per_pair}_paths_cluster_{self.cluster}_padded_edge_ids_per_path.pkl"
        try:
            padded_edge_ids_per_path = torch.load(filepath)
        except:
            paths_edges_list = []
            for key in pij.keys():
                for path in pij[key]:
                    edges_list = []
                    for edge in path:
                        index = edges_map[edge]
                        edges_list.append(index)
                    paths_edges_list.append(torch.tensor(edges_list, dtype=torch.int32))
                
            padded_edge_ids_per_path = pad_sequence(paths_edges_list, batch_first=True,
                    padding_value=-1.0)
            padded_edge_ids_per_path = padded_edge_ids_per_path.to(dtype=torch.int64)
            torch.save(padded_edge_ids_per_path, filepath)
        
        return padded_edge_ids_per_path
        
    def get_paths_to_edges_matrix(self, pij: dict) -> torch.sparse_coo_tensor:
        """
            Computes or loads a sparse matrix representing the paths-to-edges mapping for all paths in pij.

            Args:
                pij (dict): Dictionary where the keys are (src, dst) pairs and values are lists of paths 
                            represented as edge tuples.

            Returns:
                torch.sparse_coo_tensor: A sparse tensor where each row corresponds to a path, and each 
                                        column corresponds to an edge. Non-zero values indicate that a 
                                        particular edge is part of a given path.
        """
    
        filepath = f"{self.parent_dir_path}/topologies/paths/{self.props.topo}_{self.props.num_paths_per_pair}_paths_cluster_{self.cluster}.pkl"
        try:
            paths_to_edges = torch.load(filepath)
        except FileNotFoundError:
            paths_arr = []
            path_to_commodity = dict()
            path_to_idx = dict()
            cntr = 0
            
            for key in pij.keys():
                idx = 0
                i, j = key
                for p in pij[(i, j)]:
                    p_ = [self.edges_map[e] for e in p]
                    p__ = np.zeros((len(self.sp.graph.edges()),))
                    for k in p_:
                        p__[k] = 1
                    paths_arr.append(p__)
                    path_to_commodity[cntr] = (i, j)
                    path_to_idx[cntr] = idx
                    cntr += 1
                    idx += 1
                    # commodities.append((i, j))
                
            paths_to_edges = np.stack(paths_arr)
            
            paths_to_edges = csr_matrix(paths_to_edges)
            paths_to_edges = paths_to_edges.tocoo()
            paths_to_edges = torch.sparse_coo_tensor(np.vstack((paths_to_edges.row, paths_to_edges.col)), 
                            torch.FloatTensor(paths_to_edges.data), torch.Size(paths_to_edges.shape))
            torch.save(paths_to_edges, filepath)
            
        return paths_to_edges
