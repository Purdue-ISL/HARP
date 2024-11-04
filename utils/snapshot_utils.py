import numpy as np
import networkx as nx
import torch
import json
import pickle
import os

class Read_Snapshot:
    def __init__(self, props, topology_filename, pairs_filename, tm_filename):
        
        self.props = props
        self.topo = self.props.topo
        self.parent_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.failure_id = self.props.failure_id
        
        # read graph from json
        self.graph, self.capacities = self.read_graph_from_json(self.topo, topology_filename)
        self.capacities = self.capacities
        self.pairs = self.read_pairs_from_pkl(pairs_filename)
        self.tm = self.read_tms(tm_filename)
        
        if self.props.pred:
            self.tm_pred = self.read_tms_pred(tm_filename)
        else:
            self.tm_pred = np.array([0])
        
        self.num_demands = len(self.pairs)
        
        self.node_ids_map = {node: i for i, node in enumerate(self.graph.nodes())}
        if props.framework.lower() == "harp":
            self._node_features = self.get_node_features()
            
    def read_pairs_from_pkl(self, pairs_filename):
        file = open(f"{self.parent_dir_path}/pairs/{self.topo}/{pairs_filename}", "rb")
        pairs = pickle.load(file)
        file.close()
        
        return pairs
        
        
    def read_graph_from_json(self, topo: str, topology_filename):
        with open(f"{self.parent_dir_path}/topologies/{self.topo}/{topology_filename}") as f:
            data = json.load(f)
        
        graph = nx.readwrite.json_graph.node_link_graph(data)
        
        capacities = [float(data['capacity']) for u, v, data in graph.edges(data=True)]
        capacities = torch.tensor(capacities, dtype=torch.float32)
        
        if self.failure_id != None:
            undirected_graph = graph.to_undirected()
            undirected_edges = list(undirected_graph.edges())
            directed_edges = list(graph.edges())
            failed_edge = undirected_edges[self.failure_id]
            x, y = failed_edge
            idx1 = directed_edges.index((x, y))
            idx2 = directed_edges.index((y, x))
            capacities[idx1] = 0
            capacities[idx2] = 0
                    
        
        return graph, capacities
    
    def get_edge_index(self) -> torch.Tensor:        
        source_nodes = []
        target_nodes = []
        for i, j in self.graph.edges():
                x = self.node_ids_map[i]
                y = self.node_ids_map[j]
                source_nodes.append(x)
                target_nodes.append(y)
                
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.int64)
        
        return edge_index
        
    def get_node_features(self):
        degrees = dict(self.graph.in_degree())
        degrees = torch.tensor(list(degrees.values()), dtype=torch.float32)
        degrees = degrees.reshape(-1, 1)
        edge_index = self.get_edge_index()
        
        mask = torch.where(self.capacities == 0)
        if self.props.mode == "train":
            self.capacities[mask] += 1e-4
        elif self.props.mode == "test":
            self.capacities[mask] += 1e-4
        
        cap_sum_list = []
        for node in self.graph.nodes():
            node_id = self.node_ids_map[node]
            indices = (edge_index[0] == node_id).nonzero()
            cap_sum = torch.sum(self.capacities[indices])
            cap_sum_list.append(cap_sum)
        cap_sum_list = torch.tensor(cap_sum_list, dtype=torch.float32)
        cap_sum_list = cap_sum_list.reshape(-1, 1)
        node_features = torch.cat((degrees, cap_sum_list), dim=1)
        node_features = node_features.to(dtype=torch.float32)
        
        return node_features
        
    def read_tms(self, tm_filename):
        file = open(f"{self.parent_dir_path}/traffic_matrices/{self.topo}/{tm_filename}", 'rb')
        tm = pickle.load(file)
        tm = tm.reshape(-1, 1)
        tm = tm.astype(np.float32)
        tm = np.repeat(tm, repeats=self.props.num_paths_per_pair, axis=0)
        file.close()
        assert(tm.shape[0] == len(self.pairs)*self.props.num_paths_per_pair)
        
        return tm

    def read_tms_pred(self, tm_filename):
        file = open(f"{self.parent_dir_path}/traffic_matrices/{self.topo}_pred/{tm_filename}", 'rb')
        tm = pickle.load(file)
        tm = tm.reshape(-1, 1)
        tm = tm.astype(np.float32)
        tm = np.repeat(tm, repeats=self.props.num_paths_per_pair, axis=0)
        file.close()
        assert(tm.shape[0] == len(self.pairs)*self.props.num_paths_per_pair)
        
        return tm
