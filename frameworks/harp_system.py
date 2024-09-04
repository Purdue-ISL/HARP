import time
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv#, GINConv 
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch_scatter
import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

# Set Transformer
class TransformerModel(nn.Module):
    def __init__(self, in_dim: int, nhead: int, dim_feedforward: int,
                 nlayers: int, dropout: float = 0.0, activation="gelu"):
        super().__init__()
        
        encoder_layers = TransformerEncoderLayer(d_model=in_dim, nhead=nhead,
                            dim_feedforward=dim_feedforward, dropout=dropout,
                        batch_first=True, activation=activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.in_dim = in_dim
        
    def forward(self, src: Tensor, src_key_padding_mask: Tensor = None) -> Tensor:
        """
        Forward pass of the Transformer model.

        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len, in_dim), 
                                representing the source sequences.
            src_key_padding_mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_len) 
                                                        indicating which positions should be ignored 
                                                        in the source sequence. Default is None.

        Returns:
            torch.Tensor: Output tensor of the same shape as the input, after being passed through 
                        the Transformer encoder.
        """
        
        if src_key_padding_mask is not None:
            src_key_padding_mask = (~src_key_padding_mask)
            
        
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        return output

# GNN of HARP
class GNN(nn.Module):
    def __init__(self, num_features, num_gnn_layers):
        super(GNN, self).__init__()
        self.num_features = num_features
                
        self.gnns = nn.ModuleList()
        for i in range(num_gnn_layers):
            if i == 0:
                self.gnns.append(GCNConv(num_features, num_features+1))
            elif i == 1:
                self.gnns.append(GCNConv(num_features+1, num_features+2))
            else:
                self.gnns.append(GCNConv(num_features+2, num_features+2))
        self.output_dim = num_gnn_layers*(self.num_features+2) - 1
        
        
        
    def forward(self, node_features, edge_index, capacities):
        """
        Forward pass of the GNN model.

        Args:
            node_features (torch.Tensor): Node features for each graph in the batch, 
                                        shape (batch_size, num_nodes, num_features).
            edge_index (torch.Tensor): Edge indices defining the graph connectivity, 
                                    shape (2, num_edges).
            capacities (torch.Tensor): Edge capacities for each graph in the batch, 
                                    shape (batch_size, num_edges).

        Returns:
            torch.Tensor: Edge embeddings for each edge in the graph, with capacities included, 
                        shape (batch_size, num_edges, output_dim).

        Process:
            1. Iterate over the batch of node features and capacities.
            2. For each graph in the batch:
                a. Pass the node features through each GNN layer (GCNConv), applying Leaky ReLU activation.
                b. Collect the intermediate node embeddings after each GNN layer.
                c. Concatenate node embeddings from all GNN layers if more than one GNN layer is used.
            3. Stack the node embeddings for all graphs in the batch.
            4. Expand edge_index to match the batch size, so it can be used for batch processing.
            5. Use the expanded edge_index to extract edge embeddings from the node embeddings.
            6. Sum the node embeddings corresponding to each edge and concatenate them with the edge capacities.
        """


        batch_size = node_features.shape[0]
        ne_list = []
        for i in range(batch_size):
            nf = node_features[i]
            caps = capacities[i]
            sample_ne_list = []
            for j, gnn in enumerate(self.gnns):
                if j == 0:
                    ne = gnn(nf, edge_index=edge_index, edge_weight=caps)
                else:
                    ne = gnn(ne, edge_index=edge_index, edge_weight=caps)
                ne = F.leaky_relu(ne, 0.02)
                sample_ne_list.append(ne)
            if len(self.gnns) > 1:
                node_embeddings = torch.cat((sample_ne_list), dim=-1)
                ne_list.append(node_embeddings)
            else:
                ne_list = sample_ne_list
        node_embeddings = torch.stack(ne_list).contiguous()
        edge_index_expanded = edge_index.t().expand(batch_size, -1, -1)
        batch_size, num_nodes, feature_size = node_embeddings.shape
        _, num_edges, _ = edge_index_expanded.shape
        
        # Create a batch index
        batch_index = torch.arange(batch_size).view(-1, 1, 1)
        batch_index = batch_index.repeat(1, num_edges, 2)  # Repeat the batch index for each edge
        edge_embeddings = node_embeddings[batch_index, edge_index_expanded]
        capacities = capacities.unsqueeze(-1)
        edge_embeddings = edge_embeddings.sum(dim=-2)
        edge_embeddings = torch.cat((edge_embeddings, capacities), dim=-1)
        
        return edge_embeddings
        
class HARP(nn.Module):
    
    def __init__(self, props):
        
        super(HARP, self).__init__()
        
        # Define the architecture of HARP
        self.num_gnn_layers = props.num_gnn_layers
        self.num_transformer_layers = props.num_transformer_layers
        self.dropout = props.dropout
        self.num_mlp1_hidden_layers = props.num_mlp1_hidden_layers
        self.num_mlp2_hidden_layers = props.num_mlp2_hidden_layers
        self.device = props.device
        
        # Define the GNN
        self.gnn = GNN(2, self.num_gnn_layers)

        self.input_dim = self.gnn.output_dim + 1
        
        # CLS Token for the Set Transformer
        self.cls_token = nn.Parameter(torch.Tensor(1, self.input_dim))
        nn.init.kaiming_normal_(self.cls_token, nonlinearity='relu')
        
        if props.num_heads == 0:
            num_heads = self.input_dim//4
        else:
            num_heads = props.num_heads
        
        # Define the Set Transformer
        self.transformer = TransformerModel(in_dim = self.input_dim, nhead=num_heads,
                            dim_feedforward=self.input_dim, nlayers=self.num_transformer_layers, 
                            dropout=self.dropout, activation="gelu")
        
        # Define the 1st MLP
        self.mlp_1_dim = self.input_dim + 1
        self.mlp1 = nn.ModuleList()
        self.mlp1.append(nn.Linear(self.mlp_1_dim, self.mlp_1_dim))
        for i in range(self.num_mlp1_hidden_layers):
            self.mlp1.append(nn.Linear(self.mlp_1_dim, self.mlp_1_dim))
        self.mlp1.append(nn.Linear(self.mlp_1_dim, 1))
        
        # Define the 2nd MLP (Recurrent Adjustment Unit - RAU)
        self.mlp_2_dim = self.input_dim + 3
        self.mlp2 = nn.ModuleList()
        self.mlp2.append(nn.Linear(self.mlp_2_dim, self.mlp_2_dim))
        for i in range(self.num_mlp2_hidden_layers):
            self.mlp2.append(nn.Linear(self.mlp_2_dim, self.mlp_2_dim))
        self.mlp2.append(nn.Linear(self.mlp_2_dim, 1))
        
        
    def forward(self, props, node_features, edge_index, capacities, padded_edge_ids_per_path,
                tm, tm_pred, paths_to_edges):
        """
            Process:
            1. Pass the node features, edge index, and capacities through the GNN to obtain edge embeddings.
            2. Expand the edge embeddings using the padded edge IDs per path.
            3. Add a CLS token to the edge embeddings and apply masking for attention.
            4. At this point, tunnels are described as a set of edges (edge embeddings)
            5. Pass the tunnels as sets of edges through the Set Transformer.
            6. Concatenate the transformer output for path embeddings (corresponds to the CLS token) with the predicted traffic matrix.
            7. Compute initial split ratios using the first MLP (mlp1).
            8. Perform iterative adjustments of split ratios using the second MLP (RAU) within the 
            specified number of for-loops. MLP2 takes as input (per tunnel):
                i) Demand of the pair that the tunnels is associated with
                ii) Network-wide MLU
                iii) Bottleneck link utilization in the tunnel
                iv) Tunnel embeddings conditioned on the bottleneck link as generated by the Set Transformer 
        """


        num_for_loops = props.num_for_loops
        num_paths_per_pair = props.num_paths_per_pair
        edge_embeddings_with_caps = self.gnn(node_features, edge_index, capacities)
        batch_size = tm.shape[0]
        total_number_of_paths = paths_to_edges.shape[0]

        edge_embeddings_with_caps = [edge_embeddings_with_caps[i][padded_edge_ids_per_path] \
               for i in range(edge_embeddings_with_caps.shape[0])]        
        edge_embeddings_with_caps = torch.stack(edge_embeddings_with_caps)
        cls_token = self.cls_token.unsqueeze(0)
        

        # If the topology changes across time (dynamic), then probably every example has a unique edge_embeddings_with_caps
        if props.dynamic:
            edge_embeddings_with_caps = torch.cat((cls_token.repeat(batch_size, paths_to_edges.shape[0], 1).unsqueeze(-2),
                                                   edge_embeddings_with_caps), dim=-2)
        
        # If the topology does not change across examples/snapshots (static topology), just make one edge_embeddings_with_caps because it is the same for all examples
        else:
            edge_embeddings_with_caps = torch.cat((cls_token.repeat(1, paths_to_edges.shape[0], 1).unsqueeze(-2), 
                                                   edge_embeddings_with_caps), dim=-2)
        
        attention_mask = torch.cat((torch.ones((padded_edge_ids_per_path.shape[0], 1), dtype=torch.bool, device=self.device),
                                    (padded_edge_ids_per_path != -1.0)), dim=1)
        

        # If the topology changes across examples/snapshots (dynamic topology), then feed each edge_embeddings_with_caps to the transformer
        if props.dynamic:
            out_trf_list = [self.transformer(edge_embeddings_with_caps[i, :, :, :], attention_mask) for i in range(batch_size)]
        
        # If the topology does not change across examples/snapshots (static topology), then feed one edge_embeddings_with_caps to the transformer
        else: # static topology
            out_trf_list = [self.transformer(edge_embeddings_with_caps[i, :, :, :], attention_mask) for i in range(1)]

        out_trf_list = torch.stack(out_trf_list)
        
        # If the topology does not change across examples/snapshots (static topology), then make batch_size copies of the transformer output, each belongs to one example/snapshot.
        if not props.dynamic:
            out_trf_list = out_trf_list.repeat(batch_size, 1, 1, 1)
        
        path_embeddings = out_trf_list[:, :, 0, :]
        path_edge_embeddings = out_trf_list[:, :, 1:, :]
        
        # Predicted matrix        
        path_embeddings = torch.cat((path_embeddings, tm_pred), dim=-1)
        
        # Compute initial raw split ratios        
        for index, layer in enumerate(self.mlp1):
            if index == 0:
                gammas = layer(path_embeddings)
                gammas = gammas.relu()
            elif index == self.num_mlp1_hidden_layers + 1:
                gammas = layer(gammas)
            else:
                gammas = layer(gammas)
                gammas = gammas.relu()
        
        paths_to_edges = paths_to_edges.coalesce()
        indices = paths_to_edges.indices()
        values = paths_to_edges.values()
        row_indices = indices[0]
        col_indices = indices[1]
        
        for i in range(num_for_loops):
            if i > 0:
                gammas = new_gammas
            gammas = gammas.reshape(batch_size, -1, num_paths_per_pair)
            split_ratios = torch.nn.functional.softmax(gammas, dim=-1).reshape(batch_size, -1)
            
            split_ratios = split_ratios*tm_pred.squeeze(-1)
                        
            data_on_links = torch.sparse.mm(paths_to_edges.to(dtype=torch.float32).t(), split_ratios.to(dtype=torch.float32).t()).t()
            if props.dtype == torch.bfloat16:
                data_on_links = data_on_links.to(dtype=torch.bfloat16)
                        
            edges_util = data_on_links/capacities + 1e-4

            inf_mask = torch.where(edges_util == float('inf'))
            nan_mask = torch.isnan(edges_util)
            edges_util[inf_mask] = 1000 + 1e-4
            edges_util[nan_mask] = 0 + 1e-4
            
            mlu, mlu_indices = torch.max(edges_util, dim=-1)
            mlu -= 1e-4
            mlu = mlu.view(batch_size, 1, 1).repeat(1, total_number_of_paths, 1)
                        
            
            max_utilization_per_path, max_indices = torch_scatter.scatter_max((edges_util[:, col_indices] * values),
                                                                              row_indices, dim=1, dim_size=paths_to_edges.shape[0])
            max_indices = col_indices[max_indices]
            max_utilization_per_path -= 1e-4
            
            max_indices_expanded = max_indices.unsqueeze(2).expand(-1, -1,  padded_edge_ids_per_path.size(1))
            matches = (max_indices_expanded == padded_edge_ids_per_path)
            
            positions = matches.nonzero()
            positions = positions.view(batch_size, total_number_of_paths, -1)
            
            dim0_range = positions[:, :, 0].view(batch_size, total_number_of_paths, -1)
            dim1_range = positions[:, :, 1].view(batch_size, total_number_of_paths, -1)
            positions = positions[:, :, -1].view(batch_size, total_number_of_paths, -1)
            
            bottleneck_path_edge_embeddings = (path_edge_embeddings[dim0_range, dim1_range, positions]).squeeze(-2)
                                                
            dnn_2_inputs = torch.cat((bottleneck_path_edge_embeddings, 
                                      max_utilization_per_path.unsqueeze(-1),
                                      mlu,
                                      tm_pred), dim=-1).squeeze(0)
            
            for index, layer in enumerate(self.mlp2):
                if index == 0:
                    delta_gammas = layer(dnn_2_inputs)
                    delta_gammas = delta_gammas.relu()
                elif index == self.num_mlp2_hidden_layers + 1:
                    delta_gammas = layer(delta_gammas)
                else:
                    delta_gammas = layer(delta_gammas)
                    delta_gammas = delta_gammas.relu()
                    
            gammas = gammas.reshape(batch_size, -1, 1)
            new_gammas = delta_gammas + gammas
        
        if num_for_loops == 0:
            new_gammas = gammas
        new_gammas = new_gammas.reshape(batch_size, -1, num_paths_per_pair)
        split_ratios = torch.nn.functional.softmax(new_gammas, dim=-1)
        split_ratios = split_ratios.reshape(batch_size, -1) 
        
        # Actual matrix
        split_ratios = split_ratios*tm.squeeze(-1)
        data_on_links = torch.sparse.mm(paths_to_edges.to(dtype=torch.float32).t(), split_ratios.to(dtype=torch.float32).t()).t()
        if props.dtype == torch.bfloat16:
            data_on_links = data_on_links.to(dtype=torch.bfloat16)
                
        edges_util = data_on_links/capacities
        inf_mask = torch.where(edges_util == float('inf'))
        nan_mask = torch.isnan(edges_util)
        edges_util[inf_mask] = 1000
        edges_util[nan_mask] = 0
        
        return edges_util