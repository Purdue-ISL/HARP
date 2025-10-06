from torch.nn.attention import sdpa_kernel, SDPBackend
import time
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv#, GINConv 
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch_scatter
import os
from torch.utils.checkpoint import checkpoint
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

epsilon = 1e-4

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
                # ne = F.silu(ne)
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
        batch_index = batch_index.expand(-1, num_edges, 2)  # Repeat the batch index for each edge
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
                tm, tm_pred, paths_to_edges, edge_ids_dict_tensor, original_pos_edge_ids_dict_tensor):
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
        if props.checkpoint:
            edge_embeddings_with_caps = checkpoint(self.gnn, node_features, edge_index, capacities, use_reentrant=False)
        else:
            edge_embeddings_with_caps = self.gnn(node_features, edge_index, capacities)
        batch_size = tm.shape[0]
        if props.dynamic:
            batch_size_tf = batch_size
        else:
            batch_size_tf = 1
        total_number_of_paths = paths_to_edges.shape[0]
        cls_token = self.cls_token.unsqueeze(0)
        if props.mode == "train" or (props.mode == "test" and not hasattr(self, "transformer_output")):
            if props.checkpoint:
                transformer_output = checkpoint(self.compute_transformer_output, edge_embeddings_with_caps, edge_ids_dict_tensor, original_pos_edge_ids_dict_tensor, batch_size_tf, total_number_of_paths, props, cls_token, use_reentrant=False)
            else:
                transformer_output = self.compute_transformer_output(edge_embeddings_with_caps, edge_ids_dict_tensor, original_pos_edge_ids_dict_tensor, batch_size_tf, total_number_of_paths, props, cls_token)
        else:
            pass

        if props.mode == "train":
            if not props.dynamic:
                transformer_output = transformer_output.expand(batch_size, -1, -1, -1)
                capacities = capacities.expand(batch_size, -1)
        elif (props.mode == "test" and not hasattr(self, "transformer_output") and not props.dynamic):
            capacities = capacities.expand(batch_size, -1)
            self.transformer_output = transformer_output.expand(batch_size, -1, -1, -1)  
        if props.mode == "train":
            path_embeddings = transformer_output[:, :, 0, :]
            path_edge_embeddings = transformer_output[:, :, 1:, :]
        elif props.mode == "test":
            if not props.dynamic:
                path_embeddings = self.transformer_output[:, :, 0, :]
                path_edge_embeddings = self.transformer_output[:, :, 1:, :]
            else:
                path_embeddings = transformer_output[:, :, 0, :]
                path_edge_embeddings = transformer_output[:, :, 1:, :]
        
        # Predicted matrix        
        path_embeddings = torch.cat((path_embeddings, tm_pred), dim=-1)

        if props.checkpoint:
            gammas = checkpoint(self.forward_pass_mlp, path_embeddings, self.mlp1, self.num_mlp1_hidden_layers, use_reentrant=False)
        else:
            gammas = self.forward_pass_mlp(path_embeddings, self.mlp1, self.num_mlp1_hidden_layers)
                
        paths_to_edges = paths_to_edges.coalesce()
        indices = paths_to_edges.indices()
        values = paths_to_edges.values()
        row_indices = indices[0]
        col_indices = indices[1]
        pte_info = [paths_to_edges, row_indices, col_indices, values]
        
        for i in range(num_for_loops):
            if i > 0:
                gammas = new_gammas
            
            if props.checkpoint:
                edges_util = checkpoint(self.compute_edge_utils, gammas, paths_to_edges, tm_pred, capacities, props, batch_size, num_paths_per_pair, add_epsilon=True, use_reentrant=False)
            else:
                edges_util = self.compute_edge_utils(gammas, paths_to_edges, tm_pred, capacities, props, batch_size, num_paths_per_pair, add_epsilon=True)
            
            if props.checkpoint:
                mlu = checkpoint(self.compute_mlu, edges_util, batch_size, total_number_of_paths, subtract_epsilon=True, use_reentrant=False)
            else:
                mlu = self.compute_mlu(edges_util, batch_size, total_number_of_paths, subtract_epsilon=True)

            if props.checkpoint:
                bottleneck_path_edge_embeddings, max_utilization_per_path = checkpoint(self.compute_bottleneck_link_mlu_per_path, edges_util, padded_edge_ids_per_path, path_edge_embeddings, batch_size, total_number_of_paths, pte_info, use_reentrant=False)
            else:
                bottleneck_path_edge_embeddings, max_utilization_per_path = self.compute_bottleneck_link_mlu_per_path(edges_util, padded_edge_ids_per_path, path_edge_embeddings, batch_size, total_number_of_paths, pte_info)

            dnn_2_inputs = torch.cat((bottleneck_path_edge_embeddings, 
                                      max_utilization_per_path,
                                      mlu,
                                      tm_pred), dim=-1).squeeze(0)
            
            if props.checkpoint:
                delta_gammas = checkpoint(self.forward_pass_mlp, dnn_2_inputs, self.mlp2, self.num_mlp2_hidden_layers, use_reentrant=False)
            else:
                delta_gammas = self.forward_pass_mlp(dnn_2_inputs, self.mlp2, self.num_mlp2_hidden_layers)
            
            gammas = gammas.reshape(batch_size, -1, 1)
            new_gammas = delta_gammas + gammas
        
        if num_for_loops == 0:
            new_gammas = gammas
        
        if props.checkpoint:
            edges_util = checkpoint(self.compute_edge_utils, gammas, paths_to_edges, tm, capacities, props, batch_size, num_paths_per_pair, add_epsilon=False, use_reentrant=False)
        else:
            edges_util = self.compute_edge_utils(gammas, paths_to_edges, tm, capacities, props, batch_size, num_paths_per_pair, add_epsilon=False)
        
        return edges_util
    


    def compute_mlu(self, edges_util, batch_size, total_number_of_paths, subtract_epsilon=True):
            """
            Compute per-batch Maximum Link Utilization (MLU) and broadcast over paths.

            - Reduces per-edge utilizations to a single MLU per batch element via max over edges.
            - Optionally subtracts a small epsilon for numerical stability.
            - Reshapes to [B, 1, 1] and expands to [B, P, 1] to align with path-wise tensors.

            Args:
                edges_util (Tensor): Per-edge utilizations [B, E].
                batch_size (int): Batch size B.
                total_number_of_paths (int): Number of paths P.
                subtract_epsilon (bool): Whether to subtract a small epsilon from the MLU.

            Returns:
                Tensor: Broadcast MLU of shape [B, P, 1].
            """
            
            mlu, mlu_indices = torch.max(edges_util, dim=-1)
            if subtract_epsilon:
                mlu = mlu -  epsilon
            mlu = mlu.view(batch_size, 1, 1).expand(-1, total_number_of_paths, -1)
            
            return mlu

    def compute_bottleneck_link_mlu_per_path(self, edge_utils, padded_edge_ids_per_path, path_edge_embeddings, batch_size, total_number_of_paths, pte_info):
        """
        For each path, locate its bottleneck edge, fetch that edge's embedding, and return the
        bottleneck utilization.

        Steps:
        - Use `compute_bottleneck_util_per_path` logic to find the per-path max edge utilization
          and indices of the edges achieving that max.
        - Match those edge indices against `padded_edge_ids_per_path` to recover positions of the
          bottleneck edge within each path's padded edge sequence.
        - Index into `path_edge_embeddings` to extract the corresponding per-path bottleneck
          edge embeddings.

        Args:
            edge_utils (Tensor): Per-edge utilization [B, E].
            padded_edge_ids_per_path (LongTensor): Padded edge ids per path [B, P, Lmax].
            path_edge_embeddings (Tensor): Per-path per-edge embeddings [B, P, Lmax, D].
            batch_size (int): B.
            total_number_of_paths (int): P.
            pte_info (Tuple): (paths_to_edges, row_indices, col_indices, values) describing sparse
                path→edge mapping.

        Returns:
            Tuple[Tensor, Tensor]:
            - bottleneck_path_edge_embeddings: Embedding of bottleneck edge per path [B, P, D].
            - max_utilization_per_path: Bottleneck utilization per path [B, P, 1].
        """
        
        paths_to_edges, row_indices, col_indices, values = pte_info
        max_utilization_per_path, max_indices = torch_scatter.scatter_max((edge_utils[:, col_indices] * values),
                                                                        row_indices, dim=1, dim_size=paths_to_edges.shape[0])
        max_utilization_per_path = max_utilization_per_path - epsilon
        try:
            max_indices = col_indices[max_indices]
        except:
            print("max_indices.shape:", max_indices.shape)
            print("max_indices.device:", max_indices.device)
            print("max_indices.dtype:", max_indices.dtype)
            print("max_indices contains NaN:", torch.isnan(max_indices).any().item())
            print("max_indices contains Inf:", torch.isinf(max_indices).any().item())
            print(max_indices.max())
            print(col_indices.max())
            print("Out of bound indexing!!")
            exit(1)
        
        max_indices_expanded = max_indices.unsqueeze(2).expand(-1, -1,  padded_edge_ids_per_path.size(1))
        matches = (max_indices_expanded == padded_edge_ids_per_path)
        
        try:
            positions = torch.where(matches)
        except Exception as e:
            print(e)
            print(edge_utils.max())
            print("edge_utils contains NaN:", torch.isnan(edge_utils).any().item())
            print("edge_utils contains Inf:", torch.isinf(edge_utils).any().item())
            print(edge_utils.max())
            print(max_indices_expanded.shape, padded_edge_ids_per_path.shape)
            print(max_indices_expanded.max())
            print(padded_edge_ids_per_path.max())
            print(matches.max())
            print("Out of bound indexing!!")
            exit(1)
        positions = torch.stack(positions, dim=-1)
        positions = positions.view(batch_size, total_number_of_paths, -1)
                            
        dim0_range = positions[:, :, 0].view(batch_size, total_number_of_paths, -1)
        dim1_range = positions[:, :, 1].view(batch_size, total_number_of_paths, -1)
        positions = positions[:, :, -1].view(batch_size, total_number_of_paths, -1)
        
        bottleneck_path_edge_embeddings = (path_edge_embeddings[dim0_range, dim1_range, positions]).squeeze(-2)
                
        return bottleneck_path_edge_embeddings, max_utilization_per_path.unsqueeze(-1)


    def compute_transformer_output(self, edge_embeddings_with_caps, edge_ids_dict_tensor, original_pos_edge_ids_dict_tensor, batch_size_tf, total_number_of_paths, props, cls_token):
        """
        Forward pass of the Set Transformer.
        """
        
        max_path_length = max(list(edge_ids_dict_tensor.keys())) + 1 # due to CLS token
        transformer_output = torch.empty((batch_size_tf, total_number_of_paths, max_path_length, self.input_dim),
                                            device=self.device, dtype=props.dtype)
        for i, key in enumerate(sorted(edge_ids_dict_tensor.keys())):
            temp_embds = edge_embeddings_with_caps[:, edge_ids_dict_tensor[key], :]
            if props.dynamic:
                temp_cls_token = cls_token.expand(batch_size_tf, edge_ids_dict_tensor[key].shape[0], -1).unsqueeze(-2)
            else:
                temp_cls_token = cls_token.expand(1, edge_ids_dict_tensor[key].shape[0], -1).unsqueeze(-2)
            
            temp_embds = torch.cat((temp_cls_token, temp_embds), dim=-2)
            x1, x2, x3, x4 = temp_embds.shape
            temp_embds = temp_embds.reshape(x1*x2, x3, x4).contiguous()
            with sdpa_kernel([SDPBackend.MATH]):
                temp_embds = self.transformer(temp_embds)
            # with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            #     temp_embds = self.transformer(temp_embds)
            temp_embds = temp_embds.reshape(x1, x2, x3, x4)
            temp_embds = F.pad(temp_embds, (0, 0, 0, max_path_length - temp_embds.shape[2]), value=0.0)
            transformer_output[:, original_pos_edge_ids_dict_tensor[key], :, :] = temp_embds
        
        return transformer_output


    def forward_pass_mlp(self, inputs, mlp: nn.ModuleList, num_hidden_layers):
        """
        Apply a stack of Linear layers with LeakyReLU activations to produce raw path scores (gammas).

        - The first layer is applied to `inputs`, followed by LeakyReLU.
        - Each hidden layer (count = `num_hidden_layers`) is applied with LeakyReLU.
        - The final layer (index == num_hidden_layers + 1) is applied without activation.

        Args:
            inputs (Tensor): Input tensor for the MLP.
            mlp (nn.ModuleList): Sequence of Linear layers defining the MLP.
            num_hidden_layers (int): Number of hidden layers (excludes input and output layers).

        Returns:
            Tensor: Output tensor shaped by the last Linear layer's out_features.
        """
        
        for index, layer in enumerate(mlp):
            if index == 0:
                gammas_1 = layer(inputs)
                gammas_1 = F.leaky_relu(gammas_1, 0.02)
            
            elif index == (num_hidden_layers + 1):
                gammas_1 = layer(gammas_1)
            else:
                gammas_1 = layer(gammas_1)
                gammas_1 = F.leaky_relu(gammas_1, 0.02)
                        
        return gammas_1

    def compute_split_ratios(self, gammas, batch_size, num_paths_per_pair):
        gammas = gammas.reshape(batch_size, -1, num_paths_per_pair)
        split_ratios = torch.exp(torch.nn.functional.log_softmax(gammas, dim=-1))
        split_ratios = split_ratios.reshape(batch_size, -1)
        
        return split_ratios
        
    def compute_edge_utils(self, gammas, paths_to_edges, tm, capacities, props, batch_size, num_paths_per_pair, add_epsilon=True):
        """
        Convert raw path scores to traffic allocations and compute per-edge utilizations.

        Behavior:
        - Reshape `gammas` to [B, P, K], apply log_softmax then exp to get
          numerically stable split ratios per path; multiply by tm to get data on tunnels.
        - Aggregate tunnel traffic to links via sparse matmul (paths_to_edges^T · tunnels).
        - Divide by capacities to obtain edge utilizations; optionally add epsilon.

        Args:
            gammas (Tensor): Path scores.
            paths_to_edges (torch.sparse_coo_tensor): Sparse map [P, E] from paths to edges.
            tm (Tensor): Traffic matrix per path group [B, P, 1].
            capacities (Tensor): Link capacities [B, E].
            props: Config namespace (uses props.dtype for dtype control).
            batch_size (int): B.
            num_paths_per_pair (int): K, paths per source-destination pair.
            add_epsilon (bool): Add small epsilon to edge utilizations.

        Returns:
            Tuple[Tensor, Tensor]:
            - edges_util: Edge utilizations [B, E].
            - split_ratios: Per-path split ratios [B, P, 1].
        """
        split_ratios = self.compute_split_ratios(gammas, batch_size, num_paths_per_pair)
        data_on_tunnels = split_ratios*tm.squeeze(-1)
        
        # Actual matrix
        # with torch.autocast(device_type="cuda", dtype=torch.float32):
        data_on_links = torch.sparse.mm(paths_to_edges.to(dtype=torch.float32).t(), data_on_tunnels.to(dtype=torch.float32).t()).t()
        
        if props.dtype == torch.bfloat16:
            data_on_links = data_on_links.to(dtype=torch.bfloat16)
        
        if add_epsilon:
            edges_util = data_on_links/capacities + epsilon
        else:
            edges_util = data_on_links/capacities
        
        return edges_util