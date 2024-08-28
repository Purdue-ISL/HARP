def add_default_args(parser):
    
    # Topology arguments
    parser.add_argument("--topo", type=str, help="Name of the topology to be used.")
    parser.add_argument("--weight", type=str, default=None, help="name of metric used to represent weights of the edges.")
    parser.add_argument("--metric", type=str, default="MLU", help="Only supports MLU (Maximum Link Utilization) for now.")
    
    ############ HARP arguments #############
    parser.add_argument('--mode', type=str, default="train", help="Mode of operation: 'train', 'test'. Default is 'train'.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs. Default is 1.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer. Default is 0.001.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training. Default is 1.")
    parser.add_argument("--num_paths_per_pair", type=int, default=8, help="Number of paths per pair of nodes. Default is 8.")
    parser.add_argument("--framework", type=str, default="harp", help="Framework to be used. Default is 'harp'.")
    parser.add_argument("--num_transformer_layers", type=int, default=2, help="Number of Transformer layers. Default is 2.")
    parser.add_argument("--num_heads", type=int, default=0, help="Number of attention heads in the Transformer. Default is 0.")
    parser.add_argument("--num_gnn_layers", type=int, default=3, help="Number of GNN layers. Default is 3.")
    parser.add_argument("--num_mlp1_hidden_layers", type=int, default=2, help="Number of hidden layers in the first MLP. Default is 2.")
    parser.add_argument("--num_mlp2_hidden_layers", type=int, default=2, help="Number of hidden layers in the second MLP. Default is 2.")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate for the transformer. Default is 0.")
    parser.add_argument("--num_for_loops", type=int, default=3, help="Number of RAUs for HARP. Default is 3.")
    parser.add_argument("--failure_id", type=int, default=None, help="ID of the failure scenario to simulate. Optional, defaults to None.")
    parser.add_argument("--dynamic", type=int, default=1, help="Flag to indicate if the topology is dynamic. Default is 1 (True).")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type for computations. Default is 'float32'. You can also use float16 corresponding to bfloat16")
    
    parser.add_argument("--train_start_indices", type=int, nargs="+", help="HARP: Start indices for training clusters.")
    parser.add_argument("--train_end_indices", type=int, nargs="+", help="HARP: End indices for training clusters.")
    parser.add_argument("--val_start_indices", type=int, nargs="+", help="HARP: Start indices for validation clusters.")
    parser.add_argument("--val_end_indices", type=int, nargs="+", help="HARP: End indices for validation clusters.")
    parser.add_argument("--test_start_idx", type=int, help="HARP: Start index for the test cluster.")
    parser.add_argument("--test_end_idx", type=int, help="HARP: End index for the test cluster.")
    parser.add_argument("--train_clusters", type=int, nargs="+", help="HARP: Cluster IDs to be used for training.")
    parser.add_argument("--val_clusters", type=int, nargs="+", help="HARP: Cluster IDs to be used for validation.")
    parser.add_argument("--test_cluster", type=int, help="HARP: Cluster ID to be used for testing.")
    parser.add_argument("--opt_start_idx", type=int, help="Gurobi: Start index for optimal computation.")
    parser.add_argument("--opt_end_idx", type=int, help="Gurobi: End index for optimal computation.")
    
    # Prediction arguments
    parser.add_argument("--pred", type=int, default=0, help="Flag to indicate if predicted TMs are used. Default is 0 (False).")

    return parser
    
def parse_args(args):
    import argparse
    parser = argparse.ArgumentParser()
    parser = add_default_args(parser)
    args_ = parser.parse_args(args)
        
    return args_
