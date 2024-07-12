def add_default_args(parser):
    
    
    # Topology arguments
    parser.add_argument("--topo", type=str)
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--metric", type=str, default="MLU")
    
    ############ HARP arguments #############
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_paths_per_pair", type=int, default=8)
    parser.add_argument("--framework", type=str, default="harp")
    parser.add_argument("--num_transformer_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=0)
    parser.add_argument("--num_gnn_layers", type=int, default=3)
    parser.add_argument("--num_mlp1_hidden_layers", type=int, default=2)
    parser.add_argument("--num_mlp2_hidden_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--num_for_loops", type=int, default=3)
    parser.add_argument("--failure_id", type=int, default=None)
    parser.add_argument("--dynamic", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float32")
        
    parser.add_argument("--train_start_indices", type=int, nargs="+")
    parser.add_argument("--train_end_indices", type=int, nargs="+")
    parser.add_argument("--val_start_indices", type=int, nargs="+")
    parser.add_argument("--val_end_indices", type=int, nargs="+")
    parser.add_argument("--test_start_idx", type=int,)
    parser.add_argument("--test_end_idx", type=int,)
    parser.add_argument("--train_clusters", type=int, nargs="+")
    parser.add_argument("--val_clusters", type=int, nargs="+")
    parser.add_argument("--test_cluster", type=int,)
    parser.add_argument("--opt_start_idx", type=int)
    parser.add_argument("--opt_end_idx", type=int)
    
    # Prediction arguments
    parser.add_argument("--pred", type=int, default=0)

    return parser
    
    
def parse_args(args):
    import argparse
    parser = argparse.ArgumentParser()
    parser = add_default_args(parser)
    args_ = parser.parse_args(args)
        
    return args_
