import sys
import os

cwd = os.getcwd()
sys.path.append(cwd + "/utils")
from utils.args_parser import parse_args
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster
from utils.training_utils import create_dataloaders, validate, loss_mlu, move_to_device, train

import torch
device = torch.device('cuda' if  torch.cuda.is_available() else 'cpu')
from torch.utils.data import DataLoader
from tqdm import tqdm
from frameworks.harp_system import HARP
import copy

props = parse_args(sys.argv[1:])
props.device = device
props_geant = copy.deepcopy(props)

if props.dtype.lower() == "float32":
    props.dtype = torch.float32
elif props.dtype.lower() == "float16":
    props.dtype = torch.bfloat16
else:
    print("Only float32 and float16 are allowed")
    exit(1)


# Retrieve batch size and number of epochs
batch_size = props.batch_size
n_epochs = props.epochs

    
if props.mode.lower() == "train":
    # Instaniate HARP
    model = HARP(props)
    model = model.to(device=device, dtype=props.dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=props.lr)

    
    if props.meta_learning:
        print("Meta-Training Hattrick")
        rau = 7
        props_geant.topo = "geant"
        props_geant.num_paths_per_pair = 8
        props_geant.train_clusters = [0]
        props_geant.train_start_indices = [0]
        props_geant.train_end_indices = [6464]
        props_geant.epochs = 10
        props_geant.batch_size = 32
        props_geant.dynamic = 0
        props_geant.num_for_loops = rau
        props_geant.device = device
        props_geant.dtype = props.dtype
        geant_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        geant_ds_list, geant_dl_list = create_dataloaders(props_geant, props_geant.batch_size, training = True, shuffle = True)
        for i in range(props_geant.epochs):
            train(model, props_geant, geant_ds_list, geant_dl_list, geant_optimizer, i, props_geant.epochs)
        print("Meta-Training Done")


    ds_list, dl_list = create_dataloaders(props, batch_size, training = True, shuffle = True)
    val_ds_list, val_dl_list = create_dataloaders(props, 1, training = False, shuffle = False)
    
    
    # Train harp for #n_epochs
    for epoch in range(n_epochs):
        model.train()
        train(model, props, ds_list, dl_list, optimizer, epoch, n_epochs)
        # Iterate over validation clusters
        model.eval()
        for i in range(len(val_ds_list)):
            val_dataset = val_ds_list[i]
            val_dl = val_dl_list[i]
            val_dataset.pte = (val_dataset.pte).to(device=props.device, dtype=props.dtype)
            val_dataset.padded_edge_ids_per_path = val_dataset.padded_edge_ids_per_path.to(device=props.device)
            move_to_device(val_dataset.edge_ids_dict_tensor, props.device)
            move_to_device(val_dataset.original_pos_edge_ids_dict_tensor, props.device)
            val_norm_mlu = validate(model, props, val_dataset, val_dl)
            val_avg_loss = sum(val_norm_mlu)/len(val_norm_mlu)
            print(f"Validation Avg loss: {round(val_avg_loss, 5)}")
            
            val_dataset.pte = (val_dataset.pte).to(device="cpu")
            val_dataset.padded_edge_ids_per_path = val_dataset.padded_edge_ids_per_path.to(device="cpu")
            move_to_device(val_dataset.edge_ids_dict_tensor, "cpu")
            move_to_device(val_dataset.original_pos_edge_ids_dict_tensor, "cpu")
        torch.save(model, f'HARP_{props.topo}_pred_{props.pred}_{props.num_paths_per_pair}sp.pkl')
        
elif props.mode.lower() == "test": #test
    cluster = props.test_cluster
    start = props.test_start_idx
    end = props.test_end_idx
    test_dataset = DM_Dataset_within_Cluster(props, cluster, start, end)
    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_dataset.pte = (test_dataset.pte).to(props.device, dtype=props.dtype)
    test_dataset.padded_edge_ids_per_path = test_dataset.padded_edge_ids_per_path.to(device)
    move_to_device(test_dataset.edge_ids_dict_tensor, props.device)
    move_to_device(test_dataset.original_pos_edge_ids_dict_tensor, props.device)
    #load the model
    model = torch.load(f'HARP_{props.topo}_pred_{props.pred}_{props.num_paths_per_pair}sp.pkl', map_location=device)
    model = model.to(dtype=props.dtype)
    model.eval()
    
    with torch.no_grad():
        with tqdm(test_dl) as tests:
            test_losses = []
            file = open(f"results/{props.topo}/{props.num_paths_per_pair}sp/{props.test_cluster}/harp_values_{cluster}_failure_id_{props.failure_id}.txt", 'w')
            for inputs in tests:
                node_features, capacities, tms, tms_pred, opt = inputs
                node_features = node_features.to(device=device, dtype=props.dtype)
                capacities = capacities.to(device=device, dtype=props.dtype)
                tms = tms.to(device=device, dtype=props.dtype)
                opt = opt.to(device=device, dtype=props.dtype)
                
                # If prediction is on, feed the predicted matrix
                if props.pred:
                    tms_pred = tms_pred.to(device=device, dtype=props.dtype)
                    
                    predicted = model(props, node_features, test_dataset.edge_index, capacities,
                            test_dataset.padded_edge_ids_per_path,
                            tms, tms_pred, test_dataset.pte, test_dataset.edge_ids_dict_tensor, test_dataset.original_pos_edge_ids_dict_tensor)

                # If prediction is off, feed the actual matrix as predicted matrix
                else:
                    predicted = model(props, node_features, test_dataset.edge_index, capacities,
                            test_dataset.padded_edge_ids_per_path,
                            tms, tms, test_dataset.pte, test_dataset.edge_ids_dict_tensor, test_dataset.original_pos_edge_ids_dict_tensor)
                
                test_loss, test_loss_value = loss_mlu(predicted, opt)
                test_losses.append(test_loss_value)
                file.write(str(test_loss_value) + "\n")
            avg_loss = sum(test_losses) / len(test_losses)
            print(f"Test Error: \nAvg loss: {avg_loss:>8f} \n")
            file.close()
            
            with open(f"results/{props.topo}/{props.num_paths_per_pair}sp/{props.test_cluster}/harp_stats_{cluster}_failure_id_{props.failure_id}.txt", 'w') as f:
                import statistics
                dists = [float(v) for v in test_losses]
                dists.sort()
                f.write('Average: ' + str(statistics.mean(dists)) + '\n')
                f.write('Median: ' + str(dists[int(len(dists) * 0.5)]) + '\n')
                f.write('25TH: ' + str(dists[int(len(dists) * 0.25)]) + '\n')
                f.write('75TH: ' + str(dists[int(len(dists) * 0.75)]) + '\n')
                f.write('90TH: ' + str(dists[int(len(dists) * 0.90)]) + '\n')
                f.write('95TH: ' + str(dists[int(len(dists) * 0.95)]) + '\n')
                f.write('99TH: ' + str(dists[int(len(dists) * 0.99)]) + '\n')
                f.write('100TH: ' + str(dists[int(len(dists)-1)]) + '\n')
