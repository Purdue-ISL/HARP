import sys
import os

cwd = os.getcwd()
sys.path.append(cwd + "/utils")
from utils.args_parser import parse_args
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster


import torch
device = torch.device('cuda' if  torch.cuda.is_available() else 'cpu')
from torch.utils.data import DataLoader
from tqdm import tqdm
from frameworks.harp_system import HARP


props = parse_args(sys.argv[1:])
props.device = device

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


# Define the loss
def loss_mlu(y_pred_batch, y_true_batch):
    losses = []
    loss_vals = []
    batch_size = y_pred_batch.shape[0]
    for i in range(batch_size):
        y_pred = y_pred_batch[[i]]
        opt = y_true_batch[[i]]
        max_cong = torch.max(y_pred)
        loss = 1.0 - max_cong if max_cong.item() == 0.0 else max_cong/max_cong.item()
        loss_val = 1.0 if opt == 0.0 else max_cong.item() / opt.item()
        losses.append(loss)
        loss_vals.append(loss_val)
    ret = sum(losses) / len(losses)
    ret_val = sum(loss_vals) / len(loss_vals)
    return ret, ret_val
    
if props.mode.lower() == "train":
    

    # Function for validation set
    def validate(model, props, val_ds, val_dl):
        val_norm_mlu = []
        with torch.no_grad():
            with tqdm(val_dl) as vals:
                for i, inputs in enumerate(vals):
                    node_features, capacities, tms, tms_pred, opt = inputs
                                        
                    node_features = node_features.to(device=props.device, dtype=props.dtype)
                    capacities = capacities.to(device=props.device, dtype=props.dtype)
                    tms = tms.to(device=props.device, dtype=props.dtype)
                    tms_pred = tms_pred.to(device=props.device, dtype=props.dtype)
                    opt = opt.to(device=props.device, dtype=props.dtype)
                    
                    # If prediction is on, feed the predicted matrix
                    if props.pred:
                        tms_pred = tms_pred.to(device=device, dtype=props.dtype)
                        
                        predicted = model(props, node_features, val_ds.edge_index, capacities,
                                val_ds.padded_edge_ids_per_path,
                                tms, tms_pred, val_ds.pte)

                    # If prediction is off, feed the actual matrix as predicted matrix
                    else:
                        predicted = model(props, node_features, val_ds.edge_index, capacities,
                                val_ds.padded_edge_ids_per_path,
                                tms, tms, val_ds.pte)
                        
                    val_loss, value_loss_value = loss_mlu(predicted, opt)
                    val_norm_mlu.append(value_loss_value)
        
        return val_norm_mlu
        
    # create the training and validation DataLoaders
    ds_list = []
    dl_list = []
    for clstr, start, end in zip(props.train_clusters, props.train_start_indices, props.train_end_indices):
        train_dataset = DM_Dataset_within_Cluster(props, clstr, start, end)
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle = False)
        ds_list.append(train_dataset)
        dl_list.append(train_dl)
    
    val_ds_list = []
    val_dl_list = []
    for clstr, start, end in zip(props.val_clusters, props.val_start_indices, props.val_end_indices):
        val_dataset = DM_Dataset_within_Cluster(props, clstr, start, end)
        val_dl = DataLoader(val_dataset, batch_size=1, shuffle = False)
        val_ds_list.append(val_dataset)
        val_dl_list.append(val_dl)
    
    # Instaniate HARP
    model = HARP(props)
    
    # model.float()
    model = model.to(device=device, dtype=props.dtype)
        
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=props.lr)
    
    train_losses = []
    val_losses = []
    
    # Train harp for #n_epochs
    for epoch in range(n_epochs):

        # Iterate over training clusters
        for i in range(len(ds_list)):
            train_dataset = ds_list[i]
            train_dl = dl_list[i]
            train_dataset.pte = (train_dataset.pte).to(device=device, dtype=props.dtype)
            train_dataset.padded_edge_ids_per_path = train_dataset.padded_edge_ids_per_path.to(device)
            model.train()
            
            with tqdm(train_dl) as tepoch:
                epoch_train_loss = []
                loss_sum = loss_count = 0
                for i, inputs in enumerate(tepoch):
                    optimizer.zero_grad()
                    tepoch.set_description(f"Epoch {epoch+1}/{n_epochs}")
                    # Retrieve inputs to HARP
                    node_features, capacities, tms, tms_pred, opt = inputs
                    
                    # If the topology does not change across examples/snapshots (static topology), just take the first example
                    if not props.dynamic:
                        node_features = node_features[:1]
                        capacities = capacities[:1]
                        
                    node_features = node_features.to(device=device, dtype=props.dtype)
                    capacities = capacities.to(device=device, dtype=props.dtype)
                    tms = tms.to(device=device, dtype=props.dtype)
                    opt = opt.to(device=device, dtype=props.dtype)
                    
                    # If prediction is on, feed the predicted matrix
                    if props.pred:
                        tms_pred = tms_pred.to(device=device, dtype=props.dtype)
                        
                        predicted = model(props, node_features, train_dataset.edge_index, capacities,
                                train_dataset.padded_edge_ids_per_path,
                                tms, tms_pred, train_dataset.pte)

                    # If prediction is off, feed the actual matrix as predicted matrix
                    else:
                        predicted = model(props, node_features, train_dataset.edge_index, capacities,
                                train_dataset.padded_edge_ids_per_path,
                                tms, tms, train_dataset.pte)
                    
                    loss, loss_val = loss_mlu(predicted, opt)
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss.append(loss_val)
                    loss_sum += loss_val
                    loss_count += 1
                    loss_avg = loss_sum / loss_count
                    tepoch.set_postfix(loss=loss_avg)
                    
            train_dataset.pte = (train_dataset.pte).to(device="cpu")
            train_dataset.padded_edge_ids_per_path = train_dataset.padded_edge_ids_per_path.to(device="cpu")
            
        # Iterate over validation clusters
        model.eval()
        for i in range(len(val_ds_list)):
            val_dataset = val_ds_list[i]
            val_dl = val_dl_list[i]
            val_dataset.pte = (val_dataset.pte).to(device=props.device, dtype=props.dtype)
            val_dataset.padded_edge_ids_per_path = val_dataset.padded_edge_ids_per_path.to(device=props.device)
            
            val_norm_mlu = validate(model, props, val_dataset, val_dl)
            val_avg_loss = sum(val_norm_mlu)/len(val_norm_mlu)
            print(f"Validation Avg loss: {round(val_avg_loss, 5)}")
            
            val_dataset.pte = (val_dataset.pte).to(device="cpu")
            val_dataset.padded_edge_ids_per_path = val_dataset.padded_edge_ids_per_path.to(device="cpu")

        torch.save(model, f'HARP_{props.topo}_pred_{props.pred}_{props.num_paths_per_pair}sp.pkl')
        
elif props.mode.lower() == "test": #test
    cluster = props.test_cluster
    start = props.test_start_idx
    end = props.test_end_idx
    test_dataset = DM_Dataset_within_Cluster(props, cluster, start, end)
    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_dataset.pte = (test_dataset.pte).to(props.device, dtype=props.dtype)
    test_dataset.padded_edge_ids_per_path = test_dataset.padded_edge_ids_per_path.to(device)
    
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
                            tms, tms_pred, test_dataset.pte)

                # If prediction is off, feed the actual matrix as predicted matrix
                else:
                    predicted = model(props, node_features, test_dataset.edge_index, capacities,
                            test_dataset.padded_edge_ids_per_path,
                            tms, tms, test_dataset.pte)
                
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
