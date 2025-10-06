import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster

def move_to_device(dictionary, device="cpu"):
    for key in dictionary.keys():
        dictionary[key] = dictionary[key].to(device)
    return dictionary

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

def create_dataloaders(props, batch_size_dl, training = True, shuffle = True):
    ds_list = []
    dl_list = []
    if training:
        clusters = props.train_clusters
        start_indices = props.train_start_indices
        end_indices = props.train_end_indices
    else:
        clusters = props.val_clusters
        start_indices = props.val_start_indices
        end_indices = props.val_end_indices
    for clstr, start, end in zip(clusters, start_indices, end_indices):
        dataset = DM_Dataset_within_Cluster(props, clstr, start, end)
        dl = DataLoader(dataset, batch_size=batch_size_dl, shuffle = shuffle)
        ds_list.append(dataset)
        dl_list.append(dl)
    
    return ds_list, dl_list

def train(model, props, train_ds, train_dl, optimizer, epoch, n_epochs):
    for i in range(len(train_ds)):
        train_dataset = train_ds[i]
        train_dl = train_dl[i]
        train_dataset.pte = (train_dataset.pte).to(device=props.device, dtype=props.dtype)
        train_dataset.padded_edge_ids_per_path = train_dataset.padded_edge_ids_per_path.to(device=props.device)
        move_to_device(train_dataset.edge_ids_dict_tensor, props.device)
        move_to_device(train_dataset.original_pos_edge_ids_dict_tensor, props.device)
        
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
                    
                node_features = node_features.to(device=props.device, dtype=props.dtype)
                capacities = capacities.to(device=props.device, dtype=props.dtype)
                tms = tms.to(device=props.device, dtype=props.dtype)
                opt = opt.to(device=props.device, dtype=props.dtype)
                
                # If prediction is on, feed the predicted matrix
                if props.pred:
                    tms_pred = tms_pred.to(device=props.device, dtype=props.dtype)
                    
                    predicted = model(props, node_features, train_dataset.edge_index, capacities,
                            train_dataset.padded_edge_ids_per_path,
                            tms, tms_pred, train_dataset.pte, train_dataset.edge_ids_dict_tensor, train_dataset.original_pos_edge_ids_dict_tensor)

                # If prediction is off, feed the actual matrix as predicted matrix
                else:
                    predicted = model(props, node_features, train_dataset.edge_index, capacities,
                            train_dataset.padded_edge_ids_per_path,
                            tms, tms, train_dataset.pte, train_dataset.edge_ids_dict_tensor, train_dataset.original_pos_edge_ids_dict_tensor)
                
                loss, loss_val = loss_mlu(predicted, opt)
                loss.backward()
                optimizer.step()
                epoch_train_loss.append(loss_val)
                loss_sum += loss_val
                loss_count += 1
                loss_avg = loss_sum / loss_count
                tepoch.set_postfix(loss=loss_avg)
                # if torch.cuda.is_available():
                #     max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
                #     print(f"Max CUDA memory allocated: {max_allocated:.3f} GB")
                    

                
        train_dataset.pte = (train_dataset.pte).to(device="cpu")
        train_dataset.padded_edge_ids_per_path = train_dataset.padded_edge_ids_per_path.to(device="cpu")
        move_to_device(train_dataset.edge_ids_dict_tensor, "cpu")
        move_to_device(train_dataset.original_pos_edge_ids_dict_tensor, "cpu")


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
                    tms_pred = tms_pred.to(device=props.device, dtype=props.dtype)
                    
                    predicted = model(props, node_features, val_ds.edge_index, capacities,
                            val_ds.padded_edge_ids_per_path,
                            tms, tms_pred, val_ds.pte, val_ds.edge_ids_dict_tensor, val_ds.original_pos_edge_ids_dict_tensor)

                # If prediction is off, feed the actual matrix as predicted matrix
                else:
                    predicted = model(props, node_features, val_ds.edge_index, capacities,
                            val_ds.padded_edge_ids_per_path,
                            tms, tms, val_ds.pte, val_ds.edge_ids_dict_tensor, val_ds.original_pos_edge_ids_dict_tensor)
                    
                val_loss, value_loss_value = loss_mlu(predicted, opt)
                val_norm_mlu.append(value_loss_value)
    
    return val_norm_mlu
