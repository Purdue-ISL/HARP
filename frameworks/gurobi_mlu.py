import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# os.chdir("../")

from utils.snapshot_utils import Read_Snapshot
from utils.cluster_utils import Cluster_Info
from utils.args_parser import parse_args

from gurobipy import GRB, Model
import gurobipy as gp
import numpy as np
import sys
import copy
import tqdm
from scipy.sparse import csr_matrix

try:
    os.mkdir(f"{parent_dir}/results")
except:
    pass

args = sys.argv[1:]
props = parse_args(args)
topo = props.topo
num_paths_per_pair = props.num_paths_per_pair
start_index = props.opt_start_idx
end_index = props.opt_end_idx

results_path = f"{parent_dir}/results/{topo}/{num_paths_per_pair}sp"
try:
    os.makedirs(results_path)
except:
    pass


file_manifest = f"{parent_dir}/manifest/{props.topo}_manifest.txt"
manifest = np.loadtxt(file_manifest, dtype="U", delimiter=",")


num_cluster  = 0

try:
    os.mkdir(f"{results_path}/{num_cluster}")
except:
    pass


topology_filename, pairs_filename, tm_filename = manifest[start_index]

topology_filename = topology_filename.strip()
pairs_filename = pairs_filename.strip()
tm_filename = tm_filename.strip()

previous_sp = Read_Snapshot(props, topology_filename, pairs_filename, tm_filename)
current_sp = Read_Snapshot(props, topology_filename, pairs_filename, tm_filename)


cluster_info = Cluster_Info(current_sp, props, num_cluster)
cluster_info.pij = cluster_info.compute_ksp_paths(num_paths_per_pair, cluster_info.sp.pairs)
cluster_info.paths_to_edges = csr_matrix(cluster_info.get_paths_to_edges_matrix(cluster_info.pij).to_dense().numpy())

num_snapshots_in_cluster = 0

if props.failure_id == None:
    optimal_path = f"{results_path}/{num_cluster}/optimal_values.txt"
    optimal_values = open(optimal_path, "w")

    filenames_path = f"{results_path}/{num_cluster}/filenames.txt"
    filenames = open(filenames_path, "w")

else:
    optimal_path = f"{results_path}/{num_cluster}/optimal_values_failure_id_{props.failure_id}.txt"
    optimal_values = open(optimal_path, "w")
    

for i, snapshot in tqdm.tqdm(enumerate(manifest[start_index:end_index]), total=len(manifest[start_index:end_index])):
    
    index = start_index + i
    topology_filename, pairs_filename, tm_filename = snapshot
    topology_filename = topology_filename.strip()
    pairs_filename = pairs_filename.strip()
    tm_filename = tm_filename.strip()
    
    previous_sp = copy.deepcopy(current_sp)
    current_sp = Read_Snapshot(props, topology_filename, pairs_filename, tm_filename)
    
    if (len(previous_sp.graph.nodes()) != len(current_sp.graph.nodes()))\
    or (set(previous_sp.graph.nodes()) != set(current_sp.graph.nodes()))\
    or (not np.array_equal(previous_sp.pairs, current_sp.pairs))\
    or(len(previous_sp.graph.edges()) != len(current_sp.graph.edges())):
        optimal_values.close()
        filenames.close()
        
        if num_snapshots_in_cluster > 0:
            num_cluster += 1
            num_snapshots_in_cluster = 0
        elif num_snapshots_in_cluster == 0:
            path1 = f"{parent_dir}/topologies/paths"
            path2 = f"{parent_dir}/topologies/paths_dict"
            os.remove(f"{path1}/{props.topo}_{props.num_paths_per_pair}_paths_cluster_{num_cluster}.pkl")
            os.remove(f"{path2}/{props.topo}_{props.num_paths_per_pair}_paths_dict_cluster_{num_cluster}.pkl")
                    
        try:
            os.mkdir(f"{results_path}/{num_cluster}")
        except:
            pass
        
        optimal_path = f"{results_path}/{num_cluster}/optimal_values.txt"
        optimal_values = open(optimal_path, "w")
        filenames_path = f"{results_path}/{num_cluster}/filenames.txt"
        filenames = open(filenames_path, "w")
        num_pairs = current_sp.num_demands
        cluster_info = Cluster_Info(current_sp, props, num_cluster)
        cluster_info.pij = cluster_info.compute_ksp_paths(num_paths_per_pair, cluster_info.sp.pairs)
        cluster_info.paths_to_edges = csr_matrix(cluster_info.get_paths_to_edges_matrix(cluster_info.pij).to_dense().numpy())
        
    np_tm = current_sp.tm
    capacities = current_sp.capacities.numpy().astype(np.float64)
    
    #### Prepare the Gurobi model
    mlu = Model("MLU")
    mlu.setParam(GRB.Param.OutputFlag, 0)
    ## Done with Gurobi model
    
    vars_mlu = mlu.addMVar((current_sp.num_demands, num_paths_per_pair), lb=0.0,ub=GRB.INFINITY,
                       vtype=GRB.CONTINUOUS)
    
    mlu.addConstrs((vars_mlu[k, :].sum() == 1) for k in range(current_sp.num_demands))    
    max_link_util = mlu.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="max_link_util")
    # # Done with Gurobi variables
    
    vars_mlu_tm = vars_mlu.reshape(-1, 1)*np_tm
    commodities_on_links = cluster_info.paths_to_edges.T @ vars_mlu_tm
    
    capacities_constraints = []
    for k, commodities_on_link_temp in enumerate(commodities_on_links):
        rhs = gp.LinExpr(max_link_util*capacities[k])
        constraint = mlu.addConstr(commodities_on_link_temp <= rhs)
        capacities_constraints.append(constraint)
    # Done with capacity constraints
    
    obj = gp.LinExpr(max_link_util)
    model_obj = mlu.setObjective(obj, GRB.MINIMIZE)
    mlu.optimize()
    
    if mlu.status == GRB.status.OPTIMAL or mlu.status == GRB.OPTIMAL:
        if mlu.ObjVal < 0.01:
            continue
        else:
            optimal_values.write(str(round(mlu.objVal, 9))+"\n")
            num_snapshots_in_cluster += 1
            if props.failure_id == None:
                filenames.write(str(topology_filename) + "," + str(pairs_filename) + "," + str(tm_filename) + "\n")

optimal_values.close()
if props.failure_id == None:
    filenames.close()
