import json
import os
import numpy as np
import networkx as nx
import pickle
import itertools

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_tms = f"{parent_dir}/traffic_matrices/abilene"
path_topo = f"{parent_dir}/topologies/abilene"
path_pairs = f"{parent_dir}/pairs/abilene"
path_manifest = f"{parent_dir}/manifest"

try:
    os.mkdir(f"{parent_dir}/results")
except FileExistsError:
    pass

try:
    os.mkdir(path_tms)
except FileExistsError:
    pass

try:
    os.mkdir(path_topo)
except FileExistsError:
    pass

try:
    os.mkdir(path_pairs)
except FileExistsError:
    pass


file = open("topo-2003-04-10.txt")
topology = file.readlines()
file.close()

nodes = {}
node_id = 0
routers = topology[2:14]
for router in routers:
    router = router.split("\t")
    router = router[0]
    nodes[router] = node_id
    node_id += 1
    

edges = {}
links = topology[18:]
for link in links:
    link = link.split("\t")
    x = nodes[link[0]]
    y = nodes[link[1]]
    cap = int(link[2].split()[0])/1000000 # Convert unit to Gbps
    edges[(x, y)] = str(cap)
    
digraph = nx.DiGraph()
for node, node_id in nodes.items():
    digraph.add_node(node_id, id=node_id)


# print(sorted(edges.items()))
for edge, capacity in edges.items():
    digraph.add_edge(*edge, capacity=capacity)

data_digraph = nx.readwrite.json_graph.node_link_data(digraph)
with open(f"{path_topo}/t1.json", "w") as f:
    json.dump(data_digraph, f, indent=4)
    

pairs = list(itertools.product(nodes.values(), repeat=2))
pairs = [(x, y) for x, y in pairs if x != y]
pairs = np.array(pairs)
with open(f"{path_pairs}/t1.pkl", "wb")as f:
    pickle.dump(pairs, f)

manifest_file = open(f"{path_manifest}/abilene_manifest.txt", "w")
tms_files = [i for i in os.listdir() if i[0] == "X"]
tms_files = sorted(tms_files)
list_tms = []
tm_id = 1
for filename in tms_files:
    file = open(filename, "r")
    for line in file.readlines():
        matrix = line.split()
        reals = []
        matrix = np.array(matrix, float).reshape(144, 5)
        matrix = matrix[:, 0].reshape(12, 12)
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        matrix = matrix[mask]
        matrix = matrix.ravel()*8*8/(300*10*1000000) # Convert the unit to Mbps
        tm_file = open(f"{path_tms}/t{tm_id}.pkl", "wb")
        pickle.dump(matrix, tm_file)
        manifest_file.write("t1.json" + "," + "t1.pkl" + "," + f"t{tm_id}.pkl" + "\n")
        tm_file.close()
        tm_id += 1
    file.close()

manifest_file.close()
