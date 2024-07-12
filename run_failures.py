from utils.args_parser import parse_args
import sys
import subprocess

props = parse_args(sys.argv[1:])

if props.topo == "geant":
    num_edges = 36
elif props.topo == "abilene":
    num_edges = 15

for i in range(num_edges):
    subprocess.run(["python3", "frameworks/gurobi_mlu.py", "--topo", f"{props.topo}", 
                "--num_paths_per_pair", f"{props.num_paths_per_pair}",
                "--framework", "gurobi", "--failure_id", f"{i}",
                "--opt_start_idx", f"{props.test_start_idx}",
                "--opt_end_idx", f"{props.test_end_idx}",
                ])
        
    subprocess.run(["python3", "run_harp.py", "--topo", f"{props.topo}", "--mode", "test", 
                "--num_paths_per_pair", f"{props.num_paths_per_pair}", 
                "--framework", "harp", "--failure_id", f"{i}",
                "--test_start_idx", f"{props.test_start_idx}",
                "--test_end_idx", f"{props.test_end_idx}",
                "--test_cluster", f"{props.test_cluster}",
                "--pred", f"{props.pred}",
                "--num_for_loops", f"{props.num_for_loops}",
                ])