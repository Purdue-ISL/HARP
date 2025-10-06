import pandas as pd
import numpy as np 
import os
import pickle
import sys

args = sys.argv[1:]
topo = args[0]
cwd = os.getcwd()
path = f"{cwd}"


try:
    os.mkdir(f"{path}/{topo}_esm")
except:
    pass

def read_pickle(fname):
    file = open(f"{path}/{topo}/{fname}", "rb")
    data = pickle.load(file).reshape(-1)
    file.close()
    
    return data

def write_pickle(fname, data):
    file = open(f"{path}/{topo}_esm/{fname}", "wb")
    pickle.dump(data, file)
    file.close()

tms = [i for i in os.listdir(f"{path}/{topo}") if i.endswith("pkl")]
tms = sorted(tms, key=lambda x: int(x.split(".")[0][1:]))

data = []
for fname in tms:
    tm = read_pickle(fname)
    data.append(tm)
data = np.array(data)

df = pd.DataFrame(data)

esm = df.ewm(alpha=0.5, adjust=False).mean()
esm = esm.iloc[:-1, :]

x = pd.DataFrame(df.iloc[0, :]).transpose()

esm = pd.concat([x, esm], ignore_index=True)
for j, fname in enumerate(tms):
    temp_tm = esm.iloc[j, :].values.reshape(-1, 1)
    write_pickle(fname, temp_tm)
