import os
import ast
from Core.interface import parseSingleInput, parseSingleInputHeterogeneous
from Core.GCN import train
from Core.DeviceType import CircuitType

_dir = '../net'
inputs = []
labels = []
for file in os.listdir(_dir):
    with open(os.path.join(_dir, file), 'r') as f:
        data = ast.literal_eval(f.read())
        inputs.append(data['ckt_netlist'])
    labels.append(CircuitType.index(data['ckt_type'].strip()))

is_Hetero = False

graphs = []
for index, process in enumerate(inputs):
    # print(os.listdir(_dir)[index])
    graphs.append(parseSingleInput(process) if not is_Hetero else parseSingleInputHeterogeneous(process))
# print(graphs)

# g_view(graphs[0]) if not is_Hetero else g_view_hetero(graphs[0])

# g_view_detail(graphs[0]) if not is_Hetero else g_view_detail_hetero(graphs[0])

graphs = list(zip(graphs, labels))

# norm: ("none", "both", "right", "left")
train(graphs, trained_file=None, hidden_dim=32, epoch=10000, lr=0.01, batch_size=256, info=False, norm='none', out_dim=6,
      model_type='Hetero' if is_Hetero else None)
train(graphs, trained_file='GCNHetero.pt' if is_Hetero else 'GCN.pt', hidden_dim=32, epoch=1000000, lr=0.0003,
      batch_size=256, info=False, norm='none', out_dim=6, model_type='Hetero' if is_Hetero else None)
