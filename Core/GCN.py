import dgl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from Core.RunTime import cal_time
from functools import reduce
from Core.GCNModel import Classifier, ClassifierHetero
from Core.DeviceType import CircuitType


def collate(samples):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    graphs, labels = map(list, zip(*samples))
    return dgl.batch(graphs).to(device), torch.tensor(labels, dtype=torch.float32).to(device)


# Define accuracy function
def accuracy(y_pred, y):
    with torch.no_grad():
        correct = (y_pred == y).sum().item()
        total = y.size(0)
        return correct / total


@cal_time('min')
def train(data, trained_file=None, epoch=100, hidden_dim=32, lr=0.001, batch_size=256, info=False, norm='none',
          out_dim=6, model_type=None):
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if trained_file:
        if torch.cuda.is_available():
            load = torch.load(trained_file)
        else:
            load = torch.load(trained_file, map_location='cpu')
        model = load.to(device)
    else:
        model = Classifier(3, hidden_dim, out_dim, norm).to(device) if model_type is None else ClassifierHetero(
            hidden_dim, out_dim, norm).to(device)
    save_name = f'GCN.pt' if model_type is None else 'GCNHetero.pt'

    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoch):
        epoch_loss = 0
        total_accuracy = 0
        for iter, (batchg, label) in enumerate(data_loader):
            prediction = model(batchg)
            loss = loss_func(prediction, label.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().cpu().item()
            batch_accuracy = accuracy(prediction.max(dim=1).indices, label.long())
            total_accuracy += batch_accuracy

        avg_accuracy = total_accuracy / (iter + 1)
        if epoch % 100 == 0:
            torch.save(model, save_name)
            print(f'Epoch {epoch + 1}, Loss: {epoch_loss / (iter + 1):.4f}, Accuracy: {avg_accuracy:.4f}')


class GCNClassifier:
    def __init__(self, model_path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            load = torch.load(model_path).to(device)
        else:
            load = torch.load(model_path, map_location='cpu')
        self.device = device
        self.model = load.eval()

    def val(self, data):
        result = self.model(data).max(dim=1).indices.detach().cpu().numpy()[0]
        return CircuitType[result]
