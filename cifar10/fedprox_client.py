# FLower:
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common import Code, EvaluateIns, EvaluateRes, FitRes, Status
# other dependecies:
from models import CNN
import torch
from torch.utils.data import DataLoader, random_split
from typing import Dict
from util import set_filters, get_filters, params_bytes
from flwr.common import Code, EvaluateIns, EvaluateRes, FitIns, FitRes, Status
from typing import List
from copy import deepcopy
import numpy as np
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = 10
CHANNELS = 3
OTHER_PARAMS = ['bn1.num_batches_tracked', 'bn2.num_batches_tracked']

class fedprox_client(fl.client.Client):
    def __init__(self, cid, dataset, epoch, batch):
        self.cid = cid
        self.model = CNN(in_channels=CHANNELS, outputs=CLASSES).to(DEVICE)
        self.testmodel = CNN(in_channels=CHANNELS, outputs=CLASSES).to(DEVICE)
        self.local_epoch = epoch
        self.local_batch_size= batch
        len_train = int(len(dataset) * 0.7)
        len_test = len(dataset) - len_train
        ds_train, ds_val = random_split(dataset, [len_train, len_test], torch.Generator().manual_seed(2239))
        self.trainloader = DataLoader(ds_train, self.local_batch_size, shuffle=True)
        self.testloader = DataLoader(ds_val, self.local_batch_size, shuffle=False)
    
    def fit(self, ins: FitIns) -> FitRes:
        global_params_nd = parameters_to_ndarrays(ins.parameters)
        set_filters(self.model, global_params_nd)
        global_model = CNN(in_channels=CHANNELS, outputs=CLASSES).to(DEVICE)
        set_filters(global_model, global_params_nd)

        # ----------------- timed training -------------------------------
        tic = time.perf_counter()
        self.train(global_model)
        train_time = time.perf_counter() - tic

        updated_nd = get_filters(self.model)
        upload_bytes = params_bytes(updated_nd)
        download_bytes = params_bytes(global_params_nd)

        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=ndarrays_to_parameters(updated_nd),
            num_examples=len(self.trainloader),
            metrics={
                "upload_bytes": upload_bytes,
                "download_bytes": download_bytes,
                "train_time": train_time,
            },
        )
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        set_filters(self.testmodel, parameters_to_ndarrays(parameters_original))
        loss, accuracy = self.test()
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.testloader),
            metrics={"accuracy": float(accuracy)},
        )

    def train(self, globalmodel:torch.nn.Module):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.model.train()
        for e in range(self.local_epoch):
            for samples, labels in self.trainloader:
                samples, labels = samples.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                proxmal_term = torch.tensor(0.0000)
                for local_weight, global_weight in zip(self.model.parameters(), globalmodel.parameters()):
                    proxmal_term += (local_weight - global_weight).norm(2)
                outputs = self.model(samples)
                loss = criterion(outputs, labels) + 0.05 * proxmal_term
                loss.backward()
                optimizer.step()

    def test(self):
        """Evaluate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        self.testmodel.eval()
        with torch.no_grad():
            for samples, labels in self.testloader:
                samples, labels = samples.to(DEVICE), labels.to(DEVICE)
                outputs = self.testmodel(samples)
                loss = criterion(outputs, labels).item() * labels.size(0)
                total += labels.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += predicted.eq(labels).sum()
        loss = loss / total
        accuracy = correct / total
        return loss, accuracy
    