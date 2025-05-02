# FLower:
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common import Code, EvaluateIns, EvaluateRes, FitRes, Status
# other dependecies:
from models import CNN
import torch
from torch.utils.data import DataLoader, random_split
from typing import Dict
from util import set_filters, get_filters
from flwr.common import Code, EvaluateIns, EvaluateRes, FitIns, FitRes, Status
from flwr.server.client_manager import SimpleClientManager
from typing import Dict, Optional, List
from logging import INFO
from flwr.common.logger import log
from flwr.server.criterion import Criterion
import numpy as np
import time
from util import params_bytes

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")# Try "cuda" to train on GPU
CLASSES = 62
CHANNELS = 1

class FLrceSPRT_client(fl.client.Client):
    def __init__(self, cid, dataset, epoch, batch):
        self.cid = cid
        self.model = CNN(in_channels=CHANNELS, outputs=CLASSES).to(DEVICE)
        self.local_epoch = epoch
        self.local_batch_size= batch
        len_train = int(len(dataset) * 0.7)
        len_test = len(dataset) - len_train
        ds_train, ds_val = random_split(dataset, [len_train, len_test], torch.Generator().manual_seed(2239))
        self.trainloader = DataLoader(ds_train, self.local_batch_size, shuffle=True)
        self.testloader = DataLoader(ds_val, self.local_batch_size, shuffle=False)
    
    def fit(self, ins: FitIns) -> FitRes:
        # Deserialize parameters to NumPy ndarray's
        global_params_nd = parameters_to_ndarrays(ins.parameters)
        set_filters(self.model, global_params_nd)
        #_, accuracy = self.test()

        #count time
        local_time = time.perf_counter()
        self.train()
        train_time = time.perf_counter() - local_time

        updated_nd = get_filters(self.model)          # full model upload
        upload_bytes   = params_bytes(updated_nd)
        download_bytes = params_bytes(global_params_nd)

        status = Status(code=Code.OK, message="Success")
        return FitRes(status=status, 
                      parameters=ndarrays_to_parameters(updated_nd), 
                      num_examples=len(self.trainloader),         
                      metrics={
                        "upload_bytes":   upload_bytes,
                        "download_bytes": download_bytes,
                        "train_time":     train_time,
                        # "accuracy":     local_acc,   # keep if you already report it
                      },
        )
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)
        set_filters(self.model, ndarrays_original)
        loss, accuracy = self.test()
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.testloader),
            metrics={"accuracy": float(accuracy)},
        )
    
    def train(self):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=2e-4)
        self.model.train()
        for e in range(self.local_epoch):
            for samples, labels in self.trainloader:
                samples, labels = samples.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = self.model(samples)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def test(self):
        """Evaluate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        self.model.eval()
        with torch.no_grad():
            for samples, labels in self.testloader:
                samples, labels = samples.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(samples)
                loss = criterion(outputs, labels).item() * labels.size(0)
                total += labels.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += predicted.eq(labels).sum()
        loss = loss / total
        accuracy = correct / total
        return loss, accuracy

class FLrceSPRT_client_manager(SimpleClientManager):
    def sample(self, num_clients, exploit_factor:Optional[float]=None, utility_scores_map:Optional[Dict]=None, explore_map:Optional[Dict]=None, min_num_clients:Optional[int]=None, criterion:Optional[Criterion]=None):
        # Block until at least num_clients are connected.
        if exploit_factor == None or utility_scores_map == None or len(utility_scores_map) == 0:
            return super().sample(num_clients, min_num_clients, criterion)
        if min_num_clients is None:
            min_num_clients = num_clients
        super().wait_for(min_num_clients)
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]
        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []
        sampled_cids = []
        sorted_score_map = sorted(utility_scores_map.items(), key=lambda x:x[1], reverse=True)
        # select clients with the highest utility score (exploit):
        if exploit_factor:
            topk = top_k_utility_clients(num_clients, sorted_score_map)
            for cid in available_cids:
                if len(sampled_cids) >= num_clients:
                    break
                if cid in topk:
                    sampled_cids.append(self.clients[cid])
            return sampled_cids
        else:
            return super().sample(num_clients, min_num_clients, criterion)
        
def top_k_utility_clients(num_clients, map):
    IDs = []
    for k in map:
        IDs.append(k[0])
        if len(IDs) >= num_clients:
            break
    return IDs