from typing import List, Tuple, Union, Dict
from models import CNN
from fedcom_client import fedcom_client
from copy import deepcopy
import torch
import flwr as fl
import numpy as np
import random
from flwr.common import Metrics
from flwr.common import FitIns, FitRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy.aggregate import weighted_loss_avg, aggregate
from flwr.common.logger import log
from logging import WARNING
from cifar100dataset import cifar100Dataset
from util import get_filters, get_parameters, set_filters, spu_aggregation, compute_sum, parameters_to_ndarrays, top_k_sparsification
CHANNEL = 3
Batch = 128
CLASSES = 100
OTHER_PARAMS = ['bn1.num_batches_tracked', 'bn2.num_batches_tracked']

import logging
import sys
import os
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),     # Write to log file
        logging.StreamHandler(sys.stdout),       # Also print to console
    ]
)

logger = logging.getLogger()

# Create the 'results' folder if it doesn't exist
os.makedirs("results", exist_ok=True)
# Generate a unique timestamped filename
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"results/Fedcom_{timestamp_str}.txt"



class fedcom_strategy(fl.server.strategy.FedAvg):
    def __init__(self, ff, fe, mfc, mec, mac, ACC=[], ClientsSelection=[]):
        super().__init__(fraction_fit=ff, fraction_evaluate=fe, min_fit_clients=mfc, min_evaluate_clients=mec, min_available_clients=mac, evaluate_metrics_aggregation_fn=weighted_average)
        self.fraction_fit_=ff,
        self.fraction_evaluate_=fe,
        self.min_fit_clients_=mfc,
        self.min_evaluate_clients_=mec,
        self.min_available_clients_=mac
        self.global_model = CNN(CHANNEL, outputs=CLASSES)
        self.accuracy_record = ACC
        self.ClientMasks = {}
        self.local_residuals = {}
        self.local_models = {}
        self.droprate = 0.6
        self.latest_local_update = None
        self._round_resource: Dict[int, Dict[str, float]] = {}
        self.total_bytes = 0     # cumulative bytes
        self.total_wall  = 0.0
        self.efficiency_log: List[Tuple[int, float, float]] = []
        for i in range(mac):
            self.local_residuals[i] = None
        for i in range(mac):
            self.local_models[i] = get_filters(self.global_model)

    def record_test_accuracy(self, acc):
        self.accuracy_record.append(acc)

    """override"""
    def initialize_parameters(self, client_manager: ClientManager):
        return ndarrays_to_parameters(get_parameters(self.global_model))
    
    """override"""
    def configure_fit(self, server_round: int, parameters, client_manager: ClientManager):
        random.seed(random.randint(0, server_round))
        sample_size, min_num_clients = super().num_fit_clients(client_manager.num_available()) 
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        config_fit_list = []
        for client in clients:
            cid = int(client.cid)
            config = {}
            config['Residual'] = self.local_residuals[cid]
            sub_parameters = get_filters(self.global_model)
            fit_ins = FitIns(ndarrays_to_parameters(sub_parameters), config)
            config_fit_list.append((client, fit_ins))
        return config_fit_list
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]):
      """override"""
      """Aggregate fit results using weighted average."""
      if not results:
        return None, {}
      # Do not aggregate if there are failures and failures are not accepted
      if not self.accept_failures and failures:
        return None, {}
      # Convert results
      bytes_up = bytes_down = 0
      wall = 0.0
      Fit_res = []
      for client, fit_res in results:
        cid = client.cid
        param, num = parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples
        m = fit_res.metrics
        bytes_up += int(m.get("upload_bytes", 0))
        bytes_down += int(m.get("download_bytes", 0))
        wall += float(m.get("train_time", 0.0))
        Fit_res.append((param, 1))
        self.local_residuals[int(cid)] = fit_res.metrics["Residual"]
        self.local_models[int(cid)] = fit_res.metrics["personal model"]
      #for params, size, rate in weights_results:
      aggregated_updates = aggregate(Fit_res)
      current_global_model = get_filters(self.global_model)
      self.latest_local_update = top_k_sparsification(self.droprate, aggregated_updates)
      new_model = compute_sum(current_global_model, aggregated_updates)
      metrics_aggregated = {}
      self._round_resource[server_round] = {"bytes_up": bytes_up, "bytes_down": bytes_down, "wall": wall}
      if self.fit_metrics_aggregation_fn:
          fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
          metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
      elif server_round == 1:  # Only log this warning once
          log(WARNING, "No fit_metrics_aggregation_fn provided")
      set_filters(self.global_model, new_model)
      return new_model, metrics_aggregated
    
    def configure_evaluate(self, server_round: int, parameters, client_manager: ClientManager):
        """override"""
        if self.fraction_evaluate_ == 0.0:
            return []
        # Sample clients
        sample_size, min_num_clients = super().num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        config_evaluate_list = []
        parameters = get_filters(self.global_model)
        for client in clients:
            config = {}
            #parameters = self.personal_models[int(client.cid)]
            fit_ins = FitIns(ndarrays_to_parameters(parameters), config)
            config_evaluate_list.append((client, fit_ins))
        return config_evaluate_list
    
    def aggregate_evaluate(self, server_round: int, results, failures):
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(1, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
            prev_acc = self.accuracy_record[-1] if self.accuracy_record else 0.0
            curr_acc = metrics_aggregated.get("accuracy", 0.0)
            delta_acc = metrics_aggregated.get("accuracy", 0.0) - prev_acc
            self.accuracy_record.append(curr_acc)

            res = self._round_resource.get(server_round, {"bytes_up": 0, "bytes_down": 0, "wall": 0.0})
            tot_bytes = res["bytes_up"] + res["bytes_down"]
            self.total_bytes += tot_bytes
            self.total_wall  += res["wall"]
            delta_comm_eff = delta_acc / max(self.total_bytes, 1)
            delta_comp_eff = delta_acc / max(self.total_wall, 1e-6)
            comm_eff = curr_acc / max(self.total_bytes, 1)
            comp_eff = curr_acc / max(self.total_wall, 1e-6)
            self.efficiency_log.append((server_round, comm_eff, comp_eff))

            self.record_test_accuracy(metrics_aggregated['accuracy'])

            log_lines = [
                f"Round {server_round}: test_acc={curr_acc:.12f}, "
                f"bytes={self.total_bytes/1e6:.8f} MB, time={self.total_wall:.8f}s, "
                f"delta_acc/byte={delta_comm_eff:.12e}, delta_acc/sec={delta_comp_eff:.12f},"
                f"acc/byte={comm_eff:.12e}, acc/sec={comp_eff:.12f}, "
                "\n" 
            ]

            with open(log_filename, "a",  encoding="utf-8") as f:
                    f.writelines(log_lines)

        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
        return loss_aggregated, metrics_aggregated
    
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
  # Multiply accuracy of each client by number of examples used
  accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
  examples = [num_examples for num_examples, _ in metrics]
  # Aggregate and return custom metric (weighted average)
  return {"accuracy": sum(accuracies) / sum(examples)}

def fedcom_client_fn(cid) -> fedcom_client:
  Epoch = 5
  drop_rate = 0.6
  dataset = cifar100Dataset("clientdata/cifar100_client_"+ str(cid) + "_ALPHA_0.1.csv")
  return fedcom_client(cid, dataset, drop_rate, Epoch, Batch)
