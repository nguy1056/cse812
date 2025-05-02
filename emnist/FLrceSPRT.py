from typing import List, Tuple, Union, Dict
from models import CNN
from FLrceSPRT_client import FLrceSPRT_client
import random
import math
import torch
from copy import deepcopy
import flwr as fl
from flwr.common import Metrics
from flwr.common import FitIns, FitRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common.logger import log
from logging import WARNING
from dataset import EmnistDataset
from es2 import get_topk_effectiveness, max_mean_dist_split
import numpy as np
from util import get_filters, get_orthogonal_distance, compute_update, get_relationship_update_this_round, get_parameters, set_filters, highest_consensus_this_round, get_cosine_similarity
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
log_filename = f"results/FLrceSPRT_{timestamp_str}.txt"

CHANNEL = 1
Batch = 16
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MAX_EXPLOIT_RATE = 1.0
DECAY_FACTOR = 1
CLASSES = 62

class FLrceSPRT_strategy(fl.server.strategy.FedAvg):
    def __init__(self, ff, fe, mfc, mec, mac, accuracies=[], ClientsSelection=[], HighestConsensus=[], AvgConsensus=[], HCperround=[], ESCriteria=[], 
        alpha: float = 0.05,     
        beta: float = 0.10,       
        delta: float = 0.003, ):
        super().__init__(fraction_fit=ff, fraction_evaluate=fe, min_fit_clients=mfc, min_evaluate_clients=mec, min_available_clients=mac, evaluate_metrics_aggregation_fn=weighted_average)
        self.fraction_fit_=ff,
        self.fraction_evaluate_=fe,
        self.min_fit_clients_=mfc,
        self.min_evaluate_clients_=mec,
        self.min_available_clients_=mac
        self.relation_map = np.zeros(mac*mac).reshape((mac, mac))
        self.consesus = np.zeros(mac*mac).reshape((mac, mac))
        self.latest_local_updates = {}
        self.latest_globalparam = {}
        self.exploremap = {}
        self.last_update_round = {}
        self.global_model = CNN(CHANNEL, outputs=CLASSES)
        self.accuracy_record = accuracies
        for i in range(mac):
            self.exploremap[str(i)] = 0
        self.selected_clients_records = ClientsSelection
        self.highest_consensus = HighestConsensus
        self.avgconsensus = AvgConsensus
        self.hcp = HCperround
        self.is_exploit_round = False
        self.stopped = False
        self.earlystopping_round = 0
        self.earlystopping_acc = 0.0
        self.early_stopping_criteria = ESCriteria
        self.EarlyStoppingCriterias = iter([4.5, 5, 5.5, 6])
        self.conflict_threshold = 5
        self.es_criteria = next(self.EarlyStoppingCriterias, -1)
        self.best_model = CNN(CHANNEL, outputs=CLASSES)
        self.highest_test_acc = 0.4
        self.highest_test_round = 0
        self.relation_map_saving = []
        self.earlystopping_round_2 = 999
        self.non_filter_params = {}
        self.cid_to_index = {}
        self.next_index = 0
        self.total_bytes = 0
        self.total_wall = 0
        self._round_resource = {}   
        self.efficiency_log  = []

        # SPRT constants
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.A = math.log((1.0 - beta) / alpha)  
        self.B = math.log(beta / (1.0 - alpha))  
        self.LR = 0.0                          
        self.sigma2 = 1e-5                      
        self.n_var = 0     
        self.sum = 0.0   
        self.mean_r = 0.0    
    
    def get_or_create_index(self, cid: str) -> int:
        """Return a unique integer index (row/column) for this client id."""
        if cid not in self.cid_to_index:
            # If next_index has reached self.min_available_clients_, you’ve run out of space
            # You can either raise an error or handle it gracefully.
            if self.next_index >= self.min_available_clients_:
                raise ValueError(f"No more space in relation_map for cid={cid}.")
            self.cid_to_index[cid] = self.next_index
            self.next_index += 1
        return self.cid_to_index[cid]

    """override"""
    def initialize_parameters(self, client_manager: ClientManager):
        return ndarrays_to_parameters(get_parameters(self.global_model))
    
    def configure_fit(
        self, server_round: int, parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        config_fit_list = []
        scoremap = self.get_effectiveness_map()
        exploremap = self.get_explore_map()
        explore_possibility = math.pow(0.99, max(server_round-1, 0))
        exploit_threshold = min(1 - explore_possibility, MAX_EXPLOIT_RATE)
        exploit_value = random.random()
        if_exploit = exploit_value <= exploit_threshold
        self.is_exploit_round = if_exploit
        clients = client_manager.sample(num_clients=sample_size, exploit_factor=if_exploit, utility_scores_map=scoremap, explore_map=exploremap, min_num_clients=min_num_clients)
        for client in clients:
            cid = int(client.cid)
            config = {}
            parameters = get_filters(self.global_model)
            fit_ins = FitIns(ndarrays_to_parameters(parameters), config)
            config_fit_list.append((client, fit_ins))
        return config_fit_list
    
    def configure_evaluate(self, server_round: int, parameters, client_manager: ClientManager):
        """override"""
        if self.fraction_evaluate_ == 0.0:
            return []
        # Parameters and config
        config = {}
        # Sample clients
        sample_size, min_num_clients = super().num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        config_evaluate_list = []
        for client in clients:
            parameters = get_filters(self.global_model)
            fit_ins = FitIns(ndarrays_to_parameters(parameters), config)
            config_evaluate_list.append((client, fit_ins))
        return config_evaluate_list
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}
        self.latest_conflict = self.get_conflicts(results)

        selected_clients = []
        current_parameter = get_filters(self.global_model)
        oldmap = deepcopy(self.consesus)
        updateDict = deepcopy(self.latest_local_updates)
        received_params = []
        Fitres = []
        bytes_up, bytes_down, wall = 0, 0, 0.0
        for client, fit_res in results:
            m = fit_res.metrics
            bytes_up   += m.get("upload_bytes", 0)
            bytes_down += m.get("download_bytes", 0)
            wall       += m.get("train_time", 0.0)

            Fitres.append(fit_res)
            cid = client.cid
            selected_clients.append(cid)
            param, num = parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples
            self.record_latest_starting_point(cid, param)
            self.set_last_update_round(cid, server_round)
            self.record_latest_local_update(cid, param, current_parameter)
            self.update_relationship(cid, param, server_round, uDict=updateDict)
            received_params.append((param, num))
        for client_id, received_parameter in [(client.cid, parameters_to_ndarrays(fit_res.parameters)) for (client, fit_res) in results]:
            self.update_current_relationship(client_id, received_parameter, results)
        self.save_relationship()
        consensus_update = get_relationship_update_this_round(results, oldmap, self.consesus, self.cid_to_index)
        hc = highest_consensus_this_round(results, oldmap, self.consesus, self.cid_to_index)
        self.record_selected_clients(selected_clients)
        self.record_avg_consensus(consensus_update)
        self.record_hcp(hc)
        # conflicts = self.get_conflicts(results)
        # if self.is_exploit_round:
        #     topk_effectiveness = get_topk_effectiveness(self.min_available_clients_, self.relation_map, len(results))
        #     if self.es_criteria > 0 and conflicts >= self.es_criteria:
        #         self.earlystopping_round =server_round
        #         self.stopped = True
        #         self.es_criteria = next(self.EarlyStoppingCriterias, -1)
        #     elif self.es_criteria > 0 and conflicts >= self.es_criteria - 0.5:
        #         if max_mean_dist_split(topk_effectiveness) <= 1:
        #             self.earlystopping_round_2 = server_round
        #             self.record_criteria_acc_round()
        parameters_aggregated = aggregate(received_params)
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        set_filters(self.global_model, parameters_aggregated)
        self._round_resource[server_round] = {
            "bytes_up":   bytes_up,
            "bytes_down": bytes_down,
            "wall":       wall,
        }
        self.highest_consensus.append(self.get_highest_consensus())
        return parameters_aggregated, metrics_aggregated
    
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
        log_lines: List[str] = []
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(1, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)

            prev_acc = self.accuracy_record[-1] if self.accuracy_record else 0.0
            curr_acc = metrics_aggregated.get("accuracy", 0.0)
            delta_acc = metrics_aggregated.get("accuracy", 0.0) - prev_acc
            self.accuracy_record.append(curr_acc)

            #SPRT
            r_t = curr_acc - prev_acc
            sprt_stop = self.update_SPRT(r_t)
            conflict_stop = self.latest_conflict >= self.conflict_threshold
            stop_flag = sprt_stop and conflict_stop
            if stop_flag:
                self.stopped = True
                self.earlystopping_round = server_round
                self.earlystopping_acc = curr_acc
                log_lines.append(
                    "SPRT stop at round {} | acc={:.4f} | LR={:.2f}\n".format(server_round, curr_acc, self.LR)
                )

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

            log_lines.extend( [
                f"Round {server_round}: test_acc={curr_acc:.12f}, "
                f"bytes={self.total_bytes/1e6:.8f} MB, time={self.total_wall:.8f}s, "
                f"delta_acc/byte={delta_comm_eff:.12e}, delta_acc/sec={delta_comp_eff:.12f},"
                f"acc/byte={comm_eff:.12e}, acc/sec={comp_eff:.12f}, "
                "\n" 
            ]
            )

            # Append early‑stop info only once
            if self.stopped and server_round != self.earlystopping_round:
                log_lines.append(
                    f"Early Stopped at round {self.earlystopping_round}, "
                    f"test accuracy = {self.earlystopping_acc}\n",
                )
            elif self.stopped and server_round == self.earlystopping_round:
                self.earlystopping_acc = curr_acc
                self.record_criteria_acc_round()

            with open(log_filename, "a",  encoding="utf-8") as f:
                f.writelines(log_lines)

            # Console summary (one per round)
            logger.info(
                "Round %s | Exploit=%s | acc=%.4f | Δacc/byte=%.2e | Δacc/sec=%.2f",
                server_round,
                self.is_exploit_round,
                curr_acc,
                comm_eff,
                comp_eff,
            )
        elif server_round == 1:
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
            
        return loss_aggregated, metrics_aggregated

    def get_effectiveness_map(self):
        map = {}
        for i in range(self.min_available_clients_):
            map[str(i)] = sum(self.relation_map[i]) - self.relation_map[i][i]
        return map 
    
    def get_effectiveness_rank(self, id:str):
        rank = 1
        map = self.get_effectiveness_map()
        for i in map.keys():
            if i != id and map[i] > map[id]:
                rank += 1
        return rank

    def set_last_update_round(self, id:str, server_round):
        self.last_update_round[id] = server_round

    def get_last_update_round(self, id:str):
        return self.last_update_round[id]

    def get_explore_map(self):
        return self.exploremap
    
    def record_latest_local_update(self, id:str, local_param:List[np.ndarray], current_global_param:List[np.ndarray]):
        self.latest_local_updates[id] = compute_update(local_param, current_global_param)
    
    def record_latest_starting_point(self, id:str, starting_point:List[np.ndarray]):
        self.latest_globalparam[id] = starting_point

    def record_explore(self, cid):
        self.exploremap[cid] += 1

    def record_test_accuracy(self, acc):
        self.accuracy_record.append(acc)

    def save_relationship(self):
        new_relation_saving = []
        for i in range(self.min_available_clients_):
            relations = self.relation_map[i]
            values_to_s = [f'{r:.4f}' for r in relations]
            new_relation_saving.append(" ".join(values_to_s))
        self.relation_map_saving = new_relation_saving

    def record_criteria_acc_round(self):
        self.early_stopping_criteria.append(" ".join([str(self.es_criteria), str(self.earlystopping_round), str(self.earlystopping_acc), str(self.highest_test_round), '2e'+str(self.earlystopping_round_2)]))
    
    def record_selected_clients(self, clients:List[str]):
        self.selected_clients_records.append(" ".join(clients))

    def record_avg_consensus(self, value):
        self.avgconsensus.append(value)

    def record_hcp(self, cp):
        self.hcp.append(cp)

    def update_relationship(self, id:str, new_parameter:List[np.ndarray], server_round, uDict=None, Alpha=0.9, Decay_factor=DECAY_FACTOR):
        current_global_parameter = get_filters(self.global_model)
        new_parameter_merged = new_parameter
        this_update = compute_update(new_parameter_merged, current_global_parameter)
        if uDict == None:
            uDict = self.latest_local_updates
        for k in uDict.keys():
            if k != id:
                starting_point = self.latest_globalparam[k]
                last_round = self.get_last_update_round(k)
                new_update = compute_update(new_parameter_merged, starting_point)
                old_update = compute_update(current_global_parameter, starting_point)
                local_update = self.latest_local_updates[k]
                if server_round - last_round <= 1:
                    self.relation_map[self.get_or_create_index(str(id))][self.get_or_create_index(str(k))] = (1-Alpha)*self.relation_map[self.get_or_create_index(str(id))][self.get_or_create_index(str(k))] + Alpha * get_cosine_similarity(this_update, local_update)
                elif Decay_factor > 0.0:
                    distance1 = get_orthogonal_distance(old_update, local_update)
                    distance2 = get_orthogonal_distance(new_update, local_update)
                    new_value = max((distance1 - distance2) / (distance1 + 1e-5), -1)
                    old_value = self.relation_map[self.get_or_create_index(str(id))][self.get_or_create_index(str(k))]
                    if new_value >= old_value:
                        self.consesus[self.get_or_create_index(str(id))][self.get_or_create_index(str(k))] = 1
                    else:
                        self.consesus[self.get_or_create_index(str(id))][self.get_or_create_index(str(k))] = -1
                    self.relation_map[self.get_or_create_index(str(id))][self.get_or_create_index(str(k))] = (1-Alpha)*old_value + Alpha * new_value * math.pow(DECAY_FACTOR, server_round-last_round+1)

    def update_current_relationship(self, id:str, my_parameter, results, Alpha=0.9):
        if len(results) > 1:
            global_parameter = get_filters(self.global_model)
            this_update = compute_update(my_parameter, global_parameter)
            for client, fitres in results:
                k = client.cid
                new_local_parameter = parameters_to_ndarrays(fitres.parameters)
                local_update = compute_update(new_local_parameter, global_parameter)
                new_value = get_cosine_similarity(local_update, this_update)
                old_value = self.relation_map[self.get_or_create_index(str(id))][self.get_or_create_index(str(k))]
                if new_value >= old_value:
                    self.consesus[self.get_or_create_index(str(id))][self.get_or_create_index(str(k))] = 1
                else:
                    self.consesus[self.get_or_create_index(str(id))][self.get_or_create_index(str(k))] = -1
                self.relation_map[self.get_or_create_index(str(id))][self.get_or_create_index(str(k))] = (1-Alpha)*old_value + Alpha*new_value

    def get_highest_consensus(self) -> int:
        highest_value = -self.min_available_clients_
        for i in range(self.min_available_clients_):
            consesus_peers = sum(self.consesus[i])
            if consesus_peers >= highest_value:
                highest_value = consesus_peers
        return highest_value
    
    def get_conflicts(self, results):
        global_parameter = get_filters(self.global_model)
        total = 0
        num_clients = 0
        for client, fitres in results:
            num_clients += 1
            id = client.cid
            new_local_parameter = parameters_to_ndarrays(fitres.parameters)
            local_update = compute_update(new_local_parameter, get_filters(self.global_model))
            for k, f in results:
                if k != id:
                    local_parameter_k = parameters_to_ndarrays(f.parameters)
                    local_update_k = compute_update(local_parameter_k, global_parameter)
                    if get_cosine_similarity(local_update, local_update_k) <= 0.0:
                        total += 1
        return total / max(num_clients, 1) 
    
    def update_SPRT(self, r_t: float) -> bool:
        self.n_var += 1
        if self.n_var == 1:
            self.mean_r = r_t
            self.sum     = 0.0     
        else:
            delta      = r_t - self.mean_r
            self.mean_r += delta / self.n_var
            self.sum    += delta * (r_t - self.mean_r)
        self.sigma2 = max(self.sum / (self.n_var - 1 + 1e-9), 1e-6)

        ll_h1 = -((r_t - self.delta) ** 2) / (2 * self.sigma2)
        ll_h0 = -(r_t ** 2) / (2 * self.sigma2)
        self.LR += ll_h1 - ll_h0

        return self.LR >= self.A or self.LR <= self.B

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
  # Multiply accuracy of each client by number of examples used
  accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
  examples = [num_examples for num_examples, _ in metrics]
  # Aggregate and return custom metric (weighted average)
  return {"accuracy": sum(accuracies) / sum(examples)}

def FLrceSPRT_client_fn(cid) -> FLrceSPRT_client:
  Epoch = 5
  dataset = EmnistDataset("clientdata/femnist_client_"+ str(cid) + "_ALPHA_0.1.csv")
  return FLrceSPRT_client(cid, dataset, Epoch, Batch)