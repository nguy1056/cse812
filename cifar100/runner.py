from datetime import datetime
from FLrce_client import FLrce_client_manager
from FLrce import FLrce_strategy, FLrce_client_fn
from fedcom import fedcom_strategy, fedcom_client_fn
from fedprox import fedprox_strategy, fedprox_client_fn
from feddrop import dropout_strategy, feddrop_client_fn
import flwr as fl
import random

NUM_SIMS = 1
ROUNDS = 100
FF = 0.1
FE = 1
MFC = 10
MEC = 100
MAC = 100


for i in range(NUM_SIMS):
    randseed = random.randint(0, 99999)
    random.seed(randseed)
    test_acc = []
    selected_clients = []
    consensus = []
    cu = []
    hcp = []
    Inf = []
    earlystopping_records = []
    strategy = FLrce_strategy(FF, FE, MFC, MEC, MAC, accuracies=test_acc, ClientsSelection=selected_clients, ESCriteria=earlystopping_records)
    fl.simulation.start_simulation(
        client_fn=FLrce_client_fn,
        num_clients=MAC,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
        client_manager=FLrce_client_manager()
    )
    now = datetime.now()
    with open('results/FLrce_accuracies_alpha0.1_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
        for item in test_acc:
            # write each item on a new line
            fp.write("%f\n" % item)
    with open('results/FLrce_clients_alpha0.1_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
        for item in selected_clients:
            # write each item on a new line
            fp.write("%s\n" % item)
    with open('results/FLrce_inference_alpha0.1_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
        for item in Inf:
            # write each item on a new line
            fp.write("%f\n" % item)
    with open('results/FLrce_earlystopping_alpha0.1_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
        for item in earlystopping_records:
            # write each item on a new line
            fp.write("%s\n" % item)

for i in range(NUM_SIMS):
    randseed = random.randint(0, 99999)
    random.seed(randseed)
    test_acc = []
    selected_clients = []
    strategy = fedcom_strategy(FF, FE, MFC, MEC, MAC, ACC=test_acc, ClientsSelection=selected_clients)
    fl.simulation.start_simulation(
        client_fn=fedcom_client_fn,
        num_clients=MAC,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy
    )
    
    now = datetime.now()
    with open('results/fedcom_accuracies_alpha0.1_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
            for item in test_acc:
                # write each item on a new line
                fp.write("%f\n" % item)

# for i in range(NUM_SIMS):
#     randseed = random.randint(0, 99999)
#     random.seed(randseed)
#     test_acc = []
#     selected_clients = []
#     strategy = fedprox_strategy(FF, FE, MFC, MEC, MAC, ACC=test_acc, ClientsSelection=selected_clients)
#     fl.simulation.start_simulation(
#         client_fn=fedprox_client_fn,
#         num_clients=MAC,
#         config=fl.server.ServerConfig(num_rounds=ROUNDS),
#         strategy=strategy
#     )
    
#     now = datetime.now()
#     with open('results/fedprox_accuracies_alpha0.1_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
#             for item in test_acc:
#                 # write each item on a new line
#                 fp.write("%f\n" % item)

# for i in range(NUM_SIMS):
#     randseed = random.randint(0, 99999)
#     random.seed(randseed)
#     test_acc = []
#     selected_clients = []
#     strategy = dropout_strategy(FF, FE, MFC, MEC, MAC, ACC=test_acc, ClientsSelection=selected_clients)
#     fl.simulation.start_simulation(
#         client_fn=feddrop_client_fn,
#         num_clients=MAC,
#         config=fl.server.ServerConfig(num_rounds=ROUNDS),
#         strategy=strategy
#     )
    
#     now = datetime.now()
#     with open('results/feddrop_accuracies_alpha0.1_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
#             for item in test_acc:
#                 # write each item on a new line
#                 fp.write("%f\n" % item)
