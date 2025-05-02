from datetime import datetime
from FLrceSPRT_client import FLrceSPRT_client_manager
from FLrceSPRT import FLrceSPRT_strategy, FLrceSPRT_client_fn
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
    
    # Create client manager
    client_manager = FLrce_client_manager()
    SPRT_client_manager = FLrceSPRT_client_manager()
    
    strategy = FLrceSPRT_strategy(FF, FE, MFC, MEC, MAC, accuracies=test_acc, ClientsSelection=selected_clients, ESCriteria=earlystopping_records)
    fl.simulation.start_simulation(
        client_fn=FLrceSPRT_client_fn,
        client_manager=SPRT_client_manager,
        num_clients=MAC,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy
    )
    
    now = datetime.now()
    with open('results/FLrce_sprt_accuracies_alpha1.0_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
        for item in test_acc:
            # write each item on a new line
            fp.write("%f\n" % item)
    with open('results/FLrce_sprt_clients_alpha1.0_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
        for item in selected_clients:
            # write each item on a new line
            fp.write("%s\n" % item)
    with open('results/FLrce_sprt_inference_alpha1.0_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
        for item in Inf:
            # write each item on a new line
            fp.write("%f\n" % item)
    with open('results/FLrce_sprt_alpha1.0_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
        for item in earlystopping_records:
            # write each item on a new line
            fp.write("%s\n" % item)

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
    
    # Create client manager
    client_manager = FLrce_client_manager()
    
    strategy = FLrce_strategy(FF, FE, MFC, MEC, MAC, accuracies=test_acc, ClientsSelection=selected_clients, ESCriteria=earlystopping_records)
    fl.simulation.start_simulation(
        client_fn=FLrce_client_fn,
        client_manager=client_manager,
        num_clients=MAC,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy
    )
    
    now = datetime.now()
    with open('results/FLrce_accuracies_alpha1.0_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
        for item in test_acc:
            # write each item on a new line
            fp.write("%f\n" % item)
    with open('results/FLrce_clients_alpha1.0_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
        for item in selected_clients:
            # write each item on a new line
            fp.write("%s\n" % item)
    with open('results/FLrce_inference_alpha1.0_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
        for item in Inf:
            # write each item on a new line
            fp.write("%f\n" % item)
    with open('results/FLrce_earlystopping_alpha1.0_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
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

for i in range(NUM_SIMS):
    randseed = random.randint(0, 99999)
    random.seed(randseed)
    test_acc = []
    selected_clients = []
    strategy = fedprox_strategy(FF, FE, MFC, MEC, MAC, ACC=test_acc, ClientsSelection=selected_clients)
    fl.simulation.start_simulation(
        client_fn=fedprox_client_fn,
        num_clients=MAC,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy
    )
    
    now = datetime.now()
    with open('results/fedprox_accuracies_alpha0.1_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
            for item in test_acc:
                # write each item on a new line
                fp.write("%f\n" % item)

for i in range(NUM_SIMS):
    randseed = random.randint(0, 99999)
    random.seed(randseed)
    test_acc = []
    selected_clients = []
    strategy = dropout_strategy(FF, FE, MFC, MEC, MAC, ACC=test_acc, ClientsSelection=selected_clients)
    fl.simulation.start_simulation(
        client_fn=feddrop_client_fn,
        num_clients=MAC,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy
    )
    
    now = datetime.now()
    with open('results/feddrop_accuracies_alpha0.1_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
            for item in test_acc:
                # write each item on a new line
                fp.write("%f\n" % item)
