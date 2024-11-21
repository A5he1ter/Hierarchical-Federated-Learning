import torch
import torch.nn.functional as F
import os

device = None
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def Adding_Trigger(data):
    if data.shape[0] == 3:
        for i in range(3):
            data[i][1][28] = 1
            data[i][1][29] = 1
            data[i][1][30] = 1
            data[i][2][29] = 1
            data[i][3][28] = 1
            data[i][4][29] = 1
            data[i][5][28] = 1
            data[i][5][29] = 1
            data[i][5][30] = 1

    if data.shape[0] == 1:
        data[0][1][24] = 1
        data[0][1][25] = 1
        data[0][1][26] = 1
        data[0][2][24] = 1
        data[0][3][25] = 1
        data[0][4][26] = 1
        data[0][5][24] = 1
        data[0][5][25] = 1
        data[0][5][26] = 1
    return data

def euclidean_clients(param_matrix):
    dev = device
    param_tf = torch.FloatTensor(param_matrix).to(dev)
    output = torch.cdist(param_tf, param_tf, p=2)

    return output.tolist()

def launch_tensor_board(log_path, port, host):
    """Function for initiating TensorBoard.

    Args:
        log_path: Path where the log is stored.
        port: Port number used for launching TensorBoard.
        host: Address used for launching TensorBoard.
    """
    os.system(f"tensorboard --logdir={log_path} --port={port} --host={host}")
    return True

def test_accuracy(server):
    loss_collector = []
    criterion = F.cross_entropy
    with torch.no_grad():
        sum_accu = 0
        num = 0
        loss_collector.clear()
        # 载入测试集
        for data, label in server.eval_loader:
            data, label = data.to(device), label.to(device)
            preds = server.global_model(data)
            loss = criterion(preds, label.long())
            # loss = 1
            loss_collector.append(loss.item())
            preds = torch.argmax(preds, dim=1)
            sum_accu += (preds == label).float().mean()
            num += 1

        accuracy = sum_accu / num
        avg_loss = sum(loss_collector) / len(loss_collector)
    return avg_loss, accuracy

def test_attack_success_rate(server):
    with torch.no_grad():
        sum_asr = 0
        count = 0

        for data, label in server.eval_loader:
            data, label = data.to(device), label.to(device)
            for example_id in range(data.shape[0]):
                data[example_id] = Adding_Trigger(data[example_id])

            preds = server.global_model(data)

            preds = torch.argmax(preds, dim=1)

            for i, v in enumerate(preds):
                if v != label[i] and v == 0:
                    count += 1

            sum_asr += data.shape[0]

        asr = count / sum_asr

    return asr

def eval_defense_acc(clients, malicious_clients, detect_malicious_client):
    malicious = []
    for i in malicious_clients:
        malicious.append(i.client_id)
    malicious.sort()
    print("malicious clients", malicious)
    malicious = []
    for i in detect_malicious_client:
        malicious.append(i.client_id)
    malicious.sort()
    print("detected malicious clients", malicious)

    count = 0
    for c in clients:
        if (c in detect_malicious_client) == (c in malicious_clients):
            count += 1

    defense_acc = count / len(clients)

    count1 = 0
    for c in malicious_clients:
        if c in detect_malicious_client:
            count1 += 1

    if len(malicious_clients) != 0:
        malicious_precision = count1 / len(malicious_clients)
    else:
        malicious_precision = 1

    count2 = 0
    for c in detect_malicious_client:
        if c in malicious_clients:
            count2 += 1

    if len(detect_malicious_client) != 0:
        malicious_recall = count2 / len(detect_malicious_client)
    else:
        malicious_recall = 0

    return defense_acc, malicious_precision, malicious_recall