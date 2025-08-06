# server_socket.py

import socket
import torch
import time
import pickle
import struct
import csv
import os
from darknet import Darknet #
from util import * #

# --- 설정 및 초기화 ---
print("Loading network.....")
CFG_FILE = "cfg/yolov3.cfg"
WEIGHTS_FILE = "data/yolov3.weights"
model = Darknet(CFG_FILE) #
model.load_weights(WEIGHTS_FILE) #
model.eval() #
print("Network successfully loaded")

CUDA = torch.cuda.is_available() #
if CUDA:
    model.cuda() #

SAVE_DIR = "dataset"
os.makedirs(SAVE_DIR, exist_ok=True)
CSV_HEADER = [
    "repetition_id", "split_point", "client_processing_time_ms",
    "network_transmission_time_ms", "intermediate_data_size_bytes",
    "server_processing_time_ms", "total_time_ms", "timestamp"
]

def log_to_csv(log_data, split_point):
    filepath = os.path.join(SAVE_DIR, f"log_sp_{split_point}.csv")
    file_exists = os.path.exists(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)
        writer.writerow(log_data)

def recv_msg(sock):
    raw_msglen = sock.recv(4)
    if not raw_msglen: return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    data = b''
    while len(data) < msglen:
        packet = sock.recv(msglen - len(data))
        if not packet: return None
        data += packet
    return data

def server_process_and_log(model, received_data):
    server_start_time = time.time()
    experiment_data = pickle.loads(received_data)
    
    intermediate_output = experiment_data['outputs']
    client_detections = experiment_data['detections']
    split_layer = experiment_data['split_point']
    
    rep_id = experiment_data['repetition_id']
    client_time_ms = experiment_data['client_processing_time_ms']
    network_time_ms = experiment_data['network_transmission_time_ms']
    data_size_bytes = experiment_data['intermediate_data_size_bytes']

    if CUDA:
        for key, tensor in intermediate_output.items():
            intermediate_output[key] = tensor.cuda()

    with torch.no_grad():
        prediction = model.forward_server(intermediate_output, split_layer, client_detections) #
        if prediction is not None:
            write_results(prediction, confidence=0.5, num_classes=80, nms_conf=0.4) #

    server_time_ms = (time.time() - server_start_time) * 1000

    total_time_ms = client_time_ms + network_time_ms + server_time_ms
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    log_row = [
        rep_id, split_layer, f"{client_time_ms:.3f}", f"{network_time_ms:.3f}",
        data_size_bytes, f"{server_time_ms:.3f}", f"{total_time_ms:.3f}", timestamp
    ]
    log_to_csv(log_row, split_layer)
    print(f"Logged to 'log_sp_{split_layer}.csv': Rep {rep_id}, Total Time (C+N+S) {total_time_ms:.2f}ms")

def main():
    HOST = '0.0.0.0'
    PORT = 12345

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f'Server listening on {HOST}:{PORT}')

    client_socket, client_address = server_socket.accept()
    print(f'Connected to {client_address}')
    
    try:
        while True:
            data = recv_msg(client_socket)
            if data is None:
                print("Client disconnected.")
                break
            
            server_process_and_log(model, data)
            client_socket.sendall(b'DONE')
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Closing connection. Log files are saved in 'dataset' directory.")
        client_socket.close()
        server_socket.close()

if __name__ == '__main__':
    main()