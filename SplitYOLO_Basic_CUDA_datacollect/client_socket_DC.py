# client_socket.py

import socket
import torch
import time
import pickle
import struct
import cv2
import os
from torch.autograd import Variable
from darknet import Darknet #
from util import * #
from detect import LAYER_DEPENDENCIES, get_required_layers_from_client #

# --- 실험 파라미터 ---
REPETITIONS = 50
SPLIT_RANGE = range(1, 107)

# --- 설정 및 초기화 ---
print("Loading network.....")
CFG_FILE = "cfg/yolov3.cfg"
WEIGHTS_FILE = "data/yolov3.weights"
model = Darknet(CFG_FILE) #
model.load_weights(WEIGHTS_FILE) #
model.eval() #
print("Network successfully loaded")
CUDA = False # 라즈베리파이는 CPU 사용 가정

# --- 소켓 통신 및 처리 ---
def send_msg(sock, msg):
    serialized_msg = pickle.dumps(msg)
    msg_len = len(serialized_msg)
    sock.sendall(struct.pack('>I', msg_len))
    sock.sendall(serialized_msg)
    return msg_len

def client_process(model, batch, split_layer):
    required_layers = get_required_layers_from_client(split_layer, LAYER_DEPENDENCIES) #
    with torch.no_grad(): #
        outputs, detections = model.forward_client(Variable(batch), split_layer) #
        optimized_outputs = {k: v for k, v in outputs.items() if k in required_layers}
    return {'outputs': optimized_outputs, 'detections': detections}

# --- 메인 루프 ---
def main():
    SERVER_IP = '192.168.0.36'
    SERVER_PORT = 12345
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam!")
        return
        
    try:
        client_socket.connect((SERVER_IP, SERVER_PORT))
        print(f"Connected to server at {SERVER_IP}:{SERVER_PORT}.")
        
        inp_dim = int(model.net_info["height"]) #
        
        print("Starting data collection experiment using webcam stream...")

        previous_net_time_ms = 0.0

        for sp in SPLIT_RANGE:
            for rep in range(1, REPETITIONS + 1):
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame from webcam. Aborting.")
                    break
                
                cv2.imshow('Webcam Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt("Quit signal received.")

                client_start_time = time.time()
                
                batch = prep_image(frame, inp_dim) #
                if CUDA: batch = batch.cuda()
                
                data_to_send = client_process(model, batch, sp)
                client_time_ms = (time.time() - client_start_time) * 1000
                
                data_size_bytes = len(pickle.dumps(data_to_send))
                
                data_to_send.update({
                    'repetition_id': rep,
                    'split_point': sp,
                    'client_processing_time_ms': client_time_ms,
                    'intermediate_data_size_bytes': data_size_bytes,
                    'network_transmission_time_ms': previous_net_time_ms 
                })

                net_start_time = time.time()
                send_msg(client_socket, data_to_send)
                current_net_time_ms = (time.time() - net_start_time) * 1000

                previous_net_time_ms = current_net_time_ms

                print(f"Rep {rep}/{REPETITIONS}, SP {sp} | Client Time: {client_time_ms:.2f}ms | Network Time (just sent): {current_net_time_ms:.2f}ms")

                response = client_socket.recv(1024)
                if response != b'DONE':
                    print("Error: Server did not respond correctly. Aborting.")
                    break
            if response != b'DONE': break
        
    except (KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            print("\nExperiment interrupted by user.")
        else:
            print(f"An error occurred: {e}")
    finally:
        print("Experiment finished. Closing resources.")
        cap.release()
        cv2.destroyAllWindows()
        client_socket.close()

if __name__ == '__main__':
    main()