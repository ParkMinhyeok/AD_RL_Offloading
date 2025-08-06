# detect.py (수정된 최종 버전)

from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd 
import random

# --- 함수 및 전역 변수 정의 (여기까지는 그대로 둡니다) ---

def client_process(model, batch, split_layer, required_layers):
    with torch.no_grad():
        outputs, detections = model.forward_client(Variable(batch), split_layer)
        
        optimized_outputs = {}
        for layer_idx in required_layers:
            if layer_idx in outputs:
                optimized_outputs[layer_idx] = outputs[layer_idx]
            else:
                print(f"Warning: Required layer {layer_idx} not found in client outputs.")

        # 파일 저장 로직은 client_socket.py 등에서 직접 처리하므로 여기서는 데이터만 반환하도록 변경 가능
        # 이 예제에서는 원본 구조를 유지
        torch.save({
            'outputs': optimized_outputs,
            'detections': detections
        }, 'intermediate_output.pth')
  
        print(f"Total client outputs: {len(outputs)}, Sent outputs: {len(optimized_outputs)}")

def server_process(model, split_layer, confidence, num_classes, nms_thesh):    
    data_from_client = torch.load('intermediate_output.pth', weights_only=False)
    intermediate_output = data_from_client['outputs']
    client_detections = data_from_client['detections']

    with torch.no_grad():
        prediction = model.forward_server(intermediate_output, split_layer, client_detections)
        
        if prediction is None:
            return 0

        prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thesh)
        return prediction

def arg_parse():
    """
    Parse arguements to the detect module
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--images", dest = 'images', help = "Image / Directory containing images to perform detection upon", default = "./", type = str)
    parser.add_argument("--det", dest = 'det', help = "Image / Directory to store detections to", default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = "Config file", default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = "weightsfile", default = "data/yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed", default = "416", type = str)
    return parser.parse_args()

LAYER_DEPENDENCIES = [
    [0], [0], [1], [1, 2], [1, 3], [4], [5], [5, 6], [5, 7], [8], [8, 9],
    [8, 10], [11], [12], [12, 13], [12, 14], [15], [15, 16], [15, 17], [18], [18, 19],
    [18, 20], [21], [21, 22], [21, 23], [24], [24, 25], [24, 26], [27], [27, 28], [27, 29],
    [30], [30, 31], [30, 32], [33], [33, 34], [33, 35], [36], [36, 37], [36, 37, 38], [36, 37, 39],
    [36, 40], [36, 40, 41], [36, 40, 42], [36, 43], [36, 43, 44], [36, 43, 45], [36, 46], [36, 46, 47],
    [36, 46, 48], [36, 49], [36, 49, 50], [36, 49, 51], [36, 52], [36, 52, 53], [36, 52, 54],
    [36, 55], [36, 55, 56], [36, 55, 57], [36, 58], [36, 58, 59], [36, 58, 60], [36, 61],
    [36, 61, 62], [36, 61, 62, 63], [36, 61, 62, 64], [36, 61, 65], [36, 61, 65, 66], [36, 61, 65, 67],
    [36, 61, 68], [36, 61, 68, 69], [36, 61, 68, 70], [36, 61, 71], [36, 61, 71, 72], [36, 61, 71, 73],
    [36, 61, 74], [36, 61, 75], [36, 61, 76], [36, 61, 77], [36, 61, 78], [36, 61, 79],
    [36, 61, 79, 80], [36, 61, 79, 81], [36, 61, 79, 82], [36, 61, 83], [36, 61, 84], [36, 61, 85],
    [36, 86], [36, 87], [36, 88], [36, 89], [36, 90], [36, 91], [36, 91, 92], [36, 91, 93],
    [36, 91, 94], [36, 95], [36, 96], [36, 97], [98], [99], [100], [101], [102], [103], [104], [105]
]

def get_required_layers_from_client(split_layer, dependencies):
    required_from_client = set()
    # split_layer가 dependencies 리스트의 유효한 인덱스인지 확인
    if 0 <= split_layer < len(dependencies):
        deps_for_first_server_layer = dependencies[split_layer]
        for dep in deps_for_first_server_layer:
            required_from_client.add(dep)
    else:
        # split_layer가 0일 경우, 서버가 모든 것을 처리하므로 클라이언트로부터 필요한 것이 없음
        if split_layer != 0:
             print(f"Warning: split_layer ({split_layer}) is out of bounds for the dependency map.")
                
    return required_from_client


if __name__ == '__main__':
    args = arg_parse()
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()
    print(CUDA)

    num_classes = 80
    classes = load_classes("data/coco.names")

    #Set up the neural network
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    #If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    #Set the model in evaluation mode
    model.eval()

    read_dir = time.time()
    #Detection phase
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()
        
    if not os.path.exists(args.det):
        os.makedirs(args.det)

    load_batch = time.time()
    # imlist에 파일이 없을 경우 loaded_ims가 비어있게 됨
    if not imlist:
        print("Image list is empty. Please check the --images argument path.")
        exit()
        
    loaded_ims = [cv2.imread(x) for x in imlist]
    # imread 실패 시 None이 포함될 수 있으므로 필터링
    loaded_ims = [im for im in loaded_ims if im is not None]
    if not loaded_ims:
        print("Could not read any images from the specified path.")
        exit()

    im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(loaded_ims))]))
    im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)


    leftover = 0
    if (len(im_dim_list) % batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover            
        im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                            len(im_batches))]))  for i in range(num_batches)]  

    write = 0

    if CUDA:
        im_dim_list = im_dim_list.cuda()


    #!!!!!!! split_layer range: (1 ~ 106)
    split_layer = 106
    required_layers = get_required_layers_from_client(split_layer, LAYER_DEPENDENCIES)
    print('required_layers: ', required_layers)

    start_det_loop = time.time()
    for i, batch in enumerate(im_batches):
        start = time.time()
        if CUDA:
            batch = batch.cuda()
        with torch.no_grad():
            prediction = model(batch, CUDA)
        prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thesh)

        end = time.time()

        if type(prediction) == int:

            for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
                im_id = i*batch_size + im_num
                print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
                print("{0:20s} {1:s}".format("Objects Detected:", ""))
                print("----------------------------------------------------------")
            continue

        prediction[:,0] += i*batch_size

        if not write:
            output = prediction  
            write = 1
        else:
            output = torch.cat((output,prediction))

        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")

        if CUDA:
            torch.cuda.synchronize()       
    try:
        output
    except NameError:
        print ("No detections were made")
        exit()

    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
    scaling_factor = torch.min(416/im_dim_list,1)[0].view(-1,1)
    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
    output[:,1:5] /= scaling_factor
    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
        
    output_recast = time.time()
    class_load = time.time()
    colors = pkl.load(open("pallete", "rb"))
    draw = time.time()

    def write(x, results):
        c1 = (int(x[1]), int(x[2]))
        c2 = (int(x[3]), int(x[4]))
        img = results[int(x[0])]
        cls = int(x[-1])
        color = random.choice(colors)
        label = "{0}".format(classes[cls])
        cv2.rectangle(img, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
        # cv2.imshow('res', img) # imshow를 사용하면 GUI 창이 필요하므로 주석 처리
        # cv2.waitKey(0)
        return img

    list(map(lambda x: write(x, loaded_ims), output))
    det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))
    list(map(cv2.imwrite, det_names, loaded_ims))
    end = time.time()

    print("SUMMARY")
    print("----------------------------------------------------------")
    # ... (SUMMARY 출력 부분 그대로) ...
    torch.cuda.empty_cache()