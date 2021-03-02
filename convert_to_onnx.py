from __future__ import print_function
import os
import argparse
import torch
#import torch.backends.cudnn as cudnn
import numpy as np
from mask_net import MASK_CLASSIFY
#from data import cfg_mnet, cfg_slim, cfg_rfb
#from layers.functions.prior_box import PriorBox
#from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
#from models.retinaface import RetinaFace
#from models.net_slim import Slim
#from models.net_rfb import RFB
#from utils.box_utils import decode, decode_landm
#from utils.timer import Timer


#parser = argparse.ArgumentParser(description='Test')
#parser.add_argument('-m', '--trained_model', default=r'F:\demo\mask_classify\net_model\mask_classify_198.pt',
                    #type=str, help='Trained state_dict file path to open')
#parser.add_argument('--network', default='RFB', help='Backbone network mobile0.25 or slim or RFB')
#parser.add_argument('--long_side', default=320, help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
#parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')

#args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    train_model_dir = r"F:\demo\project\project\models"
    train_name = "helmet_all.pth"
    #train_model_path = r"F:\demo\mask_classify\net_model\best_mask_classify_model_S.pt" # pytorch 1.5下运行有效，其他 1.0+ 版本应该也有效
    train_model_path = os.path.join(train_model_dir, train_name)
    #save_dir = r'F:\demo\net_frame_convert\pytorch_onnx_caffe\results'
    save_dir = train_model_dir
    #mask_net = MASK_CLASSIFY(num_classes=2)
    #input_size = 112
    # load weight
    #mask_net = load_model(mask_net, train_model_path, 1)
    checkpoint = torch.load(train_model_path, map_location='cpu')  # 保存模型时连 gpu信息 一起保存了，pytorch_0.4 下需要切回 cpu 。
    mask_net = checkpoint
    #mask_net.load_state_dict(checkpoint)
    mask_net.eval()
    print('Finished loading model!')
    #print(net)
    #device = torch.device("cpu" if args.cpu else "cuda")
    device = "cpu"
    mask_net = mask_net.to(device)

    ##################export###############
    #output_onnx = 'mask_classify_S.onnx'
    output_onnx = train_name.split('.')[0] + ".onnx"
    onnx_save_path = os.path.join(save_dir, output_onnx)
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input"]
    output_names = ["output"] # 几个网络输出分支就有几个输出， 一个输入的两个维度不算是两个输出
    inputs = torch.randn(1, 3, input_size, input_size).to(device)
    torch_out = torch.onnx._export(mask_net, inputs, onnx_save_path, export_params=True, verbose=False,
                                   input_names=input_names, output_names=output_names)
    ##################end###############


