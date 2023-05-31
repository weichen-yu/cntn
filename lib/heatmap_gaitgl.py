import numpy as np
import pickle
import torch
import os
import argparse
from modeling import models
from utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr
from torch.autograd import Variable
import cv2

parser = argparse.ArgumentParser(description='Main program for opengait.')
parser.add_argument('--local_rank', type=int, default=0,
                    help="passed by torch.distributed.launch module")
parser.add_argument('--cfgs', type=str,
                    default='config/default.yaml', help="path of config file")
parser.add_argument('--phase', default='train',
                    choices=['train', 'test'], help="choose train or test phase")
parser.add_argument('--log_to_file', action='store_true',
                    help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
parser.add_argument('--iter', default=0, help="iter to restore")
opt = parser.parse_args()

def input_preprocess(path):
    pkl_path = open(path,'rb')
    seq = pickle.load(pkl_path)
    seq = seq.astype('float32') / 255.0
    seq = torch.tensor(np.array(seq),requires_grad=True)  #t,h,w
    seq_rgb = seq.unsqueeze(1)
    seq_rgb = torch.cat([seq_rgb,seq_rgb,seq_rgb],dim=1)  #t,3,h,w
    return seq, seq_rgb

def align(seq):
    device = torch.distributed.get_rank()
    t,h,w = seq.shape
    print('------t is ',t)
    seq = Variable(seq)
    seq = [seq.unsqueeze(0).to(device)] #list[1,t,h,w]
    lab = torch.tensor([0]).to(device)
    view = ['000']
    _ = ['nm-01']
    seql = torch.tensor([t]).to(device)
    ipts = [seq,lab,view,_,seql]
    return ipts

def initialization(cfgs, training):
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                               cfgs['model_cfg']['model'], engine_cfg['save_name'])
    if training:
        msg_mgr.init_manager(output_path, opt.log_to_file, engine_cfg['log_iter'],
                             engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], (int)) else 0)
    else:
        msg_mgr.init_logger(output_path, opt.log_to_file)

    msg_mgr.log_info(engine_cfg)

    seed = torch.distributed.get_rank()
    init_seeds(seed)


def model_prepare(opt):
    torch.distributed.init_process_group('nccl', init_method='env://')
    if torch.distributed.get_world_size() != torch.cuda.device_count():
        raise ValueError("Expect number of availuable GPUs({}) equals to the world size({}).".format(
            torch.distributed.get_world_size(), torch.cuda.device_count()))
    cfgs = config_loader(opt.cfgs)
    if opt.iter != 0:
        cfgs['evaluator_cfg']['restore_hint'] = int(opt.iter)
        cfgs['trainer_cfg']['restore_hint'] = int(opt.iter)

    training = (opt.phase == 'train')
    initialization(cfgs, training)

    #run_model(cfgs, training)
    msg_mgr = get_msg_mgr()
    model_cfg = cfgs['model_cfg']
    msg_mgr.log_info(model_cfg)
    Model = getattr(models, model_cfg['model'])
    model = Model(cfgs, training)
    return model


def get_gradients(model,ipts,label):
    seq,lab,_,_,_ = ipts
    seq[0].requires_grad_(True)
    retval = model.forward(ipts)
    logits = retval['training_feat']['softmax']['logits']
    print(logits.shape)
    logits = logits[:,:,label].sum()

    logits.backward(retain_graph=True)
    grad_cam = seq[0].grad[0]  #1*s*h*w
    return grad_cam

def visualize_heat(grad_cam,seq_rgb):
    heat = torch.sum(grad_cam,dim=0)  #s*h*w
    heat = heat/heat.max()*255.0
    heat = heat.abs()
    heat = cv2.applyColorMap(np.uint8(np.array(heat.cpu())), cv2.COLORMAP_JET)
    seq_pick = seq_rgb[0,0].detach().numpy()*128
    #import pdb;pdb.set_trace()
    seq_pick = np.stack([seq_pick,seq_pick,seq_pick],axis=-1)
    overlapped = seq_pick*0.01 + heat*0.99
    overlapped[overlapped > 255] = 255
    overlapped = overlapped*255/overlapped.max()
    path = '/data/ywc/hongyuan/eccv2022_sota/opengait_baseline/001.jpg'
    cv2.imwrite(path, overlapped)
if __name__ == "__main__":
    path = '/home2/ywc//workspace/output_pkl_64/001/nm-01/000/000.pkl'
    model = model_prepare(opt)
    seq,seq_rgb = input_preprocess(path)
    ipts = align(seq)
    label = 0
    grad_cam = get_gradients(model,ipts,label)
    visualize_heat(grad_cam,seq_rgb)