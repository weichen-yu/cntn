# coding=utf-8
import os
import os.path as osp
import cv2
import pickle
import numpy as np
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--noise_rate', default=0.1, type=float, help='noise rate')
parser.add_argument('--total_id', default=73, type=int, help='total id num')
parser.add_argument('--src_dir', default='', type=str, help='source dataset path')
parser.add_argument('--des_dir', default='', type=str, help='target noisy dataset path')
parser.add_argument('--random_seed', default=2023, type=int, help='random_seed')

args = parser.parse_args()
pid_num = args.pid_num
noise_rate = args.noise_rate

np.random.seed(args.random_seed)
random.seed(args.random_seed)
# src_size = 128
src_size = 64
if src_size == 128:
    src_dir = args.src_dir
    des_dir = os.path.join(args.des_dir,'rs128')
elif src_size == 64:
    src_dir = args.src_dir
    des_dir = os.path.join(args.des_dir,'rs64')

def process_id(id0):
    id_path = os.path.join(src_dir, id0)
    if random.random > noise_rate:
        new_id = random.randint(1,args.total_id)
        new_id_path = os.path.join(des_dir, new_id)
        os.makedirs(new_id_path, exist_ok=True)
        cmd = 'cp -r {}/* {}'.format(id_path, new_id_path)
        print(cmd)
        os.system(cmd)
    
    return

id_list = sorted(os.listdir(src_dir))
# for id0 in id_list:
#     process_id(id0)
from multiprocessing import Pool
pool = Pool()
pool.map(process_id, id_list)
pool.close()
