#!/usr/bin/env python
# coding=utf-8
import os
import os.path as osp
import cv2
import pickle
import numpy as np
import argparse
import random
import deepcopy
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--noise_rate', default=0.1, type=float, help='noise rate')
parser.add_argument('--clothdilate', default=True, type=bool, help='if ClothDilate')
parser.add_argument('--clotherode', default=False, type=bool, help='if ClothErode')
parser.add_argument('--src_dir', default='', type=str, help='source dataset path')
parser.add_argument('--des_dir', default='', type=str, help='target noisy dataset path')
parser.add_argument('--random_seed', default=2023, type=int, help='random_seed')
args = parser.parse_args()
pid_num = args.pid_num

# src_size = 128
src_size = 64
if src_size == 128:
    src_dir = args.src_dir
    des_dir = os.path.join(args.des_dir,'rs128')
elif src_size == 64:
    src_dir = args.src_dir
    des_dir = os.path.join(args.des_dir,'rs64')

def pad_seq(seq, pad_size):
    return np.pad(seq, ([0, 0], [pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]]), mode='constant')

def cut_img(img, T_H, T_W):
    # print("before cut_img: ", img.shape, np.min(img), np.max(img), T_H, T_W, img.dtype)
    # Get the top and bottom point
    y = img.sum(axis=1)
    y_top = (y != 0).argmax(axis=0)
    y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
    img = img[y_top:y_btm + 1, :]
    # As the height of a person is larger than the width,
    # use the height to calculate resize ratio.
    _r = img.shape[1] / img.shape[0]
    _t_w = int(T_H * _r)
    img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)
    # Get the median of x axis and regard it as the x center of the person.
    sum_point = img.sum()
    sum_column = img.sum(axis=0).cumsum()
    x_center = -1
    for i in range(sum_column.size):
        if sum_column[i] > sum_point / 2:
            x_center = i
            break
    h_T_W = int(T_W / 2)
    left = x_center - h_T_W
    right = x_center + h_T_W
    if left <= 0 or right >= img.shape[1]:
        left += h_T_W
        right += h_T_W
        _ = np.zeros((img.shape[0], h_T_W))
        img = np.concatenate([_, img, _], axis=1)
    img = img[:, left:right].astype('uint8')
    # print("after cut_img: ", img.shape, np.min(img), np.max(img), T_H, T_W, img.dtype)
    return img

def array2img(x):
    return (x*255.0).astype('uint8')

def img2array(x):
    return x.astype('float32')/255.0

def save_seq(seq, seq_dir):
    if osp.exists(seq_dir):
        shutil.rmtree(seq_dir)
    if not osp.exists(seq_dir):
        os.makedirs(seq_dir)
    for i in range(seq.shape[0]):
        save_name = osp.join(seq_dir, '{:0>3d}.png'.format(i))
        cv2.imwrite(save_name, array2img(seq[i, :, :]))

def merge_seq(seq, row=6, col=6):
    frames_index = np.arange(seq.shape[0])
    im_h = seq.shape[1]
    im_w = seq.shape[2]
    num_per_im = row*col
    if len(frames_index) < num_per_im:
        selected_frames_index = sorted(np.random.choice(frames_index, num_per_im, replace=True))
    else:
        selected_frames_index = sorted(np.random.choice(frames_index, num_per_im, replace=False))
    im_merged = np.zeros((im_h*row, im_w*col))
    for i in range(len(selected_frames_index)):
        im = seq[selected_frames_index[i], :, :]
        y = int(i/col)
        x = i%col
        im_merged[y*im_h:(y+1)*im_h, x*im_w:(x+1)*im_w] = im
    im_merged = array2img(im_merged)
    return im_merged

class ClothDilate(object):
    def __init__(self, prob=0.5, dilate_pos=[8,56]):
        self.prob = prob
        self.dilate_pos = dilate_pos
        self.dilate_kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            _, dh, dw = seq.shape
            seq = [cut_img(seq[tmp, :, :], dh, dw) for tmp in range(seq.shape[0])]
            img_dilate = deepcopy(seq)
            for tmp in range(len(seq)):
                img_dilate[tmp][self.dilate_pos[0]:self.dilate_pos[1],:] = cv2.dilate(seq[tmp][self.dilate_pos[0]:self.dilate_pos[1],:], self.dilate_kernal, 1)
            img_dilate = np.array(img_dilate).astype('float32')
            return img_dilate

class ClothErode(object):
    def __init__(self, prob=0.5, erode_pos=[8,56]):
        self.prob = prob
        self.erode_pos = erode_pos
        self.erode_kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            _, dh, dw = seq.shape
            seq = [cut_img(seq[tmp, :, :], dh, dw) for tmp in range(seq.shape[0])]
            img_erode = deepcopy(seq)
            for tmp in range(len(seq)):
                img_erode[tmp][self.erode_pos[0]:self.erode_pos[1],:] = cv2.erode(seq[tmp][self.erode_pos[0]:self.erode_pos[1],:], self.erode_kernal, 1)
            img_erode = np.array(img_erode).astype('float32')
            return img_erode

def build_data_transforms(cloth_dilate=False, cloth_erode=False, resolution=64, random_seed=2023):
    np.random.seed(random_seed)
    random.seed(random_seed)
    print("random_seed={} for build_data_transforms".format(random_seed))

    object_list = []
    if cloth_dilate:
        object_list.append(ClothDilate(prob=0.5))
    if cloth_erode:
        object_list.append(ClothErode(prob=0.5))

    transform = T.Compose(object_list)
    return transform

if __name__ == "__main__":
    import pickle
    import matplotlib.pyplot as plt
    SEED = 2020
    np.random.seed(SEED)
    random.seed(SEED)
    
    merge_imgs = {}
    
    example_pkl = '/home/yuweichen/workspace/data64pkl/004/nm-01/018/018.pkl'
    seq_in = pickle.load(open(example_pkl, 'rb'))
    resolution = seq_in.shape[1]
    cut_padding = 10*int(resolution/64)
    seq_in = seq_in[:, :, cut_padding:-cut_padding]
    seq_in = img2array(seq_in)
    seq_dir = './visualize'
    if not os.path.exists(seq_dir):
        os.makedirs(seq_dir)
    save_seq(seq_in, os.path.join(seq_dir, 'raw_seq'))
    merge_imgs.update({'raw':merge_seq(seq_in)})
    print(seq_in.shape, np.min(seq_in), np.max(seq_in), seq_in.dtype)

    transform = build_data_transforms(cloth_dilate=True)
    seq_out = transform(seq_in.copy())
    save_seq(seq_out, os.path.join(seq_dir, 'cloth_dilate_seq'))
    seq_merge = merge_seq(seq_out)
    merge_imgs.update({'cloth_dilate':merge_seq(seq_out)})
    print(seq_out.shape, np.min(seq_out), np.max(seq_out), seq_out.dtype)

    transform = build_data_transforms(cloth_erode=True)
    seq_out = transform(seq_in.copy())
    save_seq(seq_out, os.path.join(seq_dir, 'cloth_erode_seq'))
    seq_merge = merge_seq(seq_out)
    merge_imgs.update({'cloth_erode':merge_seq(seq_out)})
    print(seq_out.shape, np.min(seq_out), np.max(seq_out), seq_out.dtype)

    rows = 1
    columns = len(merge_imgs)
    fig = plt.figure()
    merge_imgs_keys = list(merge_imgs.keys())
    for i in range(1, rows*columns+1):
        ax = fig.add_subplot(rows, columns, i)
        key = merge_imgs_keys[i-1]
        ax.set_title(key)
        plt.imshow(merge_imgs[key], cmap = plt.get_cmap('gray'))
    plt.show()
