import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def map01(img):
    img_01 = (img - img.min()) / (img.max() - img.min())
    return img_01


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_auc(HSI_old, HSI_new, gt):
    n_row, n_col, n_band = HSI_old.shape
    n_pixels = n_row * n_col

    img_olds = np.reshape(HSI_old, (n_pixels, n_band), order='F')
    img_news = np.reshape(HSI_new, (n_pixels, n_band), order='F')
    sub_img = img_olds - img_news

    detectmap = np.linalg.norm(sub_img, ord=2, axis=1, keepdims=True) ** 2
    detectmap = detectmap / n_band

    # nomalization
    detectmap = map01(detectmap)

    # get auc
    label = np.reshape(gt, (n_pixels, 1), order='F')

    auc = roc_auc_score(label, detectmap)

    detectmap = np.reshape(detectmap, (n_row, n_col), order='F')

    return auc, detectmap


def TensorToHSI(img):
    HSI = img.squeeze().cpu().data.numpy().transpose((1, 2, 0))
    return HSI


# 计算一阶差分
def compute_diff1(tensor):
    # 按第二个维度计算一阶差分
    h, w = tensor.size()[2], tensor.size()[3]
    diff1 = tensor[:, 1:, :, :] - tensor[:, :-1, :, :] - tensor[:, 1:, :, :].mean() + tensor[:, :-1, :, :].mean()

    # 计算一阶差分的平方和作为损失函数的一部分
    loss_diff1 = torch.sqrt(torch.sum(diff1 ** 2)) / h / w
    return loss_diff1


def compute_diff2(tensor):
    # 计算第二通道上的二阶差分
    diff_2nd = tensor[:, 2:, :, :] - 2 * tensor[:, 1:-1, :, :] + tensor[:, :-2, :, :]
    loss_diff2 = compute_diff1(diff_2nd)
    return loss_diff2
